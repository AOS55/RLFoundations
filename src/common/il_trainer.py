from omegaconf import OmegaConf
import json
from pathlib import Path
import torch
import torch.nn as nn
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from datasets import load_dataset, load_from_disk
import numpy as np
import wandb
from typing import Dict, Any
import gymnasium as gym
from gymnasium.spaces.utils import flatten

from .base_trainer import BaseTrainer
from .callbacks import EnhancedEvalCallback

class ILTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_reward = float('-inf')
        self.setup_demonstrations()
        self.setup_model()

    def setup_demonstrations(self):
        """Load demonstrations from local path or download from Hugging Face"""
        dataset_path = Path(self.cfg.dataset.local_path)

        print(f"Loading demonstrations from local path: {dataset_path}")
        try:
            if dataset_path.exists():
                self.demonstrations = load_from_disk(str(dataset_path))
            else:
                print(f"Local path not found. Downloading from Hugging Face: {self.cfg.dataset.repo_id}")
                self.demonstrations = load_dataset(self.cfg.dataset.repo_id)
                # Save locally for future use
                self.demonstrations.save_to_disk(dataset_path)

            # Print dataset information
            print("\nDemonstrations dataset info:")
            if hasattr(self.demonstrations, 'keys'):  # DatasetDict case
                print(f"Dataset splits: {list(self.demonstrations.keys())}")
                dataset = self.demonstrations['train']
                print(f"Features: {dataset.features}")
                print(f"Number of samples: {len(dataset)}")
            else:
                print(f"Features: {self.demonstrations.features}")
                print(f"Number of samples: {len(self.demonstrations)}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load demonstrations. Please ensure demonstrations exist at {dataset_path} "
                f"or can be downloaded from {self.cfg.dataset.repo_id}\n"
                f"Error: {str(e)}"
            )

    def setup_model(self):
        """Initialize the BC model with the correct observation space"""
        transitions = self.prepare_transitions()

        # Create a flat observation space that matches our concatenated observations
        observation_dim = (
            self.env.observation_space['observation'].shape[0] +
            self.env.observation_space['achieved_goal'].shape[0] +
            self.env.observation_space['desired_goal'].shape[0]
        )

        flat_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )

        self.model = bc.BC(
            observation_space=flat_observation_space,
            action_space=self.env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(),
            device="auto",
            batch_size=self.cfg.agent.il.training.batch_size
        )

    def prepare_transitions(self):
        """Prepare transitions in a simple format for BC"""
        from imitation.data.types import Transitions

        # Get the train split
        dataset = self.demonstrations['train']

        # For Fetch environments, we need to stack the observations into a single array
        obs_array = np.array(dataset['observations/observation'])
        achieved_goals = np.array(dataset['observations/achieved_goal'])
        desired_goals = np.array(dataset['observations/desired_goal'])

        # Stack all observation components into a single array
        # This matches how the Fetch environment's observation space works
        observations = np.concatenate([
            obs_array,
            achieved_goals,
            desired_goals
        ], axis=1)

        # Create next observations (shift by 1)
        next_observations = np.roll(observations, -1, axis=0)
        next_observations[-1] = observations[-1]

        # Convert actions and terminals
        actions = np.array(dataset['actions'], dtype=np.float32)
        terminals = np.array(dataset['terminals'], dtype=bool)

        return Transitions(
            obs=observations.astype(np.float32),
            next_obs=next_observations.astype(np.float32),
            acts=actions.astype(np.float32),
            infos=np.array([{} for _ in range(len(dataset))]),
            dones=terminals
        )

    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []

        # Enhanced evaluation callback
        eval_callback = EnhancedEvalCallback(
            eval_env=self.eval_env,
            eval_freq=self.cfg.training.eval_freq,
            n_eval_episodes=self.cfg.evaluation.n_episodes,
            hydra_output_dir=self.cfg.paths.base_output_dir,
            models_dir=self.cfg.paths.model_dir,
            use_wandb=self.cfg.logging.wandb.enabled,
            save_freq=self.cfg.logging.checkpoint.save_freq,
            save_replay_buffer=False  # BC doesn't use replay buffer
        )
        callbacks.append(eval_callback)

        return callbacks

    def train(self):
        """Train the BC agent using demonstrations"""
        try:
            print(f"\nStarting BC training for {self.cfg.agent.il.training.n_epochs} epochs...")

            # Training loop
            for epoch in range(self.cfg.agent.il.training.n_epochs):
                # Train for one epoch
                self.model.train(n_epochs=1)

                # Get the current loss from the model's loss history
                if hasattr(self.model, 'train_loss'):
                    current_loss = self.model.train_loss

                    # Log training metrics
                    if self.cfg.logging.wandb.enabled:
                        wandb.log({
                            "train/loss": current_loss,
                            "train/epoch": epoch,
                        })

                # Periodic evaluation
                # if epoch % (self.cfg.agent.il.training.n_epochs // 10) == 0:
                #     mean_reward, std_reward = self.evaluate()
                #     print(f"Epoch {epoch}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

                #     if mean_reward > self.best_reward:
                #         self.best_reward = mean_reward
                #         best_model_path = Path(self.cfg.paths.model_dir) / "best_model.zip"
                #         self.model.save(best_model_path)
                #         print(f"New best model saved with reward: {mean_reward:.2f}")

            # Save final model
            final_model_path = Path(self.cfg.paths.model_dir) / "final_model.zip"
            # self.model.save(final_model_path)

            # Save training summary
            summary = {
                "total_epochs": self.cfg.agent.il.training.n_epochs,
                # "n_transitions": len(self.transitions),
                "batch_size": self.cfg.agent.il.training.batch_size,
                "best_mean_reward": float(self.best_reward),
                "final_model_path": str(final_model_path),
                "best_model_path": str(Path(self.cfg.paths.model_dir) / "best_model.zip")
            }

            summary_path = Path(self.cfg.paths.metrics_dir) / "training_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\nTraining completed!")
            print(f"Best mean reward: {self.best_reward:.2f}")
            print(f"Models saved to: {self.cfg.paths.model_dir}")

            return Path(self.cfg.paths.model_dir) / "best_model.zip"

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model...")
            interrupted_path = Path(self.cfg.paths.model_dir) / "interrupted_model.zip"
            self.model.save(interrupted_path)
            raise
        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            raise

    # def evaluate(self):
    #     """Custom evaluation method for BC policy with dictionary observations"""
    #     episode_rewards = []
    #     episode_lengths = []

    #     for _ in range(self.cfg.evaluation.n_episodes):
    #         obs = self.eval_env.reset()  # Correct way to get obs from vectorized env
    #         done = False
    #         episode_reward = 0
    #         episode_length = 0

    #         while not done:
    #             # Flatten the observation for the BC policy
    #             flat_obs = np.concatenate([
    #                 obs[0]['observation'].flatten(),  # Add [0] to get first env's obs
    #                 obs[0]['achieved_goal'].flatten(),
    #                 obs[0]['desired_goal'].flatten()
    #             ]).astype(np.float32)

    #             # Convert to tensor and add batch dimension
    #             flat_obs_tensor = torch.as_tensor(flat_obs).unsqueeze(0).to(self.model.policy.device)

    #             # Get action from policy and convert to numpy
    #             with torch.no_grad():
    #                 action = self.model.policy(flat_obs_tensor)
    #                 action = action.cpu().numpy()

    #             # Step environment with numpy array
    #             obs, reward, terminated, truncated, _ = self.eval_env.step(action)
    #             done = terminated[0] or truncated[0]  # Use first env's termination signals
    #             episode_reward += reward[0]  # Use first env's reward
    #             episode_length += 1

    #         episode_rewards.append(episode_reward)
    #         episode_lengths.append(episode_length)

    #     mean_reward = np.mean(episode_rewards)
    #     std_reward = np.std(episode_rewards)

    #     if self.cfg.logging.wandb.enabled:
    #         wandb.log({
    #             "eval/mean_reward": mean_reward,
    #             "eval/std_reward": std_reward,
    #             "eval/mean_length": np.mean(episode_lengths)
    #         })

    #     return mean_reward, std_reward
