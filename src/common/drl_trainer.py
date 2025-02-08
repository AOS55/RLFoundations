from omegaconf import OmegaConf
import json
from pathlib import Path
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her import HerReplayBuffer
from wandb.integration.sb3 import WandbCallback
import torch.nn as nn

from .base_trainer import BaseTrainer
from .callbacks import EnhancedEvalCallback

class DRLTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_model()

    def _get_activation(self, activation_str):
        """Convert activation string to PyTorch activation function"""
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU
        }
        return activation_map.get(activation_str.lower(), nn.ReLU)
    
    def setup_model(self):
        """Initialize the DRL model based on config"""
        # Map algorithm names to their classes
        ALGO_CLASSES = {
            "sac": SAC,
            "td3": TD3
        }

        # Get model class and algorithm-specific parameters
        model_class = ALGO_CLASSES[self.cfg.agent.algo]
        algo_params = OmegaConf.to_container(getattr(self.cfg.agent, self.cfg.agent.algo), resolve=True)

        # Convert activation string to actual PyTorch function
        if "policy_kwargs" in algo_params:
            activation_str = algo_params["policy_kwargs"].get("activation_fn", "relu")
            algo_params["policy_kwargs"]["activation_fn"] = self._get_activation(activation_str)

        # Setup HER if enabled
        if self.cfg.agent.drl.her.enabled:
            her_kwargs = dict(
                n_sampled_goal=self.cfg.agent.drl.her.n_sampled_goal,
                goal_selection_strategy=self.cfg.agent.drl.her.goal_selection_strategy,
            )
            algo_params = {
                **algo_params,
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": her_kwargs
            }

        if self.cfg.logging.tensorboard.enabled:
            algo_params["tensorboard_log"] = str(Path(self.cfg.paths.tensorboard_dir))
    
        # Initialize model
        self.model = model_class(
            env=self.env,
            policy=self.cfg.agent.drl.policy,
            **algo_params
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
            save_replay_buffer=self.cfg.logging.checkpoint.save_replay_buffer
        )
        callbacks.append(eval_callback)
        
        if self.cfg.logging.wandb.enabled:
            callbacks.append(WandbCallback())
        
        return callbacks

    def train(self):
        """Train the DRL agent"""
        callbacks = self.setup_callbacks()
        model_dir = Path(self.cfg.paths.model_dir)
        metrics_dir = Path(self.cfg.paths.metrics_dir)

        try:
            # Train the model
            self.model.learn(
                total_timesteps=self.cfg.training.total_timesteps,
                callback=callbacks
            )

            # Save final model and training summary
            final_model_path = model_dir / "final_model.zip"
            self.model.save(final_model_path)

            summary = {
                "total_timesteps": self.cfg.training.total_timesteps,
                "algo": self.cfg.agent.algo,
                "env_name": self.cfg.env.name,
                "final_model_path": str(final_model_path),
                "best_model_path": str(model_dir / "best_model.zip")
            }

            with open(metrics_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            return model_dir / "best_model.zip"

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current model...")
            self.model.save(model_dir / "interrupted_model.zip")
            raise

    # def evaluate(self):
    #     """Evaluate the trained model"""
    #     mean_reward, std_reward = evaluate_policy(
    #         self.model,
    #         self.eval_env,
    #         n_eval_episodes=self.cfg.training.evaluation.n_episodes,
    #         deterministic=self.cfg.training.evaluation.deterministic
    #     )

    #     metrics = {
    #         "mean_reward": float(mean_reward),
    #         "std_reward": float(std_reward),
    #         "n_episodes": self.cfg.training.evaluation.n_episodes
    #     }

    #     eval_dir = Path(self.cfg.paths.eval_dir)
    #     with open(eval_dir / "eval_metrics.json", "w") as f:
    #         json.dump(metrics, f, indent=2)

    #     print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    #     return mean_reward, std_reward