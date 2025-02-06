import os
import json
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb
from pathlib import Path

class SimpleEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, log_dir):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.metrics_path = os.path.join(log_dir, "metrics.json")
        self.metrics = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True
            )

            # Log metrics
            metrics = {
                "step": self.n_calls,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_length": float(np.mean(lengths))
            }
            self.metrics.append(metrics)

            # Save and print
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)

            print(f"\nStep {self.n_calls}")
            print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")

        return True

class EnhancedEvalCallback(BaseCallback):
    """
    Callback for evaluating and logging agent performance with optional WandB support.
    """
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        hydra_output_dir: str,
        models_dir: str,
        use_wandb: bool = False,
        save_freq: Optional[int] = None,
        save_replay_buffer: bool = False,
        verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.save_freq = save_freq
        self.save_replay_buffer = save_replay_buffer

        # Setup directories
        self.output_dir = Path(hydra_output_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging paths
        self.metrics_path = self.output_dir / "metrics.json"
        self.eval_dir = self.output_dir / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics storage
        self.metrics = []
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create video directory if the environment supports rendering
        if hasattr(self.eval_env, "render_mode"):
            self.video_dir = self.eval_dir / "videos"
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, suffix: str = "") -> None:
        """Save model checkpoint and optionally replay buffer."""
        checkpoint_path = self.models_dir / f"model_step_{self.n_calls}{suffix}"
        self.model.save(checkpoint_path)
        if self.save_replay_buffer and hasattr(self.model, "replay_buffer"):
            replay_buffer_path = self.models_dir / f"replay_buffer_step_{self.n_calls}{suffix}"
            np.save(replay_buffer_path, self.model.replay_buffer)

    def evaluate_policy(self) -> Dict[str, float]:
        """Evaluate the current policy and return metrics."""
        # Sync normalization statistics if needed
        if isinstance(self.eval_env, VecEnv):
            sync_envs_normalization(self.training_env, self.eval_env)

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True
        )

        metrics = {
            "step": self.n_calls,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "success_rate": float(np.mean(np.array(rewards) > 0))  # Assuming sparse rewards
        }

        return metrics

    def _on_step(self) -> bool:
        """Execute callback steps including evaluation, logging, and checkpointing."""
        # Regular model saving if enabled
        if self.save_freq is not None and self.n_calls % self.save_freq == 0:
            self.save_checkpoint()
        
        if self.use_wandb:
            infos = self.locals["infos"][0]
            obs = self.locals["new_obs"]
            rewards = self.locals["rewards"][0]
        
            wandb.log({
                "train/episode_reward": rewards,
                "train/success_rate": float(infos.get("is_success", 0)),
                "train/goal_distance": float(np.linalg.norm(
                    obs["achieved_goal"][0] - obs["desired_goal"][0]
                )),
                "buffer/size": self.model.replay_buffer.size(),  # Use size() method instead of len()
                "train/timesteps": self.n_calls
            }, step=self.n_calls)

            # Log losses if available in model
            if hasattr(self.model, "actor_loss") and self.model.actor_loss is not None:
                wandb.log({
                    "loss/actor": self.model.actor_loss,
                    "loss/critic": self.model.critic_loss,
                    "loss/ent_coef": self.model.ent_coef_loss
                })
                
                
                # Log histograms periodically
                if self.n_calls % (self.eval_freq * 5) == 0:
                    wandb.log({
                        "distributions/actions": wandb.Histogram(self.model.replay_buffer.actions),
                        "distributions/rewards": wandb.Histogram(self.model.replay_buffer.rewards)
                    }, step=self.n_calls)

        # Evaluation
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_policy()
            self.metrics.append(metrics)

            # Save metrics to JSON
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)

            # Log to console
            if self.verbose > 0:
                print(f"\nEvaluation at step {self.n_calls}:")
                print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"Success rate: {metrics['success_rate']:.2%}")

            # Log to WandB if enabled
            if self.use_wandb:
                wandb.log({
                    "eval/mean_reward": metrics["mean_reward"],
                    "eval/std_reward": metrics["std_reward"],
                    "eval/mean_length": metrics["mean_length"],
                    "eval/success_rate": metrics["success_rate"],
                    "eval/min_reward": metrics["min_reward"],
                    "eval/max_reward": metrics["max_reward"]
                }, step=self.n_calls)

            # Save best model
            if metrics["mean_reward"] > self.best_mean_reward:
                self.best_mean_reward = metrics["mean_reward"]
                self.save_checkpoint("_best")

        return True

    def _on_training_end(self) -> None:
        """Save final model and cleanup."""
        # Save final model
        final_model_path = self.models_dir / "final_model"
        self.model.save(final_model_path)

        # Save final metrics summary
        summary = {
            "total_timesteps": self.n_calls,
            "best_mean_reward": float(self.best_mean_reward),
            "final_eval": self.evaluate_policy()
        }

        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
