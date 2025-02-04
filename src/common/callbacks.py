import os
import json
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

class FetchRecordCallback(BaseCallback):
    """
    Callback for recording videos and logging metrics for FetchPickAndPlace environment.

    :param eval_env: Environment to use for evaluation
    :param log_dir: Directory to save videos and metrics
    :param eval_freq: Evaluate the agent every eval_freq steps
    :param n_eval_episodes: Number of episodes to evaluate
    :param record_freq: Record video every record_freq evaluations
    """
    def __init__(
        self,
        eval_env: VecEnv,
        log_dir: str,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        record_freq: int = 2,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.record_freq = record_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

        # Create directories
        self.video_folder = os.path.join(log_dir, "videos")
        self.metrics_path = os.path.join(log_dir, "metrics.json")
        os.makedirs(self.video_folder, exist_ok=True)

        # Initialize metrics list
        self.metrics = []
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.metrics = json.load(f)

    def _on_step(self) -> bool:
        """
        Called after each step in training.
        """
        if self.n_calls % self.eval_freq == 0:
            # Determine if we should record video this evaluation
            should_record = (self.n_calls // self.eval_freq) % self.record_freq == 0

            # Set up video recording if needed
            if should_record:
                base_env = self.eval_env.envs[0].unwrapped
                eval_env = DummyVecEnv([
                    lambda: RecordVideo(
                        base_env,
                        video_folder=os.path.join(self.video_folder, f"step_{self.n_calls}"),
                        episode_trigger=lambda x: True,
                        disable_logger=True
                    )
                ])

                # Sync normalization stats if needed
                if self.model.get_vec_normalize_env() is not None:
                    sync_envs_normalization(self.model.get_vec_normalize_env(), eval_env)
            else:
                eval_env = self.eval_env

            # Evaluate the agent
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards])

            # Log metrics
            metrics = {
                "timestep": self.n_calls,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "mean_episode_length": float(mean_length),
                "success_rate": float(success_rate),
                "video_recorded": should_record
            }
            self.metrics.append(metrics)

            # Save metrics to file
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)

            if self.verbose > 0:
                print(f"Step {self.n_calls}")
                print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Success rate: {success_rate:.2%}")
                print(f"Mean episode length: {mean_length:.2f}")
                if should_record:
                    print(f"Video saved to {self.video_folder}/step_{self.n_calls}")

            # Close video recording environment if it was created
            if should_record:
                eval_env.close()

        return True
