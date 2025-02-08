import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer, ILTrainer

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def run_rollout(cfg: DictConfig):
    """Run policy rollout on the robot"""
    # Select appropriate trainer based on config type
    trainer_class = {
        "drl": DRLTrainer,
        "il": ILTrainer
    }.get(cfg.type)

    if trainer_class is None:
        raise ValueError(f"Unknown training type: {cfg.type}")

    print(f"\nRunning {cfg.type.upper()} model with {cfg.algo} algorithm")
    print(f"Environment: {cfg.env.name}")

    # Initialize trainer with human-viewable rendering
    cfg.env.render_config.render_mode = "human"
    trainer = trainer_class(cfg)

    try:
        # Load the best model
        model_path = Path(cfg.paths.model_dir) / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        trainer.model.load(model_path)
        print(f"Loaded model from: {model_path}")

        # Run episodes
        total_reward = 0
        n_episodes = cfg.evaluation.n_episodes

        for episode in range(n_episodes):
            obs, _ = trainer.eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = trainer.model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = trainer.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            print(f"Episode {episode + 1}/{n_episodes}, Reward: {episode_reward:.2f}")

        print(f"\nRollout completed.")
        print(f"Average reward over {n_episodes} episodes: {total_reward / n_episodes:.2f}")

    except KeyboardInterrupt:
        print("\nRollout interrupted by user.")
    except Exception as e:
        print(f"\nRollout failed with error: {str(e)}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    run_rollout()