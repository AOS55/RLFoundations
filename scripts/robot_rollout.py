import hydra
import os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys
import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer, ILTrainer
from src.common import ensure_model_exists

def configure_for_visualization(cfg: DictConfig) -> DictConfig:
    """Configure settings for visualization only."""
    # Create a mutable copy of the config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Disable WandB
    if 'wandb' in cfg.logging:
        cfg.logging.wandb.enabled = False

    # Set render mode to human
    cfg.env.render_mode = 'human'

    # Disable output saving
    cfg.evaluation.save_video = False
    cfg.evaluation.record_video = False
    if 'save_metrics' in cfg.evaluation:
        cfg.evaluation.save_metrics = False

    # Set paths to temporary directory to avoid saving
    temp_dir = Path("/tmp/rl_visualization")
    cfg.paths.base_output_dir = str(temp_dir)
    cfg.paths.video_dir = str(temp_dir / "videos")

    return cfg

@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def run_rollout(cfg: DictConfig):
    """Run policy rollout on the robot"""
    # Configure for visualization
    cfg = configure_for_visualization(cfg)

    trainer_class = {
        "drl": DRLTrainer,
        "il": ILTrainer
    }.get(cfg.type)

    if trainer_class is None:
        raise ValueError(f"Unknown training type: {cfg.type}")

    print(f"\nRunning {cfg.type.upper()} model with {cfg.agent.algo} algorithm")
    print(f"Environment: {cfg.env.name}")

    trainer = trainer_class(cfg)

    try:
        # Load the best model
        local_model_path = Path(cfg.paths.model_dir) / "best_model.zip"
        model_path = ensure_model_exists(cfg, local_model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load model with environment
        trainer.model = trainer.model.load(model_path, env=trainer.env)
        print(f"Model loaded successfully")

        # Run episodes
        total_reward = 0
        n_episodes = cfg.evaluation.n_episodes

        for episode in range(n_episodes):
            obs_dict = trainer.eval_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                action, _ = trainer.model.predict(obs_dict, deterministic=True)
                out = trainer.eval_env.step(action)
                next_obs_dict, reward, terminated, info_list = out[0], out[1], out[2], out[3]
                obs_dict = next_obs_dict
                episode_reward += reward[0]  # Use the first reward since batch dim is 1
                episode_step += 1
                done = terminated[0]  # Use first value since batch dim is 1

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
