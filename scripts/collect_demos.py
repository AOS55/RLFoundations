import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
import minari
import numpy as np
from datetime import datetime

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer

@hydra.main(config_path="../conf", config_name="collect_demos", version_base="1.3")
def collect_demos(cfg: DictConfig):
    """Collect demonstrations from a trained model"""
    trainer = DRLTrainer(cfg)
    
    try:
        # Load the best model
        model_path = Path(cfg.paths.model_dir) / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"\nLoading model from: {model_path}")
        print(f"trainer.env: {trainer.env}")
        print(f"Current model parameters: {trainer.model.get_parameters()}")
        trainer.model.load(model_path)
        print(f"Model loaded successfully")

        # Collect demonstrations
        demonstrations = []
        total_reward = 0
        n_episodes = cfg.demos.n_episodes

        print(f"\nCollecting {n_episodes} demonstrations...")
        
        for episode in range(n_episodes):
            obs, _ = trainer.eval_env.reset()
            done = False
            episode_reward = 0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "terminals": []
            }

            while not done:
                action = trainer.model.predict(obs, deterministic=True)[0]
                next_obs, reward, terminated, truncated, info = trainer.eval_env.step(action)
                
                # Store transition
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["terminals"].append(terminated)
                
                obs = next_obs
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            demonstrations.append(episode_data)
            
            if (episode + 1) % 10 == 0:
                print(f"Collected {episode + 1}/{n_episodes} episodes. " 
                      f"Average reward: {total_reward / (episode + 1):.2f}")

        # Create dataset name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{cfg.env.name}_{cfg.algo}_{timestamp}"
        
        # Save demonstrations using minari
        dataset = minari.DatasetBuilder(
            dataset_id=dataset_name,
            algorithm_name=cfg.algo,
            environment_name=cfg.env.name,
            data=demonstrations
        )
        
        # Save and create symlink to latest
        dataset_path = Path(cfg.paths.demos_dir) / dataset_name
        dataset.save(str(dataset_path))
        
        latest_link = Path(cfg.paths.demos_dir) / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(dataset_path)

        print(f"\nDemonstrations saved to: {dataset_path}")
        print(f"Created symlink: {latest_link} -> {dataset_path}")
        print(f"Total episodes: {n_episodes}")
        print(f"Average reward: {total_reward / n_episodes:.2f}")

    except Exception as e:
        print(f"\nDemo collection failed with error: {str(e)}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    collect_demos()