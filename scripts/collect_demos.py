import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys
import numpy as np
from datetime import datetime
from datasets import Dataset
from collections import defaultdict

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
        
        trainer.model = trainer.model.load(model_path, env=trainer.env)
        print(f"Model loaded successfully")

        # Initialize lists to store all transitions
        all_data = defaultdict(list)
        
        # Collect demonstrations
        total_reward = 0
        n_episodes = cfg.demos.n_episodes
        print(f"\nCollecting {n_episodes} demonstrations...")
        
        for episode in range(n_episodes):
            # Handle the dictionary observation from reset
            obs_dict = trainer.eval_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            
            while not done:
                # Store dictionary observation components
                for key, value in obs_dict.items():
                    # Handle achieved_goal, desired_goal, and observation
                    if isinstance(value, np.ndarray):
                        all_data[f'observations/{key}'].append(value[0].tolist())  # Remove batch dim and convert to list
                
                # Store episode/step info
                all_data['episode_ids'].append(episode)
                all_data['step_ids'].append(episode_step)
                
                # Get action from the model
                action, _ = trainer.model.predict(obs_dict, deterministic=True)
                out = trainer.eval_env.step(action)
                next_obs_dict, reward, terminated, info_list = out[0], out[1], out[2], out[3]

                # Store action and reward
                all_data['actions'].append(action[0].tolist())  # Remove batch dim and convert to list
                all_data['rewards'].append(float(reward[0]))  # Convert numpy float to Python float
                all_data['terminals'].append(bool(terminated[0]))
                
                # Store additional info
                info = info_list[0]  # Get first info dict since we have batch dim of 1
                if 'is_success' in info:
                    all_data['is_success'].append(float(info['is_success']))
                
                obs_dict = next_obs_dict
                episode_reward += reward[0]  # Use the first reward since batch dim is 1
                episode_step += 1
                done = terminated[0]  # Use first value since batch dim is 1
                
            total_reward += episode_reward
            
            if (episode + 1) % 10 == 0:
                print(f"Collected {episode + 1}/{n_episodes} episodes. " 
                      f"Average reward: {total_reward / (episode + 1):.2f}")

        # Create dataset with metadata as additional columns
        all_data['metadata'] = [{
            "env_name": cfg.env.name,
            "algorithm": cfg.agent.algo,
            "num_episodes": n_episodes,
            "avg_reward": float(total_reward / n_episodes),
            "date_collected": datetime.now().isoformat(),
        }] * len(all_data['episode_ids'])  # Repeat metadata for each transition
        
        # Create HF dataset
        dataset = Dataset.from_dict(all_data)
        
        dataset_name = f"{cfg.env.name}_{cfg.agent.algo}_{cfg.demos.n_episodes}"
        dataset_path = Path(cfg.paths.demos_dir) / dataset_name
        dataset.save_to_disk(dataset_path)
        
        # Create symlink to latest
        latest_link = Path(cfg.paths.demos_dir) / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(dataset_path)
        
        print(f"\nDemonstrations saved to: {dataset_path}")
        print(f"Created symlink: {latest_link} -> {dataset_path}")
        print(f"Total episodes: {n_episodes}")
        print(f"Average reward: {total_reward / n_episodes:.2f}")
        
        # Optionally push to HF Hub
        if cfg.logging.wandb.enabled:  # Reuse wandb flag to determine if we should push
            try:
                # You need to be logged in to HF first: huggingface-cli login
                dataset.push_to_hub(
                    f"AOS55/{dataset_name}",
                    private=True
                )
                print(f"Dataset pushed to HF Hub: AOS55/{dataset_name}")
            except Exception as e:
                print(f"Failed to push to HF Hub: {e}")
        
    except Exception as e:
        print(f"\nDemo collection failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    collect_demos()