import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import os

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer, ILTrainer

def print_config_tree(cfg: DictConfig, indent: int = 0):
    """Print the config tree in a readable format"""
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            print("  " * indent + f"{key}:")
            print_config_tree(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    """Main training script with automatic trainer selection"""
    # Print training configuration
    print("\nTraining Configuration:")
    print_config_tree(cfg)

    # Determine training type and algorithm
    training_type = cfg.type
    algorithm = cfg.algo if hasattr(cfg, 'algo') else cfg.agent.algo

    print(f"\nStarting {training_type.upper()} training with {algorithm} algorithm")
    print(f"Environment: {cfg.env.name}")
    print(f"Model will be saved to: {cfg.paths.model_dir}")
    print(f"Outputs will be saved to: {cfg.paths.base_output_dir}")

    # Create necessary directories
    Path(cfg.paths.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.base_output_dir).mkdir(parents=True, exist_ok=True)

    # Select appropriate trainer based on type
    trainer_class = {
        "drl": DRLTrainer,
        "il": ILTrainer
    }.get(training_type)

    if trainer_class is None:
        raise ValueError(f"Unknown training type: {training_type}")

    # Initialize and run trainer
    trainer = None
    try:
        # Initialize trainer
        trainer = trainer_class(cfg)

        # Start training
        print("\nInitializing training...")
        model_path = trainer.train()

        print(f"\nTraining completed successfully!")
        print(f"Best model saved to: {model_path}")

        # if cfg.evaluation.save_metrics:
        #     # Run final evaluation
        #     print("\nRunning final evaluation...")
            # # mean_reward, std_reward = trainer.evaluate()
            # print(f"\nFinal evaluation results:")
            # print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            # print(f"Number of episodes: {cfg.evaluation.n_episodes}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if trainer is not None:
            print("Attempting to save interrupted model...")
            try:
                interrupted_path = Path(cfg.paths.model_dir) / "interrupted_model.zip"
                trainer.model.save(interrupted_path)
                print(f"Interrupted model saved to: {interrupted_path}")
            except Exception as e:
                print(f"Failed to save interrupted model: {e}")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        if os.getenv("HYDRA_FULL_ERROR"):
            raise
        else:
            print("\nSet HYDRA_FULL_ERROR=1 for full error traceback")
            sys.exit(1)
    finally:
        if trainer is not None:
            print("\nCleaning up...")
            # trainer.cleanup()

def main():
    """Wrapper function to handle any setup before training"""
    try:
        train()
    except Exception as e:
        print(f"Error during training setup: {e}")
        if os.getenv("HYDRA_FULL_ERROR"):
            raise
        else:
            print("\nSet HYDRA_FULL_ERROR=1 for full error traceback")
            sys.exit(1)

if __name__ == "__main__":
    main()
