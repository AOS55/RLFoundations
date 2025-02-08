import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer, ILTrainer

@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    """Main training script with automatic trainer selection"""
    # Get agent type from nested config
    agent_type = cfg.agent.drl.type if "drl" in cfg.agent else cfg.agent.il.type
    print(f"cfg: {cfg}") 
    # Select appropriate trainer based on agent type
    trainer_class = {
        "drl": DRLTrainer,
        "il": ILTrainer
    }.get(agent_type)

    if trainer_class is None:
        raise ValueError(f"Unknown training type: {agent_type}")

    # Print training information using correct config paths
    print(f"\nStarting {agent_type.upper()} training with {cfg.agent.algo} algorithm")
    print(f"Environment: {cfg.env.name}")
    print(f"Model will be saved to: {cfg.paths.model_dir}")
    print(f"Outputs will be saved to: {cfg.paths.base_output_dir}")

    # Initialize and run trainer
    trainer = trainer_class(cfg)
    try:
        model_path = trainer.train()
        print(f"\nTraining completed. Best model saved to: {model_path}")

        if cfg.evaluation.save_metrics:
            # Run final evaluation
            mean_reward, std_reward = trainer.evaluate()
            print(f"\nFinal evaluation:")
            print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    train()