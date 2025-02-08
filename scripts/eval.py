import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.common import DRLTrainer, ILTrainer

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def evaluate(cfg: DictConfig):
    """Evaluation script for trained models"""
    # Select appropriate trainer based on config type
    trainer_class = {
        "drl": DRLTrainer,
        "il": ILTrainer
    }.get(cfg.type)

    if trainer_class is None:
        raise ValueError(f"Unknown training type: {cfg.type}")

    print(f"\nEvaluating {cfg.type.upper()} model with {cfg.algo} algorithm")
    print(f"Environment: {cfg.env.name}")

    # Initialize trainer
    trainer = trainer_class(cfg)
    try:
        # Load the best model
        model_path = Path(cfg.paths.model_dir) / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load model
        trainer.model.load(model_path)
        print(f"Loaded model from: {model_path}")

        # Run evaluation
        mean_reward, std_reward = trainer.evaluate()
        print(f"\nEvaluation results:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Number of episodes: {cfg.evaluation.n_episodes}")
        print(f"\nEvaluation videos saved to: {cfg.paths.video_dir}")

    except Exception as e:
        print(f"\nEvaluation failed with error: {str(e)}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    evaluate()