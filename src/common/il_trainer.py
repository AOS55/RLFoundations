from pathlib import Path
import json
import minari
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from .base_trainer import BaseTrainer

class ILTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset_path = cfg.dataset.path
        self.best_reward = float('-inf')

    def setup_model(self):
        """Initialize the BC model"""
        # Get algorithm-specific parameters
        algo_params = getattr(self.cfg, self.cfg.algo)
        
        self.model = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **algo_params
        )

    def load_demonstrations(self):
        """Load and preprocess demonstrations"""
        dataset = minari.load_dataset(self.dataset_path)

        transitions = []
        for episode in dataset:
            transitions.extend(zip(
                episode["observations"],
                episode["actions"]
            ))

        return transitions

    def evaluate_and_save(self, epoch):
        """Evaluate current model and save if it's the best"""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.cfg.evaluation.n_episodes,
            deterministic=self.cfg.evaluation.deterministic
        )

        if mean_reward > self.best_reward:
            print(f"\nNew best model at epoch {epoch} with reward: {mean_reward:.2f}")
            self.best_reward = mean_reward
            best_model_path = Path(self.cfg.paths.model_dir) / "best_model.zip"
            self.model.save_policy(best_model_path)

            # Save best model metrics
            metrics = {
                "epoch": epoch,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "model_path": str(best_model_path)
            }

            with open(Path(self.cfg.paths.metrics_dir) / "best_model_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        return mean_reward, std_reward

    def train(self):
        """Train the IL agent with best model saving"""
        self.setup_model()
        transitions = self.load_demonstrations()

        print(f"Training on {len(transitions)} transitions...")

        # Training loop with periodic evaluation
        eval_freq = max(1, self.cfg.bc.n_epochs // 10)  # Evaluate 10 times during training

        try:
            for epoch in range(self.cfg.bc.n_epochs):
                self.model.train(
                    transitions,
                    n_epochs=1,
                    batch_size=self.cfg.bc.batch_size
                )

                if epoch % eval_freq == 0:
                    mean_reward, std_reward = self.evaluate_and_save(epoch)
                    print(f"Epoch {epoch}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

            # Final evaluation
            mean_reward, std_reward = self.evaluate_and_save(self.cfg.bc.n_epochs)

            # Save training summary
            summary = {
                "dataset_path": self.dataset_path,
                "n_transitions": len(transitions),
                "n_epochs": self.cfg.bc.n_epochs,
                "batch_size": self.cfg.bc.batch_size,
                "best_model_path": str(Path(self.cfg.paths.model_dir) / "best_model.zip"),
                "best_reward": float(self.best_reward)
            }

            with open(Path(self.cfg.paths.metrics_dir) / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            return Path(self.cfg.paths.model_dir) / "best_model.zip"

        except KeyboardInterrupt:
            print("\nTraining interrupted. Best model was already saved.")
            raise

    def evaluate(self):
        """Evaluate the trained model"""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.cfg.evaluation.n_episodes,
            deterministic=self.cfg.evaluation.deterministic
        )

        metrics = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": self.cfg.evaluation.n_episodes
        }

        with open(Path(self.cfg.paths.eval_dir) / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward