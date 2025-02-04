import hydra
from omegaconf import DictConfig
import gymnasium_robotics
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from src.common import FetchRecordCallback
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import minari
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(env_name, render_mode=None):
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        return RolloutInfoWrapper(env)
    return _init

@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # Create vectorized environment
    env = DummyVecEnv([make_env(cfg.env.name) for _ in range(cfg.experiment.num_envs)])
    eval_env = DummyVecEnv([make_env(cfg.env.name, render_mode="rgb_array")])

    # Set random seed
    rng = np.random.default_rng(cfg.experiment.seed)

    if cfg.type == "il":
        # Load demonstrations from Minari
        dataset = minari.load_dataset(cfg.agent.demonstrations_path)

        # Convert Minari dataset to transitions
        trajectories = []
        for i in range(len(dataset)):
            traj = {
                "obs": dataset[i]["observations"],
                "acts": dataset[i]["actions"],
                "infos": dataset[i]["infos"],
                "terminal": dataset[i]["terminations"][-1]
            }
            trajectories.append(traj)
        transitions = rollout.flatten_trajectories(trajectories)

        # Create BC trainer
        bc_trainer = hydra.utils.instantiate(
            cfg.agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
        )

        # Evaluate untrained policy
        print("Evaluating untrained policy...")
        reward_before, _ = evaluate_policy(
            bc_trainer.policy,
            eval_env,
            n_eval_episodes=cfg.evaluation.n_episodes,
            deterministic=cfg.evaluation.deterministic
        )
        print(f"Reward before training: {reward_before}")

        # Train the policy
        print("Training policy using Behavior Cloning...")
        bc_trainer.train(
            n_epochs=cfg.training.total_timesteps,
            progress_bar=True
        )

        # Save the trained policy
        bc_trainer.save_policy(f"{cfg.checkpoint.save_path}/final_policy")

        # Evaluate trained policy
        print("Evaluating trained policy...")
        reward_after, _ = evaluate_policy(
            bc_trainer.policy,
            eval_env,
            n_eval_episodes=cfg.evaluation.n_episodes,
            deterministic=cfg.evaluation.deterministic
        )
        print(f"Reward after training: {reward_after}")

    else:  # DRL training

         # Configure HER if enabled
        model_kwargs = {}
        if cfg.her.enabled:
            model_kwargs.update({
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": {
                    "n_sampled_goal": cfg.her.n_sampled_goal,
                    "goal_selection_strategy": cfg.her.goal_selection_strategy,
                }
            })

        # Create agent
        model = hydra.utils.instantiate(
            cfg.agent,
            env=env,
            verbose=1 if cfg.logging.wandb else 0,
            **model_kwargs
        )

        eval_callback = FetchRecordCallback(
            eval_env=eval_env,
            log_dir=f"{cfg.experiment.checkpoint.save_path}/logs",
            eval_freq=cfg.training.eval_freq,
            n_eval_episodes=cfg.training.n_eval_episodes,
            record_freq=cfg.evaluation.get("record_freq", 50000),
            deterministic=cfg.evaluation.deterministic,
            verbose=1
        )

        # Train the agent
        model.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=eval_callback
        )

        # Save the final model
        model.save(f"{cfg.checkpoint.save_path}/final_model")

        # Final evaluation
        reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=cfg.evaluation.n_episodes,
            deterministic=cfg.evaluation.deterministic
        )
        print(f"Final evaluation: {reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    train()
