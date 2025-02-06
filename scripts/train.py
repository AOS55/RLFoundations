import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import gymnasium_robotics
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path
from src.common import SimpleEvalCallback, EnhancedEvalCallback
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import minari
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo

def make_env(env_name, render_mode=None):
    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        return RolloutInfoWrapper(env)
    return _init

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    # Create vectorized environment
    env = DummyVecEnv([make_env(cfg.env.name) for _ in range(cfg.experiment.num_envs)])

    def make_eval_env(env_name):

        def _init():
            env = gym.make(env_name, render_mode="rgb_array")
            # Wrap with video recording right away
            env = RecordVideo(
                env,
                video_folder=str(Path(cfg.logging.output.video_dir)),
                episode_trigger=lambda x: x % cfg.evaluation.get("record_freq", 50000) == 0
            )
            return RolloutInfoWrapper(env)
        return _init

    eval_env = DummyVecEnv([make_eval_env(cfg.env.name)])

    if cfg.logging.wandb:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            group=cfg.logging.wandb.group,
            name=cfg.logging.wandb.name,
            config=wandb_config
        )

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

        # Check if HER is enabled
        if cfg.her.enabled:
            model = SAC(
                policy="MultiInputPolicy",
                env=env,
                replay_buffer_class=HerReplayBuffer,  # Explicitly setting HER Replay Buffer
                replay_buffer_kwargs={
                    "n_sampled_goal": cfg.her.n_sampled_goal,
                    "goal_selection_strategy": cfg.her.goal_selection_strategy,
                },
                verbose=1 if cfg.logging.wandb else 0,
            )

            env = gym.make(cfg.env.name)
            assert hasattr(env.unwrapped, "compute_reward"), "HER requires the environment to have `compute_reward()`!"

        else:
            model = SAC(
                policy="MultiInputPolicy",
                env=env,
                verbose=1 if cfg.logging.wandb else 0,
            )

        # Create callback
        eval_callback = EnhancedEvalCallback(
            eval_env=eval_env,
            eval_freq=cfg.training.eval_freq,
            n_eval_episodes=cfg.training.n_eval_episodes,
            hydra_output_dir=str(Path.cwd()),
            models_dir=str(Path(cfg.logging.checkpoint.save_path)),
            use_wandb=cfg.logging.wandb.enabled,
            save_freq=cfg.logging.checkpoint.save_freq,
            save_replay_buffer=cfg.logging.checkpoint.save_replay_buffer
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

        if cfg.logging.wandb.enabled:
            wandb.finish()

if __name__ == "__main__":
    train()
