import hydra
from omegaconf import DictConfig
import gymnasium as gym
import minari
import numpy as np
from datetime import datetime
import scipy.linalg
import control
from stable_baselines3 import SAC, TD3
from pathlib import Path

def get_mpc_action(state, target, dt=0.05, horizon=10):
    """Simple MPC controller for FetchPickAndPlace"""
    # Simplified linear dynamics for the end effector
    A = np.eye(3)  # Position only model
    B = np.eye(3) * dt  # Direct velocity control

    # Cost matrices
    Q = np.eye(3)  # State cost
    R = np.eye(3) * 0.1  # Control cost

    # Terminal cost
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Current error
    error = state['achieved_goal'] - target

    # Solve finite horizon LQR
    K, _, _ = control.dlqr(A, B, Q, R)

    # Compute control action
    u = -K @ error

    # Clip actions to environment limits
    u = np.clip(u, -1.0, 1.0)

    # Add gripper action (open/close based on distance to object)
    dist_to_object = np.linalg.norm(state['achieved_goal'] - state['observation'][:3])
    gripper_action = -1.0 if dist_to_object < 0.05 else 1.0

    return np.concatenate([u, [gripper_action]])

def load_drl_agent(cfg):
    """Load a trained DRL agent"""
    model_path = Path(cfg.demo.model_path)
    if not model_path.exists():
        raise ValueError(f"Model not found at {model_path}")

    if cfg.demo.algo == "sac":
        return SAC.load(model_path)
    elif cfg.demo.algo == "td3":
        return TD3.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.demo.algo}")

@hydra.main(config_path="../conf", config_name="config")
def collect_demonstrations(cfg: DictConfig):
    env = gym.make(cfg.env.name)

    # Load DRL agent if specified
    drl_agent = None
    if cfg.demo.source == "drl":
        drl_agent = load_drl_agent(cfg)

    # Generate unique dataset name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"fetch_pick_place_{cfg.demo.source}_{timestamp}"

    trajectories = []
    episode_count = 0
    total_success = 0

    while episode_count < cfg.demo.num_episodes:
        obs, info = env.reset()
        terminated = truncated = False

        observations = []
        actions = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        while not (terminated or truncated):
            if cfg.demo.source == "drl":
                # Use trained DRL agent
                action, _ = drl_agent.predict(obs, deterministic=True)
            elif cfg.demo.source == "mpc":
                # Use MPC controller
                target = info['desired_goal']
                state = {
                    'achieved_goal': obs['achieved_goal'],
                    'observation': obs['observation'],
                    'desired_goal': target
                }
                action = get_mpc_action(state, target)

            next_obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)

            obs = next_obs

        # Track success rate
        if info.get('is_success', 0.0) > 0:
            total_success += 1

        trajectories.append({
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "terminations": np.array(terminations),
            "truncations": np.array(truncations),
            "infos": infos
        })

        episode_count += 1
        success_rate = (total_success / episode_count) * 100
        print(f"Episode {episode_count}/{cfg.demo.num_episodes} - Success Rate: {success_rate:.2f}%")

    # Create Minari dataset
    minari.create_dataset_from_sequences(
        dataset_name=dataset_name,
        environment_name=cfg.env.name,
        sequences=trajectories,
        algorithm_name=f"{cfg.demo.source}_{cfg.demo.algo if cfg.demo.source == 'drl' else 'mpc'}",
        author=cfg.demo.author,
        code_permalink="https://github.com/yourusername/fetch_pick_and_place",
    )

    print(f"\nCreated Minari dataset: {dataset_name}")
    print(f"Final Success Rate: {success_rate:.2f}%")
    return dataset_name

if __name__ == "__main__":
    collect_demonstrations()
