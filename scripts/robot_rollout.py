import gymnasium as gym
import gymnasium_robotics
import numpy as np

def run_fetch_environment(env_name='FetchReach-v4', episodes=1000, max_steps=100000):
    env = gym.make(env_name, render_mode='human')

    for episode in range(episodes):
        obs, info = env.reset()
        print(f"Episode {episode + 1}")

        for step in range(max_steps):
            action = env.action_space.sample()  # Sample random action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step: {step + 1}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                break

    env.close()

if __name__ == "__main__":
    run_fetch_environment()
