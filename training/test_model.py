import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from rover_env import RoverEnv
from stable_baselines3 import PPO

print("Loading trained model...")
env = RoverEnv()
model = PPO.load("../models/ppo_rover_v4_wall_aware")
print("Running trained agent for 5 episodes...")
for episode in range(5):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Episode {episode+1}: Steps={steps}, Total Reward={total_reward:.2f}")

env.close()
print("Test complete!")