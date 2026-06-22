import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from rover_env import RoverEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time

print("Initializing environment...")
env = RoverEnv()

print("Checking environment compatibility...")
check_env(env, warn=True)
print("Environment check passed!")

print("Creating PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    tensorboard_log="./logs/"
)

print("Starting training...")
print("Watch the Unreal viewport - the car will start driving!")
print("Training for 10,000 steps (about 5-10 minutes)...")

model.learn(total_timesteps=10000)

print("Training complete! Saving model...")
os.makedirs("../models", exist_ok=True)
model.save("../models/ppo_rover_v1")
print("Model saved to models/ppo_rover_v1")

env.close()