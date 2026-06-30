import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from rover_env import RoverEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

print("Initializing environment...")
env = RoverEnv()

print("Creating model...")
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

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="../models/checkpoints/",
    name_prefix="ppo_rover_v4"
)

print("Starting training - 50,000 steps...")
model.learn(
    total_timesteps=50000,
    callback=checkpoint_callback
)

print("Training complete! Saving model...")
os.makedirs("../models", exist_ok=True)
model.save("../models/ppo_rover_v4_wall_aware")
print("Model saved!")
env.close()