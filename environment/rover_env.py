import cosysairsim as airsim
import gymnasium as gym
import numpy as np
import time

class RoverEnv(gym.Env):
    def __init__(self):
        super(RoverEnv, self).__init__()
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.max_steps = 500
        self.current_step = 0
        print("Environment ready!")

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.current_step = 0
        time.sleep(0.5)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        self.car_controls.throttle = float(action[0])
        self.car_controls.steering = float(action[1])
        self.car_controls.brake = 0.0
        self.client.setCarControls(self.car_controls)
        time.sleep(0.05)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        controls = self.client.getCarControls()
        obs = np.array([
            float(controls.throttle),
            float(controls.steering),
            float(controls.brake)
        ], dtype=np.float32)
        return obs

    def _get_reward(self):
        controls = self.client.getCarControls()
        reward = float(controls.throttle) * 0.1
        return reward

    def close(self):
        self.client.enableApiControl(False)
        print("Environment closed.")