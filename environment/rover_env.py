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
            low=0.0,
            high=100.0,
            shape=(7,),
            dtype=np.float32
        )
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.max_steps = 500
        self.current_step = 0
        self.steps_stuck = 0
        self.sensor_names = [
            "DistanceFront",
            "DistanceFrontLeft",
            "DistanceFrontRight",
            "DistanceLeft",
            "DistanceRight"
        ]
        print("Environment ready!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.current_step = 0
        self.steps_stuck = 0
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
        reward, terminated = self._get_reward(action, obs)
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_distances(self):
        distances = []
        for sensor in self.sensor_names:
            try:
                data = self.client.getDistanceSensorData(
                    distance_sensor_name=sensor,
                    vehicle_name="CPHatchback"
                )
                dist = float(data.distance)
                if dist <= 0 or dist > 499:
                    dist = 50.0
            except Exception:
                dist = 50.0
            distances.append(min(dist, 50.0))
        return distances

    def _get_obs(self):
        distances = self._get_distances()
        controls = self.client.getCarControls()
        obs = np.array(
            distances + [float(controls.throttle), float(controls.steering)],
            dtype=np.float32
        )
        return obs

    def _get_reward(self, action, obs):
        throttle = float(action[0])
        steering = float(action[1])
        front_dist = float(obs[0])
        front_left = float(obs[1])
        front_right = float(obs[2])
        terminated = False
        reward = 0.0
        if front_dist > 10.0:
            reward += throttle * 2.0
        elif front_dist > 5.0:
            reward += throttle * 0.5
        else:
            reward -= throttle * 2.0
        reward -= max(0.0, (15.0 - front_dist)) * 0.5
        reward -= max(0.0, (5.0 - front_left)) * 0.3
        reward -= max(0.0, (5.0 - front_right)) * 0.3
        if front_dist < 8.0:
            reward += abs(steering) * 1.0
        if front_dist < 1.5:
            reward -= 50.0
            terminated = True
            print(f"Collision at step {self.current_step}")
        if throttle < 0.1:
            self.steps_stuck += 1
            reward -= 0.5
        else:
            self.steps_stuck = 0
        if self.steps_stuck > 50:
            terminated = True
            reward -= 5.0
        return reward, terminated

    def close(self):
        self.client.enableApiControl(False)
        print("Environment closed.")