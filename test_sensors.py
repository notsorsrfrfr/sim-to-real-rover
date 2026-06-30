import sys
sys.path.append('environment')
from rover_env import RoverEnv

env = RoverEnv()
obs, _ = env.reset()
print('Sensor readings:', obs)
print('Front:', obs[0])
print('FrontLeft:', obs[1])
print('FrontRight:', obs[2])
print('Left:', obs[3])
print('Right:', obs[4])
env.close()