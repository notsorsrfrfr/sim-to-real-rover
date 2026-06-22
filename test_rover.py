import cosysairsim as airsim
import time

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connected! Driving forward...')
car_controls.throttle = 0.5
car_controls.steering = 0.0
client.setCarControls(car_controls)
time.sleep(3)
print('Turning...')
car_controls.steering = 0.5
client.setCarControls(car_controls)
time.sleep(2)
car_controls.throttle = 0.0
car_controls.brake = 1.0
client.setCarControls(car_controls)
client.enableApiControl(False)
print('Done! Check the Unreal viewport - did the car move?')