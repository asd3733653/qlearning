import gym
import time
import random


env = gym.make("MountainCar-v0", render_mode="human")
env.reset()
random_number = lambda: random.randint(0, 2)
fixed_number = lambda: 2

for i in range(2000):
    observation, reward, terminated, truncated, info = env.step(random_number())
    print("observation,", observation)
    print("reward,", reward)
    print("terminated,", terminated)
    print("truncated,", truncated)
    print("info,", info)
    env.render()

    time.sleep(0.02)

env.close()
