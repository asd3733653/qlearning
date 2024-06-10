import gym
import time
import os
import pickle
from q_learn import QLearning

env = gym.make("MountainCar-v0", render_mode="human")
observation, reset_info = env.reset()
with open(os.getcwd() + "/tmp/carmountain.model", "rb") as f:
    agent: QLearning = pickle.load(f)

agent.actions = env.action_space
agent.epsilon = 1
state = agent.digitize_state(observation=observation)

for i in range(2000):
    action = agent.choose_action(state=state)
    observation, reward, terminated, truncated, info = env.step(action)
    next_state = agent.digitize_state(observation=observation)
    print("observation,", observation)
    print("reward,", reward)
    print("terminated,", terminated)
    print("truncated,", truncated)
    print("info,", info)
    agent.learn(state=state, action=action, reward=reward, next_state=next_state)
    env.render()

    time.sleep(0.02)

env.close()
