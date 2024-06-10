import os
import gym
import pickle
from q_learn import QLearning


def train():
    env = gym.make("MountainCar-v0")
    print("action_space", env.action_space)

    agent = QLearning(action_space=env.action_space)

    # train_count
    for i in range(10000):
        observation, reset_info = env.reset()

        print("observation", observation)
        print("reset_info", reset_info)

        state = agent.digitize_state(observation=observation)

        # batch_size
        for t in range(300):
            action = agent.choose_action(state=state)
            observation, reward, terminated, truncated, info = env.step(action=action)
            next_state = agent.digitize_state(observation=observation)

            # 到達山頂
            if reward == 0:
                reward += 9999

            print(
                f"{i}. >>>>>>>>>> ",
                action,
                reward,
                terminated,
                state,
                next_state,
                truncated,
                info,
            )
            agent.learn(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )
            state = next_state
            if terminated:
                print("car terminated")
                break

    print(agent.q_table)
    env.close()

    with open(os.getcwd() + "/tmp/carmountain.model", "wb") as f:
        pickle.dump(agent, f)


train()
