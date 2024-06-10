import numpy as np


class QLearning:
    def __init__(
        self,
        action_space,
        learning_rate: float = 0.01,
        reward_decay: float = 0.99,
        e_greedy: float = 0.6,
    ) -> None:
        self.actions = action_space
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.num_pos = 20
        self.num_vel = 14
        self.q_table = np.random.uniform(
            low=-1, high=1, size=(self.num_pos * self.num_vel, self.actions.n)
        )
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            # best action
            return np.argmax(self.q_table[state])
        else:
            # explore action
            return self.actions.sample()

    def digit(self, x, bins):
        n = np.digitize(x, bins=bins)
        if x == bins[-1]:
            n = n - 1
        return n

    def digitize_state(self, observation: tuple):
        car_pos, car_vel = observation
        digitized = [
            self.digit(car_pos, self.pos_bins),
            self.digit(car_vel, self.vel_bins),
        ]
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def learn(self, state, action, reward, next_state):
        next_action = np.argmax(self.q_table[next_state])
        q_predict = self.q_table[state, action]
        q_target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.learning_rate * (q_target - q_predict)
