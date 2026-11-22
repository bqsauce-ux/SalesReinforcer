
# src/qlearningagent.py

import numpy as np


class QLearningAgent:

    def __init__(
        self,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        random_state=42
    ):

        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = np.random.default_rng(random_state)

        self.q_table = {}

    def _state_key(self, state):

        return tuple(np.asarray(state, dtype=int).tolist())

    def get_q_values(self, state):

        state = self._state_key(state)

        if state not in self.q_table:

            self.q_table[state] = np.zeros(
                self.n_actions
            )

        return self.q_table[state]

    def choose_action(self, state):

        if self.rng.random() < self.epsilon:

            return int(
                self.rng.integers(self.n_actions)
            )

        q_values = self.get_q_values(state)

        best_actions = np.flatnonzero(
            q_values == np.max(q_values)
        )

        return int(self.rng.choice(best_actions))

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q = self.get_q_values(state)

        next_q = self.get_q_values(next_state)

        if done:

            target = reward

        else:

            target = (
                reward
                + self.gamma * np.max(next_q)
            )

        current_q[action] += self.lr * (
            target - current_q[action]
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    
    def train(
        self,
        environment,
        episodes=100
        ):

        episode_rewards = []
        epsilon_history = []

        best_reward = -np.inf
        best_state = None

        for episode in range(episodes):

            state = environment.reset()

            done = False
            total_reward = 0

            while not done:

                action = self.choose_action(state)

                next_state, reward, done, info = (
                    environment.step(action)
                )

                self.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                state = next_state

                total_reward += reward

            # Track episode reward.
            episode_rewards.append(total_reward)

            # Evaluate final state.
            final_reward = environment.get_reward()

            if final_reward > best_reward:

                best_reward = final_reward

                best_state = self._state_key(
                    environment.get_state()
                )

            # Decay epsilon after completing the episode.
            self.decay_epsilon()

            # Record epsilon after decay.
            epsilon_history.append(self.epsilon)

            # Progress logging.
            if (episode + 1) % 10 == 0:

                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Reward: {total_reward:.4f} | "
                    f"Best: {best_reward:.4f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        return {
            "best_state": best_state,
            "best_reward": best_reward,
            "episode_rewards": episode_rewards,
            "epsilon_history": epsilon_history
        }

