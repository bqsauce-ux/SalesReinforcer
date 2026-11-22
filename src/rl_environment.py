# src/rl_environment.py

import numpy as np

from reward_function import calculate_reward


class FeatureSelectionEnvironment:

    def __init__(
        self,
        X,
        y,
        max_features=15,
        complexity_penalty=0.05,
    ):

        self.X = X.reset_index(drop=True)
        self.y = np.asarray(y)

        self.feature_names = list(X.columns)

        self.n_features = len(self.feature_names)

        if self.n_features == 0:
            raise ValueError("No features available.")

        if self.n_features > 20:
            raise ValueError(
                "This tabular Q-learning implementation supports "
                "at most 20 features. Reduce the feature set first."
            )

        self.max_features = min(
            max_features,
            self.n_features,
        )

        self.complexity_penalty = complexity_penalty

        # Each feature has two possible actions:
        # 0 = remove, 1 = add.
        # Final action = stop.
        self.n_actions = 2 * self.n_features + 1

        self.state = np.zeros(
            self.n_features,
            dtype=int,
        )

        self.steps = 0

        self.max_steps = self.n_features * 2

        self.done = False

        self.reward_cache = {}

    def reset(self):

        self.state = np.zeros(
            self.n_features,
            dtype=int,
        )

        self.steps = 0

        self.done = False

        return tuple(self.state)
    
    def get_state(self):
        """
        Return the current feature-selection state.
    
        1 = Feature selected
        0 = Feature not selected
        """
        return tuple(self.state)

    def get_selected_features(self):

        return [
            self.feature_names[i]
            for i in range(self.n_features)
            if self.state[i] == 1
        ]

    def get_reward(self):

        selected = self.get_selected_features()

        if len(selected) == 0:
            return -1.0

        key = tuple(sorted(selected))

        if key not in self.reward_cache:

            self.reward_cache[key] = calculate_reward(
                self.X,
                self.y,
                selected,
                complexity_penalty=self.complexity_penalty,
            )

        return self.reward_cache[key]

    def step(self, action):

        if self.done:
            raise RuntimeError(
                "Episode finished. Call reset()."
            )

        self.steps += 1

        # Last action means stop.
        if action == self.n_actions - 1:

            self.done = True

            return (
                self.get_state(),
                0.0,
                True,
                {
                    "selected_features": self.get_selected_features(),
                    "final_score": self.get_reward(),
                },
            )

        feature_index = action // 2

        operation = action % 2

        if operation == 0:
            # Remove feature.
            self.state[feature_index] = 0

        else:
            # Add feature if capacity allows.
            if (
                self.state[feature_index] == 0
                and self.state.sum() < self.max_features
            ):
                self.state[feature_index] = 1

        reward = self.get_reward()

        if self.steps >= self.max_steps:
            self.done = True

        return (
            self.get_state(),
            reward,
            self.done,
            {
                "selected_features": self.get_selected_features(),
                "score": self.get_reward(),
            },
        )