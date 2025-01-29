# Euler-Maruyama scheme for generating data of multiple short trajectories from an SDE

import numpy as np

# Drift term of the OU process


def drift_ou(x, alpha):
    return -(alpha * x)


# Drift term of triple well 2D


def drift_triple_well_2d(x):
    return np.array(
        [
            6 * x[0, :] * np.exp(-x[0, :] ** 2 - (x[1, :] - 1 / 3) ** 2)
            - 6 * x[0, :] * np.exp(-x[0, :] ** 2 - (x[1, :] - 5 / 3) ** 2)
            - 10 * (x[0, :] - 1) * np.exp(-((x[0, :] - 1) ** 2) - (x[1, :]) ** 2)
            - 10 * (x[0, :] + 1) * np.exp(-((x[0, :] + 1) ** 2) - (x[1, :]) ** 2)
            - (4 / 5) * x[0, :] ** 3,
            6 * (x[1, :] - 1 / 3) * np.exp(-((x[0, :]) ** 2) - (x[1, :] - 1 / 3) ** 2)
            - 6 * (x[1, :] - 5 / 3) * np.exp(-((x[0, :]) ** 2) - (x[0, :] - 5 / 3) ** 2)
            - 10 * x[1, :] * np.exp(-((x[0, :] - 1) ** 2) - (x[1, :]) ** 2)
            - 10 * x[1, :] * np.exp(-((x[0, :] - 1) ** 2) - (x[1, :]) ** 2)
            - (4 / 5) * (x[1, :] - 1 / 3) ** 3,
        ]
    )


# drift term double-well 2D


def drift_double_well_2d(x):
    return np.array([-4 * x[0, :] ** 3 + 4 * x[0, :], -2 * x[1, :]])


# drift term himmelblau potential


def drift_himmelblau(x):
    return np.array(
        [
            -4 * x[0, :] * (x[0, :] ** 2 + x[1, :] - 11)
            - 2 * (x[0, :] + x[1, :] ** 2 - 7),
            -2 * (x[0, :] ** 2 + x[1, :] - 11)
            - 4 * x[1, :] * (x[0, :] + x[1, :] ** 2 - 7),
        ]
    )


# drift term triple well 1d potential


def drift_triple_well_1d(x):
    return -(6 * x**5 - 20 * x**3 + 12 * x + 1)


# drift term double well 1d potential


def drift_double_well_1d(x):
    return -(8 * x**3 - 10 * x)


# drift term lemon slice potential


def drift_lemon_slice(x, n):
    return np.array(
        [
            -(
                n
                * np.sin(n * np.arctan2(x[1, :], x[0, :]))
                * x[1, :]
                / (x[0, :] ** 2 + x[1, :] ** 2)
                + (20 * (np.sqrt(x[0, :] ** 2 + x[1, :] ** 2) - 1))
                * x[0, :]
                / np.sqrt(x[0, :] ** 2 + x[1, :] ** 2)
            ),
            -(
                -n
                * np.sin(n * np.arctan2(x[1, :], x[0, :]))
                * x[0, :]
                / (x[0, :] ** 2 + x[1, :] ** 2)
                + (20 * (np.sqrt(x[0, :] ** 2 + x[1, :] ** 2) - 1))
                * x[1, :]
                / np.sqrt(x[0, :] ** 2 + x[1, :] ** 2)
            ),
        ]
    )


class EulerMaruyamaData:
    """Euler-Maruyama class for generating data"""

    def __init__(self, m: int, x0: np.ndarray, h: float, iters: int = 500):
        self.h = h
        self.iters = iters
        self.x0 = x0
        self.m = m

    # ---------------------------------------------------------------------------------------------------------------
    # Data generating funtions

    def get_ou_data(self, beta, alpha):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_ou(x[:, :, i - 1], alpha) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]

    def get_triple_well_2d_data(self, beta):
        x = np.zeros((self.x0.shape[0], self.x0.shape[1], self.iters))
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_triple_well_2d(x[:, :, i - 1]) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]  # return last value as y

    def get_double_well_2d_data(self, beta):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_double_well_2d(x[:, :, i - 1]) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]

    def get_himmelblau_data(self, beta):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_himmelblau(x[:, :, i - 1]) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]

    def get_triple_well_1d_data(self, beta):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_triple_well_1d(x[:, :, i - 1]) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]

    def get_double_well_1d_data(self, beta):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_double_well_1d(x[:, :, i - 1]) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]

    def get_lemon_slice_data(self, beta, n=5):
        x = np.zeros(
            (self.x0.shape[0], self.x0.shape[1], self.iters)
        )  # Function to generate Euler-Maruyama data
        x[:, :, 0] = self.x0  # Initial condition
        for i in range(1, self.iters):
            x[:, :, i] = (
                x[:, :, i - 1]
                + drift_lemon_slice(x[:, :, i - 1], n=n) * self.h
                + np.sqrt(2 / beta)
                * np.sqrt(self.h)
                * np.random.normal(0, 1, size=(self.x0.shape[0], self.m))
            )  # Euler-Maruyama
        return x[:, :, -1]
