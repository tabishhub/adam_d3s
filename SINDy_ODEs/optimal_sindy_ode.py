## Optimal parameters SINDy

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jax.example_libraries import optimizers
from scipy.integrate import odeint

# ==========================================================================================================================================
# Utility functions for plotting


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11
    LEGEND_SIZE = 6

    FONT = {"family": "serif", "serif": ["Computer Modern Serif"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # "axes.spines.left": True,
        # "axes.spines.bottom": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "figure.constrained_layout.use": True,
    }

    return rc_params


def set_size(
    width: float = 434.55125,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: Union[None, float] = None,
) -> tuple:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def mod_chua_attrac(y, t):
    """Define the system"""
    x1, x2, x3 = y

    # parameters
    alpha, beta = 10.2, 14.286
    a, b, c, d = 1.3, 0.11, 7, 0

    f_x = -b * np.sin(np.pi * x1 / (2 * a) + d)

    xdot = alpha * (x2 - f_x)
    ydot = x1 - x2 + x3
    zdot = -beta * x2
    sys = np.array([xdot, ydot, zdot])
    return sys


def psi_params_mod_chua(X, W):
    """Returns transformed data matrices using parametric set of basis functions"""
    return jnp.vstack(
        (
            X[0, :],
            X[1, :],
            X[2, :],
            X[2, :] * X[0, :],
            X[0, :] * X[1, :],
            X[1, :] ** 2,
            jnp.cos(X[1, :]),
            -0.11 * jnp.sin(W * X[0, :]),
        )
    )


# ================================================================================================================================================================================

# SINDy algorithm


def sindy(X, Y, psi_x, eps=0.0001, iterations=10):
    """Original SINDy for ODEs algorithm to find the optimal Xi matrix"""
    Psi = Y @ sp.linalg.pinv(psi_x)  # least−squares initial guess

    for k in range(iterations):
        s = abs(Psi) < eps  # find coefficients less than eps ...
        Psi[s] = 0  # ... and set them to zero
        for ind in range(X.shape[0]):
            b = ~s[ind, :]  # consider only coefficients greater than eps
            Psi[ind, b] = Y[ind, :] @ sp.linalg.pinv(psi_x[b, :])
    return Psi


# Cost function of SINDy


def cost_sindy(Psi, X_dot, psi_x):
    """Returns the cost of SINDy for a given Psi matrix
    Parameters-
    Psi: Estimated Psi matrix for a given basis
    X_dot: Derivatives matrix
    psi_x: transformed X matrix using basis functions
    """
    return (1 / 2) * (np.linalg.norm(X_dot - Psi @ psi_x, "fro")) ** 2


# ===========================================================================================================================================

# Class to generate data from ODE


class GenerateODEData:
    # Initialization of the class
    def __init__(self, system) -> None:
        self.system = system

    # function to solve the system of ODE
    def solve_system(self, yzero, t_span):
        X_sol = odeint(self.system, yzero, t_span)
        return X_sol

    # Get data matrices X and X_dot
    def data_matrices(self, X_sol, t):
        m = X_sol.shape[0]
        d = X_sol.shape[1]
        X_dot = np.zeros((m, d))
        for i in range(m):
            X_dot[i, :] = self.system(X_sol[i, :], t)

        return X_sol.T, X_dot.T

    # Save the data
    def save_data(self, X, X_dot):
        return sp.io.savemat("odeData", {"Xdot": X_dot, "X": X})


# =========================================================================================================================================

# Class to approximate optimal parameters of basis function for SINDy


class OptimalSindyODE:
    # Initialization of the class
    def __init__(self, epss, loops, psi_params) -> None:
        self.epss = epss
        self.loops = loops
        self.psi_params = psi_params

    # -----------------------------------------------------------------------------------------------------------------------------------------

    # SINDy algorithm

    def sindy(self, X, Y, psi_x, eps=0.0001, iterations=10):
        """Original SINDy algorithm"""

        XiT = Y @ sp.linalg.pinv(psi_x)  # least−squares initial guess

        for k in range(iterations):
            s = abs(XiT) < eps  # find coefficients less than eps ...
            XiT[s] = 0  # ... and set them to zero
            for ind in range(X.shape[0]):
                b = ~s[ind, :]  # consider only coefficients greater than eps
                XiT[ind, b] = Y[ind, :] @ sp.linalg.pinv(psi_x[b, :])
        return XiT

    # -------------------------------------------------------------------------------------------------------------------------------------------

    # Cost function of SINDy

    def cost_sindy(self, W, Psi, X, X_dot):
        """Returns the cost of SINDy for a given Psi matrix
        Parameters-
        Psi: Estimated Psi matrix for a given basis
        X, X_dot: Derivatives matrix
        W: Parameters of the basis functions

        returns-
        value of the cost function
        """
        psi_x = self.psi_params(X, W)

        return (1 / 2) * (jnp.linalg.norm(X_dot - Psi @ psi_x, "fro")) ** 2

    # ------------------------------------------------------------------------------------------------------------------------------------------

    # Gradient of the cost function of SINDy w.r.t. matrix Psi

    def grad_cost_sindy(self, W, Psi, X, X_dot):
        """
        Returns the gradient of the cost function of SINDy for a given Psi matrix
        Parameters-
        Psi: Estimated Psi matrix for a given basis
        X, X_dot: Data matrix
        psi_x: transformed X matrix using basis functions

        returns-
        gradient of the cost function w.r.t. Psi matrix
        """
        grad_calc_psi = jax.grad(self.cost_sindy, argnums=1)
        return grad_calc_psi(W, Psi, X, X_dot)

    # -----------------------------------------------------------------------------------------------------------------------------------------

    # Gradient of the cost function of SINDy w.r.t. basis parameters

    def grad_cost_sindy_params(self, W, Psi, X, X_dot):
        """
        Returns the gradient of the cost function of SINDy for a given Psi matrix
        Parameters-
        Psi: Estimated Psi matrix for a given basis
        X, X_dot: Data matrix
        psi_x: transformed X matrix using basis functions

        returns-
        gradient of the cost function w.r.t. basis parameters
        """
        grad_calc = jax.grad(self.cost_sindy, argnums=0)
        return grad_calc(W, Psi, X, X_dot)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Gradient descent for optimal parameters and Psi matrix
    def gradient_descent(self, W_init, Psi_init, X, X_dot, hw, hs, alt_loops=10):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using GD algorithm.
        Params-
        W_init: Initial basis parameters
        Psi_init: Initial Psi matrix
        X, X_dot: data matrices
        hk, hs: step size of GD
        alt_loops: number of alternating loops

        returns-
        Psi: optimal Psi matrix
        W: optimal parameters of the basis functions
        loss_K: loss in approximating psi matrix
        loss_W: loss in approximating W
        """

        # Initialize the optimizer with the initial parameters
        opt_init_psi, opt_update_psi, get_psi = optimizers.sgd(hs)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        opt_state_psi = opt_init_psi(Psi_init)
        opt_state_w = opt_init_w(W_init)

        def update_psi(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_psi = self.grad_cost_sindy(W, Psi, X, X_dot)
            opt_state = opt_update_psi(0, grads_psi, opt_state)
            new_psi = get_psi(opt_state)
            return new_psi, opt_state

        def update_w(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_sindy_params(W, Psi, X, X_dot)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        Psi_gd = Psi_init
        W_gd = W_init
        f_vals_gd_psi = []
        f_vals_gd_w = []

        l = 1

        while self.cost_sindy(W_gd, Psi_gd, X, X_dot) > self.epss and l < self.loops:
            for i in range(alt_loops):
                Psi_gd, opt_state_psi = update_psi(W_gd, Psi_gd, opt_state_psi)
                loss_psi = self.cost_sindy(W_gd, Psi_gd, X, X_dot)
                f_vals_gd_psi.append(loss_psi)

            W_gd, opt_state_w = update_w(W_gd, Psi_gd, opt_state_w)
            loss_w = self.cost_sindy(W_gd, Psi_gd, X, X_dot)
            f_vals_gd_w.append(loss_w)

            l = l + 1

        return Psi_gd, W_gd, f_vals_gd_psi, f_vals_gd_w

    # ------------------------------------------------------------------------------------------------------------------------------------------

    def stochastic_gradient_descent(
        self, W_init, Psi_init, X, X_dot, hw, hs, batch_size, alt_loops=10
    ):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using SGD algorithm.
        Params-
        W_init: Initial basis parameters
        Psi_init: Initial Psi matrix
        X, X_dot: data matrices
        hk, hs: step size of GD
        alt_loops: number of alternating loops

        returns-
        Psi: optimal Psi matrix
        W: optimal parameters of the basis functions
        loss_K: loss in approximating psi matrix
        loss_W: loss in approximating W
        """
        # Initialize the optimizer with the initial parameters
        opt_init_psi, opt_update_psi, get_psi = optimizers.sgd(hs)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        opt_state_psi = opt_init_psi(Psi_init)
        opt_state_w = opt_init_w(W_init)

        def update_psi(W, Psi, X, X_dot, opt_state):
            """Perform one update step of the optimizer."""
            grads_psi = self.grad_cost_sindy(W, Psi, X, X_dot)
            opt_state = opt_update_psi(0, grads_psi, opt_state)
            new_psi = get_psi(opt_state)
            return new_psi, opt_state

        def update_w(W, Psi, X, X_dot, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_sindy_params(W, Psi, X, X_dot)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        Psi_sgd = Psi_init
        W_sgd = W_init
        f_vals_sgd_psi = []
        f_vals_sgd_w = []

        l = 1
        idx = [i for i in range(X.shape[1])]

        while self.cost_sindy(W_sgd, Psi_sgd, X, X_dot) > self.epss and l < self.loops:
            perm_idx = np.random.permutation(idx)
            for i in range(int(X.shape[1] / batch_size)):
                chosen_idx = perm_idx[i * batch_size : (i + 1) * batch_size]

                X_s = jnp.take(X, chosen_idx, 1)
                X_dot_s = jnp.take(X_dot, chosen_idx, 1)

                for i in range(alt_loops):
                    Psi_sgd, opt_state_psi = update_psi(
                        W_sgd, Psi_sgd, X_s, X_dot_s, opt_state_psi
                    )
                    loss_psi = self.cost_sindy(W_sgd, Psi_sgd, X, X_dot)
                    f_vals_sgd_psi.append(loss_psi)

                W_sgd, opt_state_w = update_w(W_sgd, Psi_sgd, X_s, X_dot_s, opt_state_w)
                loss_w = self.cost_sindy(W_sgd, Psi_sgd, X, X_dot)
                f_vals_sgd_w.append(loss_w)

                l = l + 1

        return Psi_sgd, W_sgd, f_vals_sgd_psi, f_vals_sgd_w

    def nesterov(self, W_init, Psi_init, X, X_dot, hw, hs, alt_loops=10, mass=0.1):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using the Nesterov algorithm.
        Params-
        W_init: Initial basis parameters
        Psi_init: Initial Psi matrix
        X, X_dot: data matrices
        hk, hs: step size of GD
        alt_loops: number of alternating loops
        mass: mass in Nesterov's method

        returns-
        Psi: optimal Psi matrix
        W: optimal parameters of the basis functions
        loss_K: loss in approximating psi matrix
        loss_W: loss in approximating W
        """

        # Initialize the optimizer with the initial parameters
        opt_init_psi, opt_update_psi, get_psi = optimizers.nesterov(hs, mass)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.nesterov(hw, mass)

        opt_state_psi = opt_init_psi(Psi_init)
        opt_state_w = opt_init_w(W_init)

        def update_psi(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_psi = self.grad_cost_sindy(W, Psi, X, X_dot)
            opt_state = opt_update_psi(0, grads_psi, opt_state)
            new_psi = get_psi(opt_state)
            return new_psi, opt_state

        def update_w(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_sindy_params(W, Psi, X, X_dot)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        Psi_nest = Psi_init
        W_nest = W_init
        f_vals_nest_psi = []
        f_vals_nest_w = []

        l = 1

        while (
            self.cost_sindy(W_nest, Psi_nest, X, X_dot) > self.epss and l < self.loops
        ):
            for i in range(alt_loops):
                Psi_nest, opt_state_psi = update_psi(W_nest, Psi_nest, opt_state_psi)
                loss_psi = self.cost_sindy(W_nest, Psi_nest, X, X_dot)
                f_vals_nest_psi.append(loss_psi)

            W_nest, opt_state_w = update_w(W_nest, Psi_nest, opt_state_w)
            loss_w = self.cost_sindy(W_nest, Psi_nest, X, X_dot)
            f_vals_nest_w.append(loss_w)

            l = l + 1

        return Psi_nest, W_nest, f_vals_nest_psi, f_vals_nest_w

    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    def adam(self, W_init, Psi_init, X, X_dot, hw, hs, alt_loops=10):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using Adam algorithm.
        Params-
        W_init: Initial basis parameters
        Psi_init: Initial Psi matrix
        X, X_dot: data matrices
        hk, hs: step size of GD
        alt_loops: number of alternating loops

        returns-
        Psi: optimal Psi matrix
        W: optimal parameters of the basis functions
        loss_K: loss in approximating psi matrix
        loss_W: loss in approximating W
        """

        # Initialize the optimizer with the initial parameters
        opt_init_psi, opt_update_psi, get_psi = optimizers.adam(hs)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.adam(hw)

        opt_state_psi = opt_init_psi(Psi_init)
        opt_state_w = opt_init_w(W_init)

        def update_psi(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_psi = self.grad_cost_sindy(W, Psi, X, X_dot)
            opt_state = opt_update_psi(0, grads_psi, opt_state)
            new_psi = get_psi(opt_state)
            return new_psi, opt_state

        def update_w(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_sindy_params(W, Psi, X, X_dot)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        Psi_adam = Psi_init
        W_adam = W_init
        f_vals_adam_psi = []
        f_vals_adam_w = []

        l = 1

        while (
            self.cost_sindy(W_adam, Psi_adam, X, X_dot) > self.epss and l < self.loops
        ):
            for i in range(alt_loops):
                Psi_adam, opt_state_psi = update_psi(W_adam, Psi_adam, opt_state_psi)
                loss_psi = self.cost_sindy(W_adam, Psi_adam, X, X_dot)
                f_vals_adam_psi.append(loss_psi)

            W_adam, opt_state_w = update_w(W_adam, Psi_adam, opt_state_w)
            loss_w = self.cost_sindy(W_adam, Psi_adam, X, X_dot)
            f_vals_adam_w.append(loss_w)

            l = l + 1

        return Psi_adam, W_adam, f_vals_adam_psi, f_vals_adam_w

    def adam_decay(
        self, W_init, Psi_init, X, X_dot, hw, hs, decay_rate, decay_steps, alt_loops=10
    ):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using Adam algorithm.
        Params-
        W_init: Initial basis parameters
        Psi_init: Initial Psi matrix
        X, X_dot: data matrices
        hk, hs: step size of GD
        decay_rate, decay_steps: rate and steps of deacy in step size
        alt_loops: number of alternating loops

        returns-
        Psi: optimal Psi matrix
        W: optimal parameters of the basis functions
        loss_K: loss in approximating psi matrix
        loss_W: loss in approximating W
        """

        # Learning rate schedules with exponential decay
        schedule_psi = optimizers.exponential_decay(hs, decay_steps, decay_rate)
        schedule_w = optimizers.exponential_decay(hw, decay_steps, decay_rate)

        # Initialize the optimizer with the initial parameters
        opt_init_psi, opt_update_psi, get_psi = optimizers.adam(schedule_psi)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.adam(schedule_w)

        opt_state_psi = opt_init_psi(Psi_init)
        opt_state_w = opt_init_w(W_init)

        def update_psi(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_psi = self.grad_cost_sindy(W, Psi, X, X_dot)
            opt_state = opt_update_psi(step, grads_psi, opt_state)
            new_psi = get_psi(opt_state)
            return new_psi, opt_state

        def update_w(W, Psi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_sindy_params(W, Psi, X, X_dot)
            opt_state = opt_update_w(step, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        Psi_adam = Psi_init
        W_adam = W_init
        f_vals_adam_psi = []
        f_vals_adam_w = []

        l = 1
        step = 0

        while (
            self.cost_sindy(W_adam, Psi_adam, X, X_dot) > self.epss and l < self.loops
        ):
            step += 1
            for i in range(alt_loops):
                Psi_adam, opt_state_psi = update_psi(W_adam, Psi_adam, opt_state_psi)
                loss_psi = self.cost_sindy(W_adam, Psi_adam, X, X_dot)
                f_vals_adam_psi.append(loss_psi)

            W_adam, opt_state_w = update_w(W_adam, Psi_adam, opt_state_w)
            loss_w = self.cost_sindy(W_adam, Psi_adam, X, X_dot)
            f_vals_adam_w.append(loss_w)

            l += 1

        return Psi_adam, W_adam, f_vals_adam_psi, f_vals_adam_w
