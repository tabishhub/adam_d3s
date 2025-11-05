# import libraries

from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jax.example_libraries import optimizers

# ============================================================================================================================================================================================


# Utility function for plotting
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
    Utility function for plotting
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


# ============================================================================================================================================================================================


def multivariate_gaussian(x, sigma, mu):
    """calculate multivariate normal"""
    d = x.shape[0]
    xm = x - mu
    return np.exp(-0.5 * (xm.T.dot(xm)) / sigma**2)


def generate_surface(X, sigma, mu):
    """generate surface for given pdf"""
    m = X.shape[1]
    x1 = X[0, :]
    x2 = X[1, :]
    x1s, x2s = np.meshgrid(x1, x2)
    pdf = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            pdf[i, j] = multivariate_gaussian(
                np.matrix([[x1s[i, j]], [x2s[i, j]]]), sigma, mu
            )
    return x1s, x2s, pdf


def sortEig(A, evs=5, which="LM"):
    """
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    """
    n = A.shape[0]
    if evs < n:
        d, V = sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1]  # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])


def basis_func_1D(X, sigma, c):
    X_flat = X.flatten()
    squared_distances = (X_flat - c) ** 2
    return jnp.exp(-squared_distances / (2 * sigma**2))


def psi_1D(X, W):
    X_flat = X.flatten()

    def apply_single_basis(w):
        sigma, c = w[0], w[1]
        return basis_func_1D(X_flat, sigma, c)

    TX = jax.vmap(apply_single_basis)(W)
    return jnp.squeeze(TX)


def basis_func_nD(X, sigma, mu):
    mu = mu.reshape(-1, 1)

    diff = X - mu  # [d, m]

    basis_values = 0.5 * jnp.sum((diff / sigma) ** 2, axis=0, keepdims=True)

    return jnp.exp(-basis_values)


def psi_nD(X, W):
    sigmas = W[:, 0]  # shape [n]
    mus = W[:, 1:]  # shape [n, d]

    def single_basis(sigma, mu):
        return basis_func_nD(X, sigma, mu)

    TX = jax.vmap(single_basis)(sigmas, mus)

    return jnp.squeeze(TX)


# ==============================================================================================================================================


def inverse(x, epsilon=1e-10, ret_sqrt=False):
    """Utility function that returns the inverse of a matrix.

    Parameters-
    x:  matrix to be inverted

    Returns-
    x_inv: inverse of the matrix
    """

    # Calculate eigenvalues and eigenvectors
    eigval_all, eigvec_all = jnp.linalg.eigh(x)

    # Filter out eigenvalues below threshold and corresponding eigenvectors
    eigval = eigval_all[eigval_all > epsilon]
    eigvec = eigvec_all[:, eigval_all > epsilon]

    # diagonal matrix with the eigenvalues or their square root
    if ret_sqrt:
        diag = jnp.diag(jnp.sqrt(1 / eigval))
    else:
        diag = jnp.diag(1 / eigval)

    # Rebuild the square root of the inverse matrix
    eigvec = eigvec.astype(jnp.float32)
    diag = diag.astype(jnp.float32)
    x_inv = jnp.matmul(eigvec, jnp.matmul(diag, eigvec.T))

    return x_inv


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class OptimalKoopmanParameter:
    """
    Class to approximate optimal Koopman matrix and the basis functions using gradient descent-based algorithms

    Initialization-
    epss: tolerance of the approximation
    loops: Number of iterations of the algorithms
    psi_params: parametric set of basis functions
    """

    def __init__(self, epss, loops, psi_params) -> None:
        self.epss = epss
        self.loops = loops
        self.psi_params = psi_params

    def cost_edmd(self, K, W, X, Y):
        """
        Reconstruction error
        Parameters-
        K: Koopman matrix
        W: basis parameters
        X, Y: data matrices

        Returns-
        Reconstruction error in Frobenius norm
        """

        psi_x = self.psi_params(X, W)
        psi_y = self.psi_params(Y, W)

        # calculate the cost
        z = (1 / 2) * (jnp.linalg.norm(psi_y - K.T @ psi_x, "fro")) ** 2
        return jnp.array(z, dtype="float64")

    def cost_vamp2(self, W, X, Y):
        """Calculates the VAMP-2 score with respect to the new basis functions.

        Parameters-
        W: basis parameters
        X, Y: data matrices

        Returns-
        VAMP-2 score
        """
        psi_x = self.psi_params(X, W)
        psi_y = self.psi_params(Y, W)

        data_points = psi_x.shape[1]

        # Calculate the covariance matrices
        cov_01 = 1 / (data_points - 1) * jnp.matmul(psi_x, psi_y.T)
        cov_00 = 1 / (data_points - 1) * jnp.matmul(psi_x, psi_x.T) + 0.01 * jnp.eye(
            psi_x.shape[0]
        )
        cov_11 = 1 / (data_points - 1) * jnp.matmul(psi_y, psi_y.T) + 0.01 * jnp.eye(
            psi_x.shape[0]
        )

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = inverse(cov_00, ret_sqrt=True)
        cov_11_inv = inverse(cov_11, ret_sqrt=True)

        vamp_matrix = jnp.matmul(jnp.matmul(cov_00_inv, cov_01), cov_11_inv)

        vamp_score = jnp.linalg.norm(vamp_matrix, "fro")

        return -jnp.square(vamp_score)

    def gradient_cost_edmd(self, K, W, X, Y):
        """
        Gradient of the reconstruction error
        Parameters-
        K: Koopman matrix
        W: basis parameters
        X,Y: data matrices

        Returns:
        Gradient of the reconstruction error in shape [n,n]
        """
        grad_calc_k = jax.grad(self.cost_edmd)
        return grad_calc_k(K, W, X, Y)

    def gradient_cost_params_vamp2(self, W, X, Y):
        """
        Gradient of the VAMP-2 score
        Parameters-
        K: Koopman matrix
        W: basis parameters
        X,Y: data matrices

        Returns:
        Gradient of the VAMP-2 score w.r.t. W
        """
        grad_calc_params_vamp = jax.grad(self.cost_vamp2)
        return grad_calc_params_vamp(W, X, Y)

    def optimizer(
        self,
        optim: str = "adam",
        hk: float = 0.01,
        hw: float = 0.01,
        mass: float = 0.1,
        decay_rate: float = None,
        decay_steps: int = None,
    ) -> Callable[[jax.numpy.ndarray], jax.numpy.ndarray]:
        """
        Optimizer to be used for training

        Params -
        optim: optimizer to be used for training, available options are 'adam', 'nesterov', 'gd'

        returns -
        update functions for the optimizer in JAX
        """
        # lr_schedule = hk

        # if decay_rate is not None and decay_steps is not None:
        #     lr_schedule = optimizers.exponential_decay(
        #         init_value=hk,  # Starting learning rate
        #         transition_steps=decay_steps,  # How often to decay
        #         decay_rate=decay_rate,  # How much to decay
        #         staircase=False,  # Smooth decay if False, stepwise if True
        #     )

        if optim == "gd":
            # Initialize the optimizer with the initial parameters
            opt_init_K, opt_update_K, get_K = optimizers.sgd(hk)

            # Initialize the optimizer with the initial parameters
            opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        elif optim == "sgd":
            # Initialize the optimizer with the initial parameters
            opt_init_K, opt_update_K, get_K = optimizers.sgd(hk)

            # Initialize the optimizer with the initial parameters
            opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        elif optim == "nesterov":
            # Initialize the optimizer with the initial parameters
            opt_init_K, opt_update_K, get_K = optimizers.nesterov(hk, mass)

            # Initialize the optimizer with the initial parameters
            opt_init_w, opt_update_w, get_w = optimizers.nesterov(hw, mass)

        elif optim == "adam":
            # Initialize the optimizer with the initial parameters
            opt_init_K, opt_update_K, get_K = optimizers.adam(hk)

            # Initialize the optimizer with the initial parameters
            opt_init_w, opt_update_w, get_w = optimizers.adam(hw)

        elif optim == "rmsprop":
            # Initialize the optimizer with the initial parameters
            opt_init_K, opt_update_K, get_K = optimizers.rmsprop(hk)

            # Initialize the optimizer with the initial parameters
            opt_init_w, opt_update_w, get_w = optimizers.rmsprop(hw)
        else:
            raise ValueError(
                f"Optimizer {optim} is not available; available options are 'adam', 'nesterov', 'gd', 'rmsprop'."
            )
        return opt_init_K, opt_update_K, get_K, opt_init_w, opt_update_w, get_w

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def koopman_approximation(
        self,
        K_init: jnp.ndarray,
        w_init: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        hk: float,
        hw: float,
        optim: str = "adam",
        stochastic: bool = False,
        batch_size: int = 100,
    ):
        """
        returns the optimal Koopman matrix K and basis parameters W along with loss in approximating them using the specified optimizer.
        Params-
        K_init: Initial Koopman matrix K
        w_init: Initial parameters of the basis functions
        X, Y: data matrices
        hk, hw: step sizes of GD
        optim: optimizer to be used for training, available options are 'adam', 'nesterov', 'gd'

        returns-
        K: optimal Koopman matrix
        W: optimal basis parameters
        loss_K: loss in approximating K matrix
        loss_w: loss in approximating W matrix
        """

        # Initialize the optimizer with the initial parameters

        opt_init_K, opt_update_K, get_K, opt_init_w, opt_update_w, get_w = (
            self.optimizer(optim, hk, hw)
        )

        opt_state_K = opt_init_K(K_init)
        opt_state_w = opt_init_w(w_init)

        def update_K(K, w, X, Y, opt_state):
            """Perform one update step of the optimizer."""
            grads_K = self.gradient_cost_edmd(K, w, X, Y)
            opt_state = opt_update_K(0, grads_K, opt_state)
            new_K = get_K(opt_state)
            return new_K, opt_state

        def update_w(w, X, Y, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.gradient_cost_params_vamp2(w, X, Y)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        K_opt = K_init
        w_opt = w_init
        loss_k = []
        loss_w = []

        cost_k = self.cost_edmd(K_opt, w_opt, X, Y)
        loss_k.append(cost_k)
        cost_w = self.cost_vamp2(w_opt, X, Y)
        loss_w.append(cost_w)

        idx = [i for i in range(X.shape[1])]

        l = 1

        while (
            jnp.linalg.norm(self.gradient_cost_edmd(K_opt, w_opt, X, Y)) > self.epss
            and l < self.loops
        ):
            if stochastic:
                perm_idx = np.random.permutation(idx)
                for i in range(int(X.shape[1] / batch_size)):
                    chosen_idx = perm_idx[i * batch_size : (i + 1) * batch_size]

                    X_s = jnp.take(X, chosen_idx, 1)
                    Y_s = jnp.take(Y, chosen_idx, 1)

                    K_opt, opt_state_K = update_K(K_opt, w_opt, X_s, Y_s, opt_state_K)
                    cost_k = self.cost_edmd(K_opt, w_opt, X, Y)
                    loss_k.append(cost_k)

                    w_opt, opt_state_w = update_w(w_opt, X_s, Y_s, opt_state_w)
                    cost_w = self.cost_vamp2(w_opt, X, Y)
                    loss_w.append(cost_w)

                    l = l + 1
            else:
                # Perform one update step of the optimizer

                K_opt, opt_state_K = update_K(K_opt, w_opt, X, Y, opt_state_K)
                cost_k = self.cost_edmd(K_opt, w_opt, X, Y)
                loss_k.append(cost_k)
                w_opt, opt_state_w = update_w(w_opt, X, Y, opt_state_w)
                cost_w = self.cost_vamp2(w_opt, X, Y)
                loss_w.append(cost_w)
                l = l + 1

        return K_opt, w_opt, loss_k, loss_w


class OptimalKoopman:
    def __init__(self, epss, loops) -> None:
        self.epss = epss
        self.loops = loops

    # Cost function and its gradient

    def cost_edmd(self, K, psi_x, psi_y):
        z = (1 / 2) * (np.linalg.norm(psi_y - K.T @ psi_x, "fro")) ** 2
        return z

    def gradient_cost_edmd(self, K, psi_x, psi_y):
        z = psi_x @ psi_x.T @ K - psi_x @ psi_y.T
        return z

    def gradient_descent(self, K_init, psi_x, psi_y, hk):
        """
        returns the optimal Koopman matrix K along with loss in approximating using GD algorithm.
        Params-
        K_init: Initial Koopman matrix K
        hk: step size of Adam

        returns-
        K: optimal Koopman matrix
        loss_K: loss in approximating K matrix
        """
        # Initialize the optimizer with the initial parameters
        opt_init_K, opt_update_K, get_K = optimizers.sgd(hk)

        opt_state_K = opt_init_K(K_init)

        def update_K(K, opt_state):
            """Perform one update step of the optimizer."""
            grads_K = self.gradient_cost_edmd(K, psi_x, psi_y)
            opt_state = opt_update_K(0, grads_K, opt_state)
            new_K = get_K(opt_state)
            return new_K, opt_state

        K_gd = K_init
        f_vals_gd_K = []

        loss_K = self.cost_edmd(K_gd, psi_x, psi_y)
        f_vals_gd_K.append(loss_K)

        l = 1

        while (
            jnp.linalg.norm(self.gradient_cost_edmd(K_gd, psi_x, psi_y)) > self.epss
            and l < self.loops
        ):
            K_gd, opt_state_K = update_K(K_gd, opt_state_K)
            loss_K = self.cost_edmd(K_gd, psi_x, psi_y)
            f_vals_gd_K.append(loss_K)

            l = l + 1

        return K_gd, f_vals_gd_K

    # Stochastic Gradient Descent

    def stochastic_gradient_descent(self, K_init, psi_x, psi_y, hk, batch_size):
        """
        returns the optimal Koopman matrix K along with loss in approximating using SGD algorithm.
        Params-
        K_init: Initial Koopman matrix K
        hk: step size of Adam

        returns-
        K: optimal Koopman matrix
        loss_K: loss in approximating K matrix
        """
        l = 0  # initializing number of loops
        K_sgd = K_init
        Z = np.ones((np.shape(K_sgd)))
        f_vals_sgd = []
        idx = [i for i in range(psi_x.shape[1])]
        while np.linalg.norm(Z - K_sgd, "fro") > self.epss and l < self.loops:
            perm_idx = np.random.permutation(idx)
            for i in range(int(psi_x.shape[1] / batch_size)):
                chosen_idx = perm_idx[i * batch_size : (i + 1) * batch_size]
                v = self.cost_edmd(K_sgd, psi_x, psi_y)  # value of f at K
                f_vals_sgd.append(v)  # storing values in a list
                psi_x_s = np.take(psi_x, chosen_idx, 1)
                psi_y_s = np.take(psi_y, chosen_idx, 1)
                Z = K_sgd  # storing previous value of K
                K_sgd = K_sgd - hk * self.gradient_cost_edmd(
                    K_sgd, psi_x_s, psi_y_s
                )  # updating K using gradient descent
                l = l + 1  # keeping record of loops

        return K_sgd, f_vals_sgd

    # ------------------------------------------------------------------------------------------------------------------------
    # Nesterov

    def nesterov(self, K_init, psi_x, psi_y, hk, mass=0.1):
        """
        returns the optimal Koopman matrix K along with loss in approximating using Nesterov algorithm.
        Params-
        K_init: Initial Koopman matrix K
        hk: step size of Adam

        returns-
        K: optimal Koopman matrix
        loss_K: loss in approximating K matrix
        """

        # Initialize the optimizer with the initial parameters
        opt_init_K, opt_update_K, get_K = optimizers.nesterov(hk, mass)

        opt_state_K = opt_init_K(K_init)

        def update_K(K, opt_state):
            """Perform one update step of the optimizer."""
            grads_K = self.gradient_cost_edmd(K, psi_x, psi_y)
            opt_state = opt_update_K(0, grads_K, opt_state)
            new_K = get_K(opt_state)
            return new_K, opt_state

        K_nest = K_init
        f_vals_nest_K = []

        loss_K = self.cost_edmd(K_nest, psi_x, psi_y)
        f_vals_nest_K.append(loss_K)

        l = 1

        while (
            jnp.linalg.norm(self.gradient_cost_edmd(K_nest, psi_x, psi_y)) > self.epss
            and l < self.loops
        ):
            K_nest, opt_state_K = update_K(K_nest, opt_state_K)
            loss_K = self.cost_edmd(K_nest, psi_x, psi_y)
            f_vals_nest_K.append(loss_K)

            l = l + 1

        return K_nest, f_vals_nest_K

    # -----------------------------------------------------------------------------------------------------------------------
    # ADAM optimizer

    def adam(self, K_init, psi_x, psi_y, hk=0.01):
        """
        returns the optimal Koopman matrix K along with loss in approximating using Adam algorithm.
        Params-
        K_init: Initial Koopman matrix K
        hk: step size of Adam

        returns-
        K: optimal Koopman matrix
        loss_K: loss in approximating K matrix
        """

        # Initialize the optimizer with the initial parameters
        opt_init_K, opt_update_K, get_K = optimizers.adam(hk)

        opt_state_K = opt_init_K(K_init)

        def update_K(K, opt_state):
            """Perform one update step of the optimizer."""
            grads_K = self.gradient_cost_edmd(K, psi_x, psi_y)
            opt_state = opt_update_K(0, grads_K, opt_state)
            new_K = get_K(opt_state)
            return new_K, opt_state

        K_adam = K_init
        f_vals_adam_K = []

        loss_K = self.cost_edmd(K_adam, psi_x, psi_y)
        f_vals_adam_K.append(loss_K)

        l = 1

        while (
            jnp.linalg.norm(self.gradient_cost_edmd(K_adam, psi_x, psi_y)) > self.epss
            and l < self.loops
        ):
            K_adam, opt_state_K = update_K(K_adam, opt_state_K)
            loss_K = self.cost_edmd(K_adam, psi_x, psi_y)
            f_vals_adam_K.append(loss_K)

            l = l + 1

        return K_adam, f_vals_adam_K
