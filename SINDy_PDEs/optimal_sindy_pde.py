## Optimal parameters PDE-SINDy

import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

# ============================================================================================================================================


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Computer Modern Serif"], "size": BIGGER_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,  # fontsize of the axes title
        "axes.labelsize": SMALL_SIZE,  # fontsize of the x and y labels
        "xtick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
        "ytick.labelsize": SMALL_SIZE,  # fontsize of the tick labels
        "legend.fontsize": SMALL_SIZE,  # legend fontsize
        "figure.titlesize": BIGGER_SIZE,  # fontsize of the figure title
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,  # Disable grid
        "axes.spines.top": False,  # Hide top spine
        "axes.spines.right": False,  # Hide right spine
        "axes.spines.left": True,  # Show left spine
        "axes.spines.bottom": True,  # Show bottom spine
        "xtick.bottom": True,  # Ensure x-ticks are shown
        "ytick.left": True,  # Ensure y-ticks are shown
    }

    return rc_params


# =========================================================================================================================================

# Class to approximate optimal parameters of basis function for SINDy


class OptimalPDESINDy:

    # Initialization of the class
    def __init__(self, epss, loops, rhs_des) -> None:

        self.epss = epss
        self.loops = loops
        self.rhs_des = rhs_des  # description of rhs from PDE_FIND

    # -----------------------------------------------------------------------------------------------------------------------------------------

    def pde_sindy(self, theta, Ut, lam=0.001, eps=0.04, iterations=10):
        """PDE-FIND algorithm"""
        n, d = theta.shape

        w_best = np.linalg.inv(theta.T @ theta + lam * np.eye(d)) @ theta.T @ Ut

        # for k in range(iterations):
        #     s = abs(w_best) < eps  # find coefficients less than eps ...
        #     w_best = w_best.at[s].set(0)  # ... and set them to zero

        return w_best

    # -------------------------------------------------------------------------------------------------------------------------------------------

    def build_new_theta(self, chi, theta):
        """
        Returns the new basis functions matrix
        Parameters-
        chi: values of the parameter in the basis
        theta: old basis matrix
        """

        R_n = jnp.zeros((theta.shape[0], theta.shape[1] + 2))
        R_n = R_n.at[:, : theta.shape[1]].set(theta)

        u_ind = self.rhs_des.index("u")
        ux_ind = self.rhs_des.index("u_{x}")
        uxx_ind = self.rhs_des.index("u_{xx}")
        u = theta[:, u_ind]
        u_x = theta[:, ux_ind]
        u_xx = theta[:, uxx_ind]

        exp_u = jnp.exp(chi * u)
        new_ux_term = exp_u * u_x**2
        new_uxx_term = exp_u * u_xx

        R_n = R_n.at[:, -2].set(new_ux_term)
        R_n = R_n.at[:, -1].set(new_uxx_term)
        R_n = R_n.at[:, uxx_ind].set(jnp.zeros((u_xx.shape)))

        return R_n

    # -------------------------------------------------------------------------------------------------------------------------------------------

    # Cost function of PDE SINDy

    def cost_pde_sindy(self, w, chi, theta, Ut, lam=0.001):
        """Returns the cost of PDE SINDy for a given w coefficients vector
        Parameters-
        w: Estimated w matrix for a given theta
        theta: Dictionary matrix
        Ut: Time derivative vector
        lam: regularization parameter
        """
        theta_n = self.build_new_theta(chi, theta)
        return jnp.linalg.norm(Ut - theta_n @ w) ** 2  # + lam * jnp.linalg.norm(
        #    theta_n, "fro"
        # )

    # ------------------------------------------------------------------------------------------------------------------------------------------

    # Gradient of the cost function of PDE SINDy w.r.t. vector w

    def grad_cost_pde_sindy(self, w, chi, theta, Ut, lam):
        """
        Returns the gradient of the cost function of PDE SINDy for a given w matrix
        Parameters-
        w: Estimated w matrix for a given theta
        theta: Dictionary matrix
        Ut: Time derivative vector
        lam: regularization parameter
        """
        grad_calc_psi = jax.grad(self.cost_pde_sindy)
        return grad_calc_psi(w, chi, theta, Ut, lam)

    # -----------------------------------------------------------------------------------------------------------------------------------------

    # Gradient of the cost function of PDE SINDy w.r.t. basis parameters

    def grad_cost_pde_sindy_params(self, w, chi, theta, Ut, lam):
        """
        Returns the gradient of the cost function of SINDy for a given w vector
        Parameters-
        w: Estimated w matrix for a given theta
        theta: Dictionary matrix
        Ut: Time derivative vector
        lam: regularization parameter
        """
        grad_calc = jax.grad(self.cost_pde_sindy, 1)
        return grad_calc(w, chi, theta, Ut, lam)

    # -------------------------------------------------------------------------------------------------------------------------------------------

    # Gradient descent for parameters and optimal w vector

    def gradient_descent(self, w_init, p_init, theta, Ut, lam, hw, hp, alt_loops=10):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using GD algorithm.
        Params-
        w_init: Initial basis parameters
        p_init: Initial coeffients vector
        theta, Ut: data matrices
        hw, hp: step size of GD
        alt_loops: number of alternating loops

        returns-
        p: optimal coefficients
        w: optimal parameters of the basis functions
        loss_p: loss in approximating coefficients
        loss_w: loss in approximating w
        """

        # Initialize the optimizer with the initial parameters
        opt_init_p, opt_update_p, get_p = optimizers.sgd(hp)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        opt_state_p = opt_init_p(p_init)
        opt_state_w = opt_init_w(w_init)

        def update_p(w, p, opt_state):
            """Perform one update step of the optimizer."""
            grads_p = self.grad_cost_pde_sindy_params(w, p, theta, Ut, lam)
            opt_state = opt_update_p(0, grads_p, opt_state)
            new_p = get_p(opt_state)
            return new_p, opt_state

        def update_w(w, p, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_pde_sindy(w, p, theta, Ut, lam)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        l = 1  # initializing number of loops

        w_gd = w_init
        p_gd = p_init
        loss_gd_p = []
        loss_gd_w = []

        while (
            self.cost_pde_sindy(w_gd, p_gd, theta, Ut, lam) > self.epss
            and l < self.loops
        ):

            for i in range(alt_loops):
                w_gd, opt_state_w = update_w(w_gd, p_gd, opt_state_w)
                loss_w = self.cost_pde_sindy(w_gd, p_gd, theta, Ut, lam)
                loss_gd_w.append(loss_w)

            p_gd, opt_state_p = update_p(w_gd, p_gd, opt_state_p)
            loss_p = self.cost_pde_sindy(w_gd, p_gd, theta, Ut, lam)
            loss_gd_p.append(loss_p)

            l = l + 1  # keeping record of loops

        return w_gd, p_gd, loss_gd_p, loss_gd_w

    # ------------------------------------------------------------------------------------------------------------------------------------------

    def stochastic_gradient_descent(
        self, w_init, p_init, theta, Ut, lam, hw, hp, batch_size, alt_loops=10
    ):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using SGD algorithm.
        Params-
        w_init: Initial basis parameters
        p_init: Initial coeffients vector
        theta, Ut: data matrices
        hw, hp: step size of GD
        batch_size: number of data points in a subset of SGD
        alt_loops: number of alternating loops

        returns-
        p: optimal coefficients
        w: optimal parameters of the basis functions
        loss_p: loss in approximating coefficients
        loss_w: loss in approximating w
        """

        # Initialize the optimizer with the initial parameters
        opt_init_p, opt_update_p, get_p = optimizers.sgd(hp)

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.sgd(hw)

        opt_state_p = opt_init_p(p_init)
        opt_state_w = opt_init_w(w_init)

        def update_p(w, p, theta, Ut, opt_state):
            """Perform one update step of the optimizer."""
            grads_p = self.grad_cost_pde_sindy_params(w, p, theta, Ut, lam)
            opt_state = opt_update_p(0, grads_p, opt_state)
            new_p = get_p(opt_state)
            return new_p, opt_state

        def update_w(w, p, theta, Ut, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_pde_sindy(w, p, theta, Ut, lam)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        l = 1  # initializing number of loops

        w_sgd = w_init
        p_sgd = p_init
        loss_sgd_p = []
        loss_sgd_w = []

        idx = [i for i in range(theta.shape[0])]  # Indices

        while (
            self.cost_pde_sindy(w_sgd, p_sgd, theta, Ut, lam) > self.epss
            and l < self.loops
        ):

            perm_idx = np.random.permutation(idx)
            for i in range(int(theta.shape[0] / batch_size)):
                chosen_idx = perm_idx[i * batch_size : (i + 1) * batch_size]

                theta_s = jnp.take(theta, chosen_idx, 0)
                Ut_s = jnp.take(Ut, chosen_idx, 0)

                for i in range(alt_loops):
                    w_sgd, opt_state_w = update_w(
                        w_sgd, p_sgd, theta_s, Ut_s, opt_state_w
                    )
                    loss_w = self.cost_pde_sindy(w_sgd, p_sgd, theta, Ut, lam)
                    loss_sgd_w.append(loss_w)

                p_sgd, opt_state_p = update_p(w_sgd, p_sgd, theta_s, Ut_s, opt_state_p)
                loss_p = self.cost_pde_sindy(w_sgd, p_sgd, theta, Ut, lam)
                loss_sgd_p.append(loss_p)

            l = l + 1  # keeping record of loops

        return w_sgd, p_sgd, loss_sgd_p, loss_sgd_w

    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    def nesterov(self, w_init, p_init, theta, Ut, lam, hw, hp, alt_loops=10, mass=0.1):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using the Nesterov algorithm.
        Params-
        w_init: Initial basis parameters
        p_init: Initial coeffients vector
        theta, Ut: data matrices
        hw, hp: step size of GD
        alt_loops: number of alternating loops

        returns-
        p: optimal coefficients
        w: optimal parameters of the basis functions
        loss_p: loss in approximating coefficients
        loss_w: loss in approximating w
        """

        # Initialize the optimizer with the initial parameters
        opt_init_w, opt_update_w, get_w = optimizers.nesterov(hw, mass)

        # Initialize the optimizer with the initial parameters
        opt_init_p, opt_update_p, get_p = optimizers.nesterov(hp, mass)

        opt_state_w = opt_init_w(w_init)
        opt_state_p = opt_init_p(p_init)

        def update_w(w, chi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_pde_sindy(w, chi, theta, Ut, lam)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        def update_p(w, chi, opt_state):
            """Perform one update step of the optimizer."""
            grads_p = self.grad_cost_pde_sindy_params(w, chi, theta, Ut, lam)
            opt_state = opt_update_p(0, grads_p, opt_state)
            new_p = get_p(opt_state)
            return new_p, opt_state

        w_nest = w_init
        p_nest = p_init
        f_vals_nest_p = []
        f_vals_nest_w = []

        l = 1

        while (
            jnp.linalg.norm(self.cost_pde_sindy(w_nest, p_nest, theta, Ut, lam))
            > self.epss
            and l < self.loops
        ):

            for i in range(alt_loops):
                w_nest, opt_state_w = update_w(w_nest, p_nest, opt_state_w)
                loss_w = self.cost_pde_sindy(w_nest, p_nest, theta, Ut, lam)
                f_vals_nest_w.append(loss_w)

            p_nest, opt_state_p = update_p(w_nest, p_nest, opt_state_p)
            loss = self.cost_pde_sindy(w_nest, p_nest, theta, Ut, lam)
            f_vals_nest_p.append(loss)

            l = l + 1

        return w_nest, p_nest, f_vals_nest_w, f_vals_nest_p

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

    def adam(
        self,
        w_init,
        p_init,
        theta,
        Ut,
        lam,
        hw,
        hp,
        alt_loops=10,
    ):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using the Adam algorithm.
        Params-
        w_init: Initial basis parameters
        p_init: Initial coeffients vector
        theta, Ut: data matrices
        hw, hp: step size of GD
        decay_rate, decay_steps: rate and steps of deacy in step size
        alt_loops: number of alternating loops

        returns-
        p: optimal coefficients
        w: optimal parameters of the basis functions
        loss_p: loss in approximating coefficients
        loss_w: loss in approximating w
        """

        # Initialize the optimizer with the initial parameters and schedules
        opt_init_w, opt_update_w, get_w = optimizers.adam(hw)
        opt_init_p, opt_update_p, get_p = optimizers.adam(hp)

        opt_state_w = opt_init_w(w_init)
        opt_state_p = opt_init_p(p_init)

        def update_w(w, chi, opt_state):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_pde_sindy(w, chi, theta, Ut, lam)
            opt_state = opt_update_w(0, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        def update_p(w, chi, opt_state):
            """Perform one update step of the optimizer."""
            grads_p = self.grad_cost_pde_sindy_params(w, chi, theta, Ut, lam)
            opt_state = opt_update_p(0, grads_p, opt_state)
            new_p = get_p(opt_state)
            return new_p, opt_state

        w_adam = w_init
        p_adam = p_init
        f_vals_adam_w = []
        f_vals_adam_p = []
        p_vals = []

        l = 1
        # w_adam_temp = w_adam + 2 * jnp.ones_like(w_adam)

        while (
            self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam) > self.epss
            and l < self.loops
        ):  # > self.cost_pde_sindy(w_adam_temp, p_adam, theta, Ut, lam)

            for i in range(alt_loops):
                # w_adam_temp = w_adam
                w_adam, opt_state_w = update_w(w_adam, p_adam, opt_state_w)
                loss_w = self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam)
                f_vals_adam_w.append(loss_w)

            p_adam, opt_state_p = update_p(w_adam, p_adam, opt_state_p)
            loss = self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam)
            f_vals_adam_p.append(loss)
            p_vals.append(p_adam)

            l += 1

        return w_adam, p_adam, f_vals_adam_w, f_vals_adam_p, p_vals

    def adam_decay(
        self,
        w_init,
        p_init,
        theta,
        Ut,
        lam,
        hw,
        hp,
        decay_rate,
        decay_steps,
        alt_loops=10,
    ):
        """
        returns the optimal Psi matrix and basis parameters along with loss in approximating them using the Adam algorithm.
        Params-
        w_init: Initial basis parameters
        p_init: Initial coeffients vector
        theta, Ut: data matrices
        hw, hp: step size of GD
        decay_rate, decay_steps: rate and steps of deacy in step size
        alt_loops: number of alternating loops

        returns-
        p: optimal coefficients
        w: optimal parameters of the basis functions
        loss_p: loss in approximating coefficients
        loss_w: loss in approximating w
        """

        # Learning rate schedules with exponential decay
        schedule_w = optimizers.exponential_decay(hw, decay_steps, decay_rate)
        schedule_p = optimizers.exponential_decay(hp, decay_steps, decay_rate)

        # Initialize the optimizer with the initial parameters and schedules
        opt_init_w, opt_update_w, get_w = optimizers.adam(schedule_w)
        opt_init_p, opt_update_p, get_p = optimizers.adam(schedule_p)

        opt_state_w = opt_init_w(w_init)
        opt_state_p = opt_init_p(p_init)

        def update_w(w, chi, opt_state, step):
            """Perform one update step of the optimizer."""
            grads_w = self.grad_cost_pde_sindy(w, chi, theta, Ut, lam)
            opt_state = opt_update_w(step, grads_w, opt_state)
            new_w = get_w(opt_state)
            return new_w, opt_state

        def update_p(w, chi, opt_state, step):
            """Perform one update step of the optimizer."""
            grads_p = self.grad_cost_pde_sindy_params(w, chi, theta, Ut, lam)
            opt_state = opt_update_p(step, grads_p, opt_state)
            new_p = get_p(opt_state)
            return new_p, opt_state

        w_adam = w_init
        p_adam = p_init
        f_vals_adam_w = []
        f_vals_adam_p = []
        p_vals = []

        l = 1
        step = 0

        while (
            jnp.linalg.norm(self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam))
            > self.epss
            and l < self.loops
        ):

            step += 1
            for i in range(alt_loops):

                w_adam, opt_state_w = update_w(w_adam, p_adam, opt_state_w, step)
                loss_w = self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam)
                f_vals_adam_w.append(loss_w)

            p_adam, opt_state_p = update_p(w_adam, p_adam, opt_state_p, step)
            loss = self.cost_pde_sindy(w_adam, p_adam, theta, Ut, lam)
            f_vals_adam_p.append(loss)
            p_vals.append(p_adam)

            l += 1

        return w_adam, p_adam, f_vals_adam_w, f_vals_adam_p, p_vals
