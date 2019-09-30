"""
Implementation of safety criterion.

Returns -dV/dq, where V is the criterion function that we
want to maximize, with V = 0.5*Kc*(d-d0)^2,
where d is the distance from the center of an obstacle,
d0 the safety distance,
Kc a control gain.
"""
import numpy as np
from src.utils import (
    compute_link_loss,
    compute_slope,
    derivative_of_dist_to_obstacle,
    minimum_dist_to_obstacle
)


class SafetyCriterion():
    """
    Class that implements the safety criterion.

    Evaluate a safety criterion (whether all link points are more than
    a safety distance further than the obstacles) to compute a loss
    in order to control the robotic arm.
    Create an instance of this class (optionally) providing the
    following arguments:
         x_obs1: float, the x coordinate of the first obstacle's
            position, default 5.0
        - y_obs1: float, the y coordinate of the first obstacle's
            position, default 0.3
        - x_obs2: float, the x coordinate of the second obstacle's
            position, default 5.0
        - y_obs2: float, the y coordinate of the second obstacle's
            position, default 1.7
        -safety_dist: float, the minimum distance from an obstacle
            in order for a trajectory to be considered safe,
            default 0.65
    """

    def __init__(self, x_obs0=5, y_obs0=0.3, x_obs1=5, y_obs1=1.7,
                 safety_dist=0.65):
        """Initialize giving obstacles' (x, y) and safety distance."""
        self.obstacles = [(x_obs0, y_obs0), (x_obs1, y_obs1)]
        self.safety_dist = safety_dist

    def compute(self, jnts_x, jnts_y, djnts_x, djnts_y):
        """
        Compute loss due to safety distance criterion.

        Inputs:
            - jnts_x: array (8,), x coordinate of angles' positions
            - jnts_y: array (8,), y coordinate of angles' positions
            - djnts_x: list of arrays (8,), x coordinates of the
                 derivatives of angles' positions
            - djnts_y: list of arrays (8,), y coordinates of the
                 derivatives of angles' positions
        Returns:
            - array (8,) of loss w.r.t. joint angles
        """
        self.jnts_x = jnts_x[3:]
        self.jnts_y = jnts_y[3:]
        self.djnts_x = djnts_x[3:]
        self.djnts_y = djnts_y[3:]
        self._compute_link_slopes()
        loss = np.zeros((8,))
        for x_obs, y_obs in self.obstacles:
            self._compute_minimum_dist_per_link(x_obs, y_obs)
            self._compute_derivatives_of_dists_per_link(x_obs, y_obs)
            loss += np.sum(self._compute_link_losses(), axis=0)
        return loss

    def _compute_link_slopes(self):
        """Compute link_slopes, a list of floats."""
        self.link_slopes = [
            compute_slope(
                self.jnts_x[jnt],
                self.jnts_y[jnt],
                self.jnts_x[jnt + 1],
                self.jnts_y[jnt + 1]
            )
            for jnt in range(len(self.jnts_x) - 1)
        ]

    def _compute_minimum_dist_per_link(self, x_obs, y_obs):
        """Compute min distance per link, list of (dist, point_type)."""
        self.minimum_dists_per_link = [
            minimum_dist_to_obstacle(
                self.jnts_x[jnt],
                self.jnts_y[jnt],
                self.jnts_x[jnt + 1],
                self.jnts_y[jnt + 1],
                self.link_slopes[jnt],
                x_obs,
                y_obs
            )
            for jnt in range(len(self.jnts_x) - 1)
        ]

    def _compute_derivatives_of_dists_per_link(self, x_obs, y_obs):
        """Compute derivates of distances, a list of arrays (8,)."""
        self.derivatives_of_dists_per_link = [
            derivative_of_dist_to_obstacle(
                self.minimum_dists_per_link[jnt],
                self.jnts_x[jnt],
                self.jnts_y[jnt],
                self.jnts_x[jnt + 1],
                self.jnts_y[jnt + 1],
                self.djnts_x[jnt],
                self.djnts_y[jnt],
                self.djnts_x[jnt + 1],
                self.djnts_y[jnt + 1],
                self.link_slopes[jnt],
                x_obs,
                y_obs
            )
            for jnt in range(len(self.jnts_x) - 1)
        ]

    def _compute_link_losses(self):
        """Return an array (joints, 8)."""
        return np.array([
            compute_link_loss(
                self.minimum_dists_per_link[jnt],
                self.derivatives_of_dists_per_link[jnt],
                self.safety_dist
            )
            for jnt in range(len(self.jnts_x) - 1)
        ])
