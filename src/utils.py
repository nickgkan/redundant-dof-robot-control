'''
A set of functions to support the robotic arm control, motsly
concerning the computation of the controller loss function.
'''
from math import sqrt

import numpy as np

def compute_link_loss(min_dist, dist_der, safety_dist):
    '''
    Compute loss due to a certain link, given its minimum
    distance from an obstacle and the distance's derivative.

    Inputs:
        - min_dist: tuple (dist (float), point_type (int))
            dist is the minimum distance from the obstacle
            point_type denotes the type of the point that
            is closest to the obstacle and is
                * 0, if the point is the first joint
                * 1, if the point is the second joint
                * 2, if it is an intermediate point
        - dist_der: array (8,), the derivatives of distances
            w.r.t. joint angles
        - safety_dist: float, minimum safe distance from an obstacle
    Returns:
        - array (8,), the loss due to this link being close to
            an obstacle, if such a case
    '''
    dist = min_dist[0]
    if dist < safety_dist:
        return (safety_dist-dist) * dist_der
    return np.zeros_like(dist_der)

def compute_slope(x_jnt_0, y_jnt_0, x_jnt_1, y_jnt_1):
    '''
    Compute the slope of a link.

    Inputs:
        - x_jnt0: float, x coordinate of the first joint's position
        - y_jnt0: float, y coordinate of the first joint's position
        - x_jnt1: float, x coordinate of the second joint's position
        - y_jnt1: float, x coordinate of the second joint's position
    Returns:
        - float or None, the slope of the link
    '''
    if x_jnt_0 == x_jnt_1:
        return None
    return (y_jnt_1 - y_jnt_0) / (x_jnt_1 - x_jnt_0)

def derivative_of_dist_to_obstacle(min_dist, x_jnt0, y_jnt0, x_jnt1, y_jnt1,
                                   dx_jnt0, dy_jnt0, dx_jnt1, dy_jnt1,
                                   link_slope, x_obs, y_obs):
    '''
    Compute derivatives of distance to an obstacle w.r.t. joint angles.

    Inputs:
        - min_dist: tuple (dist (float), point_type (int))
            dist is the minimum distance from the obstacle
            point_type denotes the type of the point that
            is closest to the obstacle and is
                * 0, if the point is the first joint
                * 1, if the point is the second joint
                * 2, if it is an intermediate point
        - x_jnt0: float, x coordinate of the first joint's position
        - y_jnt0: float, y coordinate of the first joint's position
        - x_jnt1: float, x coordinate of the second joint's position
        - y_jnt1: float, x coordinate of the second joint's position
        - dx_jnt0: array (8,), x coordinate of the derivatives of the
            first joint's position
        - dy_jnt0: array (8,), y coordinate of the derivatives of the
            first joint's position
        - dx_jnt1: array (8,), x coordinate of the derivatives of the
            second joint's position
        - dy_jnt1: array (8,), y coordinate of the derivatives of the
            second joint's position
        - link_slope: float, link's slope
        - x_obs: float, x coordinate of the obstacle's position
        - y_obs: float, y coordinate of the obstacle's position
    Returns:
        - derivatives (8,) w.r.t. joint angles
    '''
    dist, point_type = min_dist
    if point_type == 0:
        dist_der = ((x_jnt0 - x_obs)*dx_jnt0 + (y_jnt0 - y_obs)*dy_jnt0) / dist
    elif point_type == 1:
        dist_der = ((x_jnt1 - x_obs)*dx_jnt1 + (y_jnt1 - y_obs)*dy_jnt1) / dist
    elif point_type == 2:
        if link_slope is None:
            dist_der = dx_jnt0 if x_jnt0 > x_obs else -dx_jnt0
        elif link_slope == 0:
            dist_der = dy_jnt0 if y_jnt0 > y_obs else -dy_jnt0
        else:
            x_intersect = (
                x_obs/link_slope + y_obs + link_slope*x_jnt0 - y_jnt0
            ) / (link_slope + 1/link_slope)
            y_intersect = link_slope*(x_intersect - x_jnt0) + y_jnt0
            dlink_slope = ((1 / (x_jnt1 - x_jnt0))
                           * (
                               dy_jnt1 - dy_jnt0
                               + link_slope * (dx_jnt1-dx_jnt0)
                           ))
            dx_intersect = (
                link_slope**4 * dx_jnt0
                + link_slope**2 * dlink_slope * (y_jnt0-y_obs)
                - link_slope**3 * dy_jnt0
                + dlink_slope * (y_obs-y_jnt0)
                + 2 * link_slope * dlink_slope * (x_jnt0-x_obs)
                + link_slope**2 * dx_jnt0
                - link_slope * dy_jnt0
            ) / (1 + link_slope**2) ** 2
            dy_intersect = (link_slope * (dx_intersect-dx_jnt0)
                            + dlink_slope * (x_intersect-x_jnt0)
                            + dy_jnt0)
            dist_der = (
                (x_intersect-x_obs) * dx_intersect
                + (y_intersect-y_obs) * dy_intersect
                ) / dist
    return dist_der

def eucl_dist(x_0, y_0, x_1, y_1):
    '''Euclidean distance of two points (x_0, y_0) and (x_1, y_1)'''
    return sqrt((x_1 - x_0)**2 + (y_1 - y_0)**2)

def minimum_dist_to_obstacle(x_jnt0, y_jnt0, x_jnt1, y_jnt1, link_slope,
                             x_obs, y_obs):
    '''
    Compute minimum distance of a link from a given obstacle.

    Inputs:
        - x_jnt0: float, x coordinate of the first joint's position
        - y_jnt0: float, y coordinate of the first joint's position
        - x_jnt1: float, x coordinate of the second joint's position
        - y_jnt1: float, x coordinate of the second joint's position
        - link_slope: float, link's slope
        - x_obs: float, x coordinate of the obstacle's position
        - y_obs: float, y coordinate of the obstacle's position
    Returns:
        - min_dist: tuple (dist (float), point_type (int))
            dist is the minimum distance from the obstacle
            point_type denotes the type of the point that
            is closest to the obstacle and is
                * 0, if the point is the first joint
                * 1, if the point is the second joint
                * 2, if it is an intermediate point
    '''
    # Search for an intermediate closest point
    if link_slope is None:
        x_poss_closest = x_jnt0
        y_poss_closest = y_obs
    elif link_slope == 0:
        x_poss_closest = x_obs
        y_poss_closest = y_jnt0
    else:
        x_poss_closest = (
            x_obs/link_slope + y_obs + link_slope*x_jnt0 - y_jnt0
        ) / (link_slope + 1/link_slope)
        y_poss_closest = link_slope*(x_poss_closest - x_jnt0) + y_jnt0

    # Check if it is indeed an intermediate point, else return a joint
    if (
            x_poss_closest >= min(x_jnt0, x_jnt1)
            and x_poss_closest <= max(x_jnt0, x_jnt1)
            and y_poss_closest >= min(y_jnt0, y_jnt1)
            and y_poss_closest <= max(y_jnt0, y_jnt1)
        ):
        return eucl_dist(x_poss_closest, y_poss_closest, x_obs, y_obs), 2
    jnt_dists = np.array([
        eucl_dist(x_jnt0, y_jnt0, x_obs, y_obs),
        eucl_dist(x_jnt1, y_jnt1, x_obs, y_obs)
    ])
    return np.min(jnt_dists), np.argmin(jnt_dists)
