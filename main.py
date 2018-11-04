'''
Class Redundant8dofController controls the operation of a robotic arm
in order to perform a fine-grained task.
'''

from __future__ import division

from imageio import mimsave
from matplotlib import pyplot as plt
import numpy as np

from src.criterion import SafetyCriterion

class Redundant8dofController():
    '''
    A class to configure and simulate an 8-DOF robotic arm to perform
    a fine-grained task: ride its body through an obstacle topology in
    order to reach a given end point.

    Initialize an instance of this class (optionally) providing the
    following arguments:
        - x_obs0: float, the x coordinate of the first obstacle's
            position.
        - y_obs0: float, the y coordinate of the first obstacle's
            position.
        - x_obs1: float, the x coordinate of the second obstacle's
            position.
        - y_obs1: float, the y coordinate of the second obstacle's
            position.
    Note that the arm has 8 degrees of freedom across 8 links of unit
    length. Thus, considering the arm is based on (0, 0), its
    effective region is the disk x^2 + y^2 <= 64. Both obstacles' and
    goal's coordinates should be selected inside this area. Obstacles
    are circles with a radius equal to 0.2.
    '''
    def __init__(self, x_obs0=5, y_obs0=0.3, x_obs1=5, y_obs1=1.7):
        self.x_obs0 = x_obs0
        self.y_obs0 = y_obs0
        self.x_obs1 = x_obs1
        self.y_obs1 = y_obs1
        self.x_start = 5
        self.y_start = 1
        self.dt = 0.001 #sampling period

    def simulate(self, x_goal, y_goal, T_f=1, kappa_c=80, kappa_p=10,
                 safety_dist=0.65, gif_name=None, gif_fps=10,
                 plot_arm=True, plot_fps=10):
        '''
        Simulate the robotic arm's task to reach the goal while
        avoiding the obstacles.

        Inputs:
            - x_goal: float, the goal's x coordinate
            - y_goal: float, the goal's y_coordinate
            - T_f: float (positive), the total duration of the
                simulation in seconds, default 1.0
            - kappa_c: float, control gain for redundant DOF,
                default 80.0
            - kappa_p: float, PD control gain, default 10.0
            -safety_dist: float, the minimum distance from an obstacle
                in order for a trajectory to be considered safe,
                default 0.65
            - gif_name: str or None, the name of the .gif file to create
                showing the arm's movement. If None, no fole is created,
                default None
            - gif_fps: int, frames per second on the saved .gif file,
                to be ignored if gif_fps is None,
                default 10
            - plot_arm: boolean, whether to plot arm configurations,
                defalt True
            - plot_fps: int, frames per second on when plotting,
                ignored if plot_arm=False
                default 10
        Returns:
            - self: Redundant8dofController object
        Note that obstacle have a diameter of 0.4, thus the safety
        distance, s measured from the obstacle's center, must be
        greater than 0.2.
        '''
        # Initialize parameters
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.T_f = T_f
        self.kappa_c = kappa_c
        self.kappa_p = kappa_p
        self.safety_dist = safety_dist
        self.gif_name = gif_name
        self.gif_fps = gif_fps
        self.plot_arm = plot_arm
        self.plot_fps = plot_fps
        self.joint_angles = np.zeros(8)
        self.x_effector = self.x_start
        self.y_effector = self.y_start

        # Compute desired trajectory (positions and velocities)
        self.x_desired, self.y_desired, self.vx_desired, self.vy_desired = \
            self._compute_desired_trajectory()

        # Simulate and save angles and effector's position
        self._config_plots = []
        self.fig, self.axs = plt.subplots()
        if self.plot_arm:
            self.fig.show()
        self._joint_angles_on_time, self._x_effector_on_time, \
        self._y_effector_on_time = zip(*[
            self._run_kinematics_and_criteria(t)
            for t, _ in enumerate(self.x_desired)
        ])
        if self.gif_name is not None:
            mimsave(gif_name, self._config_plots, fps=self.gif_fps)
        return self

    def _run_kinematics_and_criteria(self, timestamp):
        '''
        Perform one iteration of the kinematic model with respect to
        the defined criteria for safety and return kinematic info for
        the timestamp examined.

        Inputs:
            - timestamp: int, discrete time value (or current number
                of iteration)
        Outputs:
            - joint_angles: array 8x1, the computed values for joint
                angles
            - x_effector: float, the computed x coordinate of effector's
                position
            - y_effector: float, the computed y coordinate of effector's
                position
        '''
        # Forward kinematics
        jnts_x, jnts_y, djnts_x, djnts_y, jacobian, jacobian_psinv = \
            self.compute_joint_positions(np.cumsum(self.joint_angles))
        self.x_effector = jnts_x[-1]
        self.y_effector = jnts_y[-1]

        # Subtasks
        self.joint_angles += \
            self.dt \
            * (
                self._subtask_trajectory(jacobian_psinv, timestamp)
                + self._subtask_criterion(jnts_x, jnts_y, djnts_x, djnts_y,
                                          jacobian, jacobian_psinv)
            )

        # Plot and return
        if timestamp % 50 == 0:
            self._config_plots.append(self._plot_configuration(jnts_x, jnts_y))
        return np.copy(self.joint_angles), self.x_effector, self.y_effector

    def _subtask_criterion(self, jnts_x, jnts_y, djnts_x, djnts_y, jacobian,
                           jacobian_psinv):
        '''
        Act for 2nd subtask of a fine-grained robotic manipulation,
        compute angle speed due to the control criterion.

        Inputs:
            - jnts_x: list 8 float numbers, the x coordinates of joints'
                positions
            - jnts_y: list 8 float numbers, the y coordinates of joints'
                positions
            - djnts_x: list of arrays (8,), x coordinates of the
                 derivatives of angles' positions
            - djnts_y: list of arrays (8,), y coordinates of the
                 derivatives of angles' positions
            - jacobian: array (2, 8), the jacobian matrix
            - jacobn_psinv: array (8, 2), the Jacobian pseud-inverse of
                the kinematic model
        Outputs:
            - The angle speed due to criterion evaluation
        '''
        criterion = SafetyCriterion(
            self.x_obs0, self.y_obs0,
            self.x_obs1, self.y_obs1,
            self.safety_dist
        )
        dq_criterion = criterion.compute(jnts_x, jnts_y, djnts_x, djnts_y)
        return (
            self.kappa_c
            * np.dot(
                (np.eye(8) - np.matmul(jacobian_psinv, jacobian)),
                dq_criterion
            )
        )

    def _subtask_trajectory(self, jacobian_psinv, timestamp):
        '''
        Act for 1st subtask of a fine-grained robotic manipulation,
        compute angle speed due to the proclivity to follow the desired
        trajectory.

        Inputs:
            - jacobian_psinv: array (8, 2), the Jacobian pseud-inverse of
                the kinematic model
            - timestamp: int, discrete time value (or current number
                of iteration)
        Outputs:
            - The angle speed due to trajectory
        '''
        v = np.array([self.vx_desired[timestamp], self.vy_desired[timestamp]])
        p = np.array([self.x_desired[timestamp], self.y_desired[timestamp]])
        p_true = np.array([self.x_effector, self.y_effector])
        return np.matmul(jacobian_psinv, v + self.kappa_p*(p-p_true))

    def _compute_desired_trajectory(self):
        '''
        Given the goal coordinates and the desired time of total task,
        compute the ideal trajectory (position and speed) to be
        followed.

        Polynomial space equation for velocity-acceleration control is
        applied.
        '''
        t = np.arange(0, self.T_f+self.dt, self.dt)
        slope = (self.y_goal-self.y_start) / (self.x_goal-self.x_start)
        x_desired = self.x_start \
                    + (3 * (self.x_goal-self.x_start) / (self.T_f**2)) * t**2\
                    - (2 * (self.x_goal-self.x_start)/(self.T_f**3)) * t**3
        y_desired = self.y_start + slope*(x_desired-self.x_start)
        vx_desired = (6 * (self.x_goal-self.x_start)/(self.T_f**2)) * t \
                     - (6* (self.x_goal-self.x_start)/(self.T_f**3)) * t**2
        vy_desired = slope * vx_desired
        return x_desired, y_desired, vx_desired, vy_desired

    def _plot_configuration(self, jnts_x, jnts_y):
        '''
        Plot current configuration.

        Inputs:
            - jnts_x: list 8 float numbers, the x coordinates of joints'
                positions
            - jnts_y: list 8 float numbers, the y coordinates of joints'
                positions
        Returns:
            - plot image: array, the plotted image in numpy array form
        '''
        self.axs.cla()
        self.axs.axis([-2, 8, -2, 5])

        # Obstacles
        obs_1 = plt.Circle((self.x_obs0, self.y_obs0), radius=0.2, color='r')
        self.axs.add_artist(obs_1)
        obs_2 = plt.Circle((self.x_obs1, self.y_obs1), radius=0.2, color='r')
        self.axs.add_artist(obs_2)

        # Arm
        self.axs.plot(self.x_desired, self.y_desired, 'm:')
        self.axs.plot([0], [0], 'o')
        self.axs.plot([0, jnts_x[0]], [0, jnts_y[0]])
        self.axs.plot([jnts_x[0]], [jnts_y[0]], '*')
        self.axs.plot([jnts_x[0], jnts_x[1]], [jnts_y[0], jnts_y[1]], 'b')
        self.axs.plot([jnts_x[1]], [jnts_y[1]], '*')
        self.axs.plot([jnts_x[1], jnts_x[2]], [jnts_y[1], jnts_y[2]], 'b')
        self.axs.plot([jnts_x[2]], [jnts_y[2]], '*')
        self.axs.plot([jnts_x[2], jnts_x[3]], [jnts_y[2], jnts_y[3]], 'b')
        self.axs.plot([jnts_x[3]], [jnts_y[3]], '*')
        self.axs.plot([jnts_x[3], jnts_x[4]], [jnts_y[3], jnts_y[4]], 'b')
        self.axs.plot([jnts_x[4]], [jnts_y[4]], '*')
        self.axs.plot([jnts_x[4], jnts_x[5]], [jnts_y[4], jnts_y[5]], 'b')
        self.axs.plot([jnts_x[5]], [jnts_y[5]], '*')
        self.axs.plot([jnts_x[5], jnts_x[6]], [jnts_y[5], jnts_y[6]], 'b')
        self.axs.plot([jnts_x[6]], [jnts_y[6]], '*')
        self.axs.plot([jnts_x[6], jnts_x[7]], [jnts_y[6], jnts_y[7]], 'b')
        self.axs.plot([jnts_x[7]], [jnts_y[7]], 'y*')
        self.axs.plot([jnts_x[7]], [jnts_y[7]], 'g+')
        self.axs.plot([self.x_goal], [self.y_goal], 'o')

        # Draw and create image
        self.fig.canvas.draw()
        if self.plot_arm:
            plt.pause(1 / self.plot_fps)
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        return image.reshape(self.fig.canvas.get_width_height()[::-1] + (3, ))

    @staticmethod
    def compute_joint_positions(angle_cumsum):
        '''
        Given the cummulative sum of joint angles, compute joint
        positions.

        Inputs:
            - angle_cumsum: array (8,), the cummulative sum of angles
        Returns:
            - jnts_x: array (8,), x coordinate of angles' positions
            - jnts_y: array (8,), y coordinate of angles' positions
            - djnts_x: list of arrays (8,), x coordinates of the
                 derivatives of angles' positions
            - djnts_y: list of arrays (8,), y coordinates of the
                 derivatives of angles' positions
            - jacobian: array (2, 8), the jacobian matrix
            - jacobian_psinv: array (8, 2), pseudo-inverse of jacobian
        '''
        (s1, s12, s123, s1234, s12345, s123456, s1234567, s12345678) = \
            np.sin(angle_cumsum).tolist()
        (c1, c12, c123, c1234, c12345, c123456, c1234567, c12345678) = \
            np.cos(angle_cumsum).tolist()

        # Jacobian and pseudo-inverse Jacobian computation
        jacobian = np.zeros((2, 8))
        jacobian[0, :] = np.cumsum([
            -c1, -s12, -c123, -s1234, -s12345, -s123456, c1234567, -s12345678
        ][::-1])[::-1]
        jacobian[1, :] = np.cumsum([
            -s1, c12, -s123, c1234, c12345, c123456, s1234567, c12345678
        ][::-1])[::-1]
        jacobian_psinv = np.matmul(
            jacobian.T,
            np.linalg.inv(np.matmul(jacobian, jacobian.T))
        )

        # Forward kinematics (joints' x and y positions)
        jnts_x = np.cumsum([
            -s1, c12, -s123, c1234, c12345, c123456, s1234567, c12345678
        ])
        jnts_y = np.cumsum([
            c1, s12, c123, s1234, s12345, s123456, -c1234567, s12345678
        ])

        # Derivatives of forward kinematics w.r.t. joint angles
        djnts_x = np.cumsum([
            -c1, -s12, -c123, -s1234, -s12345, -s123456, c1234567, -s12345678
        ]).tolist()
        djnts_y = np.cumsum([
            -s1, c12, -s123, c1234, c12345, c123456, s1234567, c12345678
        ]).tolist()
        djnts_x = [
            np.array(djnts_x[:jnt+1][::-1] + [0 for _ in range(7 - jnt)])
            for jnt, _ in enumerate(djnts_x)
        ]
        djnts_y = [
            np.array(djnts_y[:jnt+1][::-1] + [0 for _ in range(7 - jnt)])
            for jnt, _ in enumerate(djnts_y)
        ]
        return jnts_x, jnts_y, djnts_x, djnts_y, jacobian, jacobian_psinv

    def effector_coords_on_time(self):
        '''
        Returns a list of 2 lists, the x and y coordinates of the
        robot's effector.
        '''
        return [self._x_effector_on_time, self._y_effector_on_time]

    @property
    def config_plots(self):
        '''Returns the list of configuration plots'''
        return self._config_plots

    @property
    def joint_angles_on_time(self):
        '''Return a list of arrays (8,) containing the angles
        of the 8 joints as changing with time.'''
        return self._joint_angles_on_time

if __name__ == "__main__":
    Redundant8dofController().simulate(7, 3, gif_name='robot_arm.gif', plot_arm=True)
    images = [
        image
        for x_goal, y_goal in [(7, 3), (5.5, 1.2), (6, 2)]
        for image in
        Redundant8dofController().simulate(x_goal, y_goal, plot_arm=True, kappa_p=3, kappa_c=246).config_plots
    ]
    mimsave(
        'robot_sims.gif',
        images,
        fps=7
    )
