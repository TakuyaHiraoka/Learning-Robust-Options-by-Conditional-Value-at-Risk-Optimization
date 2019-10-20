import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# to deal with exception in iniztialization
import sys

class Walker2dRandomParamsDiscreteEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        # Parameter Randomization # replica of previouse walker2D environment
        #  add missing clip part @ 20190219
        # -torso mass
        #torso_mass = min(9.0, max(3.0, self.np_random.normal(loc=6.0, scale=1.5)))
        if self.np_random.rand() < 0.1:
            torso_mass = 9.0
        else:
            torso_mass = 3.0
        self.model.body_mass[1] = torso_mass
        # -ground friction
        #ground_friction = min(2.9, max(0.9, self.np_random.normal(loc=1.9, scale=0.4)))
        if self.np_random.rand() < 0.1:
            ground_friction = 2.9
        else:
            ground_friction = 0.9
        self.model.geom_friction[7][0] = ground_friction

        try: # add try-exception syntax to deal with exceptions in initialization (primary arised by dummy_env of WorstCase)
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
            )
        except:
            self.set_state(
                self.init_qpos,
                self.init_qvel
            )
            print("Got following exception in walker2D initialization: ")
            print(sys.exc_info()[0])
            print("The state is initialized with a default values for initial positions and velocities")

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
