import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahRandomParamsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        #self.model.body_mass[1] = min(9.0, max(3.0, self.np_random.normal(loc=6.0, scale=1.5)))
        self.model.body_mass[1] = min(13.0, max(1.0, self.np_random.normal(loc=7.0, scale=3.0))) # mod
        ground_friction = min(3.1, max(0.1, self.np_random.normal(loc=1.6, scale=0.8)))
        for i in range(0, 9):
            self.model.geom_friction[i][0] = ground_friction

        # add joint damping and armature 2019/03/15
        for i in range(3,9):
            self.model.dof_damping[i] = min(11.0, max(1.0, self.np_random.normal(loc=6.0, scale=2.5)))
        #for i in range(3,9):
        #    self.model.dof_armature[i] = min(1.0, max(0.01, self.np_random.normal(loc=5.0, scale=2.5)))

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
