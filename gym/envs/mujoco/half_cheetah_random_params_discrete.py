import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahRandomParamsDiscreteEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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
        # torso mass
        if self.np_random.rand() < 0.1:
            torso_mass = 13.0
        else:
            torso_mass = 1.0
        self.model.body_mass[1] = torso_mass
        # ground friction
        if self.np_random.rand() < 0.1:
            ground_friction = 3.1
        else:
            ground_friction = 0.1
        for i in range(0, 9):
            self.model.geom_friction[i][0] = ground_friction
        # joint damping
        if self.np_random.rand() < 0.1:
            dof_damping = 11.0
        else:
            dof_damping = 1.0
        for i in range(3,9):
            self.model.dof_damping[i] = dof_damping
        #print(torso_mass)
        #print(ground_friction)
        #print(dof_damping)
        #print()

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
