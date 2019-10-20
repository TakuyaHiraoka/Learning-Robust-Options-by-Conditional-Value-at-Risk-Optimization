import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math

import numpy as np

import gym
import random
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

import pyrr
from pyrr.utils import all_parameters_as_numpy_arrays
import math
import six


def isclose(a, b, rel_tol=1e-04, abs_tol=0.0):
    # TODO: move to util func
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def rotate_vector(v, axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    R = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return np.dot(R, v)


def IceWallFactory(class_type,):
    """class_type should be an OpenAI gym time"""

    class IceWallEnv(class_type, utils.EzPickle):
        # Using https://github.com/bstadie/third_person_im/blob/88516c1703221586099062053af696f0b4a31cda/rllab/envs/mujoco/maze/maze_env.py
        # as a base
        # ORI_IND = None

        # TODO: remove this
        MAZE_MAKE_CONTACTS = False

        # manually give a penalty if the torso comes into contact with the wall
        MANUAL_COLLISION = False
        # TODO: this suckssssss, maye shouldn't use mujoco at all.

        def __init__(
                self,
                model_path,
                ori_ind,
                wall_height = .12,
                wall_pos_range = ([2.8, 0.0], [2.8, 0.0]),
                n_bins=10,
                sensor_range=10.,
                sensor_span=math.pi/2,
                *args,
                **kwargs):

            self._n_bins = n_bins
            self.ori_ind = ori_ind
            # Add a sensor
            self._sensor_range = sensor_range
            self._sensor_span = sensor_span

            # model_path = os.path.dirname(gym.envs.mujoco.__file__) + "/assets/" + xml_name
            # model_path = osp.join(MODEL_DIR, path)
            tree = ET.parse(model_path)
            worldbody = tree.find(".//worldbody")

            height = wall_height
            self.w_height= wall_height
            self.wall_pos_range = wall_pos_range
            rand_x = random.uniform(wall_pos_range[0][0], wall_pos_range[1][0]) #self.np_random.uniform(low=wall_pos_range[0][0], high=wall_pos_range[1][0], size=1)[0]
            rand_y = random.uniform(wall_pos_range[0][1], wall_pos_range[1][1]) #self.np_random.uniform(low=wall_pos_range[0][1], high=wall_pos_range[1][1], size=1)[0]
            self.wall_pos = wall_pos = (rand_x, rand_y)
            torso_x, torso_y = 0, 0
            self._init_torso_x = torso_x
            self.class_type = class_type
            self._init_torso_y = torso_y
            self.wall_size = (0.25, 0.4, height)


            self.num_walls = 2
            self.space_between = 5

            for i in range(self.num_walls):
                ET.SubElement(
                    worldbody, "geom",
                    name="wall %i" % i,
                    pos="%f %f %f" % (wall_pos[0]+i*self.space_between,
                                      wall_pos[1],
                                      height / 2.),
                    size="%f %f %f" % self.wall_size,
                    type="box",
                    material="",
                    density="5.",
                    rgba="1.0 0. 1. 1",
                    contype="1",
                    conaffinity="1",
                    condim="1",
                )



            _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
            tree.write(file_path)

            # self._goal_range = self._find_goal_range()
            self._cached_segments = None

            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

            # import pdb; pdb.set_trace()

        def get_body_xquat(self, body_name):
            
            idx = self.model.body_names.index(six.b(body_name))
            return self.model.data.xquat[idx]


        #
        IS_DISCRETE = True

        def _reset(self):

            temp = np.copy(self.model.geom_pos)

            rand_x = random.uniform(self.wall_pos_range[0][0], self.wall_pos_range[1][0])
            rand_y = random.uniform(self.wall_pos_range[0][1], self.wall_pos_range[1][1])

            # TODO: make this more robust,
            # hardcoding that the second geom is the wall,
            # but we should do something more robust??
            # assert isclose(temp[1][0], self.wall_pos[0])
            # assert isclose(temp[1][1], self.wall_pos[1])

            self.wall_pos = wall_pos = (rand_x, rand_y)

            temp[1][0] = self.wall_pos[0]
            temp[1][1] = self.wall_pos[1]
            #self.model.geom_pos = temp
            #self.model._compute_subtree()
            #self.model.forward()
            #------------------------------
            for i in range(0, self.model.geom_pos.shape[0]):
                for j in range(0, self.model.geom_pos.shape[1]):
                    self.model.geom_pos[i][j] = temp[i][j]
            #------------------------------
            #--- TODO: this randomization part must be separated from the this code
            if self.IS_DISCRETE:
                APPEARANCE_PROB_OF_FROZEN = 0.1
                prob = np.random.rand()
                if prob < APPEARANCE_PROB_OF_FROZEN:
                    ground_fric = 0.1
                else:
                    ground_fric = 2.0
                #for i in range(self.num_walls+1):
                for i in range(self.model.geom_friction.shape[0]):
                    self.model.geom_friction[i][0] = ground_fric
                #---
            else:
                ground_fric = max(0.1, min(2.0, np.random.normal(1.05, 0.475)))
                for i in range(self.model.geom_friction.shape[0]):
                    self.model.geom_friction[i][0] = ground_fric
            #print(ground_fric)

            ob = super(IceWallEnv, self).reset()
            return ob

        def _get_obs(self):
            # The observation would include both information about the robot itself as well as the sensors around its
            # environment


            terrain_read = np.zeros((6,))
            index_ratio = 2/1 # number of indices per meter. a ratio of 2 means each index is 0.5 long in mujoco coordinates

            robot_x, robot_y, robot_z = robot_coords = self.get_body_com("foot")
            # import pdb;pdb.set_trace()

            wall_length = self.wall_size[0] * 2

            for i in range(self.num_walls):
                wall_startx = self.wall_pos[0]+i*self.space_between - self.wall_size[0]
                wall_endx = self.wall_pos[0] + self.wall_size[0]
                diff = wall_startx - (robot_x - 1/index_ratio)

                if diff > 0.:
                    start_index = int(round(diff * index_ratio))
                    end_index = start_index + int(wall_length * index_ratio)

                elif diff < 0. and diff >= -wall_length:
                    start_index = 0

                    end_diff = wall_endx - (robot_x-1/index_ratio)
                    end_index = int(round(end_diff * index_ratio))
                    # end_index = max(int(round(end_diff * index_ratio)),1)
                elif diff < -wall_length:
                    start_index=end_index =-1

                terrain_read[start_index:end_index] = 1.




            obs = np.concatenate([
                self.class_type._get_obs(self),
                terrain_read
            ])

            return obs

        def _is_in_collision(self, pos):
            x, y = pos

            minx = self.wall_pos[0] * 1 - 1 * 0.5 - self._init_torso_x
            maxx = self.wall_pos[0] * 1 + 1 * 0.5 - self._init_torso_x
            miny = self.wall_pos[1] * 1 - 1 * 0.5 - self._init_torso_y
            maxy = self.wall_pos[1] * 1 + 1 * 0.5 - self._init_torso_y
            if minx <= x <= maxx and miny <= y <= maxy:
                return True
            return False


        def get_xy(self):
            return self.get_body_com("torso")[:2]

        def _step(self, action):
            # import pdb; pdb.set_trace()
            if self.MANUAL_COLLISION:
                old_pos = self.get_xy()
                state, reward, done, info = super(IceWallEnv, self).step(action)
                new_pos = self.get_xy()
                if self._is_in_collision(new_pos):
                    # print("Collision " + new_pos)
                    reward = -10.0
            else:
                state, reward, done, info = super(IceWallEnv, self).step(action)


            next_obs = self._get_obs()

            x, y = self.get_body_com("torso")[:2]
            return next_obs, reward, done, info

        def action_from_key(self, key):
            return self.action_from_key(key)
    return IceWallEnv












def MazeFactory(class_type):
    """class_type should be an OpenAI gym time"""

    class MazeEnv(class_type, utils.EzPickle):
        # Using https://github.com/bstadie/third_person_im/blob/88516c1703221586099062053af696f0b4a31cda/rllab/envs/mujoco/maze/maze_env.py
        # as a base
        # ORI_IND = None

        # TODO: remove this
        MAZE_MAKE_CONTACTS = False

        # manually give a penalty if the torso comes into contact with the wall
        MANUAL_COLLISION = False
        # TODO: this suckssssss, maye shouldn't use mujoco at all.

        def __init__(
                self,
                model_path,
                ori_ind,
                wall_height = 5,
                wall_pos_range = ([2.8, 0.0], [2.8, 0.0]),
                n_bins=10,
                sensor_range=10.,
                sensor_span=math.pi/2,
                *args,
                **kwargs):

            self._n_bins = n_bins
            self.ori_ind = ori_ind
            # Add a sensor
            self._sensor_range = sensor_range
            self._sensor_span = sensor_span

            # model_path = os.path.dirname(gym.envs.mujoco.__file__) + "/assets/" + xml_name
            # model_path = osp.join(MODEL_DIR, path)
            tree = ET.parse(model_path)
            worldbody = tree.find(".//worldbody")

            height = wall_height
            self.w_height= wall_height
            self.wall_pos_range = wall_pos_range
            rand_x = random.uniform(wall_pos_range[0][0], wall_pos_range[1][0]) #self.np_random.uniform(low=wall_pos_range[0][0], high=wall_pos_range[1][0], size=1)[0]
            rand_y = random.uniform(wall_pos_range[0][1], wall_pos_range[1][1]) #self.np_random.uniform(low=wall_pos_range[0][1], high=wall_pos_range[1][1], size=1)[0]
            self.wall_pos = wall_pos = (rand_x, rand_y)
            torso_x, torso_y = 0, 0
            self._init_torso_x = torso_x
            self.class_type = class_type
            self._init_torso_y = torso_y


            self.wall_size = (0.25, 2., height)
            self.side_wall_size = (20., .25, .75)


            self.num_walls = 5
            self.space_between = 5
            self.init_y = 1.5

            for i in range(self.num_walls):
                ET.SubElement(
                    worldbody, "geom",
                    name="wall %i" % i,
                    pos="%f %f %f" % (wall_pos[0]+i*self.space_between,
                                      self.init_y * (-1)**i,
                                      height / 2.),
                    size="%f %f %f" % self.wall_size,
                    type="box",
                    material="",
                    density="5.",
                    rgba="1.0 0. 1. 1",
                    contype="1",
                    conaffinity="1",
                    condim="1",
                )

            for i in range(2):
                ET.SubElement(
                    worldbody, "geom",
                    name="sidewall %i" % i,
                    pos="%f %f %f" % (self.side_wall_size[0]/2,
                                      (self.init_y+self.wall_size[1]) * (-1)**i,
                                      self.side_wall_size[2]/2),
                    size="%f %f %f" % self.side_wall_size,
                    type="box",
                    material="",
                    density="5.",
                    rgba=".0 .0 .0 .2",
                    contype="1",
                    conaffinity="1",
                    condim="1",
                )        


            _, file_path = tempfile.mkstemp(text=True)
            tree.write(file_path)

            # self._goal_range = self._find_goal_range()
            self._cached_segments = None

            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

            # import pdb; pdb.set_trace()

        def get_body_xquat(self, body_name):
            
            idx = self.model.body_names.index(six.b(body_name))
            return self.model.data.xquat[idx]



        def _get_obs(self):
            # The observation would include both information about the robot itself as well as the sensors around its
            # environment
            max_x=2
            max_y=5
            terrain_read = np.zeros((max_y,max_x))

            # terrain_read = np.full((6,),0.1)

            index_ratio = 1/1 # number of indices per meter. a ratio of 2 means each index is 0.5 long in mujoco coordinates

            robot_x, robot_y, robot_z = robot_coords = self.get_body_com("torso")
            

            wall_length = self.wall_size[1] * 2

            for i in range(self.num_walls):

                diff_x = self.wall_pos[0]+i*self.space_between - (robot_x + 1/index_ratio)
                index_x =  int(round(diff_x * index_ratio))

                if index_x < 2 and index_x >=0 and i%2==0: 

                    wall_starty =   self.init_y * (-1)**i - self.wall_size[1]
                    diff = (robot_y +  2/index_ratio) - wall_starty

                    if diff >= 0.:
                        end_index = int(round(diff * index_ratio))
                        terrain_read[:end_index,index_x] = 1.

                elif index_x < 2 and index_x >=0 and i%2==1: 

                    wall_endy =   self.init_y * (-1)**i + self.wall_size[1]
                    diff = wall_endy - (robot_y -  2/index_ratio) 

                    if diff >= 0.:
                        end_index = int(round(diff * index_ratio))
                        terrain_read[-end_index:,index_x] = 1.                    



            for i in range(2):


                if i==0:
                    wall_starty =   (self.init_y+self.wall_size[1]) * (-1)**i - self.side_wall_size[1]
                    diff = (robot_y +  2/index_ratio) - wall_starty

                    if diff >= 0.:
                        end_index = int(round(diff * index_ratio))
                        terrain_read[:end_index,:] = 1.

                if i==1:
                    wall_endy =   (self.init_y+self.wall_size[1]) * (-1)**i + self.side_wall_size[1]
                    diff = wall_endy - (robot_y -  2/index_ratio) 

                    if diff >= 0.:                    
                        end_index = int(round(diff * index_ratio)) 
                        if end_index == 0:
                            end_index = 1 # due to the way negative slicing works
                        terrain_read[-end_index:,:] = 1.   



            obs = np.concatenate([
                self.class_type._get_obs(self),
                terrain_read.flatten()
            ])

            return obs

        def _is_in_collision(self, pos):
            x, y = pos

            minx = self.wall_pos[0] * 1 - 1 * 0.5 - self._init_torso_x
            maxx = self.wall_pos[0] * 1 + 1 * 0.5 - self._init_torso_x
            miny = self.wall_pos[1] * 1 - 1 * 0.5 - self._init_torso_y
            maxy = self.wall_pos[1] * 1 + 1 * 0.5 - self._init_torso_y
            if minx <= x <= maxx and miny <= y <= maxy:
                return True
            return False


        def get_xy(self):
            return self.get_body_com("torso")[:2]

        def _step(self, action):
            # import pdb; pdb.set_trace()
            if self.MANUAL_COLLISION:
                old_pos = self.get_xy()
                state, reward, done, info = super(MazeEnv, self)._step(action)
                new_pos = self.get_xy()
                if self._is_in_collision(new_pos):
                    # print("Collision " + new_pos)
                    reward = -10.0
            else:
                state, reward, done, info = super(MazeEnv, self)._step(action)


            next_obs = self._get_obs()

            x, y = self.get_body_com("torso")[:2]
            return next_obs, reward, done, info

        def action_from_key(self, key):
            return self.action_from_key(key)
    return MazeEnv


















def StairsFactory(class_type):
    """class_type should be an OpenAI gym time"""

    class StairsEnv(class_type, utils.EzPickle):
        # Using https://github.com/bstadie/third_person_im/blob/88516c1703221586099062053af696f0b4a31cda/rllab/envs/mujoco/maze/maze_env.py
        # as a base
        # ORI_IND = None

        # TODO: remove this
        MAZE_MAKE_CONTACTS = False

        # manually give a penalty if the torso comes into contact with the wall
        MANUAL_COLLISION = False
        # TODO: this suckssssss, maye shouldn't use mujoco at all.

        def __init__(
                self,
                model_path,
                ori_ind,
                wall_height = .15,
                wall_pos_range = ([3., 0.0], [3., 0.0]),
                n_bins=10,
                sensor_range=10.,
                sensor_span=math.pi/2,
                *args,
                **kwargs):

            self._n_bins = n_bins
            self.ori_ind = ori_ind
            # Add a sensor
            self._sensor_range = sensor_range
            self._sensor_span = sensor_span

            # model_path = os.path.dirname(gym.envs.mujoco.__file__) + "/assets/" + xml_name
            # model_path = osp.join(MODEL_DIR, path)
            tree = ET.parse(model_path)
            worldbody = tree.find(".//worldbody")

            height = wall_height
            self.w_height= wall_height
            self.wall_pos_range = wall_pos_range
            rand_x = random.uniform(wall_pos_range[0][0], wall_pos_range[1][0]) #self.np_random.uniform(low=wall_pos_range[0][0], high=wall_pos_range[1][0], size=1)[0]
            rand_y = random.uniform(wall_pos_range[0][1], wall_pos_range[1][1]) #self.np_random.uniform(low=wall_pos_range[0][1], high=wall_pos_range[1][1], size=1)[0]
            self.wall_pos = wall_pos = (rand_x, rand_y)
            torso_x, torso_y = 0, 0
            self._init_torso_x = torso_x
            self.class_type = class_type
            self._init_torso_y = torso_y


            self.wall_size = [(2.25, 0.4, height),
                                (1.5, 0.4, height),
                                (0.75, 0.4, height)]


            self.num_stairs = 3

            for i in range(self.num_stairs):
                ET.SubElement(
                    worldbody, "geom",
                    name="level%i" % i,
                    pos="%f %f %f" % (wall_pos[0],
                                      wall_pos[1],
                                      height / 2. + height*i),
                    size="%f %f %f" % self.wall_size[i],
                    type="box",
                    material="",
                    density="1.",
                    rgba="1.0 0. 1. 1",
                    contype="1",
                    conaffinity="1",
                    condim="3",
                )




            _, file_path = tempfile.mkstemp(text=True)
            tree.write(file_path)

            # self._goal_range = self._find_goal_range()
            self._cached_segments = None
            # import pdb;pdb.set_trace()
            class_type.__init__(self, model_path=file_path)
            utils.EzPickle.__init__(self)

            # import pdb; pdb.set_trace()

        def get_body_xquat(self, body_name):
            
            idx = self.model.body_names.index(six.b(body_name))
            return self.model.data.xquat[idx]

        def _reset(self):

            temp = np.copy(self.model.geom_pos)

            rand_x = random.uniform(self.wall_pos_range[0][0], self.wall_pos_range[1][0])
            rand_y = random.uniform(self.wall_pos_range[0][1], self.wall_pos_range[1][1])

            # TODO: make this more robust,
            # hardcoding that the second geom is the wall,
            # but we should do something more robust??
            # assert isclose(temp[1][0], self.wall_pos[0])
            # assert isclose(temp[1][1], self.wall_pos[1])

            self.wall_pos = wall_pos = (rand_x, rand_y)

            temp[1][0] = self.wall_pos[0]
            temp[1][1] = self.wall_pos[1]
            self.model.geom_pos = temp
            self.model._compute_subtree()
            self.model.forward()
            ob = super(StairsEnv, self)._reset()
            return ob

        def _get_obs(self):
            # The observation would include both information about the robot itself as well as the sensors around its
            # environment
            terrain_read = np.zeros((6,))

            robot_x, robot_y, robot_z = robot_coords = self.get_body_com("foot")

            index_ratio = 2/1 # number of indices per meter. a ratio of 2 means each index is 0.5 long in mujoco coordinates

            for i in range(self.num_stairs):

                wall_startx = self.wall_pos[0] - self.wall_size[i][0]
                wall_endx = self.wall_pos[0] + self.wall_size[i][0]

                wall_length = self.wall_size[i][0] * 2


                diff = wall_startx - (robot_x - 1/index_ratio)

                if diff > 0.:
                    start_index = int(round(diff * index_ratio))
                    end_index = start_index + int(wall_length * index_ratio)

                elif diff < 0. and diff >= -wall_length:
                    start_index = 0

                    end_diff = wall_endx - (robot_x-1/index_ratio)
                    end_index = int(round(end_diff * index_ratio))

                elif diff < -wall_length:
                    start_index=end_index =-1



                terrain_read[start_index:end_index] += 0.5             



            obs = np.concatenate([
                self.class_type._get_obs(self),
                terrain_read

            ])


            return obs

        def _is_in_collision(self, pos):
            x, y = pos

            minx = self.wall_pos[0] * 1 - 1 * 0.5 - self._init_torso_x
            maxx = self.wall_pos[0] * 1 + 1 * 0.5 - self._init_torso_x
            miny = self.wall_pos[1] * 1 - 1 * 0.5 - self._init_torso_y
            maxy = self.wall_pos[1] * 1 + 1 * 0.5 - self._init_torso_y
            if minx <= x <= maxx and miny <= y <= maxy:
                return True
            return False


        def get_xy(self):
            return self.get_body_com("torso")[:2]

        def _step(self, a):
            posbefore = self.model.data.qpos[0, 0]
            self.do_simulation(a, self.frame_skip)
            posafter, height, ang = self.model.data.qpos[0:3, 0]
            alive_bonus = 1.0
            reward = (posafter - posbefore) / self.dt
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()
            s = self.state_vector()
            done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                        (height > .7) and (abs(ang) < .27))
                        # and (abs(ang) < .2))

            ob = self._get_obs()

            # done=False
            return ob, reward, done, {}


        def action_from_key(self, key):
            return self.action_from_key(key)
    return StairsEnv














