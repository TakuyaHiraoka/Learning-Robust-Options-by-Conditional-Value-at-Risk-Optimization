from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import pdb
import os
import shutil
from scipy import spatial
import gym

import copy
import sys

#
from gym.spaces.box import Box
#
import pickle
import statistics
import math

import time

import random

#Envirtonment
WARMING_UP_EPOCHS = 300
#WARMING_UP_EPOCHS = 0
split_size = 2.0
# Hopper
P_IN_HOPPER = []
for torso_mass in np.arange(3.0, 9.1, (9.0 - 3.0) / split_size):
    for ground_friction in np.arange(1.5, 2.6, (2.5 - 1.5) / split_size):
        for joint_damping in np.arange(0.0, 2.1, (2.0 - 0.0) / split_size):
            for armature in np.arange(0.5, 1.6, (1.5 - 0.5) / split_size):
                P_IN_HOPPER.append({"torso_mass": torso_mass, "ground_friction": ground_friction,
                                    "joint_damping": joint_damping, "armature": armature})
# HaflCheetah-cont
P_IN_HALF_CHEETAH = []
for torso_mass in np.arange(1.0, 13.1, (13.0 - 1.0) / split_size):
    for ground_friction in np.arange(0.1, 3.2, (3.1 - 0.5) / split_size):
        for joint_damping in np.arange(1.0, 11.1, (11.0 - 1.0) / split_size):
            P_IN_HALF_CHEETAH.append({"torso_mass": torso_mass, "ground_friction": ground_friction,
                                      "joint_damping": joint_damping})
# Walker2D-cont
P_IN_WALKER2D = []
for torso_mass in np.arange(3.0, 9.1, (9.0 - 3.0) / split_size):
    for ground_friction in np.arange(0.9, 3.0, (2.9 - 0.9) / split_size):
        P_IN_WALKER2D.append({"torso_mass": torso_mass, "ground_friction": ground_friction})
# Humanoid
P_IN_HUMANOID = [] # appended @ 20190225
for torso_mass in np.arange(5.0, 11.1, (11.0 - 5.0) / split_size):
    for ground_friction in np.arange(0.2, 1.9, (1.8 - 0.2) / split_size):
        P_IN_HUMANOID.append({"torso_mass": torso_mass, "ground_friction": ground_friction})
# HopperIceBlock-disc
P_IN_HOPPER_WALL = []
for is_fronzed in range(0,2):
    P_IN_HOPPER_WALL.append({"frozen": is_fronzed})
# HopperIceBlock-cont
P_IN_HOPPER_WALL_CONTINUOUS = []
for ground_friction in np.arange(0.1,2.1, (2.0 - 0.1) / split_size):
    P_IN_HOPPER_WALL_CONTINUOUS.append({"ground_friction": ground_friction})
# Walker2D-disc
P_IN_WALKER2D_DISCRETE = []
for torso_mass in range(0,2):
    for ground_friction in range(0,2):
        P_IN_WALKER2D_DISCRETE.append({"torso_mass": torso_mass, "ground_friction": ground_friction})
# Halfcheetah-disc
P_IN_HALF_CHEETAH_DISCRETE = []
for torso_mass in range(0,2):
    for ground_friction in range(0,2):
        for joint_damping in range(0,2):
            P_IN_HALF_CHEETAH_DISCRETE.append({"torso_mass": torso_mass,
                                      "ground_friction": ground_friction,
                                      "joint_damping": joint_damping})



def traj_segment_generator(pi, env, horizon, stochastic, num_options, saves, results, rewbuffer, dc, method, additional_info):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    if method == "CVaR": # Extend observation
        v = additional_info["v"]
        ob = np.append(ob, v)
        prev_v = v  # for  reward shaping 20190215


    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    option = pi.get_option(ob)

    optpol_p = []
    term_p = []
    value_val = []
    opt_duration = [[] for _ in range(num_options)]
    logstds = [[] for _ in range(num_options)]
    curr_opt_duration = 0.

    if method == "WORST_CASE":
        sim_data = env.env.env.sim.data
        sim_states = [{"pos": sim_data.qpos.copy(), "vel": sim_data.qvel.copy()} for _ in range(horizon)]# for WORST_CASE method
        # to save mamory. only position and velocity are memories are saved.
    else:
        sim_states = None

    if method == "CVaR": 
        exceeding_risk = []
    else:
        exceeding_risk = None
    ep_rets_origin_CVaR = []
    cur_ep_ret_origin_CVaR = 0.0


    while True:
        prevac = ac
        ac, vpred, feats, logstd = pi.act(stochastic, ob, option)
        logstds[option].append(logstd)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "realrew": realrews, "vpred": vpreds, "new": news,
                   "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': term_p, 'value_val': value_val,
                   "opt_dur": opt_duration, "optpol_p": optpol_p, "logstds": logstds,
                   "sim_state": sim_states, # 4 worst case
                   "e_risk": exceeding_risk, "ep_rets_origin_CVaR": ep_rets_origin_CVaR} # 4 CVaR
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            term_p = []
            value_val = []
            opt_duration = [[] for _ in range(num_options)]
            logstds = [[] for _ in range(num_options)]
            curr_opt_duration = 0.

            if method == "CVaR": 
                exceeding_risk = []
            else:
                exceeding_risk = None
            ep_rets_origin_CVaR = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac

        #
        if method == "WORST_CASE":
            #sim_states[i] = copy.deepcopy(env.env.env.sim.data) # for WORST_CASE method
            sim_data = env.env.env.sim.data
            sim_states[i] = {"pos": sim_data.qpos.copy(), "vel": sim_data.qvel.copy()}


        ob, rew, new, _ = env.step(ac)
        #env.render(mode="human")

        cur_ep_ret_origin_CVaR += rew
        if method == "CVaR":  #  Extends observation and reward
            v = (v + rew) / additional_info["gamma"]
            # TODO: since the following part is adhoc, it will be removed when it does not affect the learning result. 
            # clipping value (added 20190205)
            clipped_v = min(math.fabs(additional_info["bounds_risk"]), max(-math.fabs(additional_info["bounds_risk"]), v))
            clipped_v = clipped_v / float(math.fabs(additional_info["bounds_risk"]))
            #ob = np.append(ob, v)
            ob = np.append(ob, clipped_v)

            #
            penalty = 0.0
            if new:
                #rew = rew + (additional_info["lambda"] * max(0.0, -v*(additional_info["gamma"]**cur_ep_len)) / additional_info["epsilon"])
                penalty = additional_info["lambda"] * max(0.0, -v) / additional_info["epsilon"] # fixed reward representation 20190205
                penalty = max(additional_info["bounds_risk"], penalty)
                assert penalty <= 0.0, "penalty term should be always equal or smaller than zero."

            prev_v = v # reward shaping

            #rew = rew + penalty + shaping_reward
            rew = rew + penalty

        rew = rew / 10 if num_options > 1 else rew  # To stabilize learning.
        rews[i] = rew
        realrews[i] = rew

        curr_opt_duration += 1

        ### Book-keeping
        t_p = []
        v_val = []
        for oopt in range(num_options):
            v_val.append(pi.get_vpred([ob], [oopt])[0][0])
            t_p.append(pi.get_tpred([ob], [oopt])[0][0])
        term_p.append(t_p)
        optpol_p.append(pi._get_op([ob])[0][0])
        value_val.append(v_val)
        term = pi.get_term([ob], [option])[0][0]
        ###

        if term:
            if num_options > 1:
                rews[i] -= dc
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            option = pi.get_option(ob)

        cur_ep_ret += rew * 10 if num_options > 1 else rew
        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            if method == "CVaR":
                e_risk = max(0.0, -v)
                assert e_risk >= 0.0, "Illegal val " + str(e_risk)
                exceeding_risk.append(e_risk)
                # creat new obs
                v = additional_info["v"]
                ob = np.append(ob, v)
                prev_v = v  # for reward shaping 20190215

            ep_rets_origin_CVaR.append(cur_ep_ret_origin_CVaR)
            cur_ep_ret_origin_CVaR = 0.0

            option = pi.get_option(ob)
        t += 1



def traj_segment_generator_for_test(pi, env, horizon, stochastic, num_options, saves, results, rewbuffer, dc, param, method, additional_info):
    env_id = env.spec.id

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    if method == "CVaR": # Extend observation
        v = additional_info["v"]
        ob = np.append(ob, v)

    # set model param.
    if env_id == "Hopper-Random-Params-v1":
        env.env.model.body_mass[1] = param["torso_mass"]
        env.env.model.geom_friction[4][0] = param["ground_friction"]
        env.env.model.dof_damping[5] = env.env.model.dof_damping[5] * param["joint_damping"]
        for i in range(3, 6):
            env.env.model.dof_armature[i] = env.env.model.dof_armature[i] * param["armature"]
    elif env_id == "HalfCheetah-Random-Params-v1":
        env.env.model.body_mass[1] = param["torso_mass"]
        for i in range(0, 9):
            env.env.model.geom_friction[i][0] = param["ground_friction"]
        #
        for i in range(3,9):
            env.env.model.dof_damping[i] = param["joint_damping"]
        #for i in range(3,9):
        #    env.env.model.dof_armature[i] = param["armature"]
    elif env_id == 'Walker2d-Random-Params-v1':
        env.env.model.body_mass[1] = param["torso_mass"]
        env.env.model.geom_friction[7][0] = param["ground_friction"]
    elif env_id == 'Humanoid-Random-Params-v1':
        env.env.model.body_mass[1] = param["torso_mass"]
        for i in range(env.env.model.geom_friction.shape[0]):
            env.env.model.geom_friction[i][0] = param["ground_friction"]
    elif env_id == 'HopperIceWall-v0':
        if param["frozen"] == 1:
            ground_fric = 0.1
        else:
            ground_fric = 2.0
        for i in range(env.env.model.geom_friction.shape[0]):
            env.env.model.geom_friction[i][0] = ground_fric

    elif env_id == "Walker2d-Random-Params-discrete-v1":
        if param["torso_mass"] == 1:
            env.env.model.body_mass[1] = 9.0
        else:
            env.env.model.body_mass[1] = 3.0
        if param["ground_friction"] == 1:
            env.env.model.geom_friction[7][0] = 2.9
        else:
            env.env.model.geom_friction[7][0] = 0.9
    elif env_id == "HopperIceWall-continuous-v0":
        for i in range(env.env.model.geom_friction.shape[0]):
            env.env.model.geom_friction[i][0] = param["ground_friction"]

    elif env_id == "HalfCheetah-Random-Params-discrete-v1":
        # torso mass
        if param["torso_mass"] == 0:
            torso_mass = 13.0
        else:
            torso_mass = 1.0
        env.env.model.body_mass[1] = torso_mass
        # ground friction
        if param["ground_friction"] == 0:
            ground_friction = 3.1
        else:
            ground_friction = 0.1
        for i in range(0, 9):
            env.env.model.geom_friction[i][0] = ground_friction
        # joint damping
        if param["joint_damping"] == 0:
            dof_damping = 11.0
        else:
            dof_damping = 1.0
        for i in range(3,9):
            env.env.model.dof_damping[i] = dof_damping
    else:
        assert False, "Unsupported environment"


    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    option = pi.get_option(ob)

    optpol_p = []
    term_p = []
    value_val = []
    opt_duration = [[] for _ in range(num_options)]
    logstds = [[] for _ in range(num_options)]
    curr_opt_duration = 0.

    while True:
        prevac = ac
        ac, vpred, feats, logstd = pi.act(stochastic, ob, option)
        logstds[option].append(logstd)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "realrew": realrews, "vpred": vpreds, "new": news,
                   "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': term_p, 'value_val': value_val,
                   "opt_dur": opt_duration, "optpol_p": optpol_p, "logstds": logstds}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            term_p = []
            value_val = []
            opt_duration = [[] for _ in range(num_options)]
            logstds = [[] for _ in range(num_options)]
            curr_opt_duration = 0.


        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        #env.render(mode="human")

        if method == "CVaR":
            v = (v + rew) / additional_info["gamma"]
            clipped_v = min(math.fabs(additional_info["bounds_risk"]), max(-math.fabs(additional_info["bounds_risk"]), v))
            clipped_v = clipped_v / float(math.fabs(additional_info["bounds_risk"]))
            #ob = np.append(ob, v)
            ob = np.append(ob, clipped_v)

        rew = rew / 10 if num_options > 1 else rew  # To stabilize learning.
        rews[i] = rew
        realrews[i] = rew

        curr_opt_duration += 1

        ### Book-keeping
        t_p = []
        v_val = []
        for oopt in range(num_options):
            v_val.append(pi.get_vpred([ob], [oopt])[0][0])
            t_p.append(pi.get_tpred([ob], [oopt])[0][0])
        term_p.append(t_p)
        optpol_p.append(pi._get_op([ob])[0][0])
        value_val.append(v_val)
        term = pi.get_term([ob], [option])[0][0]
        ###

        if term:
            if num_options > 1:
                rews[i] -= dc
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            option = pi.get_option(ob)

        cur_ep_ret += rew * 10 if num_options > 1 else rew
        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            if method == "CVaR":
                # creat new obs
                v = additional_info["v"]
                ob = np.append(ob, v)

            # set model param.
            if env_id == "Hopper-Random-Params-v1":
                env.env.model.body_mass[1] = param["torso_mass"]
                env.env.model.geom_friction[4][0] = param["ground_friction"]
                env.env.model.dof_damping[5] = env.env.model.dof_damping[5] * param["joint_damping"]
                for i in range(3, 6):
                    env.env.model.dof_armature[i] = env.env.model.dof_armature[i] * param["armature"]
            elif env_id == "HalfCheetah-Random-Params-v1":
                env.env.model.body_mass[1] = param["torso_mass"]
                for i in range(0, 9):
                    env.env.model.geom_friction[i][0] = param["ground_friction"]
                #
                for i in range(3, 9):
                    env.env.model.dof_damping[i] = param["joint_damping"]
                #for i in range(3, 9):
                #    env.env.model.dof_armature[i] = param["armature"]

            elif env_id == 'Walker2d-Random-Params-v1':
                env.env.model.body_mass[1] = param["torso_mass"]
                env.env.model.geom_friction[7][0] = param["ground_friction"]
            elif env_id == 'Humanoid-Random-Params-v1':
                env.env.model.body_mass[1] = param["torso_mass"]
                for i in range(env.env.model.geom_friction.shape[0]):
                    env.env.model.geom_friction[i][0] = param["ground_friction"]
            elif env_id == 'HopperIceWall-v0':
                if param["frozen"] == 1:
                    ground_fric = 0.1
                else:
                    ground_fric = 2.0
                #ground_fric = 0.1
                #ground_fric = 2.0

                for i in range(env.env.model.geom_friction.shape[0]):
                    env.env.model.geom_friction[i][0] = ground_fric
            elif env_id == "Walker2d-Random-Params-discrete-v1":
                if param["torso_mass"] == 1:
                    env.env.model.body_mass[1] = 9.0
                else:
                    env.env.model.body_mass[1] = 3.0
                if param["ground_friction"] == 1:
                    env.env.model.geom_friction[7][0] = 2.9
                else:
                    env.env.model.geom_friction[7][0] = 0.9
            elif env_id == "HopperIceWall-continuous-v0":
                for i in range(env.env.model.geom_friction.shape[0]):
                    env.env.model.geom_friction[i][0] = param["ground_friction"]
            elif env_id == "HalfCheetah-Random-Params-discrete-v1":
                # torso mass
                if param["torso_mass"] == 0:
                    torso_mass = 13.0
                else:
                    torso_mass = 1.0
                env.env.model.body_mass[1] = torso_mass
                # ground friction
                if param["ground_friction"] == 0:
                    ground_friction = 3.1
                else:
                    ground_friction = 0.1
                for i in range(0, 9):
                    env.env.model.geom_friction[i][0] = ground_friction
                # joint damping
                if param["joint_damping"] == 0:
                    dof_damping = 11.0
                else:
                    dof_damping = 1.0
                for i in range(3, 9):
                    env.env.model.dof_damping[i] = dof_damping
            else:
                assert False, "Unsupported environment"

            option = pi.get_option(ob)
        t += 1



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def add_vtarg_and_adv_in_worst_case(seg, gamma, lam, dummy_env, pi): # @ 20190120
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0

    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        # original delta
        #delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]

        # the worst case version
        # - for all possible transition pattern
        # -- sample next state and pick up a minimum case
        min_vpred = sys.float_info.max

        if dummy_env.spec.id == "Hopper-Random-Params-v1":
            for p in P_IN_HOPPER:
                dummy_env.reset()
                # set model param.
                dummy_env.env.model.body_mass[1] = p["torso_mass"]
                # - ground friction
                dummy_env.env.model.geom_friction[4][0] = p["ground_friction"]

                dummy_env.env.model.dof_damping[5] = dummy_env.env.model.dof_damping[5] * p["joint_damping"]
                for i in range(3, 6):
                    dummy_env.env.model.dof_armature[i] = dummy_env.env.model.dof_armature[i] * p["armature"]

                # set state
                for i in range(dummy_env.env.data.qpos.size):
                   dummy_env.env.data.qpos[i] = seg["sim_state"][t]["pos"][i]
                for i in range(dummy_env.env.data.qvel.size):
                    dummy_env.env.data.qvel[i] = seg["sim_state"][t]["vel"][i]

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id == "HalfCheetah-Random-Params-v1":
            for p in P_IN_HALF_CHEETAH:
                dummy_env.reset()
                # set model param.
                dummy_env.env.model.body_mass[1] = p["torso_mass"]
                # - ground friction
                for i in range(0, 9):
                    dummy_env.env.model.geom_friction[i][0] = p["ground_friction"]

                # add joint damping and armature 2019/03/15
                for i in range(3, 9):
                    dummy_env.env.model.dof_damping[i] = p["joint_damping"]

                # set state
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id  == 'Walker2d-Random-Params-v1':
            for p in P_IN_WALKER2D:
                dummy_env.reset()
                # set model param.
                dummy_env.env.model.body_mass[1] = p["torso_mass"]
                # - ground friction
                dummy_env.env.model.geom_friction[7][0] = p["ground_friction"]

                # set state
                # new version @ 20190219
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id  == 'Humanoid-Random-Params-v1':
            for p in P_IN_HUMANOID:
                dummy_env.reset()
                # set model param.
                dummy_env.env.model.body_mass[1] = p["torso_mass"]
                # - ground friction
                for i in range(dummy_env.env.model.geom_friction.shape[0]):
                    dummy_env.env.model.geom_friction[i][0] = p["ground_friction"]

                # set state (old style)
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id  == 'HopperIceWall-v0':
            for p in P_IN_HOPPER_WALL:
                dummy_env.reset()
                if p["frozen"] == 1:
                    ground_fric = 0.1
                else:
                    ground_fric = 2.0
                for i in range(dummy_env.env.model.geom_friction.shape[0]):
                    dummy_env.env.model.geom_friction[i][0] = ground_fric

                # set state
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id == "Walker2d-Random-Params-discrete-v1":
            for p in P_IN_WALKER2D_DISCRETE:
                dummy_env.reset()
                # set model param.
                if p["torso_mass"] == 1:
                    dummy_env.env.model.body_mass[1] = 9.0
                else:
                    dummy_env.env.model.body_mass[1] = 3.0
                if p["ground_friction"] == 1:
                    dummy_env.env.model.geom_friction[7][0] = 2.9
                else:
                    dummy_env.env.model.geom_friction[7][0] = 0.9
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        elif dummy_env.spec.id == 'HopperIceWall-continuous-v0':
            for p in P_IN_HOPPER_WALL_CONTINUOUS:
                dummy_env.reset()
                for i in range(dummy_env.env.model.geom_friction.shape[0]):
                    dummy_env.env.model.geom_friction[i][0] = p["ground_friction"]

                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred

        elif dummy_env.spec.id == "HalfCheetah-Random-Params-discrete-v1":
            for p in P_IN_HALF_CHEETAH_DISCRETE:
                dummy_env.reset()
                # torso mass
                if p["torso_mass"] == 0:
                    torso_mass = 13.0
                else:
                    torso_mass = 1.0
                dummy_env.env.model.body_mass[1] = torso_mass
                # ground friction
                if p["ground_friction"] == 0:
                    ground_friction = 3.1
                else:
                    ground_friction = 0.1
                for i in range(0, 9):
                    dummy_env.env.model.geom_friction[i][0] = ground_friction
                # joint damping
                if p["joint_damping"] == 0:
                    dof_damping = 11.0
                else:
                    dof_damping = 1.0
                for i in range(3, 9):
                    dummy_env.env.model.dof_damping[i] = dof_damping

                # set state
                dummy_env.env.set_state(
                    seg["sim_state"][t]["pos"],
                    seg["sim_state"][t]["vel"]
                )

                dummy_ob, dummy_rew, dummy_new, _ = dummy_env.step(seg["ac"][t])
                # -eval v
                dummy_vpred = pi.get_vpred([dummy_ob], [seg["opts"][t]])[0][0]
                if min_vpred > dummy_vpred:
                    min_vpred = dummy_vpred
        else:
            assert False, "Unsuported environment"
        # -- compute delta with the selected value
        delta = rew[t] + gamma * min_vpred * nonterminal - vpred[t]

        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize, # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant', # annealing for stepsize parameters (epsilon and adam)
          num_options=1,
          app='',
          saves=False,
          wsaves=False,
          epoch=-1,
          seed=1,
          dc=0,
          method,
          mpath
          ):
    if env.spec.id == "HopperIceWall-continuous-v0":
        env.env.env.IS_DISCRETE = False

    optim_batchsize_ideal = optim_batchsize
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # env._seed(seed)

    ### Book-keeping
    gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)
    gamename += app

    dirname = '{}_{}opts_saves/'.format(gamename, num_options)

    if wsaves and (mpath is None):
        first = True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False

        files = ['pposgd_simple.py', 'mlp_policy.py', 'run_mujoco.py']
        for i in range(len(files)):
            src = os.path.expanduser('./') + files[i]
            dest = os.path.expanduser('./') + dirname

            shutil.copy2(src, dest)



    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    if method == "CVaR": # Extend observation space
        n_high = np.append(copy.deepcopy(ob_space.high), float('inf'))
        n_low = np.append(copy.deepcopy(ob_space.low), -float('inf'))
        n_ob_space = Box(low=n_low, high=n_high)
        pi = policy_func("pi", n_ob_space, ac_space)  # Construct network for new policy
        oldpi = policy_func("oldpi", n_ob_space, ac_space)  # Network for old policy
    else:
        pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
        oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    # option = tf.placeholder(dtype=tf.int32, shape=[None])

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    # pdb.set_trace()
    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    term_loss = pi.tpred * term_adv

    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    entropy = -tf.reduce_sum(pi.op_pi * log_pi, reduction_indices=1)
    op_loss = - tf.reduce_sum(log_pi[0][option[0]] * atarg + entropy * 0.1)

    total_loss += op_loss

    var_list = pi.get_trainable_variables()
    term_list = var_list[6:8]

    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option, term_adv],
                             losses + [U.flatgrad(total_loss, var_list)])
    termloss = U.function([ob, option, term_adv],
                          [U.flatgrad(term_loss, var_list)])  # Since we will use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)

    # Add learnable lambda and v (CVaR)
    cvars_over_training = []
    rets_over_training = []
    best_cvars = -9999.0
    best_cvars_pol_id = -1
    BETA = -1500.0

    if method == "CVaR":
        if mpath is not None:
            # Load pickles if any
            with open(mpath + ".pcl", "rb") as f_cvar:
                dic_avars4cvar=pickle.load(f_cvar)
            print("CVaR dictionary is loaded from " + mpath)

        else:
            dic_avars4cvar = {"lambda": (-1.0 * np.random.rand()), 
                              "v": 1000, 
                              "beta": BETA, "gamma": gamma, 
                              "epsilon": 0.1, 
                              #"bounds_risk": -500.0}  # FOR HALF CHEETAH Case and WALKER2D case and Humanoid case
                              "bounds_risk": -2000.0}  # added to stabilize learning 2019/ FOR WALKER2D-discrete case

    if mpath is not None:
        U.load_state(mpath)
        print("TEST MODE; The trained model was loaded from " + mpath)
    else:
        U.initialize()
        adam.sync()
        print("TRAINING MODE")

    # test
    if mpath is not None:
        env4test = gym.make(env.spec.id) # this is to evaluate the value at the worst case setting.
        avr_over_all_params = 0.0
        min_avr = sys.float_info.max
        param_id = 0

        if env.spec.id == "Hopper-Random-Params-v1":
            test_params = P_IN_HOPPER
        elif env.spec.id == "HalfCheetah-Random-Params-v1":
            test_params = P_IN_HALF_CHEETAH
        elif env.spec.id == "Walker2d-Random-Params-v1":
            test_params = P_IN_WALKER2D
        elif env.spec.id == "HopperIceWall-v0":
            test_params = P_IN_HOPPER_WALL
        elif env.spec.id == 'Humanoid-Random-Params-v1':
            test_params = P_IN_HUMANOID
        elif env.spec.id == "Walker2d-Random-Params-discrete-v1":
            test_params = P_IN_WALKER2D_DISCRETE
        elif env.spec.id == "HopperIceWall-continuous-v0":
            test_params = P_IN_HOPPER_WALL_CONTINUOUS
        elif env.spec.id == "HalfCheetah-Random-Params-discrete-v1":
            test_params = P_IN_HALF_CHEETAH_DISCRETE
        else:
            assert False, "Unsupported test case"


        for param in test_params:
            rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

            # orginal test
            if method == "CVaR":
                seg_gen = traj_segment_generator_for_test(pi, env4test, timesteps_per_batch, stochastic=True, 
                                                                 num_options=num_options,
                                                                 saves=None, results=None, rewbuffer=None, dc=dc,
                                                                 param=param, method=method, additional_info=dic_avars4cvar)
            else:
                seg_gen = traj_segment_generator_for_test(pi, env4test, timesteps_per_batch, stochastic=True,
                                                          num_options=num_options,
                                                          saves=None, results=None, rewbuffer=None, dc=dc,
                                                          param=param, method=method, additional_info=None)

            seg = seg_gen.__next__()
            avr = 0.0
            for rets in seg["ep_rets"]:
                avr += rets
            avr /= len(seg["ep_rets"])
            print("return: " + str(avr) + ", " + str(param))
            avr_over_all_params += avr
            if min_avr > avr:
                min_avr = avr
            if param_id == int(round(len(test_params)/2 - 1)):
                most_freq_avr = avr
            param_id += 1
        avr_over_all_params /= len(test_params)
        print("Test summary:")
        print("Average at Most frequent params: " + str(most_freq_avr))
        print("Average^2 return over all params: " + str(avr_over_all_params))
        print("Minimum average return:" + str(min_avr))
        sys.exit()

    saver = tf.train.Saver(max_to_keep=10000)

    #
    if method == "WORST_CASE":
        dummy_env = gym.make(env.spec.id) # this is to evaluate the value at the worst case setting.

    ### More book-kepping
    results = []
    if saves:
        results = open(gamename + '_' + str(num_options) + 'opts_' + '_results.csv', 'w')

        out = 'epoch,avg_reward'

        for opt in range(num_options): out += ',option {} dur'.format(opt)
        for opt in range(num_options): out += ',option {} std'.format(opt)
        for opt in range(num_options): out += ',option {} term'.format(opt)
        for opt in range(num_options): out += ',option {} adv'.format(opt)
        if method == "CVaR":
            out += ',lambda,v,e_risk'
        out += ",estmd_01pCVaR,original_ret_CVaR"
        out += '\n'
        results.write(out)
        results.flush()

    if epoch >= 0:
        dirname = '{}_{}opts_saves/'.format(gamename, num_options)
        print("Loading weights from iteration: " + str(epoch))

        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename, epoch)
        saver.restore(U.get_session(), filename)
    ###

    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    #
    ret_of_best_pol = -1.0
    best_pol_id = -1

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    # Prepare for rollouts
    # ----------------------------------------
    if method == "CVaR":
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_options=num_options,
                                         saves=saves, results=results, rewbuffer=rewbuffer, dc=dc, method=method, additional_info=dic_avars4cvar)
    else:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_options=num_options,
                                         saves=saves, results=results, rewbuffer=rewbuffer, dc=dc, method=method, additional_info=None)

    datas = [0 for _ in range(num_options)]

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()

        # WORST case @ 20190121
        if method == "WORST_CASE" and (iters_so_far > WARMING_UP_EPOCHS): # This additional condition is to use warm up
            add_vtarg_and_adv_in_worst_case(seg, gamma, lam, dummy_env, pi)
        else:
            add_vtarg_and_adv(seg, gamma, lam)

        opt_d = []
        for i in range(num_options):
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)

        std = []
        for i in range(num_options):
            logstd = np.mean(seg['logstds'][i]) if len(seg['logstds'][i]) > 0 else 0.
            std.append(np.exp(logstd))
        print("mean opt dur:", opt_d)
        print("mean op pol:", np.mean(np.array(seg['optpol_p']), axis=0))
        print("mean term p:", np.mean(np.array(seg['term_p']), axis=0))
        print("mean value val:", np.mean(np.array(seg['value_val']), axis=0))

        # EOOpt-eps @ 20190123
        if method == "EOOpt" and (iters_so_far > WARMING_UP_EPOCHS): # The second condition is for warming up
            epsilon = 0.1 #

            # back up
            original_ep_ret = copy.deepcopy(seg["ep_rets"])  # returns of completed episodes in this segment
            original_ep_len = copy.deepcopy(seg["ep_lens"])  # lengths of ...

            #- find Q val at eps percentile
            list_pair_ret_id = []
            for i in range(len(seg["ep_rets"])):
                list_pair_ret_id.append((seg["ep_rets"][i], i))
            list_pair_ret_id = sorted(list_pair_ret_id)
            index = int(len(list_pair_ret_id) * epsilon)
            if index >= len(list_pair_ret_id):
                index = index - 1
            Q_eps = list_pair_ret_id[index][0]

            # Initialize history arrays
            n_ep_rets = []  # returns of completed episodes in this segment
            n_ep_lens = []  # lengths of ...

            n_obs = None
            n_rews = np.array([]).astype("float32")
            n_realrews = np.array([]).astype("float32")
            n_vpreds = np.array([]).astype("float32")
            n_news = np.array([]).astype("float32")
            n_opts = np.array([]).astype("int32")
            n_acs = None
            #
            n_adv = np.array([]).astype("float32")
            n_lamret = np.array([]).astype("float32")
            #
            n_optpol_p = []
            n_term_p = []
            n_value_val = []
            #opt_duration = []
            #logstds = []
            #n_curr_opt_duration = 0.
            #- discard trajectories with higher values than the threshold Q-val
            is_discard = None
            ep_ind = 0
            if seg["ep_rets"][ep_ind] > Q_eps:
                is_discard = True
            else:
                is_discard = False
                n_ep_rets.append(seg["ep_rets"][ep_ind])  # returns of completed episodes in this segment
                n_ep_lens.append(seg["ep_lens"][ep_ind])  # lengths of ...

            for i in range(len(seg["rew"])):
                if not is_discard:
                    #- copy params
                    if n_obs is None:
                        n_obs = copy.deepcopy(seg["ob"][i]).reshape((1, -1))
                    else:
                        n_obs = np.append(n_obs, copy.deepcopy(seg["ob"][i]).reshape((1,-1)), axis=0)
                    n_rews = np.append(n_rews, seg["rew"][i])
                    n_realrews = np.append(n_realrews, seg["realrew"][i])
                    n_vpreds = np.append(n_vpreds, seg["vpred"][i])
                    n_news = np.append(n_news, seg["new"][i])
                    n_opts = np.append(n_opts, seg["opts"][i])
                    if n_acs is None:
                        n_acs = copy.deepcopy(seg["ac"][i]).reshape((1, -1))
                    else:
                        n_acs = np.append(n_acs, copy.deepcopy(seg["ac"][i]).reshape((1, -1)), axis=0)
                    #
                    n_adv = np.append(n_adv, seg["adv"][i])
                    #
                    n_optpol_p.append(seg["optpol_p"][i])
                    n_term_p.append(seg["term_p"][i])
                    n_value_val.append(seg["value_val"][i])
                    #
                    n_lamret = np.append(n_lamret, seg["tdlamret"][i])

                if i != 0 and (seg["new"][i] == 1) and ep_ind < (len(seg["ep_rets"]) - 1):
                    ep_ind += 1
                    if seg["ep_rets"][ep_ind] > Q_eps:
                        is_discard = True
                    else:
                        is_discard = False
                        #- copy params
                        n_ep_rets.append(seg["ep_rets"][ep_ind]) # returns of completed episodes in this segment
                        n_ep_lens.append(seg["ep_lens"][ep_ind]) # lengths of ...

            #
            seg["ob"] = n_obs
            seg["rew"] = n_rews
            seg["realrew"] = n_realrews
            seg["vpred"] = n_vpreds
            seg["new"] = n_news
            seg["opts"] = n_opts
            seg["ac"] = n_acs
            seg["adv"] = n_adv
            seg["optpol_p"] = n_optpol_p
            seg["term_p"] = n_term_p
            seg["value_val"] = n_value_val
            seg["tdlamret"] = n_lamret


        ob, ac, opts, atarg, tdlamret = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values


        #for finding best policy satisfying certain level of CVaR @ added 20190401
        sorted_original_ret_4_display = sorted(seg["ep_rets_origin_CVaR"])
        CVaR4display = 0.0
        for o_ret in range(0, int(len(sorted_original_ret_4_display) * 0.1)):
            CVaR4display += sorted_original_ret_4_display[o_ret]
        CVaR4display /= (len(sorted_original_ret_4_display) * 0.1)
        cvars_over_training.append(CVaR4display)
        rets_over_training.append(statistics.mean(seg["ep_rets_origin_CVaR"]))

        if iters_so_far % 5 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename, iters_so_far)
            # save
            U.save_state(filename)
            if method == "CVaR":
                filename_cvar = dirname + '{}_epoch_{}.ckpt.pcl'.format(gamename, iters_so_far)
                with open(filename_cvar, "wb") as f_cvar:
                    pickle.dump(dic_avars4cvar, f_cvar)

            # best policty indicator
            isWarmUpEnd = ((method == "EOOpt" or method == "WORST_CASE") and (iters_so_far > WARMING_UP_EPOCHS))
            if not isWarmUpEnd:
                isWarmUpEnd = ((method == "SOFT_ROBUST") or (method == "CVaR"))
            if isWarmUpEnd:
                if method == "EOOpt":
                    if ret_of_best_pol < (sum(original_ep_ret))/len(original_ep_ret):
                        ret_of_best_pol = (sum(original_ep_ret))/len(original_ep_ret)
                        best_pol_id = iters_so_far
                        f_bestpol = open("bestpolid.txt", "w")
                        f_bestpol.write(filename)
                        f_bestpol.close()
                else:
                    if ret_of_best_pol < (sum(seg["ep_rets"])/len(seg["ep_rets"])):
                        ret_of_best_pol = (sum(seg["ep_rets"])/len(seg["ep_rets"]))
                        best_pol_id = iters_so_far
                        f_bestpol = open("bestpolid.txt", "w")
                        f_bestpol.write(filename)
                        f_bestpol.close()
                print("The policy id with maximum return is " + str(best_pol_id))
                print("The best return is " + str(ret_of_best_pol))

                # update when the new policy acquired
                f_latestpol = open("latestpolid.txt", "w")
                f_latestpol.write(filename)
                f_latestpol.close()

            # Best policy satisfying Beta @ added 20190401@ changed 201904
            if isWarmUpEnd:
                if len(cvars_over_training) >= 6:
                    avr_cvar = statistics.mean(cvars_over_training[-5:])
                    avr_ret = statistics.mean(rets_over_training[-5:])
                    if avr_cvar >= (-BETA) and avr_ret > best_cvars:
                        best_cvars = avr_ret
                        best_cvars_pol_id = iters_so_far

                    filename_cvar_pol = dirname + "bestpol-cvar.txt"
                    with open(filename_cvar_pol, "w") as f_cvar:
                        f_cvar.write(str(best_cvars_pol_id))

        # "batch update" of CVaR additional parameters
        if method == "CVaR":
            bu_lambda = dic_avars4cvar["lambda"]
            bu_v = dic_avars4cvar["v"]

            ave_num_pos_e_risks = 0.0
            ave_e_risk = 0.0
            for e_risk in seg["e_risk"]:
                ave_e_risk += e_risk
                if e_risk > 0.0:
                    ave_num_pos_e_risks += 1.0
            ave_num_pos_e_risks /= float(len(seg["e_risk"]))
            ave_e_risk /= float(len(seg["e_risk"]))

            # updated hyper params as
            learning_rate_lambda = (5e-7) / 1.0 * 2.0 * 10 
            learning_rate_v = 0.05 

            n_lambda = dic_avars4cvar["lambda"] - learning_rate_lambda * (bu_v + ave_e_risk/dic_avars4cvar["epsilon"] - dic_avars4cvar["beta"]) 
            if n_lambda < -10.0: 
                n_lambda = -10.0

            n_v = dic_avars4cvar["v"] + learning_rate_v * bu_lambda * (1.0 - (ave_num_pos_e_risks/dic_avars4cvar["epsilon"]))

            sorted_original_ret = sorted(seg["ep_rets_origin_CVaR"])
            CVaR = 0.0
            for o_ret in range(0, int(len(sorted_original_ret)*dic_avars4cvar["epsilon"])):
                CVaR += sorted_original_ret[o_ret]
            CVaR /= (len(sorted_original_ret)*dic_avars4cvar["epsilon"])

            if (iters_so_far == 0):  # if the current v is too far away from ret in the worst case, or initialization phase
                print("Adjust v to fit it to speed up the learning")
                n_v = -1.0*sorted_original_ret[int(len(sorted_original_ret)*dic_avars4cvar["epsilon"])]# Modified to update v with the estimated VaR 20190214
            elif (math.fabs(sorted_original_ret[int(len(sorted_original_ret)*dic_avars4cvar["epsilon"])] - (-1.0 * n_v)) > 300) and (-CVaR > dic_avars4cvar["beta"]):  # if the current v is too far away from ret in the worst case, or initialization phase
                print("Adjust v to fit it to speed up the learning")
                n_v += 0.01*(-1.0*sorted_original_ret[int(len(sorted_original_ret)*dic_avars4cvar["epsilon"])] - n_v)  # Modified to update v with the estimated VaR 20190214

            dic_avars4cvar["lambda"] = n_lambda
            dic_avars4cvar["v"] = n_v
            print(n_lambda)
            print(n_v)

        min_batch = 160  # Arbitrary
        t_advs = [[] for _ in range(num_options)]
        for opt in range(num_options):
            indices = np.where(opts == opt)[0]
            print("batch size:", indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                t_advs[opt].append(0.)
                continue

            ### This part is only necessary when we use options. We proceed to these verifications in order not to discard any collected trajectories.
            if datas[opt] != 0:
                if (indices.size < min_batch and datas[opt].n > min_batch):
                    datas[opt] = Dataset(
                        dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                        shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif indices.size + datas[opt].n < min_batch:
                    # pdb.set_trace()
                    oldmap = datas[opt].data_map

                    cat_ob = np.concatenate((oldmap['ob'], ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'], ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'], atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'], tdlamret[indices]))
                    datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                         shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (
                        indices.size > min_batch and datas[opt].n < min_batch):

                    oldmap = datas[opt].data_map
                    cat_ob = np.concatenate((oldmap['ob'], ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'], ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'], atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'], tdlamret[indices]))
                    datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                             shuffle=not pi.recurrent)

                if (indices.size > min_batch and datas[opt].n > min_batch):
                    datas[opt] = d = Dataset(
                        dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                        shuffle=not pi.recurrent)

            elif datas[opt] == 0:
                datas[opt] = d = Dataset(
                    dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                    shuffle=not pi.recurrent)


            optim_batchsize = optim_batchsize or ob.shape[0]
            optim_epochs = np.clip(np.int(10 * (indices.size / (timesteps_per_batch / num_options))), 10,
                                   10) if num_options > 1 else optim_epochs
            print("optim epochs:", optim_epochs)
            logger.log("Optimizing...")

            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    tadv, nodc_adv = pi.get_term_adv(batch["ob"], [opt])
                    tadv = tadv if num_options > 1 else np.zeros_like(tadv)
                    t_advs[opt].append(nodc_adv)

                    *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                    cur_lrmult, [opt], tadv)
                    termg = termloss(batch["ob"], [opt], tadv)
                    adam.update(termg[0], 5e-7 * cur_lrmult)
                    adam.update(grads, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        if method == "EOOpt" and (iters_so_far > WARMING_UP_EPOCHS):
            lrlocal = (original_ep_len, original_ep_ret)

        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

        ### Book keeping
        if saves:
            out = "{},{}"
            for _ in range(num_options): out += ",{},{},{},{}"
            if method == "CVaR":
                out += ",{},{},{}"
            out += ",{},{}"
            out += "\n"

            info = [iters_so_far, np.mean(rewbuffer)]
            for i in range(num_options): info.append(opt_d[i])
            for i in range(num_options): info.append(std[i])
            for i in range(num_options): info.append(np.mean(np.array(seg['term_p']), axis=0)[i])
            for i in range(num_options):
                info.append(np.mean(t_advs[i]))
            if method == "CVaR":
                info.append(dic_avars4cvar["lambda"])
                info.append(dic_avars4cvar["v"])
                mean_e_risk = statistics.mean(seg["e_risk"])
                info.append(mean_e_risk)
            #
            sorted_original_ret_4_display = sorted(seg["ep_rets_origin_CVaR"])
            CVaR4display = 0.0
            for o_ret in range(0, int(len(sorted_original_ret_4_display)*0.1)):
                CVaR4display += sorted_original_ret_4_display[o_ret]
            CVaR4display /= (len(sorted_original_ret_4_display)*0.1)
            info.append(CVaR4display)

            mean_ep_ret_origin = statistics.mean(seg["ep_rets_origin_CVaR"])
            info.append(mean_ep_ret_origin)

            results.write(out.format(*info))
            results.flush()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
