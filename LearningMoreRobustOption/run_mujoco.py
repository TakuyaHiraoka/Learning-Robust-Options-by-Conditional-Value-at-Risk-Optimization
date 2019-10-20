# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import pdb

from baselines import logger
import sys
import time

from gym_extensions.continuous import mujoco

# Experimental Conditions 20190120
METHODS = ["WORST_CASE", "SOFT_ROBUST", "EOOpt", "CVaR"]


def train(env_id, num_timesteps, seed, num_options, app, saves, wsaves, epoch, dc, method, mpath):
    #from baselines.ppo1
    import mlp_policy
    import pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    if num_options ==1:
        optimsize=64
    elif num_options ==2:
        optimsize=32
    else:
        print("Only two options or primitive actions is currently supported.")
        sys.exit()


    assert method in METHODS, "Method should be either of " + str(METHODS)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            #timesteps_per_batch=2048,
            timesteps_per_batch=(2048*5), # this part is changed to realize more stable learning 2019/01/31
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc,
            method=method,
            mpath=mpath
        )
    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Task environments
    #parser.add_argument('--env', help='environment ID', default='HopperIceWall-v0') #HopperIceBlock-cont
    #parser.add_argument('--env', help='environment ID', default='Walker2d-Random-Params-v1') # Walker2d-cont
    #parser.add_argument('--env', help='environment ID', default='HalfCheetah-Random-Params-v1') # HalfCheetah-cont
    # 4 additional experiment
    #parser.add_argument('--env', help='environment ID', default='Walker2d-Random-Params-discrete-v1') # Walker2d-disc
    #parser.add_argument('--env', help='environment ID', default='HopperIceWall-continuous-v0') # HopperIceBlock-cont
    parser.add_argument('--env', help='environment ID', default='HalfCheetah-Random-Params-discrete-v1') # HalfCheetah-disc

    parser.add_argument('--seed', help='RNG seed', type=int, default=int(time.time()))

    parser.add_argument('--opt', help='number of options', type=int, default=2)
    parser.add_argument('--app', help='Append to folder name', type=str, default='')

    parser.add_argument('--saves', dest='saves', action='store_true', default=True)
    #parser.add_argument('--saves', dest='saves', action='store_true', default=False)

    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=True)
    #parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)

    parser.add_argument('--epoch', help='Epoch', type=int, default=-1)
    parser.add_argument('--dc', type=float, default=0.)

    # Learning methods
    #parser.add_argument('--method', help='Method name:' + str(METHODS), type=str, default="WORST_CASE") # worst-case
    #parser.add_argument('--method', help='Method name:' + str(METHODS), type=str, default="SOFT_ROBUST") # sofrobustt
    #parser.add_argument('--method', help='Method name:' + str(METHODS), type=str, default="EOOpt") # EOOpt
    parser.add_argument('--method', help='Method name:' + str(METHODS), type=str, default="CVaR") # OC3

    # 4 test
    parser.add_argument('--mpath', help='path for learnt model to be evaluated (dont set for training)', type=str, default=None)
    #
    # 4 behavior analysis
    #parser.add_argument('--mpath', help='path for learnt model to be evaluated  (dont set for training)',
    #                    type=str, default="./Example-WorstCase-HopperIceBlock/example1-average/hoppericewallseed1550311239_epoch_740.ckpt")
                        #type=str, default = "./Example-WorstCase-HopperIceBlock/example2-best/hoppericewallseed1550371166_epoch_750.ckpt")
    #
    #parser.add_argument('--mpath', help='path for learnt model to be evaluated  (dont set for training)',
    #                   type=str, default="./Example-SoftRobust-HopperIceBlock/hoppericewallseed1550360970_epoch_975.ckpt")
    #parser.add_argument('--mpath', help='path for learnt model to be evaluated  (dont set for training)',
    #                    type=str, default="./Example-CVaR-HopperIceBlock/hoppericewallseed1550429614_epoch_525.ckpt")
    #
    #parser.add_argument('--mpath', help='path for learnt model to be evaluated  (dont set for training)',
    #                    type=str, default="./halfcheetah-softrobust-learnt/halfcheetah-random-paramsseed1549650953_epoch_965.ckpt")



    args = parser.parse_args()


    train(args.env,
          #num_timesteps=(1.5 * (1e6) * 5), # for Hopper case
          num_timesteps=(2.0*(1e6)*5), # for Half Cheetah, Walker2D, and HopperWall cases.
          #num_timesteps=(1.0 * (1e6) * 5),  # for Prelimnary test case
          seed=args.seed, num_options=args.opt,
          app=args.app, saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,
          dc=args.dc, method=args.method, mpath=args.mpath)

if __name__ == '__main__':
    main()

