from tabnanny import verbose
import numpy as np
import gym
import argparse

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gym_unity.envs import UnityToGymWrapper

from stable_baselines3 import DQN
from stable_baselines3 import PPO

from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib import TRPO
from sb3_contrib.common.wrappers import ActionMasker

from CallbackActions import CallbackActions
from StepInfoChannel import StepInfoChannel
from MaskInfoChannel import MaskInfoChannel

from RPPO import RPPO

PATH_EVN_5 = "./build_beaming/lego_stacking"
PATH_EVN_10 = "./build_beaming/lego_stacking_10"
PATH_EVN_25 = "./build_beaming/lego_stacking_25"
LOG_PATH = "../Logging/"

fakeactions = {"blockNr": 0, "blockPos": 0}

def create_unity_env(reverse, virtualactions, usemask, blockcount):
    """
    Function to create a simulation environment.

    Keyword arguments:
        reverse -- Whether to run the simulation backwards in time
        virtualactions -- Use virtualactions. In reverse mode: Enable lazy solving
        usemask -- Use action masking
        blockcount -- Number of blocks: 5, 10, or 25
    """
    channel_step = StepInfoChannel(fakeactions)
    channel_mask = MaskInfoChannel(actionmask)
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()

    # Using the blockcount to switch envs
    if blockcount == 5:
        path_env = PATH_EVN_5
    if blockcount == 10:
        path_env = PATH_EVN_10
    if blockcount == 25:
        path_env = PATH_EVN_25

    unity_env = UnityEnvironment(path_env,  no_graphics=True, \
        side_channels=[channel_engine, channel_env, channel_step, channel_mask])
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=False )

    channel_engine.set_configuration_parameters(time_scale = 10.0)
    channel_env.set_float_parameter("reverse" , reverse)
    channel_env.set_float_parameter("virtualactions" , virtualactions)
    channel_env.set_float_parameter("actionmask" , usemask)
    return env

def calc_rew_threshold(blockcount):
    """
    Use blockcount to set reward thresholds for success
    """
    rew_threshold = 1000
    if blockcount == 5:
        rew_threshold = 0.35
    if blockcount == 10:
        rew_threshold = 0.85
    if blockcount == 25:
        rew_threshold = 2.35
    return rew_threshold

actionmask = bytearray(5)

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Masking function for the env. Requires an actionmask in the workspace

    Keyword arguments:
        env -- The environment to be used, currently unused.
    """
    validActionMask = np.ones(17)
    i = 0
    for block in actionmask:
        if(block == 0):
            validActionMask[i] = 0
        i += 1
    return validActionMask

def forwards(algo, blockcount, num_timesteps, SEED):
    """
    Run the experiments. 
    """
    RUN_NAME= algo.__name__ + "_Forward_Baseline_" + str(blockcount) + "_"
    print(LOG_PATH + RUN_NAME + str(SEED))
    logger = configure(LOG_PATH + RUN_NAME + str(SEED), ["stdout", "csv", "log", "tensorboard", "json"]) # Logger needed to get JSONs
    env =  create_unity_env( 0.0 , 0.0, 0.0, blockcount)
    model = algo("MlpPolicy", env = env, verbose=1, seed=SEED)
    model.set_logger(logger)
    eval_callback = EvalCallback(env, best_model_save_path=LOG_PATH + RUN_NAME + str(SEED),
                                log_path=LOG_PATH + RUN_NAME + str(SEED), eval_freq=1000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    env.reset()
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    reward_threshold = calc_rew_threshold(blockcount)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    np.savez(LOG_PATH + RUN_NAME + str(SEED)+ "/eval_values.npz", rew=rew, std=std)
    model.save("./"+ LOG_PATH + RUN_NAME + str(SEED) + "/last_model") 
    env.close()

def backwards(algo, blockcount, num_timesteps, SEED):
    """
    Run the experiments. 
    """
    RUN_NAME= algo.__name__ + "_Backwards_Baseline_" + str(blockcount) + "_"
    print(LOG_PATH + RUN_NAME + str(SEED))
    logger = configure(LOG_PATH + RUN_NAME + str(SEED), ["stdout", "csv", "log", "tensorboard", "json"]) # Logger needed to get JSONs
    env =  create_unity_env( 1.0 , 1.0, 0.0, blockcount)

    env = Monitor(env)

    model = algo("MlpPolicy", env = env, verbose=1, seed=SEED)
    model.set_logger(logger)
    eval_callback = EvalCallback(env, best_model_save_path=LOG_PATH + RUN_NAME + str(SEED),
                                log_path=LOG_PATH + RUN_NAME + str(SEED), eval_freq=1000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    env.reset()
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    reward_threshold = calc_rew_threshold(blockcount)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    np.savez(LOG_PATH + RUN_NAME + str(SEED)+ "/eval_values.npz", rew=rew, std=std)
    model.save("./"+ LOG_PATH + RUN_NAME + str(SEED) + "/last_model") 
    env.close()

def backwards_then_forwards(algo, blockcount, num_timesteps, SEED):
    """
    Run the experiments. 
    """

    cb_actions = CallbackActions(fakeactions)

    RUN_NAME= algo.__name__ + "_BW_Then_FW_" + str(blockcount) + "_"
    print(LOG_PATH + RUN_NAME + str(SEED))
    logger = configure(LOG_PATH + RUN_NAME + str(SEED), ["stdout", "csv", "log", "tensorboard", "json"]) # Logger needed to get JSONs
    env =  create_unity_env( 1.0 , 1.0, 0.0, blockcount)

    env = Monitor(env)

    model = RPPO("MlpPolicy", env = env, verbose=1, seed=SEED)
    model.set_logger(logger)
    eval_callback = EvalCallback(env, best_model_save_path=LOG_PATH + RUN_NAME + str(SEED),
                                log_path=LOG_PATH + RUN_NAME + str(SEED), eval_freq=1000,
                                deterministic=True, render=False)
    model.learn(total_timesteps=num_timesteps, callback=[cb_actions, eval_callback])
    env.reset()
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    np.savez(LOG_PATH + RUN_NAME + str(SEED)+ "/eval_values.npz", rew=rew, std=std)
    model.save("./"+ LOG_PATH + RUN_NAME + str(SEED) + "/last_model") 
    model.policy_rev.save("bw_policy")
    env.close()


    #FW Eval
    env =  create_unity_env( 0.0 , 0.0, 0.0, blockcount)
    model = PPO("MlpPolicy", env = env, verbose=1, seed=SEED)
    model.policy = model.policy.load("bw_policy")
    model.set_logger(logger)
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    model.save("./"+ LOG_PATH + RUN_NAME + str(SEED) + "/last_model_fw") 
    env.reset()
    env.close()

if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the three arguments
    parser.add_argument('algo', help='Algorithm to be used for execution: PPO, TRPO or RPPO currently')
    parser.add_argument('blockcount', type=int, help='Number of blocks for execution: 5, 10 or 25 currently')
    parser.add_argument('num_timesteps', type=int, help='Number of time steps for execution')
    parser.add_argument('mode', type=int, help='1: forward, 2: backward, 3: backward then forward')
    parser.add_argument('SEED', type=int, help='Seed for the RL Algo. Recommended: 100, 200, 300, 400, 500')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    if args.mode == 1:
        forwards(eval(args.algo), args.blockcount, args.num_timesteps, args.SEED)
    if args.mode == 2:
        backwards(eval(args.algo), args.blockcount, args.num_timesteps, args.SEED)
    if args.mode == 3:
        backwards_then_forwards(eval(args.algo), args.blockcount, args.num_timesteps, args.SEED)

