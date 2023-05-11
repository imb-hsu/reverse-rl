import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common import logger

MAX_REW_5 = 0.4
MAX_REW_10 = 0.9
MAX_REW_25 = 02.4

def collectData(experiment_name="RPPO_BW_then_FW_25"):
    """
    Fuction to collect all the data of the experiment runs. The experiment name should match the folders with the runs.
    The data is averaged over the individual runs. 
    """
    # set directory paths for training data in JSON format and evaluation data in NPZ format
    training_dir = "./BackwardResultsWR"

    #keys for timesteps, reward, and ep length
    timesteps_key = "time/total_timesteps"
    reward_key = "eval/mean_reward"
    ep_length_key = "eval/mean_ep_length"

    # create list of data frames per training run
    data = []

    # find max reward, max timesteps for standardization
    if experiment_name.endswith('5'):
        max_rew = MAX_REW_5
    if experiment_name.endswith('10'):
        max_rew = MAX_REW_10
    if experiment_name.endswith('25'):
        max_rew = MAX_REW_25

    # collect training data from JSON files in subfolders that match the experiment name, for each experiment name
    for root, dirs, files in os.walk(training_dir):
        for dir in dirs:
            if experiment_name in dir:
                dir_path = os.path.join(root, dir)
                for filename in os.listdir(dir_path):
                    if filename.endswith(".csv"):
                        with open(os.path.join(dir_path, filename), 'r') as f:
                            frame = pd.read_csv(f)
                            data.append(frame)

    # sanitize panda dataframes
    data_rewards = []
    data_ep_lengths = []
    data_time_steps = []
    for run in data:
        run.dropna(axis=0, subset=reward_key, inplace = True)
        data_rewards.append(run[reward_key])
        data_ep_lengths.append(run[ep_length_key])
        data_time_steps.append(run[timesteps_key])

    # concatenate data frames along the time axis, and normalize according to max_rew
    rewards_concat = pd.concat(data_rewards, axis=1)/max_rew
    ep_lengths_concat = pd.concat(data_ep_lengths, axis=1)

    # calculate mean and standard deviation along the time axis (axis=1)
    #mean_rewards = rewards_concat.mean(axis=1)
    #mean_ep_lengths = ep_lengths_concat.mean(axis=1)
    #std_rewards = rewards_concat.std(axis=1)
    #std_ep_lengths = ep_lengths_concat.std(axis=1)

    #reshape timesteps, assume timesteps are uniform between runs
    timesteps = np.array(data_time_steps[0])

    # convert to numpy arrays
    rewards_concat = np.array(rewards_concat)
    ep_lengths_concat = np.array(ep_lengths_concat)

    #return everything
    return timesteps, rewards_concat, ep_lengths_concat

def drawPlots(experiment_names = ["PPO_Forward_Baseline_5", "PPO_Forward_Baseline_10"]):
    """
    Function calls the collect data function of each experiment and then draws the data into a single plot
    """

    SMALL_SIZE = 26
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 26

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    #rewards
    for experiment_name in experiment_names:
        #run collect data
        plt.figure(figsize=(8,6))
        timesteps, rewards_concat, ep_lengths_concat = collectData(experiment_name)
        # plot training data
        #plt.subplot(1, len(experiment_names), i)
        plt.plot(timesteps, rewards_concat)
        plt.ylabel("Mean Episode Reward")
        plt.xlabel("Timesteps")

        plt.locator_params(nbins = 4) # number of ticks
        plt.tight_layout(pad=0.5)
        #plt.suptitle("Training Data")
        #plt.style.use('fivethirtyeight')
        # plt.show() # commented out on remote pc


        plt.savefig('./Logging/' + experiment_name +'_all_wr_rewards.png')
        plt.close()

    #episode lengths
    for experiment_name in experiment_names:
        plt.figure(figsize=(8,6))
        #run collect data
        timesteps, rewards_concat, ep_lengths_concat = collectData(experiment_name)
        # plot training data
        #plt.subplot(1, len(experiment_names), i)
        plt.plot(timesteps, ep_lengths_concat)
        plt.ylabel("Mean Episode Length")
        plt.xlabel("Timesteps")

        plt.locator_params(nbins = 4) # number of ticks
        plt.tight_layout(pad=0.5)
        #plt.suptitle("Training Data")
        #plt.style.use('fivethirtyeight')
        # plt.show() # commented out on remote pc
        plt.savefig('./Logging/' + experiment_name +'_all_wr_timesteps.png')
        plt.close()

def main():
    #drawPlots(["PPO_Forward_Baseline_5", "PPO_Forward_Baseline_10", "PPO_Forward_Baseline_25"])
    #drawPlots(["RPPO_BW_Then_FW_5","RPPO_BW_Then_FW_10","RPPO_BW_Then_FW_25"])
    drawPlots(["RPPO_Backwards_Baseline_25"])

if __name__ == '__main__':
    main()

