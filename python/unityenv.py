

import numpy as np
import gym

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from gym_unity.envs import UnityToGymWrapper

from StepInfoChannel import StepInfoChannel
from MaskInfoChannel import MaskInfoChannel

PATH_EVN = "/home/niklas/legoblocks/LegoSimulationAIGym/build_beaming/lego_stacking"
PATH_EVN_10 = "/home/niklas/legoblocks/LegoSimulationAIGym/build_beaming/lego_stacking_10"
PATH_EVN_25 = "/home/niklas/legoblocks/LegoSimulationAIGym/build_beaming/lego_stacking_25"


#CURRENTLY NOT USED
def create_unity_env(reverse, virtualactions, usemask, fakeactions, actionmask):
    """
    Function to create a wrapped environment of the Lego simulation
    """
    fakeactions[:] = fakeactions
    actionmask[:] = actionmask
    channel_step = StepInfoChannel(fakeactions)
    channel_mask = MaskInfoChannel(actionmask)
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(PATH_EVN,  no_graphics=True, \
        side_channels=[channel_engine, channel_env, channel_step, channel_mask])
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=False )

    channel_engine.set_configuration_parameters(time_scale = 10.0)
    channel_env.set_float_parameter("reverse" , reverse)
    channel_env.set_float_parameter("virtualactions" , virtualactions)
    channel_env.set_float_parameter("actionmask" , usemask)
    return env
