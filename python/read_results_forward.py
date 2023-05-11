import os
import numpy as np

directory_path = "./ForwardResults" 
experiment_name = "PPO_Forward_Baseline_5"


assembly_list = []
disassembly_list = []

#Collect results at the end of Logfiles. BEWARE: Std are actually the episode lengths
for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            if experiment_name in dir:
                dir_path = os.path.join(root, dir)
                for filename in os.listdir(dir_path):
                    if filename.endswith(".txt"):  
                        file_path = os.path.join(dir_path, filename)
                        with open(file_path, "r") as file:
                            lines = file.readlines()
                            last_line = lines[-1].strip()
                            line_dict_1 = eval(last_line)
                            assembly_list.append(line_dict_1)

#Merge the dictionaries 
merged_dict_assembly = {}

for dictionary in assembly_list:
    for key, value in dictionary.items():
        if key not in merged_dict_assembly:
            merged_dict_assembly[key] = []
        merged_dict_assembly[key].append(value)

#Flatten the Arrays
flat_dict_assembly = {key: [item for sublist in value for item in sublist] for key, value in merged_dict_assembly.items()}

# Calculate Averages and Stds, print out results
print(experiment_name)
possible_reward = 0.4 
#possible_reward = 0.9 
#possible_reward = 2.4 

rewards_assembly = np.array(flat_dict_assembly["reward"])/possible_reward
print("Reward Assembly")
#print(rewards_disassembly)
print("Average Reward Assembly")
print(np.average(rewards_assembly))
print("Std Dev. Reward Assembly")
print(np.std(rewards_assembly))

timesteps_assembly = np.array(flat_dict_assembly["std"])
print("Episode Lengths Assembly")
#print(rewards_disassembly)
print("Average Episode Lengths Assembly")
print(np.average(timesteps_assembly))
print("Std Dev. Episode Lengths Assembly")
print(np.std(timesteps_assembly))