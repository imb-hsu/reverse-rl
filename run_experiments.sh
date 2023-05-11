#!/bin/bash

# Define the parameter configurations
# ALGO BLOCKCOUNT TIMESTEPS SEED
CONFIGS=(
  #"RPPO 25 50000 2 100"
  #"RPPO 25 50000 2 200"
  #"RPPO 25 50000 2 300"
  #"RPPO 25 50000 2 400"

  #"RPPO 25 250000 3 110"
  #"RPPO 25 250000 3 210"
  #"RPPO 25 250000 3 310"
  #"RPPO 25 250000 3 410"
  #"RPPO 25 250000 3 510"
  #"RPPO 25 250000 3 610"
  #"RPPO 25 250000 3 710"
  #"RPPO 25 250000 3 810"
  
  "RPPO 10 50000 3 100"
  "RPPO 10 50000 3 200"
  "RPPO 10 50000 3 300"
  "RPPO 10 50000 3 400"
  "RPPO 10 50000 3 500"
  "RPPO 10 50000 3 600"
  "RPPO 10 50000 3 700"
  "RPPO 10 50000 3 800"
)

# Loop through the configurations and run the Docker container
for config in "${CONFIGS[@]}"; do
  # Extract the parameter values from the configuration
  ALGO=$(echo $config | cut -d " " -f 1)
  BLOCKCOUNT=$(echo $config | cut -d " " -f 2)
  NUM_TIMESTEPS=$(echo $config | cut -d " " -f 3)
  DIRECTION=$(echo $config | cut -d " " -f 4)
  SEED=$(echo $config | cut -d " " -f 5)

  # Set the volume name to the current config name with underscores
  VOLUME_NAME=${config// /_}

  # Run the Docker container with the parameter values.
  # Use the designated Entrypoint of the dockerfile
  sudo docker run -d --gpus all -v $VOLUME_NAME:/Logging niklas/legoarguments ./code/main.py $ALGO $BLOCKCOUNT $NUM_TIMESTEPS $DIRECTION $SEED

  # echo current config
  echo $ALGO
  echo $BLOCKCOUNT
  echo $NUM_TIMESTEPS
  echo $DIRECTION
  echo $SEED
done
