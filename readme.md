## Reverse RL Implementation and Brick Assembly Simulation

This repository is complementary to a pending publication.

### Reverse RL

It contains adaptations of Stable Baselines 3 code  (https://github.com/DLR-RM/stable-baselines3), also published under MIT.

The implementation plus additional scripts can be found under /python.

RPPO contains the RRL implementation while main.py is the execution fine and entry point for the Docker container. 

### Simulation Environment

The whole repository serves as a Unity project. It contains a simulation environment for toy brick assembly. Precompiled executables for Ubuntu Linux are included under build_beaming.

The scripts and additional assets of the simulation can be found under /Assets

### Docker

For parallelisation and reuse the whole simulation can be run inside Docker containters. The Dockerfile is included on top level. 

In order to run the simulation in various configurations the run_experiments file can be adapted to the created Docker Container.
