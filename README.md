# Project Name

## Overview

This project utilizes Unity ML-Agents version 3.0.0-exp.1 and Neat version 0.92 for training and testing 3DBall environments.

## Requirements

- Unity ML-Agents version: 3.0.0-exp.1
- Neat version: 0.92
## Environment Paths

Make sure to set the correct paths for your environment in the `run_trainer.py` script:

```python
# Specify the path for the single-agent environment
single_agent_env_path = "./Builds/SingleAgent/3DBallBalancing.exe"

# Specify the path for the multi-agent environment
multi_agent_env_path = "./Builds/60Agents/3DBallBalancing.exe"
```
## Getting Started

To run 3DBall training, follow these steps:

1. Execute `run_trainer.py` to obtain parameters for either training or testing environments.
```python
# [PARAMETERS]  
max_generations = 200       # Max number of generations
is_training = False         # Whether we are training (True) or testing (False)
no_graphics = is_training     # Usually, when training we do not want to see graphics window
is_multi = True               # either True or False, whether we use multiple or single agent environment
is_debug = False               # debugging option with prints
load_from_checkpoint = False   # load from neat checkpoint
```
