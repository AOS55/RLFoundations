# Robotic Learning Foundations

Welcome to RLFoundations! This repository is designed to help you get started with robotic learning without getting lost in implementation details. Whether you're a student, researcher, or robotics enthusiast, you'll find everything you need to start experimenting with fundamental robotic learning techniques.

The repository focuses on the [Fetch](https://robotics.farama.org/envs/fetch/) robotics environments using [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) and [imitation](https://imitation.readthedocs.io/en/latest/) The Hydra configs used to run the various experiments can easily be edited to help you understand how each parameter effects the environment. 

To get started run:
```bash
python scripts/robot_rollout.py
```
![Pick and Place Demo](https://github.com/AOS55/RLFoundations/blob/assets/PickAndPlaceDemo.gif)


## Features

### Multiple Learning Approaches:

- Deep Reinforcement Learning (SAC, TD3)
- Imitation Learning (Behavioral Cloning)
- 🏗️ Coming Soon: Model Predictive Control (MPC) 🏗️

### Integrated Tools:
- [Weights & Biases](https://wandb.ai/site/experiment-tracking/) logging.
- Flexible configuration using [Hydra](https://hydra.cc/).
- [Hugging Face](https://huggingface.co/) pretrained models and datasets.
- [TensorBoard](https://www.tensorflow.org/tensorboard) support.

## Installation:
1. Clone the repository:
  ```bash
  git clone https://github.com/AOS55/RLFoundations.git
  cd RLFoundations
  ```
2. Create [Conda](https://docs.anaconda.com/miniconda/install/) environment:
  ```bash
  conda create --name robotics-env python=3.10
  conda activate robotics-env
  ```
3. Install required packages:
  ```bash
  pip install .
  ```

## Usage
Configs are managed in the [conf](conf) directory using Hydra. Experiments are pre-baked and ready to run.

### Training
- Train an SAC Agent on PickAndPlace environment

```bash
python scripts/train.py
```

- Train an agent using Behavioral Cloning

```bash
python scripts/train.py --config-name train_il
```

### Collect Demonstrations
```bash
python scripts/collect_demos.py
```
