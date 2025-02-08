from setuptools import setup, find_packages

setup(
    name="rlfoundations",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "imitation @ git+https://github.com/AOS55/imitation.git@master",
        "mujoco",
        "gymnasium-robotics",
        "hydra-core",
        "omegaconf",
        "wandb",  # optional for logging
        "minari",
        "shimmy",
        "moviepy",
    ],
    extras_require={
        "dev": [  # Optional development dependencies
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
)
