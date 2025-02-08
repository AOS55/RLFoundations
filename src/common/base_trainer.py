import os
from pathlib import Path
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.wrappers import RecordVideo, TimeLimit

class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        gym.register_envs(gymnasium_robotics)
        self.cfg = cfg
        self.setup_directories()
        self.setup_wandb()
        self.setup_environments()

    def setup_directories(self):
        orig_cwd = hydra.utils.get_original_cwd()
        for path_name, path_value in self.cfg.paths.items():
            if isinstance(path_value, str):
                Path(os.path.join(orig_cwd, path_value)).mkdir(parents=True, exist_ok=True)

    def make_env(self, render_mode="rgb_array", record_video=False):
        """Creates a wrapped environment for training or evaluation"""
        print(f'env name: {self.cfg.env}')
        def _init():
            env = gym.make(
                self.cfg.env.name,
                render_mode=render_mode,
                max_episode_steps=self.cfg.env.max_episode_steps, 
            )

            # Apply configured wrappers
            if self.cfg.env.wrappers.time_limit:
                env = TimeLimit(env, max_episode_steps=self.cfg.env.max_episode_steps)

            if record_video:
                env = RecordVideo(
                    env,
                    video_folder=str(self.cfg.paths.video_dir),
                    episode_trigger=lambda x: x % self.cfg.evaluation.video_freq == 0
                )

            return RolloutInfoWrapper(env)
        return _init

    def setup_environments(self):
        """Setup training and evaluation environments"""
        # Training environment with configured number of envs
        self.env = SubprocVecEnv(
            [self.make_env() for _ in range(self.cfg.training.num_envs)]
        )

        # Single evaluation environment with video recording if enabled
        render_mode = "rgb_array"
        self.eval_env = DummyVecEnv([
            self.make_env(
                render_mode=render_mode,
                record_video=self.cfg.evaluation.record_video
            )
        ])

    def setup_wandb(self):
        """Initialize WandB if enabled"""
        
        resolved_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        
        if self.cfg.logging.wandb.enabled:
            os.environ["WANDB_DIR"] = str(Path(self.cfg.paths.base_output_dir)) 
            wandb.init(
                project=self.cfg.logging.wandb.project,
                entity=self.cfg.logging.wandb.entity,
                group=self.cfg.logging.wandb.group,
                name=self.cfg.logging.wandb.name,
                config=resolved_cfg,
                tags=self.cfg.logging.wandb.tags,
                monitor_gym=self.cfg.logging.wandb.monitor_gym,
                save_code=self.cfg.logging.wandb.save_code
            )

    def cleanup(self):
        """Cleanup resources"""
        self.env.close()
        self.eval_env.close()
        if self.cfg.logging.wandb.enabled:
            wandb.finish()