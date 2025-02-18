import os
from pathlib import Path
from omegaconf import DictConfig
from huggingface_hub import HfApi, upload_file, hf_hub_download
from typing import List, Optional
import shutil

class HFModelManager:
    def __init__(self, repo_id: str, token: Optional[str] = None):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.token = token

    def push_models(self, models_dir: str, root_level: bool = False) -> None:
        """
        Push models to Hugging Face.

        Args:
            models_dir (str): Path to the models directory
            root_level (bool): If True, pushes files directly from the root directory
        """
        models_path = Path(models_dir).resolve()
        print(f"Looking for files in: {models_path}")

        try:
            self.api.create_repo(repo_id=self.repo_id, exist_ok=True)
        except Exception as e:
            print(f"Error creating repository: {e}")
            return

        # Find all .zip and .npy files recursively
        files = list(models_path.rglob('*.zip')) + list(models_path.rglob('*.npy'))
        print(f"Found files: {[str(f.relative_to(models_path)) for f in files]}")

        for file in files:
            if file.is_file():
                # Preserve the directory structure
                relative_path = file.relative_to(models_path)
                try:
                    print(f"Attempting to upload: {relative_path}")
                    upload_file(
                        path_or_fileobj=str(file),
                        path_in_repo=str(relative_path),
                        repo_id=self.repo_id,
                        token=self.token
                    )
                    print(f"Successfully uploaded {relative_path}")
                except Exception as e:
                    print(f"Error uploading {relative_path}: {e}")
        else:
            # Original nested directory structure logic
            for algorithm in os.listdir(models_path):
                algo_path = models_path / algorithm
                if not algo_path.is_dir():
                    continue

                for env in os.listdir(algo_path):
                    env_path = algo_path / env
                    if not env_path.is_dir():
                        continue

                    model_files = ['best_model.zip', 'final_model.zip', 'interrupted_model.zip']
                    for model_file in model_files:
                        model_path = env_path / model_file
                        if model_path.exists():
                            remote_path = f"{algorithm}/{env}/{model_file}"
                            try:
                                upload_file(
                                    path_or_fileobj=str(model_path),
                                    path_in_repo=remote_path,
                                    repo_id=self.repo_id,
                                    token=self.token
                                )
                                print(f"Uploaded {remote_path}")
                            except Exception as e:
                                print(f"Error uploading {remote_path}: {e}")

    def pull_models(self, output_dir: str, root_level: bool = False, algorithm: Optional[str] = None, env: Optional[str] = None) -> None:
        """
        Pull models from Hugging Face.

        Args:
            output_dir (str): Directory to save the downloaded models
            root_level (bool): If True, downloads files to root directory
            algorithm (str, optional): Specific algorithm to download
            env (str, optional): Specific environment to download
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            files = self.api.list_repo_files(self.repo_id)

            for file in files:
                if root_level:
                    # Download directly to output directory
                    try:
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=file,
                            local_dir=output_path,
                            local_dir_use_symlinks=False
                        )
                        print(f"Downloaded {file}")
                    except Exception as e:
                        print(f"Error downloading {file}: {e}")
                else:
                    # Original nested directory logic
                    if not file.endswith('.zip'):
                        continue

                    parts = file.split('/')
                    if len(parts) != 3:
                        continue

                    file_algo, file_env, model_file = parts

                    if algorithm and file_algo != algorithm:
                        continue
                    if env and file_env != env:
                        continue

                    save_dir = output_path / file_algo / file_env
                    save_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=file,
                            local_dir=output_path,
                            local_dir_use_symlinks=False
                        )
                        print(f"Downloaded {file}")
                    except Exception as e:
                        print(f"Error downloading {file}: {e}")

        except Exception as e:
            print(f"Error accessing repository: {e}")

def ensure_model_exists(cfg: DictConfig, local_path: Path) -> Path:
    """
    Ensure model exists locally, downloading from HF if necessary.
    """
    if local_path.exists():
        print(f"Using local model from: {local_path}")
        return local_path

    print(f"Local model not found at {local_path}. Attempting to download from Hugging Face...")

    # Setup HF model manager
    token = os.getenv("HF_TOKEN")
    repo_id = "AOS55/RLFoundations"

    # Create the directory structure if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Construct the remote path and local path
        algo = cfg.agent.algo.lower()
        env_name = cfg.env.name
        filename = f"{algo}/{env_name}/best_model.zip"

        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=local_path.parent,
            local_dir_use_symlinks=False
        )

        # Move the file to the correct location if needed
        downloaded_path = Path(downloaded_path)
        if downloaded_path != local_path:
            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # Move the file to the correct location
            shutil.move(str(downloaded_path), str(local_path))

            # Clean up any empty directories
            current_dir = downloaded_path.parent
            while current_dir != local_path.parent:
                try:
                    current_dir.rmdir()
                    current_dir = current_dir.parent
                except OSError:
                    break  # Directory not empty or other error

        if local_path.exists():
            print(f"Successfully downloaded model to: {local_path}")
            return local_path
        else:
            raise FileNotFoundError("Model was not found after download attempt")

    except Exception as e:
        raise FileNotFoundError(
            f"Could not find model locally or on Hugging Face: {str(e)}\n"
            f"Local path: {local_path}"
        )

def setup_model_sharing(repo_id: str, token: str) -> HFModelManager:
    """
    Create a HFModelManager instance with the given credentials.

    Args:
        repo_id (str): The Hugging Face repository ID
        token (str): Hugging Face API token

    Returns:
        HFModelManager: Configured model manager instance
    """
    return HFModelManager(repo_id=repo_id, token=token)

# Example usage:
if __name__ == "__main__":
    token = os.getenv("HF_TOKEN")
    repo_id = "AOS55/RLFoundations"

    manager = setup_model_sharing(repo_id=repo_id, token=token)

    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models"

    print(f"Looking for models in: {models_dir}")
    print("Files in directory:")
    for file in models_dir.rglob('*'):  # Use rglob here too to show all files
        if file.is_file():
            print(f"  {file.relative_to(models_dir)}")

    manager.push_models(str(models_dir), root_level=True)
