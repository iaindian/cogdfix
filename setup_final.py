"""
Production-grade headless ComfyUI installer (with CUDA-enabled PyTorch and plugin root-level dependency installs)
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Optional: GitPython fallback
try:
    from git import Repo
    GITPYTHON_AVAILABLE = True
except ImportError:
    GITPYTHON_AVAILABLE = False

# Logging configuration
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, cwd: Path = None, shell: bool = False, input_text: str = None):
    """
    Execute a command, optionally providing input_text to handle prompts.
    """
    if isinstance(cmd, (list, tuple)):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    logger.info(f"$ {cmd_str}")
    subprocess.run(cmd, cwd=cwd, shell=shell, check=True, input=input_text, text=True)


def install_torch_cuda(cuda_version: str = "cu118"):
    # """
    # Uninstall existing torch packages and install CUDA-enabled builds non-interactively.
    # """
    packages = ["torch", "torchvision", "torchaudio"]

    # # Uninstall existing
    # logger.info("Uninstalling existing torch packages (if any)")
    # run_command([sys.executable, "-m", "pip", "uninstall", "-y"] + packages)

    # # Clear cache
    # logger.info("Clearing pip cache to avoid stale wheels")
    # run_command([sys.executable, "-m", "pip", "cache", "purge"])

    # Install with forced confirmation (Y) in case of custom wrappers
    installer_cmd = [
        sys.executable, "-m", "pip", "install"
    ] + packages + ["--index-url", f"https://download.pytorch.org/whl/{cuda_version}"]
    logger.info(f"Installing torch packages with CUDA {cuda_version} (auto-confirm)")
    run_command(installer_cmd, input_text="Y\n")


def clone_repo(repo_url: str, target_dir: Path) -> bool:
    """
    Clone a Git repo to target_dir or pull if it already exists.

    Returns:
        True if the repo was freshly cloned, False if it already existed and was updated.
    """
    if target_dir.exists():
        logger.info(f"Updating existing repo at {target_dir}")
        if GITPYTHON_AVAILABLE:
            Repo(target_dir).remotes.origin.pull()
        else:
            run_command(["git", "-C", str(target_dir), "pull"])
        return False
    else:
        logger.info(f"Cloning {repo_url} -> {target_dir}")
        if GITPYTHON_AVAILABLE:
            Repo.clone_from(repo_url, str(target_dir))
        else:
            run_command(["git", "clone", repo_url, str(target_dir)])
        return True


def install_requirements(path: Path):
    """
    Install Python dependencies from requirements.txt at the given path.
    """
    req_file = path / "requirements.txt"
    if req_file.is_file():
        logger.info(f"Installing requirements for {path.name}")
        run_command([sys.executable, "-m", "pip", "install", "-r", str(req_file)])


def run_install_scripts(path: Path):
    """
    Execute install.py or install.bat if present at the given path.
    """
    py = path / "install.py"
    bat = path / "install.bat"
    if py.is_file():
        logger.info(f"Running install.py in {path}")
        run_command([sys.executable, str(py)], cwd=path)
    if bat.is_file() and os.name == 'nt':
        logger.info(f"Running install.bat in {path}")
        run_command(str(bat), cwd=path, shell=True)


def setup_plugin_root(plugin_dir: Path):
    """
    Install only the root-level requirements and scripts for a plugin.
    """
    install_requirements(plugin_dir)
    run_install_scripts(plugin_dir)


def install_comfyui(core_url: str, install_dir: Path, cuda_version: str = "cu118"):
    """
    Clone and set up the ComfyUI core, including CUDA-enabled PyTorch.
    """
    clone_repo(core_url, install_dir)
    install_torch_cuda(cuda_version)
    install_requirements(install_dir)


def install_plugins(plugin_urls: list, comfy_root: Path):
    """
    Clone and install each plugin under ComfyUI/custom_nodes at the plugin root only.
    Skips setup if the directory already existed.
    """
    custom_nodes_dir = comfy_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    for url in plugin_urls:
        name = Path(url.rstrip('/')).stem
        dest = custom_nodes_dir / name
        try:
            is_new = clone_repo(url, dest)
            if is_new:
                setup_plugin_root(dest)
                logger.info(f"✅ Fresh plugin installed: {name}")
            else:
                logger.info(f"ℹ️ Plugin already present, skipping root setup: {name}")
        except Exception as exc:
            logger.error(f"❌ Failed installing {name}: {exc}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    COMFY_URL = "https://github.com/comfyanonymous/ComfyUI.git"
    CUDA_VERSION = "cu121"   # Update to your CUDA toolkit version
    PLUGINS = [
      "https://github.com/storyicon/comfyui_segment_anything",
      "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
      "https://github.com/WASasquatch/was-node-suite-comfyui",
      "https://github.com/kijai/ComfyUI-KJNodes",
      "https://github.com/Acly/comfyui-inpaint-nodes",
      "https://github.com/cubiq/ComfyUI_essentials",
      "https://github.com/pamparamm/sd-perturbed-attention"
      
        # Add more plugin repositories here
    ]

    base_dir = Path(__file__).parent.resolve()
    comfy_root = base_dir / "ComfyUI"

    logger.info("--- Setting up ComfyUI core ---")
    install_comfyui(COMFY_URL, comfy_root, CUDA_VERSION)

    logger.info("--- Installing custom nodes ---")
    install_plugins(PLUGINS, comfy_root)

    logger.info("All done! Headless ComfyUI with CUDA and custom nodes is ready.")
