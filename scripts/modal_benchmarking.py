"""Runs a training command on a Modal B200, the leaderboard's GPU. Run from the repo root:

    uv run modal run --detach scripts/modal_train.py --dataset owt-32k --args "--device cuda --run_name ... <training.py flags>"
    uv run modal run scripts/modal_train.py::check     # builds the image and checks the environment, without a GPU

Data and tokenizers live in the Modal volume cs336-data (data/<dataset>/, tokenizers/<name>/), uploaded once with
`modal volume put`. Finished runs are saved there under runs/<run_name>/; fetch them with
`modal volume get cs336-data runs/<run_name> runs/modal/`. --detach keeps the run going if this terminal closes.

modal is a dev dependency of the project. Only modal is imported at the top: everything else runs remotely in the image,
so this file must not need torch or the project's modules locally.
"""
import modal

GPU = "B200"
TIMEOUT_MINUTES = 60  # hard stop; training.py's --time_budget_minutes should end the run well before
PROJECT_DIR = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "libc6-dev")  # torch.compile's Triton compiles a small C helper at runtime
    .uv_sync(extra_options="--no-dev --no-install-local")  # the locked dependencies, Linux torch built for CUDA 13
    # expandable_segments lets PyTorch's allocator join scattered free memory, so a large batch doesn't fail on
    # fragmentation (an A100 run once ran out of memory with 4.2 GiB reserved but unusable).
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{PROJECT_DIR}:{PROJECT_DIR}/cs336-basics-own", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("cs336-basics-own", f"{PROJECT_DIR}/cs336-basics-own", ignore=["**/__pycache__", "**/.DS_Store", "**/.ipynb_checkpoints"])
    .add_local_dir("cs336_systems", f"{PROJECT_DIR}/cs336_systems", ignore=["**/__pycache__", "**/.DS_Store", "**/.ipynb_checkpoints"])
)
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
app = modal.App("cs336-train", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_MINUTES * 60, volumes={"/vol": volume}, secrets=[modal.Secret.from_name("wandb")])
def train(dataset: str, args: list[str], git_commit: str):
    import os
    import shutil
    import subprocess
    import sys

    os.chdir(PROJECT_DIR)
    # Local copies, like on Vertex: np.memmap reads small random pieces, which is slow over a network volume.
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=False)

    returncode = 1
    try:
        returncode = subprocess.run([sys.executable, "cs336_systems/benchmarking.py", *args], env=dict(os.environ, GIT_COMMIT=git_commit)).returncode
    finally:
        # Save whatever the run produced, also when it failed.
        if os.path.isdir("runs"):
            shutil.copytree("runs", "/vol/runs", dirs_exist_ok=True)
            volume.commit()
    if returncode != 0:
        raise RuntimeError(f"training.py exited with code {returncode}")


@app.function(timeout=10 * 60)
def check():
    import sys

    import torch

    print(f"python {sys.executable}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    # The GPU architectures this torch has kernels for, readable without a GPU. A B200 needs sm_100.
    print(f"architectures: {torch._C._cuda_getArchFlags()}")


@app.local_entrypoint()
def main(dataset: str, args: str, gpu: str = GPU):
    """--gpu B200+ lets Modal pick a B200 or a B300 at the B200 price: shorter queues, but B300 timings aren't B200 ones."""
    import shlex
    import subprocess

    # Runs locally, where Git is available: the version goes into the run's config.json via $GIT_COMMIT.
    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0:
        git_commit += "-dirty"
    train.with_options(gpu=gpu).remote(dataset, shlex.split(args), git_commit)
