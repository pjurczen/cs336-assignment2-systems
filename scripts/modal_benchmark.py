"""Runs a training command on a Modal B200, the leaderboard's GPU. Run from the repo root:

    uv run modal run --detach scripts/modal_benchmark.py --args "--device cuda --run_name ... <benchmark.py flags>"
    uv run modal run scripts/modal_benchmark.py::check     # builds the image and checks the environment, without a GPU

modal is a dev dependency of the project. Only modal is imported at the top: everything else runs remotely in the image,
so this file must not need torch or the project's modules locally.
"""
import modal

GPU = "B200"
TIMEOUT_MINUTES = 60  # hard stop; training.py's --time_budget_minutes should end the run well before
PROJECT_DIR = "/root/project"
# Summary tables exported as CSV next to each nsys report (nsys stats doesn't run on macOS).
NSYS_STATS_REPORTS = ["cuda_gpu_kern_sum", "cuda_gpu_trace", "cuda_api_sum", "cuda_kern_exec_sum", "nvtx_sum", "nvtx_gpu_proj_sum", "nvtx_kern_sum"]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "libc6-dev", "wget")  # torch.compile's Triton compiles a small C helper at runtime
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -qq && apt-get install -y -qq --no-install-recommends nsight-systems-2026.3.2")
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
def benchmark(nsys_args: list[str], args: list[str], git_commit: str, env: str):
    import os
    import shutil
    import subprocess
    import sys
    import glob

    os.chdir(PROJECT_DIR)
    # Local copies, like on Vertex: np.memmap reads small random pieces, which is slow over a network volume.
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=False)

    returncode = 1
    try:
        prefix = ["nsys", "profile", *nsys_args] if nsys_args else []
        os.makedirs("runs/nsys", exist_ok=True)
        # --nvtx-capture only matches registered NVTX strings by default; torch.cuda.nvtx sends plain ones.
        returncode = subprocess.run([*prefix, sys.executable, "cs336_systems/benchmark.py", *args],
                                    env=dict(os.environ, GIT_COMMIT=git_commit, NSYS_NVTX_PROFILER_REGISTER_ONLY="0", **dict(kv.split("=", 1) for kv in env.split()))).returncode
        for report in glob.glob("runs/nsys/*.nsys-rep"):
            stem = report.removesuffix(".nsys-rep")
            subprocess.run(["nsys", "stats", "--report", ",".join(NSYS_STATS_REPORTS), "--format", "csv", "--output", stem, report], check=False)
    finally:
        # Save whatever the run produced, also when it failed.
        if os.path.isdir("runs"):
            shutil.copytree("runs", "/vol/runs", dirs_exist_ok=True)
            volume.commit()
    if returncode != 0:
        raise RuntimeError(f"benchmark.py exited with code {returncode}")


@app.function(timeout=10 * 60)
def check():
    import sys

    import torch

    print(f"python {sys.executable}, torch {torch.__version__}, CUDA {torch.version.cuda}")
    # The GPU architectures this torch has kernels for, readable without a GPU. A B200 needs sm_100.
    print(f"architectures: {torch._C._cuda_getArchFlags()}")


@app.local_entrypoint()
def main(args: str, gpu: str = GPU, nsys_args: str = "", env: str = ""):
    """--gpu B200+ lets Modal pick a B200 or a B300 at the B200 price: shorter queues, but B300 timings aren't B200 ones."""
    import shlex
    import subprocess

    # Runs locally, where Git is available: the version goes into the run's config.json via $GIT_COMMIT.
    git_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"]).returncode != 0:
        git_commit += "-dirty"
    benchmark.with_options(gpu=gpu).remote(shlex.split(nsys_args), shlex.split(args), git_commit, env)
