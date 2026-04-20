"""
Quick PyTorch / CUDA sanity check for the auto-rig training pipeline.

Run:    python ml_training\check_device.py

Verifies that:
  1. torch is installed and importable
  2. torch was built with CUDA support
  3. At least one CUDA device is visible
  4. Kernels for this device's compute capability are present
     (the important one for RTX 50-series — if torch was built without
     sm_120 kernels, .is_available() returns True but any op on the GPU
     raises "no kernel image is available for execution on the device")
  5. Training-relevant ops (matmul, conv2d) actually run on the GPU
  6. Quick CPU vs GPU micro-benchmark so the speedup is visible
"""
import sys
import time
import platform


def green(s):  return f"\033[32m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"


def main() -> int:
    print("=" * 60)
    print("  Auto-Rig Device Check")
    print("=" * 60)
    print(f"Python:    {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

    # 1) torch importable?
    try:
        import torch
    except ImportError as e:
        print(red(f"torch not installed: {e}"))
        print("    Fix:  python -m pip install -r ml_training/requirements.txt")
        return 1

    print(f"torch:     {torch.__version__}")
    cuda_built = torch.backends.cuda.is_built()
    print(f"torch.backends.cuda.is_built():     {cuda_built}")
    print(f"torch.cuda.is_available():          {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count():          {torch.cuda.device_count()}")
    print(f"torch.version.cuda:                 {torch.version.cuda}")
    print(f"torch.backends.cudnn.version():     {torch.backends.cudnn.version()}")

    if not cuda_built:
        print()
        print(red("This is a CPU-only torch build."))
        print("    You need the CUDA wheel. For RTX 50-series (Blackwell, sm_120):")
        print("      pip uninstall -y torch torchvision")
        print("      pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        print("    (use nightly/cu128 if stable isn't available for your Python yet)")
        return 2

    if not torch.cuda.is_available():
        print()
        print(red("torch was built with CUDA but no device is visible."))
        print("    Checks: nvidia-smi, NVIDIA driver install, CUDA_VISIBLE_DEVICES env.")
        return 3

    # 2) Per-device info
    print()
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU[{i}]: {name}")
        print(f"        compute capability: sm_{major}{minor}  ({major}.{minor})")
        print(f"        VRAM:               {vram_gb:.1f} GB")
        print(f"        SM count:           {props.multi_processor_count}")

    # 3) Check torch's compiled architectures vs the device's capability.
    # If torch has no kernels for this sm_NN, ops will crash at runtime.
    arch_list = torch.cuda.get_arch_list()
    print(f"\ntorch compiled for:  {arch_list}")
    major, minor = torch.cuda.get_device_capability(0)
    needed = f"sm_{major}{minor}"
    if any(needed in a for a in arch_list):
        print(green(f"  OK: wheel contains {needed} kernels."))
    else:
        print(yellow(
            f"  WARNING: wheel has no {needed} kernels. "
            f"Ops on GPU[0] may crash with 'no kernel image is available.'"))
        print(yellow(
            f"  For an {needed} card you need a CUDA-12.8+ wheel (cu128 or newer)."))

    # 4) Smoke test: actually run matmul + conv2d on the GPU.
    print()
    print("Smoke test (matmul + conv2d on GPU)...")
    try:
        dev = torch.device("cuda:0")
        a = torch.randn(512, 512, device=dev)
        c = (a @ a).sum().item()
        x = torch.randn(1, 7, 256, 256, device=dev)
        w = torch.randn(32, 7, 3, 3, device=dev)
        y = torch.nn.functional.conv2d(x, w).sum().item()
        torch.cuda.synchronize()
        print(green(f"  OK: matmul -> {c:.2f},  conv2d -> {y:.2f}"))
    except RuntimeError as e:
        print(red(f"  FAILED: {e}"))
        print(yellow("  This almost always means the torch wheel lacks kernels"))
        print(yellow("  for your GPU's compute capability. Install a newer wheel."))
        return 4

    # 5) Quick CPU vs GPU micro-benchmark so the speedup is visible.
    print()
    print("Quick benchmark (conv2d 1x7x256x256 -> 64 filters, 20 iters)...")
    x_cpu = torch.randn(1, 7, 256, 256)
    w_cpu = torch.randn(64, 7, 3, 3)

    # Warmup
    for _ in range(3):
        torch.nn.functional.conv2d(x_cpu, w_cpu)
    t0 = time.perf_counter()
    for _ in range(20):
        torch.nn.functional.conv2d(x_cpu, w_cpu)
    cpu_ms = (time.perf_counter() - t0) * 1000 / 20

    x_gpu, w_gpu = x_cpu.to(dev), w_cpu.to(dev)
    for _ in range(3):
        torch.nn.functional.conv2d(x_gpu, w_gpu)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        torch.nn.functional.conv2d(x_gpu, w_gpu)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000 / 20

    speedup = cpu_ms / max(gpu_ms, 1e-6)
    print(f"  CPU:  {cpu_ms:6.2f} ms/iter")
    print(f"  GPU:  {gpu_ms:6.2f} ms/iter   ({green(f'{speedup:.0f}x faster')})")

    print()
    print(green("All checks passed. Training will use the GPU."))
    print(dim("  Tip: with 32GB VRAM on a 5090 you can raise batch_size"))
    print(dim("       in train_from_captures.py from 8 to 64+ for extra speed."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
