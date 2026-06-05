#!/usr/bin/env python3
"""
setup_flux.py -- one-shot installer for the 4-bit FLUX.2 text-to-image
generator used by the editor's Content Browser.

What it does
------------
1. Creates a dedicated virtual-env at  realworld/tools/flux/.venv
   (the editor launches `<.venv>/python flux_generate.py`, so the deps
    must live HERE, isolated from the ml_training env).
2. Installs CUDA PyTorch + diffusers/transformers/accelerate/bitsandbytes.
3. Verifies / performs the gated Hugging Face login (FLUX.2-dev is gated).
   If not already logged in it PAUSES for you to accept the licence, then
   reads your token interactively.
4. Pre-downloads the NF4 4-bit weights: diffusers/FLUX.2-dev-bnb-4bit
   (DiT + text-encoders 4-bit, VAE full precision -- ~18-20 GB VRAM).

Run with the SYSTEM python; it bootstraps the venv itself. Idempotent.

Usage
-----
    python realworld/tools/flux/setup_flux.py
    python realworld/tools/flux/setup_flux.py --cuda cu128
    python realworld/tools/flux/setup_flux.py --cpu
    python realworld/tools/flux/setup_flux.py --hf-token hf_xxx
    python realworld/tools/flux/setup_flux.py --skip-download
    python realworld/tools/flux/setup_flux.py --selftest
    python realworld/tools/flux/setup_flux.py --recreate-venv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
VENV_DIR = THIS_DIR / ".venv"
DEFAULT_MODEL = "diffusers/FLUX.2-dev-bnb-4bit"
BASE_MODEL = "black-forest-labs/FLUX.2-dev"

PIP_PACKAGES = [
    "diffusers>=0.36.0",
    "transformers>=4.46.0",
    "accelerate>=1.0.0",
    "bitsandbytes>=0.45.0",
    "safetensors>=0.4.0",
    "huggingface_hub>=0.27.0",
    "sentencepiece",
    "protobuf",
    "pillow>=10.0.0",
]


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd, **kw):
    printable = " ".join(str(c) for c in cmd)
    print("[setup_flux] $ " + printable, flush=True)
    subprocess.check_call([str(c) for c in cmd], **kw)


def ensure_venv(recreate: bool) -> Path:
    py = venv_python(VENV_DIR)
    if recreate and VENV_DIR.exists():
        import shutil
        print("[setup_flux] removing existing venv " + str(VENV_DIR))
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    if not py.exists():
        print("[setup_flux] creating venv at " + str(VENV_DIR))
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("[setup_flux] reusing venv at " + str(VENV_DIR))
    if not py.exists():
        raise RuntimeError("venv python not found after creation: " + str(py))
    return py


def pip_install(py: Path, args):
    run([py, "-m", "pip", "install", "--disable-pip-version-check", *args])


def install_torch(py: Path, cuda: str, cpu: bool):
    print("\n[setup_flux] -- installing PyTorch --")
    if cpu or cuda == "cpu":
        pip_install(py, ["torch", "torchvision"])
        return
    index = "https://download.pytorch.org/whl/" + cuda
    pip_install(py, ["torch", "torchvision", "--index-url", index])


def _whoami(py: Path):
    try:
        out = subprocess.run(
            [str(py), "-c", "from huggingface_hub import whoami; print(whoami()['name'])"],
            capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _login_with_token(py: Path, token: str):
    """Persist an HF login from token. Token is passed via env (not argv)
    so it never appears in the process list. Returns username or None."""
    env = dict(os.environ, _FLUX_HF_TOKEN=token)
    code = (
        "import os;"
        "from huggingface_hub import login, whoami;"
        "login(token=os.environ['_FLUX_HF_TOKEN'], add_to_git_credential=False);"
        "print(whoami()['name'])"
    )
    try:
        out = subprocess.run([str(py), "-c", code], env=env,
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
        if out.stderr.strip():
            print("[setup_flux] " + out.stderr.strip().splitlines()[-1])
    except Exception as e:
        print("[setup_flux] login error: " + str(e))
    return None


def hf_login(py: Path, token: str | None):
    """token arg / HF_TOKEN -> existing login -> INTERACTIVE pause + token."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("[setup_flux] logging into Hugging Face with the provided token...")
        who = _login_with_token(py, token)
        if who:
            print("[setup_flux] logged in as: " + who)
            return True
        print("[setup_flux][WARN] token login failed; switching to interactive prompt.")
    else:
        who = _whoami(py)
        if who:
            print("[setup_flux] using existing HF login: " + who)
            return True

    print("\n" + "=" * 60)
    print(" FLUX.2-dev is GATED -- one-time licence acceptance required.")
    print(" 1) Sign in at huggingface.co, then open BOTH pages and click")
    print("    'Agree and access repository':")
    print("      https://huggingface.co/" + BASE_MODEL)
    print("      https://huggingface.co/" + DEFAULT_MODEL)
    print(" 2) Create a READ token: https://huggingface.co/settings/tokens")
    print("=" * 60, flush=True)
    try:
        input("  Press Enter once you've accepted the licence on both pages... ")
    except EOFError:
        print("[setup_flux][WARN] no interactive console (stdin closed).")
        print("                    Set HF_TOKEN or pass --hf-token, then re-run.")
        return False

    import getpass
    for attempt in range(1, 4):
        try:
            tok = getpass.getpass("  Paste your HF token (hf_...), or Enter to skip: ").strip()
        except Exception:
            try:
                tok = input("  Paste your HF token (hf_...), or Enter to skip: ").strip()
            except EOFError:
                tok = ""
        if not tok:
            print("[setup_flux][WARN] no token entered -- skipping the gated download.")
            print("                    Re-run later:  python " + Path(__file__).name)
            return False
        who = _login_with_token(py, tok)
        if who:
            print("[setup_flux] logged in as: " + who)
            return True
        print("[setup_flux][WARN] login failed (attempt %d/3). "
              "Check the token has READ scope, then try again." % attempt)
    print("[setup_flux][WARN] could not authenticate after 3 attempts.")
    return False


def download_weights(py: Path, model: str):
    print("\n[setup_flux] -- downloading weights: " + model + " --")
    code = (
        "from huggingface_hub import snapshot_download;"
        "p=snapshot_download(%r);"
        "print('[setup_flux] cached at', p)" % model
    )
    run([py, "-c", code])


def selftest(py: Path, model: str, cpu: bool):
    print("\n[setup_flux] -- self-test render (tiny image) --")
    gen = THIS_DIR / "flux_generate.py"
    out = THIS_DIR / "_selftest.png"
    cmd = [py, str(gen), "--selftest", "--model", model,
           "--out", str(out), "--width", "512", "--height", "512", "--steps", "8"]
    if cpu:
        cmd.append("--offload")
    run(cmd)
    print("[setup_flux] self-test image written to " + str(out))


def main():
    ap = argparse.ArgumentParser(description="Install the 4-bit FLUX.2 generator.")
    ap.add_argument("--cuda", default="cu128",
                    help="CUDA wheel tag (cu128 default; cu121/cu118; 'cpu').")
    ap.add_argument("--cpu", action="store_true", help="Force CPU torch (slow).")
    ap.add_argument("--hf-token", default=None, help="HF token for the gated download.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id of the 4-bit model.")
    ap.add_argument("--skip-download", action="store_true", help="Install deps only.")
    ap.add_argument("--selftest", action="store_true", help="Render a tiny image to verify.")
    ap.add_argument("--recreate-venv", action="store_true", help="Delete and rebuild the venv.")
    args = ap.parse_args()

    print("=" * 60)
    print(" FLUX.2 (4-bit / NF4) generator setup")
    print("   venv : " + str(VENV_DIR))
    print("   model: " + args.model)
    print("=" * 60)

    py = ensure_venv(args.recreate_venv)
    pip_install(py, ["--upgrade", "pip", "wheel", "setuptools<82"])
    install_torch(py, args.cuda, args.cpu)
    print("\n[setup_flux] -- installing diffusers stack --")
    pip_install(py, ["--upgrade", *PIP_PACKAGES])

    # Optional: triton-windows provides Triton on Windows, which silences
    # the "triton not found; flop counting will not work" warning at
    # generation time. Purely cosmetic, and best-effort -- a failed install
    # (no matching wheel for this torch/python) must never block setup.
    if os.name == "nt" and not args.cpu:
        print("\n[setup_flux] -- installing triton-windows (optional) --")
        try:
            pip_install(py, ["triton-windows"])
        except Exception:
            print("[setup_flux][WARN] triton-windows install failed "
                  "(optional) -- skipping; the warning is harmless.")

    logged_in = hf_login(py, args.hf_token)

    if args.skip_download:
        print("[setup_flux] --skip-download: leaving weights for first generation.")
    elif not logged_in:
        print("[setup_flux] not authenticated -- skipping weight download for now.")
        sys.exit(2)
    else:
        download_weights(py, args.model)
        if args.selftest:
            selftest(py, args.model, args.cpu)

    print("\n[setup_flux] DONE. The editor will call:")
    print("             " + str(venv_python(VENV_DIR)) + " " + str(THIS_DIR / "flux_generate.py") + " ...")


if __name__ == "__main__":
    main()
