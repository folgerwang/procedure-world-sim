#!/usr/bin/env python3
"""
setup_audiogen.py - one-time setup for the text-to-audio generator.

Installs the Python dependencies into the CURRENT interpreter (the same one
Setup.bat configured with CUDA torch for the FLUX image generator) and
prefetches the Stable Audio Open 1.0 weights so the first in-editor
generation doesn't stall on a download.

The weights are gated: accept the licence at
    https://huggingface.co/stabilityai/stable-audio-open-1.0
and log in first (huggingface-cli login, or set HF_TOKEN).
"""
import subprocess
import sys

DEPS = [
    "diffusers>=0.30.0",
    "transformers",
    "accelerate",
    "soundfile",
    "torchsde",         # CosineDPMSolverMultistepScheduler (Stable Audio)
    "optimum-quanto",   # fp8 weight quantization for the DiT (--quant fp8)
]

MODEL = "stabilityai/stable-audio-open-1.0"


def main() -> int:
    print(f"[audiogen-setup] interpreter: {sys.executable}")

    print("[audiogen-setup] installing deps:", ", ".join(DEPS))
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-U"] + DEPS)
    if rc != 0:
        print("[audiogen-setup] pip install FAILED")
        return rc

    # torch should already be present (CUDA build from the main setup); only
    # install a fallback if it's completely missing.
    try:
        import torch  # noqa: F401
        print(f"[audiogen-setup] torch present: {torch.__version__}, "
              f"cuda={torch.cuda.is_available()}")
    except ImportError:
        print("[audiogen-setup] torch missing - installing CUDA build "
              "(cu121)...")
        rc = subprocess.call([
            sys.executable, "-m", "pip", "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cu121"])
        if rc != 0:
            return rc

    print(f"[audiogen-setup] prefetching {MODEL} (gated - needs HF login)...")
    if not _prefetch_weights():
        # Non-zero exit so Setup.bat reports the failure instead of "ready".
        print("[audiogen-setup] FAILED - weights not cached.")
        return 1

    print("[audiogen-setup] done.")
    return 0


def _prefetch_weights() -> bool:
    """Download the gated weights, interactively walking the user through the
    licence gate when access is denied.  Returns True once the snapshot is
    fully cached (snapshot_download blocks until every file is on disk)."""
    from huggingface_hub import snapshot_download

    url = f"https://huggingface.co/{MODEL}"
    for attempt in range(3):
        try:
            snapshot_download(MODEL)
            print("[audiogen-setup] weights cached.")
            return True
        except Exception as e:
            msg = str(e)
            gated = "gated" in msg.lower() or "403" in msg
            print(f"[audiogen-setup] prefetch failed: {e}")
            if not gated or attempt == 2:
                break
            # Licence gate: open the model page, wait for the user to accept,
            # then retry.  Access is granted immediately after the form.
            print()
            print("  This model is GATED. In the browser page that just")
            print("  opened, log into the SAME Hugging Face account as your")
            print("  token and click 'Agree and access repository':")
            print(f"     {url}")
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass
            try:
                input("  Press Enter AFTER accepting to retry the download "
                      "(Ctrl+C to abort)... ")
            except EOFError:
                # Non-interactive invocation — can't wait for the user.
                break

    print("  Fix and re-run: python realworld\\tools\\audiogen\\"
          "setup_audiogen.py")
    print("  (generation will also retry the download on first use)")
    return False


if __name__ == "__main__":
    sys.exit(main())
