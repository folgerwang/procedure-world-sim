#!/usr/bin/env python3
"""
setup_flux.py -- install deps and download weights for the FLUX.2-klein-4B
image generator used by the editor's Content Browser.

Installs into the CURRENT Python interpreter (the same `python` the editor
launches as `python flux_generate.py`, and the same one Setup.bat set up with
CUDA torch), so everything lives in one env -- no separate venv.

klein-4B uses a Qwen3 text encoder and its own Flux2KleinPipeline, which needs
diffusers from source (>=0.37.0.dev0) and a Qwen3-capable transformers; both are
pinned in requirements.txt.

Needs a Hugging Face login (FLUX.2-klein-4B is gated): accept the licence at
https://huggingface.co/black-forest-labs/FLUX.2-klein-4B then log in. If you are
not already authenticated this script PROMPTS for a token and continues.
"""
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = pathlib.Path(__file__).resolve().parent
MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def run(cmd) -> int:
    print("[flux-setup] $", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def install_deps(py: str) -> None:
    # NOT torch -- Setup.bat installs the CUDA torch wheel already; if you run
    # this standalone, install torch first from pytorch.org.
    rc = run([py, "-m", "pip", "install", "--disable-pip-version-check",
              "-r", str(HERE / "requirements.txt")])
    if rc != 0:
        print("[flux-setup] dependency install FAILED. If torch is missing, "
              "install the CUDA wheel from https://pytorch.org/get-started/locally/",
              file=sys.stderr)
        sys.exit(1)


def verify_import(py: str, module: str) -> None:
    """Fail fast (with a clear message) if a key package didn't install into
    the SAME interpreter the editor will use."""
    rc = subprocess.call([py, "-c", "import " + module])
    if rc != 0:
        print("[flux-setup] '" + module + "' did NOT install into " + py +
              "\n  This is the interpreter the editor launches as `python`."
              "\n  If it's a git-install failure, you do NOT need git now -- "
              "requirements.txt uses the GitHub zipball. Re-run this script.",
              file=sys.stderr)
        sys.exit(1)
    print("[flux-setup] verified: " + module + " importable", flush=True)


def _existing_token():
    for var in TOKEN_ENV_VARS:
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    return None


def ensure_authenticated() -> None:
    from huggingface_hub import login, whoami

    token = _existing_token()
    try:
        info = whoami(token=token) if token else whoami()
        print(f"[flux-setup] Hugging Face: authenticated as {info.get('name', '?')}",
              flush=True)
        if token:
            login(token=token, add_to_git_credential=False)
        return
    except Exception:
        pass

    print("\n[flux-setup] FLUX.2-klein-4B is a GATED model and you are not logged in.")
    print(f"  1) Accept the licence (once, in a browser):  https://huggingface.co/{MODEL}")
    print("  2) Create a READ token:  https://huggingface.co/settings/tokens")
    print("  3) Paste the token below and press Enter.")
    print("     Paste tips:  Command Prompt -> right-click  |  "
          "Windows Terminal/PowerShell -> Ctrl+V or right-click\n")

    for _ in range(3):
        try:
            token = input("[flux-setup] Hugging Face token (hf_...): ").strip()
        except EOFError:
            token = ""
        if not token:
            print("[flux-setup] no token entered.")
            continue
        if not token.startswith("hf_"):
            print("[flux-setup] that doesn't look like an HF token (should start "
                  "with 'hf_') -- check the paste and try again.")
            continue
        try:
            login(token=token, add_to_git_credential=False)
            info = whoami(token=token)
            print(f"[flux-setup] logged in as {info.get('name', '?')}", flush=True)
            os.environ["HF_TOKEN"] = token
            return
        except Exception as e:
            print(f"[flux-setup] that token did not work: {e}\n", file=sys.stderr)

    print("[flux-setup] could not authenticate after 3 attempts -- aborting.",
          file=sys.stderr)
    sys.exit(1)


def download_weights() -> None:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.errors import GatedRepoError
    except Exception:
        from huggingface_hub.utils import GatedRepoError  # type: ignore

    print(f"[flux-setup] downloading weights: {MODEL}", flush=True)
    try:
        snapshot_download(MODEL)
    except GatedRepoError:
        print("\n[flux-setup] ACCESS DENIED to the gated repo even though you are "
              "logged in.")
        print(f"  -> Open https://huggingface.co/{MODEL} and click "
              "'Agree and access repository', then re-run the setup.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[flux-setup] WEIGHT DOWNLOAD FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    py = sys.executable
    print(f"[flux-setup] using interpreter: {py}", flush=True)

    install_deps(py)
    verify_import(py, "diffusers")
    ensure_authenticated()
    download_weights()

    print("\n[flux-setup] done.  The editor's Content Browser can now generate "
          "images (right-click a folder -> Generate).")


if __name__ == "__main__":
    main()
