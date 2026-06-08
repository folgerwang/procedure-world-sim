#!/usr/bin/env python3
"""
download_voices.py - bulk-download sherpa-onnx TTS voice models.

Enumerates the assets of the sherpa-onnx "tts-models" GitHub release (so the
list never goes stale) and downloads + extracts the matching voice archives
into realworld/assets/ml_models/tts/.  Each voice becomes a folder the engine's
resolveModelDir() discovers automatically; switch between them at runtime with
the RW_TTS_VOICE env var.

Examples:
    # every voice in the release (~130+, several GB)
    python download_voices.py --all

    # just the English Piper voices
    python download_voices.py --filter vits-piper-en

    # a couple of specific ones
    python download_voices.py --filter amy --filter ryan

    # see what's available without downloading
    python download_voices.py --list
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request

REPO = "k2-fsa/sherpa-onnx"
TAG = "tts-models"
API = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"


def _http_json(url: str):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "rw-tts-downloader",
    })
    # A token lifts the 60/hr unauthenticated API limit; optional.
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_assets():
    """All downloadable voice archives (name, url, size) for the release.

    The GitHub assets endpoint paginates at 100; the tts-models release has
    more than that, so walk the pages until one comes back short.
    """
    rel = _http_json(API)
    assets_url = rel["assets_url"]
    out = []
    page = 1
    while True:
        batch = _http_json(f"{assets_url}?per_page=100&page={page}")
        if not batch:
            break
        for a in batch:
            name = a["name"]
            if name.endswith(".tar.bz2") or name.endswith(".tar.gz"):
                out.append((name, a["browser_download_url"], a["size"]))
        if len(batch) < 100:
            break
        page += 1
    return out


def _stem(name: str) -> str:
    for suf in (".tar.bz2", ".tar.gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


# A voice asset stem looks like "vits-piper-en_US-amy-low-int8": the speaker
# is followed by an optional QUALITY suffix (x_low/low/medium/high) and an
# optional PRECISION suffix (fp16/fp32/int8).  Strip those trailing tokens to
# get the speaker "base", then keep the single SMALLEST file per base.
_QUALITY = {"x_low", "x-low", "low", "medium", "high"}
_PRECISION = {"fp16", "fp32", "int8"}


def _voice_base(stem: str) -> str:
    """The voice/speaker identity with quality + precision suffixes removed,
    so all size/quality variants of one voice collapse to a single key."""
    parts = stem.split("-")
    while len(parts) > 1 and parts[-1].lower() in (_QUALITY | _PRECISION):
        parts.pop()
    return "-".join(parts)


def keep_smallest_per_voice(sel):
    """From (name, url, size) tuples, keep only the SMALLEST file for each
    distinct voice (lowest quality + most-quantized precision)."""
    best = {}   # base -> (size, tuple)
    for item in sel:
        base = _voice_base(_stem(item[0]))
        size = item[2]
        if base not in best or size < best[base][0]:
            best[base] = (size, item)
    return [v[1] for v in best.values()]


def _download_once(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": "rw-tts-downloader"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stdout.write(f"\r    {pct:3d}%  ({done >> 20} / "
                                 f"{total >> 20} MB)")
                sys.stdout.flush()
    sys.stdout.write("\n")


def download(url: str, dest: str, retries: int = 4):
    """Download with retry + backoff — GitHub's CDN returns intermittent
    504s under load, which would otherwise drop a voice permanently."""
    import time
    last = None
    for attempt in range(1, retries + 1):
        try:
            _download_once(url, dest)
            return
        except Exception as e:
            last = e
            try:
                os.remove(dest)        # discard the partial file
            except OSError:
                pass
            if attempt < retries:
                wait = 2 ** attempt    # 2, 4, 8 s
                sys.stdout.write(f"\r    retry {attempt}/{retries - 1} "
                                 f"after {wait}s ({e})\n")
                sys.stdout.flush()
                time.sleep(wait)
    raise last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "ml_models", "tts"),
        help="destination directory (default: assets/ml_models/tts)")
    ap.add_argument("--filter", action="append", default=[],
                    help="substring(s) an asset name must contain "
                         "(repeatable; ANY match wins). Omit with --all.")
    ap.add_argument("--all", action="store_true",
                    help="download EVERY voice (~130+, several GB)")
    ap.add_argument("--all-qualities", action="store_true",
                    help="keep every variant of a voice.  By default only the "
                         "SMALLEST model per voice is kept (lowest quality + "
                         "most-quantized precision) to save space.")
    ap.add_argument("--list", action="store_true",
                    help="print matching voices + sizes, download nothing")
    args = ap.parse_args()

    if not args.all and not args.filter and not args.list:
        ap.error("specify --all, one or more --filter, or --list")

    try:
        assets = list_assets()
    except Exception as e:
        print(f"[tts-dl] failed to query the release: {e}")
        print("  (GitHub API rate-limited?  Set GITHUB_TOKEN and retry.)")
        return 1

    def matches(name: str) -> bool:
        if args.all:
            return True
        n = name.lower()
        return any(f.lower() in n for f in args.filter)

    sel = [(n, u, s) for (n, u, s) in assets if matches(n)]

    if not args.all_qualities:
        before = len(sel)
        sel = keep_smallest_per_voice(sel)
        dropped = before - len(sel)
        if dropped:
            print(f"[tts-dl] keeping smallest model per voice "
                  f"(dropped {dropped} larger variant(s); "
                  f"--all-qualities to keep them)")

    sel.sort()

    if not sel:
        print("[tts-dl] no voices matched.  Try --list to see all names.")
        return 1

    total_mb = sum(s for _, _, s in sel) >> 20
    print(f"[tts-dl] {len(sel)} voice(s), ~{total_mb} MB total:")
    for n, _, s in sel:
        print(f"    {_stem(n):50s} {s >> 20:5d} MB")
    if args.list:
        return 0

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[tts-dl] destination: {out_dir}")

    ok = 0
    failed = []
    for name, url, _ in sel:
        stem = _stem(name)
        if os.path.isdir(os.path.join(out_dir, stem)):
            print(f"  [skip] {stem} (already present)")
            ok += 1
            continue
        print(f"  [get ] {stem}")
        tmp = os.path.join(out_dir, name)
        try:
            download(url, tmp)
            mode = "r:gz" if name.endswith(".gz") else "r:bz2"
            with tarfile.open(tmp, mode) as tf:
                # filter="data" silences the Python 3.14 deprecation and is
                # the safe extraction policy (no absolute paths / symlinks).
                try:
                    tf.extractall(out_dir, filter="data")
                except TypeError:        # older Python without the filter arg
                    tf.extractall(out_dir)
            os.remove(tmp)
            ok += 1
        except Exception as e:
            print(f"  [fail] {stem}: {e}")
            failed.append(stem)
            try:
                os.remove(tmp)
            except OSError:
                pass

    print(f"[tts-dl] done: {ok}/{len(sel)} voice(s) installed in {out_dir}")
    if failed:
        print(f"[tts-dl] {len(failed)} still failed after retries — re-run the "
              f"same command to resume (already-downloaded voices are "
              f"skipped):")
        for f in failed:
            print(f"    {f}")
    print("  Pick one at runtime with the RW_TTS_VOICE env var "
          "(e.g. set RW_TTS_VOICE=ryan).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
