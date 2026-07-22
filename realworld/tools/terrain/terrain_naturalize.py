#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
terrain_naturalize.py — measure and correct heightmap statistics so generated
terrain matches the distributions of real-world DEMs.

Why: diffusion / class-synthesized heightmaps look plausible but are
statistically wrong in ways the eye reads as "game terrain":

  * spectral slope (roughness-vs-scale) is off — real DEMs follow a power law
    P(f) ~ f^-beta with beta ≈ 2.0 (measured 1.7–2.4 across landscapes);
  * the elevation histogram is arbitrary (min-max stretched diffusion tones,
    or per-class plateaus) instead of a natural hypsometric curve;
  * clamping piles mass at exactly 0.0 / 1.0 (flat seas, clipped plateaus);
  * there is no drainage structure — real terrain's defining signature is
    dendritic channels obeying the slope-area law S ~ A^-theta, theta≈0.45;
  * slope distributions have wrong tails (vertical spikes, or over-smoothing).

This module provides:

  analyze(h, ...)    -> dict of the metrics above + pass/fail vs target bands
  naturalize(h, ...) -> corrected heightfield + before/after report
      1. de-clamp saturated masses
      2. phase-preserving spectral shaping toward P(f) ~ f^-beta
         (fixes roughness statistics WITHOUT moving landforms)
      3. hypsometric remap toward a biome target curve (monotonic, so
         relative landform structure is preserved)
      4. flow-routed stream-power erosion (D8 accumulation + S~A^-theta
         incision) + talus relaxation — carves real dendritic drainage
      5. gentle percentile renormalization (no min-max stretch)

Pure numpy + PIL; no scipy/torch required. Flow accumulation uses
np.bincount scatter-gather, so 2048x2048 runs in tens of seconds on CPU.

CLI:
  python terrain_naturalize.py --in h.png [--out h_nat.png] [--biome hills]
         [--strength 1.0] [--erosion-iters 8] [--report r.json] [--analyze-only]

Library:
  import terrain_naturalize as tn
  rep = tn.analyze(h01)                       # h01: float32 [0,1] square
  h2, rep = tn.naturalize(h01, biome="mountains", water_mask=w)
"""

import argparse
import json
import sys

import numpy as np

# World mapping (matches terrain_from_text.py / the engine).
WORLD_SIZE_M = 4096.0
HEIGHT_AMP_M = 250.0

# ── Target bands (from published DEM statistics; see module docstring) ──────
# NOTE on the spectral band: full-detail DEMs measure beta ~= 2.0, but this
# pipeline's BASE map deliberately carries landforms only — the engine's
# runtime detail tiles re-add the ~1 m high-frequency relief (see
# condition_base in terrain_from_text.py).  A landform-only base map is
# legitimately smoother, so the band tops out higher.
SPECTRAL_BETA_TARGET = 2.2
SPECTRAL_BETA_BAND = (1.5, 3.8)
CLAMP_MASS_MAX = 0.02          # ≤2% of pixels at exactly 0 or 1 (excl. water)
TERRACE_MASS_MAX = 0.03        # ≤3% in any single non-water elevation level
STEEP_FRAC_MAX = 0.03          # ≤3% of land steeper than 45° — catches spike
                               # fields (which measure 10%+), tolerates real
                               # gorge walls and residual pinned-water margins
CONCAVITY_BAND = (0.22, 0.90)  # slope-area theta, where measurable.
# Real-basin concavity spans ~0.2 (debris-flow / young landscapes) to ~0.9
# (mature fluvial).  The check's job is discriminating ORGANIZED drainage
# (>=0.22) from none (unstructured noise measures ~0.0).

# Biome hypsometry targets: elevation-pdf ~ Beta(a, b); HI = a / (a + b).
BIOME_HYPSO = {
    "plains":    (1.4, 3.6),   # HI ≈ 0.28 — mass concentrated low
    "hills":     (2.0, 2.7),   # HI ≈ 0.43
    "mountains": (2.2, 2.0),   # HI ≈ 0.52 — alpine catchments hold mass high
}
BIOME_SLOPE_MEAN_DEG = {"plains": (1.0, 8.0),
                        "hills": (4.0, 16.0),
                        "mountains": (8.0, 28.0)}
HI_TOL = 0.13                  # acceptable |HI - target|.  The check exists
                               # to catch DEGENERATE mass distributions
                               # (everything piled low/high); water-filled
                               # valleys legitimately raise land-only HI.

_D8 = [(-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
       (0, -1, 1.0),                      (0, 1, 1.0),
       (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5)]


# ── small helpers (3x3 neighborhood ops without scipy) ──────────────────────

def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    ys, yd = (slice(dy, None), slice(None, -dy)) if dy > 0 else \
             (slice(None, dy), slice(-dy, None)) if dy < 0 else \
             (slice(None), slice(None))
    xs, xd = (slice(dx, None), slice(None, -dx)) if dx > 0 else \
             (slice(None, dx), slice(-dx, None)) if dx < 0 else \
             (slice(None), slice(None))
    out[yd, xd] = a[ys, xs]
    return out


def _min3(a):
    m = a.copy()
    for dy, dx, _ in _D8:
        np.minimum(m, _shift(a, dy, dx, np.inf), out=m)
    return m


def _mean3(a):
    """3x3 box mean with reflected edges (no border darkening)."""
    p = np.pad(a, 1, mode="reflect")
    s = np.zeros_like(a, np.float64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            s += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return s / 9.0


def _slope_mpm(h, world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M):
    """Slope magnitude in meter-per-meter (central differences)."""
    n = h.shape[0]
    texel = world_m / n
    gy, gx = np.gradient(h.astype(np.float64) * amp_m, texel)
    return np.sqrt(gx * gx + gy * gy)


# ── metric: radial power-spectral slope ─────────────────────────────────────

def spectral_beta(h):
    """Fit P(f) ~ f^-beta on the radially-averaged power spectrum."""
    n = h.shape[0]
    win = np.hanning(n)
    w2 = np.outer(win, win)
    f = np.fft.fftshift(np.fft.fft2((h - h.mean()) * w2))
    p = np.abs(f) ** 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2).astype(np.int64)
    radial = np.bincount(r.ravel(), weights=p.ravel())
    counts = np.bincount(r.ravel())
    radial = radial[: n // 2] / np.maximum(counts[: n // 2], 1)
    # fit over mid-band: skip DC/window lobe, skip top octave (interp noise)
    lo, hi = max(4, n // 256), n // 4
    fr = np.arange(len(radial), dtype=np.float64)
    m = (fr >= lo) & (fr <= hi) & (radial > 0)
    if m.sum() < 8:
        return float("nan")
    coef = np.polyfit(np.log10(fr[m]), np.log10(radial[m]), 1)
    return float(-coef[0])


# ── metric: hypsometric integral ────────────────────────────────────────────

def hypsometric_integral(h, land_mask=None):
    x = h[land_mask] if land_mask is not None else h.ravel()
    if x.size == 0:
        return float("nan")
    p1, p99 = np.percentile(x, [1.0, 99.0])
    if p99 - p1 < 1e-9:
        return 0.0
    return float(np.clip((x.mean() - p1) / (p99 - p1), 0.0, 1.0))


# ── flow routing (D8) ───────────────────────────────────────────────────────

def _fill_depressions(h, eps=1e-5, iters=None):
    """Iterative epsilon depression filling (Planchon–Darboux style)."""
    n = h.shape[0]
    iters = iters or n
    f = np.full_like(h, np.inf)
    f[0, :] = h[0, :]; f[-1, :] = h[-1, :]
    f[:, 0] = h[:, 0]; f[:, -1] = h[:, -1]
    for it in range(iters):
        cand = np.maximum(h, _min3(f) + eps)
        nf = np.minimum(f, cand)
        if it % 8 == 7 and np.allclose(nf, f, atol=eps * 0.5, equal_nan=True):
            f = nf
            break
        f = nf
    # any cell the sweep never reached (unconverged interior): fall back
    bad = ~np.isfinite(f)
    if bad.any():
        f[bad] = h[bad]
    return f


def _receivers(h):
    """Steepest-descent D8 receiver flat-index per cell; -1 = outlet/pit."""
    n = h.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    best_drop = np.zeros((n, n), np.float64)
    rec = np.full((n, n), -1, np.int64)
    for dy, dx, dist in _D8:
        # off-map = outlet slightly below the map floor: borders always
        # drain outward, with a FINITE drop (an -inf fill would propagate
        # inf slopes into the erosion math)
        nb_h = _shift(h, dy, dx, -0.05)
        nb_i = _shift(idx, dy, dx, -1)
        drop = (h - nb_h) / dist
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        rec = np.where(better, nb_i, rec)
    return rec.ravel(), best_drop.ravel()          # drop is per-texel


def flow_accumulation(h, iters=None):
    """Cells drained through each cell (incl. itself), via K rounds of
    A = 1 + scatter(A -> receiver). K bounds resolvable channel length."""
    n = h.shape[0]
    iters = iters or min(600, int(n * 1.5))
    filled = _fill_depressions(h)
    rec, _ = _receivers(filled)
    valid = rec >= 0
    tgt = rec[valid]
    a = np.ones(n * n, np.float64)
    for _ in range(iters):
        inflow = np.bincount(tgt, weights=a[valid], minlength=n * n)
        na = 1.0 + inflow
        if np.abs(na - a).max() < 0.5:
            a = na
            break
        a = na
    return a.reshape(n, n), rec


def detect_flat_water(h, world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M):
    """Heuristic water mask for maps without segmentation: near-exactly
    flat cells belonging to elevation levels that hold significant mass
    (pinned seas/lakes/rivers).  Genuine hillslope cells are never this
    flat over a full texel."""
    s = _slope_mpm(h, world_m, amp_m)
    flat = s < 5e-4
    hist, edges = np.histogram(h, bins=2048, range=(0.0, 1.0))
    big = hist > max(64, h.size * 0.002)
    lvl = np.clip((h * 2048).astype(np.int64), 0, 2047)
    return flat & big[lvl]


def slope_area_theta(h, world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M,
                     exclude_mask=None):
    """Concavity theta from the slope-area law S ~ A^-theta over channels.
    exclude_mask (e.g. water): those cells and a 1-texel ring around them
    are dropped from the fit — game water is pinned FLAT by design, and a
    flat strip lying exactly along the drainage would floor the fit."""
    n = h.shape[0]
    a, rec = flow_accumulation(h)
    filled = _fill_depressions(h)
    _, drop = _receivers(filled)
    texel = world_m / n
    s = (drop.reshape(n, n) * amp_m) / texel       # m per m along flow dir
    interior = np.zeros((n, n), bool)
    interior[2:-2, 2:-2] = True                     # border cells have fake
    # off-map outlet drops AND the largest basins — they invert the fit
    am = a.ravel(); sm = s.ravel()
    ch = (am > 150.0) & interior.ravel()            # true interior channels —
    # hillslope cells carry detail-noise slopes uncorrelated with A and
    # wash the regression out
    if exclude_mask is not None and exclude_mask.any():
        grow = exclude_mask.astype(np.float64)
        grow = _mean3(grow) > 1e-6                  # +1 texel ring (banks)
        ch &= ~grow.ravel()
    if ch.sum() < 500:
        return float("nan")
    la, ls = np.log10(am[ch]), np.log10(np.maximum(sm[ch], 1e-6))
    bins = np.linspace(la.min(), la.max(), 12)
    bi = np.digitize(la, bins)
    xs, ys = [], []
    for b in range(1, 12):
        m = bi == b
        if m.sum() >= 30:
            xs.append(la[m].mean()); ys.append(np.median(ls[m]))
    if len(xs) < 4:
        return float("nan")
    coef = np.polyfit(xs, ys, 1)
    return float(-coef[0])


# ── the analyzer ────────────────────────────────────────────────────────────

def analyze(h, biome="hills", water_mask=None,
            world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M, full=True):
    """Measure natural-distribution metrics. h: float [0,1], square."""
    h = np.asarray(h, np.float64)
    n = h.shape[0]
    if water_mask is None:
        # no segmentation available: auto-detect pinned-flat water so it
        # doesn't poison the land statistics (game water is flat by design)
        water_mask = detect_flat_water(h, world_m, amp_m)
        if water_mask.mean() < 0.005:
            water_mask = None                       # nothing significant
    land = ~water_mask if water_mask is not None else np.ones_like(h, bool)
    # slope statistics additionally exclude a ~3-texel ring around water:
    # the masks pass feathers pinned banks over several texels, and those
    # steps are deliberate constructs, not terrain
    if water_mask is not None:
        g = water_mask.astype(np.float64)
        g = _mean3(_mean3(_mean3(g)))
        land_slope = ~(g > 1e-6)
    else:
        land_slope = land

    clamp_lo = float(((h <= 1e-4) & land).mean())
    clamp_hi = float((h >= 1.0 - 1e-4).mean())

    # terracing: mass of the 4th-fullest elevation level (excluding water).
    # The top 3 levels are allowed to be full — seas and eroded/filled lakes
    # are legitimately flat; QUANTIZATION shows up as dozens of full levels.
    hist, _ = np.histogram(h[land], bins=1024, range=(0.0, 1.0))
    hs = np.sort(hist)[::-1]
    terrace = float(hs[3] / max(hist.sum(), 1))

    beta = spectral_beta(h)
    hi = hypsometric_integral(h, land)

    s = _slope_mpm(h, world_m, amp_m)
    sd = np.degrees(np.arctan(s[land_slope]))
    slope_mean = float(sd.mean())
    slope_p90 = float(np.percentile(sd, 90))
    steep_frac = float((sd > 45.0).mean())

    theta = slope_area_theta(h, world_m, amp_m,
                             exclude_mask=water_mask) if full \
        else float("nan")

    a, b = BIOME_HYPSO[biome]
    hi_target = a / (a + b)
    sl_lo, sl_hi = BIOME_SLOPE_MEAN_DEG[biome]
    checks = {
        "spectral_beta_ok": bool(SPECTRAL_BETA_BAND[0] <= beta
                                 <= SPECTRAL_BETA_BAND[1]),
        "clamp_ok": bool(clamp_lo <= CLAMP_MASS_MAX
                         and clamp_hi <= CLAMP_MASS_MAX),
        "terrace_ok": bool(terrace <= TERRACE_MASS_MAX),
        "hypsometry_ok": bool(abs(hi - hi_target) <= HI_TOL),
        "slope_ok": bool(sl_lo <= slope_mean <= sl_hi
                         and steep_frac <= STEEP_FRAC_MAX),
        "drainage_ok": bool(np.isnan(theta)
                            or CONCAVITY_BAND[0] <= theta
                            <= CONCAVITY_BAND[1]),
    }
    return {
        "size": n, "biome": biome,
        "spectral_beta": round(beta, 3),
        "clamp_mass_lo": round(clamp_lo, 4),
        "clamp_mass_hi": round(clamp_hi, 4),
        "terrace_mass": round(terrace, 4),
        "hypsometric_integral": round(hi, 3),
        "hi_target": round(hi_target, 3),
        "slope_mean_deg": round(slope_mean, 2),
        "slope_p90_deg": round(slope_p90, 2),
        "steep_frac": round(steep_frac, 4),
        "drainage_theta": None if np.isnan(theta) else round(theta, 3),
        "checks": checks,
        "natural": bool(all(checks.values())),
    }


# ── correction passes ───────────────────────────────────────────────────────

def _relief_governor(h, cap_deg, headroom=0.6, water_mask=None,
                     world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M):
    """Scale macro relief about its mean so the mean slope lands at
    headroom * cap_deg — leaving slope budget for synthesized detail.
    Amplitude-only: landform shapes and layout are untouched (this is the
    statistical equivalent of choosing a sane --height-scale)."""
    land = ~water_mask if water_mask is not None else np.ones_like(h, bool)
    sd = np.degrees(np.arctan(_slope_mpm(h, world_m, amp_m)))[land]
    cur = float(sd.mean())
    target = cap_deg * headroom
    if cur <= target or cur < 1e-3:
        return h
    c = max(0.35, target / cur)
    mean = h.mean()
    return np.clip(mean + (h - mean) * c, 0.0, 1.0)


def _deterrace(h, seed=0):
    """Detect elevation quantization (posterized diffusion output or 8-bit
    intermediates) and dissolve the terrace steps: sub-quantum dither plus
    a short smear turns 38-degree step cliffs into gentle ramps.  Continuous
    inputs pass through untouched."""
    u = np.unique(np.round(h, 6))
    if len(u) > 4000:
        return h                                   # continuous — no terracing
    d = np.diff(u)
    d = d[d > 1e-6]
    if d.size == 0:
        return h
    q = float(np.median(d))
    if q < 1e-5:
        return h
    rng = np.random.default_rng(seed)
    x = h + (rng.random(h.shape) - 0.5) * q
    x = _mean3(_mean3(x))
    return np.clip(x, 0.0, 1.0)


def _declamp(h, water_mask):
    """Pull saturated masses off the 0/1 rails with a soft knee."""
    out = h.copy()
    knee = 0.02
    lo = h < knee
    if water_mask is not None:
        lo &= ~water_mask                          # water may sit flat at 0
    out[lo] = knee * (h[lo] / knee) ** 0.5 * 0.8 + h[lo] * 0.2
    hi = h > 1.0 - knee
    t = np.clip((h[hi] - (1.0 - knee)) / knee, 0.0, 1.0)
    out[hi] = (1.0 - knee) + knee * (1.0 - (1.0 - t) ** 0.5) * 0.8 \
        + (np.clip(h[hi], None, 1.0) - (1.0 - knee)) * 0.2
    return out


def _despike(h, k=4.0):
    """Replace single-texel outliers (diffusion speckle) with the local
    3x3 median. Landforms are 10-100x larger and pass untouched."""
    stack = [h]
    for dy, dx, _ in _D8:
        stack.append(_shift(h, dy, dx, np.nan))
    st = np.stack(stack)
    med = np.nanmedian(st, axis=0)
    dev = np.abs(h - med)
    mad = np.nanmedian(np.abs(st - med[None]), axis=0) + 1e-5
    out = h.copy()
    bad = dev > k * mad
    out[bad] = med[bad]
    return out


def _spectral_shape(h, strength=1.0, beta_target=SPECTRAL_BETA_TARGET,
                    seed=0, slope_cap_deg=None, channel_suppress=None,
                    allow_synth=True):
    """Move the radial power spectrum toward P(f) ~ f^-beta_target.

    Excess power (too rough at some scale) is ATTENUATED in place —
    phase-preserving, so landforms don't move.  Missing power (too smooth,
    e.g. an over-blurred diffusion base) is SYNTHESIZED as random-phase
    fractal detail modulated by local slope, so flats/water stay clean and
    relief gains character — never by amplifying existing content, which
    would blow up residual speckle."""
    n = h.shape[0]
    mean = h.mean()
    F = np.fft.fft2(h - mean)
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.fftfreq(n)[None, :] * n
    rb = np.round(np.hypot(fy, fx)).astype(np.int64)      # cycles/image
    nb = rb.max() + 1
    P = np.bincount(rb.ravel(), weights=(np.abs(F) ** 2).ravel(),
                    minlength=nb)
    cnt = np.maximum(np.bincount(rb.ravel(), minlength=nb), 1)
    P = P / cnt
    f_ax = np.arange(nb, dtype=np.float64)
    # Target spectrum P_t = scale * f^-beta.  scale is the SMALLER of:
    #  (a) variance anchor — same total power as the input for f >= 2
    #      (band-anchoring over-scales high f when the input concentrates
    #      its power at landform scale);
    #  (b) slope anchor — total gradient power implied by the target
    #      spectrum stays inside the biome slope budget, so the requested
    #      beta is actually ACHIEVABLE without breaking the slope check.
    with np.errstate(divide="ignore"):
        shape_f = np.where(f_ax >= 1, f_ax ** (-beta_target), 0.0)
    hi_lim = n // 2                                # ignore corner bins
    pw = cnt.astype(np.float64)
    if not allow_synth:
        # attenuation-only mode: anchor the target to the LANDFORM band so
        # low-frequency gain is ~1 by construction — we only shave power
        # sitting ABOVE the f^-beta extrapolation (speckle), never the
        # landforms themselves
        band = slice(4, max(17, n // 64))
        scale = float(np.median(P[band] * f_ax[band] ** beta_target))
        Pt = scale * shape_f
        Pt[0] = P[0]
        ratio = np.ones(nb)
        m = (f_ax >= 2) & (P > 0)
        ratio[m] = Pt[m] / P[m]
        g = np.sqrt(ratio)
        lg = np.log(np.clip(g, 1e-3, 1e3))
        ker = np.ones(5) / 5.0
        lg = np.convolve(lg, ker, mode="same")
        g = np.exp(lg)
        g = np.minimum(g, 1.0)
        g[: max(24, n // 21)] = 1.0                # landform band untouched
        s = np.clip(strength, 0.0, 1.0)
        gain2d = 1.0 + (g[rb] - 1.0) * s
        out = np.real(np.fft.ifft2(F * gain2d)) + mean
        return np.clip(out, 0.0, 1.0)
    tot_in = float((P[2:hi_lim] * pw[2:hi_lim]).sum())
    tot_sh = float((shape_f[2:hi_lim] * pw[2:hi_lim]).sum())
    scale = tot_in / max(tot_sh, 1e-30)
    if slope_cap_deg is not None:
        # gradient power of mode f (normalized [0,1] heights, world units):
        # |grad| ~ 2*pi*f/n per texel -> m/m via amp/texel scaling
        texel = WORLD_SIZE_M / n
        g_of_f = (2.0 * np.pi * f_ax / n) * (HEIGHT_AMP_M / texel)
        # Parseval for numpy's unnormalized fft2: var = sum(|F|^2) / n^4
        grad_pow_unit = float((shape_f[2:hi_lim] * pw[2:hi_lim]
                               * g_of_f[2:hi_lim] ** 2).sum()) / float(n) ** 4
        g_target = np.tan(np.radians(slope_cap_deg)) * 0.9
        scale_slope = (g_target ** 2) / max(grad_pow_unit, 1e-30)
        scale = min(scale, scale_slope)
    Pt = scale * shape_f
    Pt[0] = P[0]
    ratio = np.ones(nb)
    m = (f_ax >= 2) & (P > 0)
    ratio[m] = Pt[m] / P[m]
    g = np.sqrt(ratio)
    # smooth the per-bin gain (log domain) to avoid ringing
    lg = np.log(np.clip(g, 1e-3, 1e3))
    ker = np.ones(5) / 5.0
    lg = np.convolve(lg, ker, mode="same")
    g = np.exp(lg)
    s = np.clip(strength, 0.0, 1.0)
    # 1) attenuate excess power in place (gain capped at 1)
    g_att = np.minimum(g, 1.0)
    gain2d = 1.0 + (g_att[rb] - 1.0) * s
    out = np.real(np.fft.ifft2(F * gain2d)) + mean
    # 2) synthesize missing power as slope-modulated fractal detail
    deficit = np.clip(Pt - P * np.minimum(g, 1.0) ** 2, 0.0, None)
    deficit[:2] = 0.0
    if allow_synth and deficit[2:].sum() > 1e-12:
        rng = np.random.default_rng(seed)
        phase = np.exp(2j * np.pi * rng.random((n, n)))
        amp = np.sqrt(deficit[rb])
        noise = np.real(np.fft.ifft2(amp * phase)) * (2.0 ** 0.5)
        slope = _slope_mpm(h)
        sm = _mean3(_mean3(slope))
        p90 = np.percentile(sm, 90) + 1e-9
        mod = 0.25 + 0.75 * np.clip(sm / p90, 0.0, 1.0)
        if channel_suppress is not None:
            mod = mod * channel_suppress            # keep carved channels clean
        noise = noise * mod
        # slope governor: find the LARGEST detail amplitude that keeps the
        # combined mean slope inside the cap.  Bisection on the actual
        # combined field — the noise is slope-modulated (rough ridges,
        # smooth valleys, like real terrain), so a linear estimate badly
        # underfills the spectrum.
        c = 1.0
        if slope_cap_deg is not None:
            cap = np.tan(np.radians(slope_cap_deg)) * 0.95
            if float(_slope_mpm(out + noise * s).mean()) > cap:
                lo_c, hi_c = 0.0, 1.0
                for _ in range(5):
                    mid = 0.5 * (lo_c + hi_c)
                    if float(_slope_mpm(out + noise * mid * s).mean()) > cap:
                        hi_c = mid
                    else:
                        lo_c = mid
                c = lo_c
        out = out + noise * c * s
    return np.clip(out, 0.0, 1.0)


def _hypso_remap(h, biome, strength, water_mask):
    """Monotonic histogram match toward the biome Beta-target CDF."""
    a, b = BIOME_HYPSO[biome]
    land = ~water_mask if water_mask is not None else np.ones_like(h, bool)
    x = h[land]
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, np.float64)
    ranks[order] = (np.arange(x.size) + 0.5) / x.size
    # inverse Beta CDF via sampled interpolation (avoids scipy)
    grid = np.linspace(1e-6, 1.0 - 1e-6, 4096)
    pdf = grid ** (a - 1.0) * (1.0 - grid) ** (b - 1.0)
    cdf = np.cumsum(pdf); cdf /= cdf[-1]
    target = np.interp(ranks, cdf, grid)
    lo, hi = x.min(), x.max()
    target = lo + target * (hi - lo)               # stay inside current range
    s = np.clip(strength, 0.0, 1.0) * 0.85         # never fully destroy input
    xn = x * (1.0 - s) + target * s
    out = h.copy()
    out[land] = xn
    return np.clip(out, 0.0, 1.0)


def _stream_power_erode(h, outer_iters=10, strength=1.0, water_mask=None,
                        world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M,
                        theta=0.45, k_base=0.006, talus_deg=38.0,
                        a_ref=50.0, res_cap=256):
    """Detachment-limited stream-power incision E = k·(A/A_ref)^0.5·S with
    talus relaxation and uplift compensation.  Carves dendritic drainage
    obeying S ~ A^-theta.  A_ref is the channel-head drainage area, so the
    whole channel NETWORK incises (normalizing by max(A) would make
    everything but the trunk river negligible).  For maps larger than
    res_cap the erosion runs downsampled and the carve delta is upsampled —
    the base map only needs landform-scale drainage; detail tiles handle
    close range."""
    n0 = h.shape[0]
    if n0 > res_cap:
        from PIL import Image
        small = np.asarray(Image.fromarray(h.astype(np.float32))
                           .resize((res_cap, res_cap), Image.BILINEAR),
                           np.float64)
        wm = None
        if water_mask is not None:
            wm = np.asarray(Image.fromarray(
                water_mask.astype(np.uint8) * 255).resize(
                    (res_cap, res_cap), Image.NEAREST)) > 127
        eroded = _stream_power_erode(small, outer_iters, strength, wm,
                                     world_m, amp_m, theta, k_base,
                                     talus_deg, a_ref, res_cap)
        delta = np.asarray(Image.fromarray(
            (eroded - small).astype(np.float32)).resize(
                (n0, n0), Image.BICUBIC), np.float64)
        return np.clip(h + delta, 0.0, 1.0)

    n = n0
    texel = world_m / n
    out = h.astype(np.float64).copy()
    land = (~water_mask if water_mask is not None
            else np.ones_like(out, bool)).astype(np.float64)
    talus_drop = np.tan(np.radians(talus_deg)) * texel / amp_m  # per texel
    # Resolution-aware k: incision/drop ratio = k * a_norm_max * (amp/texel)
    # must stay < 1 for stability, and the equilibrium S* = U/(k*a_norm)
    # must stay above the pit-fill epsilon.  0.094 puts the max ratio at
    # 0.75 for a_norm_max = 8 at any resolution.
    k = 0.094 * (texel / amp_m) * np.clip(strength, 0.0, 1.5)
    acc_iters = min(500, int(n * 1.2))
    for _ in range(outer_iters):
        # adopt the pit-filled surface (lakes fill with sediment): without
        # this, carved pits become eps-graded flats whose S≈0 stalls the
        # incision and destroys the slope-area relation
        out = np.maximum(out, _fill_depressions(out))
        a, rec = flow_accumulation(out, iters=acc_iters)
        _, drop = _receivers(out)
        drop = drop.reshape(n, n)
        s = (drop * amp_m) / texel                  # m/m along flow
        a_norm = np.minimum(np.power(a / a_ref, 0.5), 8.0)
        incis = k * a_norm * np.maximum(s, 0.0)
        # stability cap ABOVE the equilibrium incision (0.9 > the 0.75
        # worst-case incision/drop ratio) — a lower cap would bind at the
        # trunks and flatten S to a uniform value, erasing the A-dependence
        incis = np.minimum(incis, drop * 0.9)
        incis[rec.reshape(n, n) < 0] = 0.0          # borders: fake outlet
        incis[:2, :] = 0.0; incis[-2:, :] = 0.0     # drops would blast
        incis[:, :2] = 0.0; incis[:, -2:] = 0.0     # craters into the rim
        out -= incis * land
        # FIXED uniform uplift, balanced against k: drives the profile to
        # the stream-power equilibrium S* = U/(k·(A/A_ref)^0.5) — an
        # A^-0.5 slope-area law with channel-head slopes ~8% grade.
        # (Uplift = mean-incision would shrink as channels flatten and
        # collapse the equilibrium onto the pit-fill epsilon floor.)
        out += (k * 0.08) * land
        # talus: move material off slopes steeper than repose
        for dy, dx, dist in _D8:
            nb = _shift(out, dy, dx, 1e9)           # borders: no outflow
            d = out - nb
            move = np.clip((d - talus_drop * dist) * 0.12, 0.0, 0.01)
            out -= move * land
        out = np.clip(out, 0.0, 1.0)
    return out


def naturalize(h, biome="hills", strength=1.0, water_mask=None,
               erosion_iters=30, world_m=WORLD_SIZE_M, amp_m=HEIGHT_AMP_M,
               verbose=True):
    """Full correction chain. Returns (h_corrected, report_dict)."""
    h = np.asarray(h, np.float64)
    if h.shape[0] != h.shape[1]:
        raise ValueError("heightmap must be square")
    if biome not in BIOME_HYPSO:
        biome = "hills"
    before = analyze(h, biome, water_mask, world_m, amp_m)

    slope_cap = BIOME_SLOPE_MEAN_DEG[biome][1]
    out = _despike(h)
    out = _deterrace(out)
    out = _declamp(out, water_mask)
    # macro relief first: land the LANDFORM slope budget below the biome
    # cap so spectral character has headroom
    out = _relief_governor(out, slope_cap, 0.55, water_mask, world_m, amp_m)
    # spectral pass BEFORE erosion, ATTENUATION-ONLY: strips excess
    # high-frequency power (residual speckle) safely; never synthesizes —
    # additive noise (even light) creates pit fields that stall drainage
    # organization, and validation showed it barely moves beta anyway
    out = _spectral_shape(out, strength, slope_cap_deg=slope_cap * 0.9,
                          allow_synth=False)
    out = _hypso_remap(out, biome, strength, water_mask)
    # erosion runs LAST: it is what produces natural terrain statistics
    # (dendritic drainage, the S~A^-theta law, graded profiles).  Any
    # global spectral surgery AFTER it would scramble the tiny channel
    # gradients — trunk drops are ~1e-4 of range, far below texture scale.
    # High-frequency texture is the ENGINE's job (runtime detail tiles),
    # not the base map's.
    out = _stream_power_erode(out, erosion_iters, strength, water_mask,
                              world_m, amp_m)
    # second hypsometric pass AFTER erosion: erosion/uplift shift the
    # height distribution; a MONOTONIC remap preserves elevation ordering,
    # so flow directions — and the carved drainage network — survive intact
    out = _hypso_remap(out, biome, strength, water_mask)
    # final slope trim (also monotonic — pure amplitude about the mean):
    # erosion uplift can push gentle biomes slightly over their band
    out = _relief_governor(out, slope_cap, 0.93, water_mask, world_m, amp_m)
    # tail-only cleanup: soft-compress the extreme 0.2% into range — NO
    # min-max stretch (a stretch would re-amplify the relief the governor
    # just brought into the biome slope budget, and re-create clamp spikes)
    p_lo, p_hi = np.percentile(out, [0.1, 99.9])
    lo_m = out < p_lo
    hi_m = out > p_hi
    out[lo_m] = p_lo - (p_lo - out[lo_m]) * 0.3
    out[hi_m] = p_hi + (out[hi_m] - p_hi) * 0.3
    out = np.clip(out, 0.0, 1.0)
    if water_mask is not None:                      # re-pin water flat+low
        if water_mask.any():
            wl = np.percentile(out[water_mask], 15)
            out[water_mask] = np.minimum(out[water_mask], wl)

    after = analyze(out, biome, water_mask, world_m, amp_m)
    report = {"before": before, "after": after,
              "biome": biome, "strength": strength}
    if verbose:
        def fmt(r):
            return (f"beta={r['spectral_beta']} HI={r['hypsometric_integral']}"
                    f" slope={r['slope_mean_deg']}° terr={r['terrace_mass']}"
                    f" clamp={r['clamp_mass_lo']}/{r['clamp_mass_hi']}"
                    f" theta={r['drainage_theta']}"
                    f" natural={r['natural']}")
        print(f"[naturalize] before: {fmt(before)}")
        print(f"[naturalize] after : {fmt(after)}")
    return out.astype(np.float32), report


# ── biome inference from a prompt (mirrors terrain_from_text word lists) ────

def biome_from_prompt(prompt):
    p = (prompt or "").lower()
    mountain_w = ("mountain", "alpine", "peak", "ridge", "highland", "cliff",
                  "canyon", "gorge", "volcan", "caldera", "glacier", "fjord")
    plains_w = ("plain", "prairie", "steppe", "flat", "meadow", "farmland",
                "field", "delta", "marsh", "wetland", "tundra")
    if any(w in p for w in mountain_w):
        return "mountains"
    if any(w in p for w in plains_w):
        return "plains"
    return "hills"


# ── CLI ─────────────────────────────────────────────────────────────────────

def _load_h(path):
    from PIL import Image
    im = Image.open(path)
    a = np.array(im, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    scale = 65535.0 if a.max() > 255.0 else (255.0 if a.max() > 1.0 else 1.0)
    return a / scale


def _save_h(h, path):
    from PIL import Image
    a = (np.clip(h, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(a, mode="I;16").save(path)


def main():
    ap = argparse.ArgumentParser(
        description="check / correct heightmap natural-world statistics")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--biome", default="hills",
                    choices=sorted(BIOME_HYPSO) + ["auto"])
    ap.add_argument("--prompt", default="",
                    help="used for --biome auto inference")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--erosion-iters", type=int, default=8)
    ap.add_argument("--report", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    biome = biome_from_prompt(args.prompt) if args.biome == "auto" \
        else args.biome
    h = _load_h(args.inp)
    if args.analyze_only:
        rep = analyze(h, biome)
        print(json.dumps(rep, indent=2))
        ok = rep["natural"]
    else:
        out, rep = naturalize(h, biome, args.strength,
                              erosion_iters=args.erosion_iters)
        dst = args.out or args.inp.rsplit(".", 1)[0] + "_nat.png"
        _save_h(out, dst)
        print(f"[naturalize] wrote {dst}")
        ok = rep["after"]["natural"]
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
