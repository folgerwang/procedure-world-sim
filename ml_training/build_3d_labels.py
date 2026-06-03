"""
build_3d_labels.py — lift multi-view 2D rigs to a 3D node graph by triangulation.

Two uses:
  1. Generate 3D training labels from your existing per-view captures.
  2. Standalone "2D rigs -> 3D rig, no depth buffer" conversion.

Each sample directory holds <base>_view<N>_meta.json files (one per view).
For the 3D lift each meta.json MUST carry the view's camera matrix:

    "view_proj": [16 floats, column-major]   # world->clip, as used by the renderer

(The C++ exporter change adds this.  Older captures that only have
azimuth/elevation can't be triangulated without the bounding-sphere centre +
radius, so re-export them once the view_proj field is in place.)

Joints are triangulated with a robust DLT (drop the worst-reprojecting view
and re-solve), identical in spirit to the C++ fusion path.  Output is a
rig_graph JSON.
"""

from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import rig_graph as rg


def _dlt(obs):
    """obs: list of (P(4x4 col-major np), x_ndc, y_ndc, w). Returns (3,) or None."""
    if len(obs) < 2:
        return None
    M = np.zeros((4, 4))
    for P, x, y, w in obs:
        # rows of the world->clip matrix (math row i = P[:, i] for column-major)
        r0, r1, r3 = P[:, 0], P[:, 1], P[:, 3]
        rx = x * r3 - r0
        ry = y * r3 - r1
        M += w * (np.outer(rx, rx) + np.outer(ry, ry))
    # smallest eigenvector of the 4x4 symmetric system
    evals, evecs = np.linalg.eigh(M)
    X = evecs[:, 0]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def _reproj(P, x, y, p3):
    # P is column-major (P = VP_math^T), so the math projection is P.T @ v.
    c = P.T @ np.array([p3[0], p3[1], p3[2], 1.0])
    if abs(c[3]) < 1e-20:
        return 1e9
    return float(np.hypot(c[0] / c[3] - x, c[1] / c[3] - y))


def triangulate_robust(obs, tol=0.02):
    obs = list(obs)
    if len(obs) < 2:
        return None, 0
    while True:
        p = _dlt(obs)
        if p is None:
            return None, 0
        if len(obs) <= 2:
            break
        errs = [_reproj(P, x, y, p) for (P, x, y, w) in obs]
        wi = int(np.argmax(errs))
        if errs[wi] > tol:
            obs.pop(wi)
        else:
            break
    return p, len(obs)


def _perspective(fov, aspect, n, f, yflip=True):
    """glm::perspective (RH, depth [-1,1]); yflip mirrors proj[1][1] *= -1."""
    t = 1.0 / np.tan(fov / 2)
    P = np.zeros((4, 4))
    P[0, 0] = t / aspect
    P[1, 1] = (-t if yflip else t)        # Vulkan y-flip used by the renderer
    P[2, 2] = (f + n) / (n - f)
    P[2, 3] = (2 * f * n) / (n - f)
    P[3, 2] = -1.0
    return P                               # math (row-major)


def _look_at(eye, c, up):
    f = c - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def reconstruct_view_proj(azimuth_deg, elevation_deg, radius=1.0, centre=(0, 0, 0)):
    """Rebuild a view's world->clip matrix, mirroring SimpleRasterizer::captureOrbit.

    Triangulation is invariant to a uniform scale+offset of all cameras, so the
    canonical defaults (radius=1, centre=origin) recover the skeleton in a
    normalized frame without needing the mesh.  Returns a COLUMN-MAJOR 4x4
    (same layout as glm value_ptr), matching what the DLT code expects.
    """
    centre = np.asarray(centre, float)
    az = np.radians(azimuth_deg); el = np.radians(elevation_deg)
    eye = centre + radius * np.array([np.cos(el) * np.cos(az),
                                      np.sin(el),
                                      np.cos(el) * np.sin(az)])
    view = _look_at(eye, centre, np.array([0.0, 1.0, 0.0]))
    proj = _perspective(np.radians(45.0), 1.0, radius * 0.01, radius * 4.0, yflip=True)
    vp_math = proj @ view                  # world->clip (row-major)
    return vp_math.T                        # store column-major (P[:,i] = math row i)


def load_views(sample_dir):
    """Return list of (view_proj 4x4 col-major, joint_uvs (19,2)) for one sample.

    Uses the stored view_proj when present; otherwise reconstructs canonical
    orbit cameras from azimuth/elevation (lets us reuse existing captures that
    predate the view_proj export field)."""
    metas = sorted(glob.glob(os.path.join(sample_dir, "*_meta.json")))
    views = []
    for mp in metas:
        m = json.load(open(mp))
        if "view_proj" in m:
            P = np.array(m["view_proj"], dtype=np.float64).reshape(4, 4)
        else:
            P = reconstruct_view_proj(m.get("azimuth_deg", 0.0),
                                      m.get("elevation_deg", 0.0))
        uvs = np.full((rg.NUM_JOINTS, 2), np.nan)
        for j, jd in enumerate(m["joints"][: rg.NUM_JOINTS]):
            if jd.get("edited", True):
                uvs[j] = jd["uv"]
        views.append((P, uvs))
    return views


def lift(sample_dir):
    views = load_views(sample_dir)
    if len(views) < 2:
        raise SystemExit(f"need >=2 views with view_proj in {sample_dir} "
                         f"(found {len(views)}); re-export with the view_proj field")
    pos = np.zeros((rg.NUM_JOINTS, 3))
    support = np.zeros(rg.NUM_JOINTS, dtype=int)
    for j in range(rg.NUM_JOINTS):
        obs = []
        for P, uvs in views:
            if np.any(np.isnan(uvs[j])):
                continue
            u, v = uvs[j]
            obs.append((P, u * 2 - 1, v * 2 - 1, 1.0))   # uv -> ndc
        p, n = triangulate_robust(obs)
        if p is not None:
            pos[j], support[j] = p, n
    return pos, support


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", nargs="?", help="dir with *_meta.json views")
    ap.add_argument("-o", "--out", default="rig3d.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.sample_dir:
        # Project a known skeleton through synthetic orbit cameras, add one
        # outlier view, then verify triangulation recovers it.
        def perspective(fov, aspect, n, f):
            t = 1 / np.tan(fov / 2)
            P = np.zeros((4, 4))
            P[0, 0] = t / aspect; P[1, 1] = t
            P[2, 2] = (f + n) / (n - f); P[2, 3] = (2 * f * n) / (n - f)
            P[3, 2] = -1
            return P  # row-major (math)
        def look_at(eye, c, up):
            f = (c - eye); f /= np.linalg.norm(f)
            s = np.cross(f, up); s /= np.linalg.norm(s)
            u = np.cross(s, f)
            M = np.eye(4)
            M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
            M[:3, 3] = -M[:3, :3] @ eye
            return M
        rng = np.random.default_rng(1)
        true = rng.normal(size=(rg.NUM_JOINTS, 3)) * [0.25, 0.8, 0.15] + [0, 1, 0]
        proj = perspective(np.radians(45), 1, 0.1, 50)
        cams = []
        for i in range(8):
            az = np.radians(45 * i)
            eye = np.array([3*np.sin(az), 1.0, 3*np.cos(az)])
            VP = proj @ look_at(eye, np.array([0, 1., 0]), np.array([0, 1., 0]))
            cams.append(VP.T.reshape(-1))   # store column-major like the engine
        maxerr = 0.0
        for j in range(rg.NUM_JOINTS):
            obs = []
            for k, vpflat in enumerate(cams):
                P = np.array(vpflat).reshape(4, 4)        # column-major
                c = P.T @ np.array([*true[j], 1.0])       # math: row-major @ vec
                x, y = c[0] / c[3], c[1] / c[3]
                if k == 3:                                # inject an outlier view
                    x += 0.7; y -= 0.5
                obs.append((P, x, y, 1.0))
            p, n = triangulate_robust(obs)
            maxerr = max(maxerr, np.linalg.norm(p - true[j]))
        print(f"selftest: max joint error={maxerr:.6e}  ->",
              "PASS" if maxerr < 1e-3 else "FAIL")
    else:
        pos, support = lift(args.sample_dir)
        rg.save(args.out, pos, norm=rg.normalize(pos)[1:])
        print(f"wrote {args.out}")
        for j in range(rg.NUM_JOINTS):
            print(f"  {j:2d} {rg.JOINT_NAMES[j]:16s} {pos[j].round(3)}  views={support[j]}")
