// ============================================================================
//  skinning_weights.cpp  —  see header for the high-level description.
//
//  Everything here is self-contained: glm + the C++ standard library only.
//  Numerics are done in double; results are written back as the float
//  VertexSkinData expected by the exporter.
// ============================================================================

#include "plugins/auto_rig/skinning_weights.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace plugins {
namespace auto_rig {
namespace {

using dvec3 = glm::dvec3;

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
//  Small geometry helper: distance from point p to segment [a,b].
// ---------------------------------------------------------------------------
double distPointSegment(const dvec3& p, const dvec3& a, const dvec3& b) {
    dvec3 ab = b - a;
    double len2 = glm::dot(ab, ab);
    double t = (len2 > 1e-18) ? glm::clamp(glm::dot(p - a, ab) / len2, 0.0, 1.0) : 0.0;
    return glm::length(p - (a + t * ab));
}

// ---------------------------------------------------------------------------
//  Per-joint "outgoing" bone segments.
//
//  A joint owns the geometry that extends from itself toward its children
//  (so the upper-arm mesh follows the shoulder/upper_arm joint, not the
//  elbow).  Leaf joints (head, hands, feet) get a virtual TIP segment that
//  extends past the joint along the incoming bone direction, so terminal
//  geometry like the skull is anchored to something.
// ---------------------------------------------------------------------------
struct JointBones {
    std::vector<std::pair<dvec3, dvec3>> segs;
};

std::vector<JointBones> buildJointBones(const Skeleton& sk, double scale) {
    const int nj = static_cast<int>(sk.joints.size());
    std::vector<std::vector<int>> children(nj);
    for (int j = 0; j < nj; ++j) {
        int p = sk.joints[j].parent;
        if (p >= 0 && p < nj) children[p].push_back(j);
    }
    std::vector<JointBones> jb(nj);
    for (int j = 0; j < nj; ++j) {
        dvec3 pj = dvec3(sk.joints[j].position);
        if (!children[j].empty()) {
            for (int c : children[j])
                jb[j].segs.push_back({pj, dvec3(sk.joints[c].position)});
        } else {
            // Leaf: synthesize a tip bone past the joint.
            int p = sk.joints[j].parent;
            dvec3 dir(0, 1, 0);
            double len = 0.15 * scale;
            if (p >= 0) {
                dvec3 d = pj - dvec3(sk.joints[p].position);
                double l = glm::length(d);
                if (l > 1e-9) { dir = d / l; len = 0.5 * l; }
            }
            jb[j].segs.push_back({pj, pj + dir * len});
        }
    }
    return jb;
}

// Distance from point p to joint j (min over its outgoing segments).
double distToJoint(const dvec3& p, const JointBones& jb) {
    double d = kInf;
    for (auto& s : jb.segs) d = std::min(d, distPointSegment(p, s.first, s.second));
    return d;
}

// Index (and distance) of the nearest joint to point p.
std::pair<int, double> nearestJoint(const dvec3& p, const std::vector<JointBones>& jbs) {
    int best = 0;
    double bd = kInf;
    for (int j = 0; j < (int)jbs.size(); ++j) {
        double d = distToJoint(p, jbs[j]);
        if (d < bd) { bd = d; best = j; }
    }
    return {best, bd};
}

// ---------------------------------------------------------------------------
//  Sparse mesh operators built once and shared by the solvers.
//
//   - graph      : undirected adjacency with Euclidean edge lengths (Dijkstra)
//   - Loff/Ldiag : clamped cotangent Laplacian  L = D - W,  W >= 0
//                  SpMV:  (L x)_i = Ldiag_i*x_i + sum_j Loff_ij.val * x_col
//   - mass/minv  : lumped (barycentric) vertex areas and their inverse
// ---------------------------------------------------------------------------
struct MeshOps {
    int nv = 0;
    std::vector<std::vector<std::pair<int, double>>> graph;
    std::vector<std::vector<std::pair<int, double>>> Loff;   // (col, -w_ij)
    std::vector<double> Ldiag;
    std::vector<double> mass, minv;
    double avg_edge = 0.0;
};

MeshOps buildMeshOps(const TriangleMesh& mesh) {
    MeshOps ops;
    const int nv = static_cast<int>(mesh.positions.size());
    ops.nv = nv;
    ops.graph.assign(nv, {});
    ops.mass.assign(nv, 0.0);

    std::vector<std::unordered_map<int, double>> wmap(nv);   // accumulated cotan w_ij
    std::vector<std::unordered_map<int, double>> gmap(nv);   // edge lengths (dedup)

    auto addEdge = [&](int i, int j) {
        if (i == j) return;
        if (gmap[i].find(j) == gmap[i].end()) {
            double len = glm::length(dvec3(mesh.positions[i]) - dvec3(mesh.positions[j]));
            gmap[i][j] = len;
            gmap[j][i] = len;
        }
    };
    auto cotAt = [](const dvec3& v, const dvec3& a, const dvec3& b) {
        dvec3 e0 = a - v, e1 = b - v;
        double cr = glm::length(glm::cross(e0, e1));
        if (cr < 1e-15) return 0.0;
        return glm::dot(e0, e1) / cr;
    };

    const auto& idx = mesh.indices;
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        int a = idx[t], b = idx[t + 1], c = idx[t + 2];
        if (a < 0 || b < 0 || c < 0 || a >= nv || b >= nv || c >= nv) continue;
        dvec3 pa(mesh.positions[a]), pb(mesh.positions[b]), pc(mesh.positions[c]);

        double area = 0.5 * glm::length(glm::cross(pb - pa, pc - pa));
        double third = area / 3.0;
        ops.mass[a] += third; ops.mass[b] += third; ops.mass[c] += third;

        // cotan(angle at X) weights the OPPOSITE edge.
        double cA = cotAt(pa, pb, pc);   // -> edge (b,c)
        double cB = cotAt(pb, pc, pa);   // -> edge (c,a)
        double cC = cotAt(pc, pa, pb);   // -> edge (a,b)
        wmap[b][c] += 0.5 * cA; wmap[c][b] += 0.5 * cA;
        wmap[c][a] += 0.5 * cB; wmap[a][c] += 0.5 * cB;
        wmap[a][b] += 0.5 * cC; wmap[b][a] += 0.5 * cC;

        addEdge(a, b); addEdge(b, c); addEdge(c, a);
    }

    ops.Loff.assign(nv, {});
    ops.Ldiag.assign(nv, 0.0);
    ops.minv.assign(nv, 0.0);
    double edge_sum = 0.0; size_t edge_cnt = 0;
    for (int i = 0; i < nv; ++i) {
        double diag = 0.0;
        for (auto& kv : wmap[i]) {
            double w = std::max(kv.second, 0.0);   // clamp -> M-matrix / stable
            if (w <= 0.0) continue;
            ops.Loff[i].push_back({kv.first, -w});
            diag += w;
        }
        ops.Ldiag[i] = diag;
        ops.mass[i] = std::max(ops.mass[i], 1e-12);
        ops.minv[i] = 1.0 / ops.mass[i];
        for (auto& kv : gmap[i]) { ops.graph[i].push_back(kv); edge_sum += kv.second; ++edge_cnt; }
    }
    ops.avg_edge = (edge_cnt ? edge_sum / double(edge_cnt) : 1.0);
    return ops;
}

inline void applyL(const MeshOps& ops, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < ops.nv; ++i) {
        double s = ops.Ldiag[i] * x[i];
        for (auto& e : ops.Loff[i]) s += e.second * x[e.first];
        y[i] = s;
    }
}

// ---------------------------------------------------------------------------
//  Conjugate gradient for SPD systems, Jacobi-preconditioned.
//  matvec(x,y): y = A x.   diag: A's diagonal (for the preconditioner).
//  Returns false if the result is non-finite.
// ---------------------------------------------------------------------------
template <class MatVec>
bool cg(MatVec&& matvec, const std::vector<double>& diag, const std::vector<double>& b,
        std::vector<double>& x, int max_iter, double tol) {
    const int n = (int)b.size();
    std::vector<double> r(n), z(n), p(n), Ap(n);
    matvec(x, Ap);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
    auto precond = [&](const std::vector<double>& in, std::vector<double>& out) {
        for (int i = 0; i < n; ++i) out[i] = (diag[i] > 1e-30) ? in[i] / diag[i] : in[i];
    };
    precond(r, z);
    p = z;
    double rz = 0.0; for (int i = 0; i < n; ++i) rz += r[i] * z[i];
    double bnorm = 0.0; for (int i = 0; i < n; ++i) bnorm += b[i] * b[i];
    bnorm = std::sqrt(bnorm) + 1e-30;

    for (int it = 0; it < max_iter; ++it) {
        matvec(p, Ap);
        double pAp = 0.0; for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < 1e-300) break;
        double alpha = rz / pAp;
        double rnorm2 = 0.0;
        for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; rnorm2 += r[i] * r[i]; }
        if (std::sqrt(rnorm2) / bnorm < tol) break;
        precond(r, z);
        double rz_new = 0.0; for (int i = 0; i < n; ++i) rz_new += r[i] * z[i];
        double beta = rz_new / (rz + 1e-300);
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }
    for (double v : x) if (!std::isfinite(v)) return false;
    return true;
}

// ---------------------------------------------------------------------------
//  Dense per-joint weight matrix -> pruned/normalised SkinWeights.
//  Picks the top `k` joints per vertex and renormalises to a partition of 1.
// ---------------------------------------------------------------------------
SkinWeights finalize(const std::vector<std::vector<double>>& W /*[nj][nv]*/,
                     int nv, int nj, int k,
                     const std::vector<int>& nearest /*fallback owner per vertex*/) {
    k = glm::clamp(k, 1, kMaxVertexInfluences);
    SkinWeights out;
    out.per_vertex.resize(nv);
    for (int v = 0; v < nv; ++v) {
        std::vector<std::pair<double, int>> col;
        col.reserve(nj);
        for (int j = 0; j < nj; ++j) {
            double w = W[j][v];
            if (std::isfinite(w) && w > 1e-7) col.push_back({w, j});
        }
        std::sort(col.begin(), col.end(), [](auto& a, auto& b) { return a.first > b.first; });
        auto& vsd = out.per_vertex[v];
        double sum = 0.0;
        int n = std::min((int)col.size(), k);
        for (int i = 0; i < n; ++i) { vsd.joint_indices[i] = col[i].second; vsd.weights[i] = (float)col[i].first; sum += col[i].first; }
        if (sum > 1e-12) {
            for (int i = 0; i < kMaxVertexInfluences; ++i) vsd.weights[i] /= (float)sum;
        } else {
            // Degenerate: hard-bind to nearest joint.
            vsd.joint_indices[0] = (v < (int)nearest.size()) ? nearest[v] : 0;
            vsd.weights[0] = 1.0f;
            for (int i = 1; i < kMaxVertexInfluences; ++i) { vsd.joint_indices[i] = 0; vsd.weights[i] = 0.0f; }
        }
    }
    return out;
}

// A few Laplacian smoothing passes on the dense weights (then renormalised
// in finalize()).  Cheap way to remove boundary speckle for geodesic.
void smoothWeights(const MeshOps& ops, std::vector<std::vector<double>>& W, int nj, int passes) {
    const int nv = ops.nv;
    std::vector<double> tmp(nv);
    for (int p = 0; p < passes; ++p) {
        for (int j = 0; j < nj; ++j) {
            auto& w = W[j];
            for (int i = 0; i < nv; ++i) {
                double acc = w[i]; double wsum = 1.0;
                for (auto& e : ops.graph[i]) { acc += w[e.first]; wsum += 1.0; }
                tmp[i] = acc / wsum;
            }
            w.swap(tmp);
        }
    }
}

// ---------------------------------------------------------------------------
//  Algorithm 0 (legacy / fallback): nearest-bone inverse distance.
// ---------------------------------------------------------------------------
std::vector<std::vector<double>> weightsNearestBone(
    const TriangleMesh& mesh, const std::vector<JointBones>& jbs, int nj) {
    const int nv = (int)mesh.positions.size();
    std::vector<std::vector<double>> W(nj, std::vector<double>(nv, 0.0));
    for (int v = 0; v < nv; ++v) {
        dvec3 p(mesh.positions[v]);
        for (int j = 0; j < nj; ++j) {
            double d = distToJoint(p, jbs[j]);
            W[j][v] = 1.0 / (d * d + 1e-6);
        }
    }
    return W;
}

// ---------------------------------------------------------------------------
//  Algorithm: Bone Heat.   (L + D) w_j = D p_j ,  D_i = c * mass_i / d_i^2 .
//  Provably non-negative and a partition of unity (sum_j w_j == 1).
// ---------------------------------------------------------------------------
std::vector<std::vector<double>> weightsBoneHeat(
    const TriangleMesh& mesh, const MeshOps& ops, const std::vector<JointBones>& jbs,
    int nj, const std::vector<int>& owner, double scale) {
    const int nv = ops.nv;
    const double c = 1.5;                          // anchoring strength
    const double dfloor = 1e-3 * scale;            // distance floor (avoid div0)

    std::vector<double> D(nv, 0.0);
    for (int v = 0; v < nv; ++v) {
        dvec3 p(mesh.positions[v]);
        double d = distToJoint(p, jbs[owner[v]]);
        d = std::max(d, dfloor);
        D[v] = c * ops.mass[v] / (d * d);
    }
    std::vector<double> Adiag(nv);
    for (int i = 0; i < nv; ++i) Adiag[i] = ops.Ldiag[i] + D[i];
    auto matvec = [&](const std::vector<double>& x, std::vector<double>& y) {
        for (int i = 0; i < nv; ++i) {
            double s = Adiag[i] * x[i];
            for (auto& e : ops.Loff[i]) s += e.second * x[e.first];
            y[i] = s;
        }
    };

    std::vector<std::vector<double>> W(nj, std::vector<double>(nv, 0.0));
    for (int j = 0; j < nj; ++j) {
        std::vector<double> b(nv, 0.0), x(nv, 0.0);
        bool any = false;
        for (int v = 0; v < nv; ++v) if (owner[v] == j) { b[v] = D[v]; x[v] = 1.0; any = true; }
        if (!any) continue;                       // no territory -> leave zeros
        if (!cg(matvec, Adiag, b, x, 1500, 1e-5)) { W.clear(); return W; }  // signal fail
        for (int v = 0; v < nv; ++v) W[j][v] = std::max(0.0, x[v]);
    }
    return W;
}

// ---------------------------------------------------------------------------
//  Algorithm: Geodesic.  Multi-source Dijkstra to each joint's territory,
//  then a smooth soft-min falloff in world units.
// ---------------------------------------------------------------------------
std::vector<std::vector<double>> weightsGeodesic(
    const TriangleMesh& mesh, const MeshOps& ops, const std::vector<JointBones>& jbs,
    int nj, const std::vector<int>& owner, double scale) {
    const int nv = ops.nv;
    const double tau = 0.04 * scale;              // blend width (world units)

    std::vector<std::vector<double>> dist(nj, std::vector<double>(nv, kInf));
    using QN = std::pair<double, int>;
    for (int j = 0; j < nj; ++j) {
        auto& dj = dist[j];
        std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
        for (int v = 0; v < nv; ++v) if (owner[v] == j) { dj[v] = 0.0; pq.push({0.0, v}); }
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dj[u]) continue;
            for (auto& e : ops.graph[u]) {
                double nd = d + e.second;
                if (nd < dj[e.first]) { dj[e.first] = nd; pq.push({nd, e.first}); }
            }
        }
    }
    std::vector<std::vector<double>> W(nj, std::vector<double>(nv, 0.0));
    for (int v = 0; v < nv; ++v) {
        double dmin = kInf;
        for (int j = 0; j < nj; ++j) dmin = std::min(dmin, dist[j][v]);
        for (int j = 0; j < nj; ++j) {
            double d = dist[j][v];
            W[j][v] = std::isinf(d) ? 0.0 : std::exp(-(d - dmin) / tau);
        }
    }
    smoothWeights(ops, W, nj, 2);
    return W;
}

// ---------------------------------------------------------------------------
//  Algorithm: Bounded Biharmonic.  Minimise biharmonic energy w^T L M^-1 L w
//  with Dirichlet anchors (1 on a joint's territory, 0 on others), bounds
//  enforced by clamp + renormalise.  Partition of unity holds by construction.
// ---------------------------------------------------------------------------
std::vector<std::vector<double>> weightsBiharmonic(
    const TriangleMesh& mesh, const MeshOps& ops, const std::vector<JointBones>& jbs,
    int nj, const std::vector<int>& owner, double scale) {
    const int nv = ops.nv;
    const double r = 0.05 * scale;                // anchor radius

    // anchor[v] = owning joint if v is within r of its bone, else -1 (free).
    std::vector<int> anchor(nv, -1);
    for (int v = 0; v < nv; ++v) {
        dvec3 p(mesh.positions[v]);
        double d = distToJoint(p, jbs[owner[v]]);
        if (d <= r) anchor[v] = owner[v];
    }
    // Ensure every joint with territory has at least one anchor (nearest vert).
    std::vector<int> bestV(nj, -1); std::vector<double> bestD(nj, kInf);
    for (int v = 0; v < nv; ++v) {
        int j = owner[v];
        dvec3 p(mesh.positions[v]);
        double d = distToJoint(p, jbs[j]);
        if (d < bestD[j]) { bestD[j] = d; bestV[j] = v; }
    }
    for (int j = 0; j < nj; ++j) if (bestV[j] >= 0 && anchor[bestV[j]] < 0) anchor[bestV[j]] = j;

    auto applyQ = [&](const std::vector<double>& x, std::vector<double>& y) {
        std::vector<double> t(nv);
        applyL(ops, x, t);
        for (int i = 0; i < nv; ++i) t[i] *= ops.minv[i];
        applyL(ops, t, y);
    };
    // Jacobi diagonal estimate for Q ~ Ldiag^2 * minv.
    std::vector<double> qdiag(nv);
    for (int i = 0; i < nv; ++i) qdiag[i] = std::max(ops.Ldiag[i] * ops.Ldiag[i] * ops.minv[i], 1e-12);

    std::vector<std::vector<double>> W(nj, std::vector<double>(nv, 0.0));
    for (int j = 0; j < nj; ++j) {
        // g = boundary values; free mask.
        std::vector<double> g(nv, 0.0);
        bool any = false;
        for (int v = 0; v < nv; ++v) if (anchor[v] == j) { g[v] = 1.0; any = true; }
        if (!any) continue;
        // rhs_free = -(Q * g_only)_free ; operator zeroes constrained dofs.
        std::vector<double> Qg(nv); applyQ(g, Qg);
        std::vector<double> b(nv), x(nv, 0.0);
        for (int v = 0; v < nv; ++v) { b[v] = (anchor[v] < 0) ? -Qg[v] : 0.0; }
        auto matvecFree = [&](const std::vector<double>& in, std::vector<double>& out) {
            std::vector<double> z(nv);
            for (int v = 0; v < nv; ++v) z[v] = (anchor[v] < 0) ? in[v] : 0.0;
            applyQ(z, out);
            for (int v = 0; v < nv; ++v) if (anchor[v] >= 0) out[v] = 0.0;
        };
        if (!cg(matvecFree, qdiag, b, x, 3000, 1e-4)) { W.clear(); return W; }
        for (int v = 0; v < nv; ++v) {
            double val = (anchor[v] >= 0) ? g[v] : x[v];
            W[j][v] = glm::clamp(val, 0.0, 1.0);          // enforce bounds
        }
    }
    return W;
}

}  // namespace

// ===========================================================================
//  Public entry point.
// ===========================================================================
SkinWeights computeSkinWeightsAlgo(const TriangleMesh& mesh, const Skeleton& skeleton,
                                   SkinWeightAlgo algo, int max_influences) {
    const int nv = (int)mesh.positions.size();
    const int nj = (int)skeleton.joints.size();
    if (nv == 0 || nj == 0) return {};

    // Model scale = bounding-box diagonal (used for all relative radii).
    dvec3 lo(kInf), hi(-kInf);
    for (auto& p : mesh.positions) { lo = glm::min(lo, dvec3(p)); hi = glm::max(hi, dvec3(p)); }
    double scale = glm::length(hi - lo);
    if (!(scale > 1e-9)) scale = 1.0;

    std::vector<JointBones> jbs = buildJointBones(skeleton, scale);

    // owner[v] = nearest joint (the vertex's "territory").
    std::vector<int> owner(nv);
    for (int v = 0; v < nv; ++v) owner[v] = nearestJoint(dvec3(mesh.positions[v]), jbs).first;

    auto runFallback = [&]() {
        fprintf(stderr, "[AutoRig] weight solve fell back to nearest-bone.\n");
        auto W = weightsNearestBone(mesh, jbs, nj);
        return finalize(W, nv, nj, max_influences, owner);
    };

    if (algo == SkinWeightAlgo::kNearestBone) {
        auto W = weightsNearestBone(mesh, jbs, nj);
        return finalize(W, nv, nj, max_influences, owner);
    }

    MeshOps ops = buildMeshOps(mesh);

    std::vector<std::vector<double>> W;
    switch (algo) {
        case SkinWeightAlgo::kBoneHeat:   W = weightsBoneHeat(mesh, ops, jbs, nj, owner, scale); break;
        case SkinWeightAlgo::kGeodesic:   W = weightsGeodesic(mesh, ops, jbs, nj, owner, scale); break;
        case SkinWeightAlgo::kBiharmonic: W = weightsBiharmonic(mesh, ops, jbs, nj, owner, scale); break;
        default: break;
    }
    if (W.empty()) return runFallback();          // solver signalled failure

    const char* nm = skinWeightAlgoName(algo);
    fprintf(stderr, "[AutoRig] skinning: %d verts, %d joints, algo=%s\n", nv, nj, nm);
    return finalize(W, nv, nj, max_influences, owner);
}

}  // namespace auto_rig
}  // namespace plugins
