#pragma once
// ============================================================================
//  skinning_weights.h
//
//  Self-contained (no external deps beyond glm/std) skin-weight solvers used
//  by the auto-rig pipeline.  Three quality algorithms plus the legacy
//  nearest-bone fallback, all sharing the same surface-topology front end:
//
//    * Bone Heat   — surface heat-diffusion (Pinocchio-style, no embree
//                    visibility): solve (L + lambda*H) w_j = lambda*H*p_j.
//                    Provably non-negative and a partition of unity.
//    * Geodesic    — multi-source Dijkstra distance to each joint's surface
//                    "territory" + smooth falloff.  No bleeding across gaps.
//    * Biharmonic  — constrained bi-Laplacian solve (bounded biharmonic
//                    weights, bounds enforced by clamp+renormalise).
//
//  All paths:
//    - bind limb geometry to the PROXIMAL (parent) joint via per-joint
//      "outgoing" bone segments, and add a virtual tip bone at every leaf
//      joint (head / hands / feet) so terminal geometry is anchored;
//    - prune to <= max_influences per vertex and renormalise to sum 1.
//
//  If a solver hits a degenerate mesh / non-finite result it transparently
//  falls back to the nearest-bone weights so the pipeline never produces NaNs.
// ============================================================================

#include "plugins/auto_rig/rig_types.h"

namespace plugins {
namespace auto_rig {

// Compute per-vertex skin weights for `mesh` against `skeleton` using `algo`.
// `max_influences` is clamped to [1,4] (the VertexSkinData capacity).
SkinWeights computeSkinWeightsAlgo(const TriangleMesh& mesh,
                                   const Skeleton&     skeleton,
                                   SkinWeightAlgo      algo,
                                   int                 max_influences = 4);

}  // namespace auto_rig
}  // namespace plugins
