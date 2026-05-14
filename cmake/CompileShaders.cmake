# ─── cmake/CompileShaders.cmake ─────────────────────────────────────────────
# Compiles every GLSL shader to SPIR-V at CMake build time using glslc.
#
# Usage (in CMakeLists.txt):
#   include(cmake/CompileShaders.cmake)
#   add_dependencies(RealWorld Shaders)
#
# Output: realworld/lib/shaders/**/*.spv  (same location the runtime expects;
#         the binary runs with working-dir = realworld/).
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Locate glslc ──────────────────────────────────────────────────────────
find_program(GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VK_SDK_PATH}/Bin"
    DOC "Vulkan glslc shader compiler"
)

if(NOT GLSLC_EXECUTABLE)
    message(WARNING
        "glslc not found — shader compilation skipped.\n"
        "Install the Vulkan SDK and ensure VULKAN_SDK env var is set.")
    return()
endif()
message(STATUS "glslc: ${GLSLC_EXECUTABLE}")

# ── 2. Paths ─────────────────────────────────────────────────────────────────
set(SHADER_SRC_DIR  "${CMAKE_SOURCE_DIR}/realworld/src/sim_engine/shaders")
set(SHADER_OUT_DIR  "${CMAKE_SOURCE_DIR}/realworld/lib/shaders")

# All .glsl.h headers — any change to these invalidates every shader.
file(GLOB_RECURSE SHADER_HEADERS "${SHADER_SRC_DIR}/*.glsl.h")

# ── 3. Helper function ───────────────────────────────────────────────────────
# add_spirv(LIST_VAR output_spv source_rel stage [define ...])
#
#   LIST_VAR   — name of a CMake list variable to append the output path to
#   output_spv — output filename relative to SHADER_OUT_DIR (e.g. "foo_frag.spv"
#                or "subdir/foo_frag.spv")
#   source_rel — source file relative to SHADER_SRC_DIR
#   stage      — glslc stage: vertex fragment compute geometry
#                raygen miss closesthit callable mesh task
#   [define …] — bare macro names, e.g. HAS_UV_SET0 DOUBLE_SIDED
#
function(add_spirv LIST_VAR output_spv source_rel stage)
    set(src  "${SHADER_SRC_DIR}/${source_rel}")
    set(spv  "${SHADER_OUT_DIR}/${output_spv}")

    # Build -DFOO flags from remaining positional args.
    set(define_flags "")
    foreach(def IN LISTS ARGN)
        list(APPEND define_flags "-D${def}")
    endforeach()

    # Ensure the output sub-directory exists at build time.
    get_filename_component(spv_dir "${spv}" DIRECTORY)
    file(MAKE_DIRECTORY "${spv_dir}")

    add_custom_command(
        OUTPUT  "${spv}"
        COMMAND "${GLSLC_EXECUTABLE}"
                --target-env=vulkan1.3
                -fshader-stage=${stage}
                -I "${SHADER_SRC_DIR}"
                ${define_flags}
                -o "${spv}"
                "${src}"
        DEPENDS "${src}" ${SHADER_HEADERS}
        COMMENT "glslc ${output_spv}"
        VERBATIM
    )

    set(${LIST_VAR} ${${LIST_VAR}} "${spv}" PARENT_SCOPE)
endfunction()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Enumerate all shaders
# ─────────────────────────────────────────────────────────────────────────────
set(SPIRV_FILES "")

# Note: USE_IBL and USE_PUNCTUAL are defined in global_definition.glsl.h lines 2/7.
# ALPHAMODE_MASK is defined at line 8 of base.frag and base_depthonly.frag.
# None of these need to be passed on the glslc command line — the shader sources
# already define them, and repeating them would cause a "Macro redefined" error.

# ── base.vert variants ───────────────────────────────────────────────────────
# Suffix → compile-time define(s)
#   _TEX    HAS_UV_SET0
#   _N      HAS_NORMALS
#   _TN     HAS_TANGENT only — global_definition.glsl.h line 4-6 auto-derives
#           HAS_NORMALS from HAS_TANGENT, so passing both would cause a
#           "Macro redefined" error. Never pass HAS_NORMALS alongside HAS_TANGENT.
#   _NOMTL  NO_MTL                    (no material descriptor)
#   _SKIN   HAS_SKIN_SET_0 USE_SKINNING
#
# All 14 vertex variants (7 geometry combos × {no skin, skin}):
add_spirv(SPIRV_FILES "base_vert.spv"             "base.vert" vertex)
add_spirv(SPIRV_FILES "base_vert_TEX.spv"         "base.vert" vertex  HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_vert_N.spv"           "base.vert" vertex  HAS_NORMALS)
add_spirv(SPIRV_FILES "base_vert_TN.spv"          "base.vert" vertex  HAS_TANGENT)
add_spirv(SPIRV_FILES "base_vert_TEX_N.spv"       "base.vert" vertex  HAS_UV_SET0 HAS_NORMALS)
add_spirv(SPIRV_FILES "base_vert_TEX_TN.spv"      "base.vert" vertex  HAS_UV_SET0 HAS_TANGENT)
add_spirv(SPIRV_FILES "base_vert_NOMTL.spv"       "base.vert" vertex  NO_MTL)
add_spirv(SPIRV_FILES "base_vert_SKIN.spv"        "base.vert" vertex  HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_N_SKIN.spv"      "base.vert" vertex  HAS_NORMALS HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_TN_SKIN.spv"     "base.vert" vertex  HAS_TANGENT HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_TEX_SKIN.spv"    "base.vert" vertex  HAS_UV_SET0 HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_TEX_N_SKIN.spv"  "base.vert" vertex  HAS_UV_SET0 HAS_NORMALS HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_TEX_TN_SKIN.spv" "base.vert" vertex  HAS_UV_SET0 HAS_TANGENT HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_vert_NOMTL_SKIN.spv"  "base.vert" vertex  NO_MTL HAS_SKIN_SET_0 USE_SKINNING)

# ── base.frag variants ───────────────────────────────────────────────────────
# _DS → DOUBLE_SIDED  (fragment-only; no _SKIN variant for frag)
# All 14 fragment variants (7 geometry combos × {front-only, double-sided}):
add_spirv(SPIRV_FILES "base_frag.spv"           "base.frag" fragment)
add_spirv(SPIRV_FILES "base_frag_DS.spv"        "base.frag" fragment  DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_N.spv"         "base.frag" fragment  HAS_NORMALS)
add_spirv(SPIRV_FILES "base_frag_N_DS.spv"      "base.frag" fragment  HAS_NORMALS DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_TN.spv"        "base.frag" fragment  HAS_TANGENT)
add_spirv(SPIRV_FILES "base_frag_TN_DS.spv"     "base.frag" fragment  HAS_TANGENT DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_TEX.spv"       "base.frag" fragment  HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_frag_TEX_DS.spv"    "base.frag" fragment  HAS_UV_SET0 DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_TEX_N.spv"     "base.frag" fragment  HAS_UV_SET0 HAS_NORMALS)
add_spirv(SPIRV_FILES "base_frag_TEX_N_DS.spv"  "base.frag" fragment  HAS_UV_SET0 HAS_NORMALS DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_TEX_TN.spv"    "base.frag" fragment  HAS_UV_SET0 HAS_TANGENT)
add_spirv(SPIRV_FILES "base_frag_TEX_TN_DS.spv" "base.frag" fragment  HAS_UV_SET0 HAS_TANGENT DOUBLE_SIDED)
add_spirv(SPIRV_FILES "base_frag_NOMTL.spv"     "base.frag" fragment  NO_MTL)
add_spirv(SPIRV_FILES "base_frag_NOMTL_DS.spv"  "base.frag" fragment  NO_MTL DOUBLE_SIDED)

# ── base_depthonly.vert variants ─────────────────────────────────────────────
add_spirv(SPIRV_FILES "base_depthonly_vert.spv"          "base_depthonly.vert" vertex)
add_spirv(SPIRV_FILES "base_depthonly_vert_TEX.spv"      "base_depthonly.vert" vertex  HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_depthonly_vert_NOMTL.spv"    "base_depthonly.vert" vertex  NO_MTL)
add_spirv(SPIRV_FILES "base_depthonly_vert_SKIN.spv"     "base_depthonly.vert" vertex  HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_depthonly_vert_TEX_SKIN.spv" "base_depthonly.vert" vertex  HAS_UV_SET0 HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_depthonly_vert_NOMTL_SKIN.spv" "base_depthonly.vert" vertex  NO_MTL HAS_SKIN_SET_0 USE_SKINNING)

# ── CSM_PER_CASCADE permutations of base_depthonly.vert ───────────────────
# Same vertex shader, but reads RuntimeLightsParams.light_view_proj[
# model_params.cascade_idx] instead of VIEW_PARAMS_SET's camera_info
# .view_proj.  Consumed by the DrawMode::kCsmPerCascade pipeline (the
# "Regular" option on the shadow draw-mode menu).  The host loops over
# CSM_CASCADE_COUNT cascades and pushes cascade_idx per draw — no GS,
# no mesh shader, no extra UBO upload (the lights UBO is already bound
# for the GS path).  TEX / SKIN matrix mirrors the four non-CSMCASC
# permutations above; NOMTL is omitted because CSM shadow draws always
# have a material context.
add_spirv(SPIRV_FILES "base_depthonly_vert_CSMCASC.spv"          "base_depthonly.vert" vertex  CSM_PER_CASCADE)
add_spirv(SPIRV_FILES "base_depthonly_vert_TEX_CSMCASC.spv"      "base_depthonly.vert" vertex  CSM_PER_CASCADE HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_depthonly_vert_SKIN_CSMCASC.spv"     "base_depthonly.vert" vertex  CSM_PER_CASCADE HAS_SKIN_SET_0 USE_SKINNING)
add_spirv(SPIRV_FILES "base_depthonly_vert_TEX_SKIN_CSMCASC.spv" "base_depthonly.vert" vertex  CSM_PER_CASCADE HAS_UV_SET0 HAS_SKIN_SET_0 USE_SKINNING)

# ── base_depthonly.frag variants ─────────────────────────────────────────────
add_spirv(SPIRV_FILES "base_depthonly_frag.spv"       "base_depthonly.frag" fragment)
add_spirv(SPIRV_FILES "base_depthonly_frag_TEX.spv"   "base_depthonly.frag" fragment  HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_depthonly_frag_NOMTL.spv" "base_depthonly.frag" fragment  NO_MTL)

# ── base_depthonly CSM geometry shader ───────────────────────────────────────
add_spirv(SPIRV_FILES "base_depthonly_csm_geom.spv" "base_depthonly_csm.geom" geometry)

# ── base_depthonly CSM mesh-shader task + mesh pair ──────────────────────────
# Drawable-path mesh-shader CSM (DrawMode::kCsmMeshShader).  task: one
# workgroup per drawcall, amplifies to CSM_CASCADE_COUNT mesh workgroups
# (one per cascade layer).  mesh: fetches VB/IB/instance via per-
# primitive SSBO bindings, transforms with light_view_proj[cascade] *
# instance_transform * model_mat, emits to gl_Layer=cascade.  Supports
# opaque non-skinned UINT32-indexed primitives with <=256 verts/tris;
# everything else falls back to the GS pipeline inside drawMesh.
add_spirv(SPIRV_FILES "base_depthonly_csm_task.spv" "base_depthonly_csm.task" task)
add_spirv(SPIRV_FILES "base_depthonly_csm_mesh.spv" "base_depthonly_csm.mesh" mesh)

# ── Cluster bindless pass ─────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "cluster_bindless_vert.spv"     "cluster_bindless.vert" vertex)
add_spirv(SPIRV_FILES "cluster_bindless_frag.spv"     "cluster_bindless.frag" fragment)
# OIT_OUTPUT variant: same shader compiled with -DOIT_OUTPUT so it writes
# the McGuire-Bavoil weighted-blended pair (accum + reveal) instead of a
# single colour.  Used by the cluster translucent pipeline; resolved by
# oit_composite_frag.spv in a fullscreen pass.
add_spirv(SPIRV_FILES "cluster_bindless_oit_frag.spv" "cluster_bindless.frag" fragment OIT_OUTPUT)
# Depth-only CSM shadow variants for cluster rendering.
#   _vert / _geom  — legacy VS+GS broadcast path (kept compileable for A/B).
#   _mesh          — production mesh-shader path; one workgroup per
#                    (cluster, cascade), emits surviving triangles
#                    directly to the matching depth-array layer.
add_spirv(SPIRV_FILES "cluster_bindless_shadow_vert.spv" "cluster_bindless_shadow.vert" vertex)
add_spirv(SPIRV_FILES "cluster_bindless_shadow_geom.spv" "cluster_bindless_shadow.geom" geometry)
add_spirv(SPIRV_FILES "cluster_bindless_shadow_mesh.spv" "cluster_bindless_shadow.mesh" mesh)
# Task shader companion — drives the mesh shader's dispatch by culling
# clusters against all CSM_CASCADE_COUNT cascade frustums and emitting
# mesh workgroups only for surviving (cluster, cascade) pairs.
add_spirv(SPIRV_FILES "cluster_bindless_shadow_task.spv" "cluster_bindless_shadow.task" task)
# Silhouette prepass — fills each cascade's in-camera-frustum region of
# the shadow map with depth=1.0 (forced via gl_Position.z = gl_Position.w)
# so the cleared 0.0 outside the silhouette rejects every later shadow
# caster via the LESS_OR_EQUAL depth test.  Hi-Z then drops whole tiles.
add_spirv(SPIRV_FILES "csm_silhouette_prepass_mesh.spv" "csm_silhouette_prepass.mesh" mesh)
add_spirv(SPIRV_FILES "cluster_cull_comp.spv"         "cluster_cull.comp"     compute)
# WBOIT resolve — fullscreen pass that reads accum+reveal and writes the
# composited translucent layer back over the scene colour buffer.
add_spirv(SPIRV_FILES "oit_composite_frag.spv"        "oit_composite.frag"    fragment)

# ── Dynamic reflection cubemap ───────────────────────────────────────────────
# Depth-aware reprojection compute pass for the temporally-amortised
# reflection probe (see scene_rendering/dynamic_cubemap.h).  One face is
# freshly rendered per frame; this shader warps the other 5 to the new
# camera position.
add_spirv(SPIRV_FILES "cube_reproject_comp.spv"       "cube_reproject.comp"   compute)

# Per-face depth → linear-distance conversion for the dynamic cubemap.
# Runs once per face render (just after the face render pass) to feed
# the linear distances that sh_project.comp's parallax-aware probe
# sampling reads from depth_cube_.
add_spirv(SPIRV_FILES "depth_to_linear_comp.spv"      "depth_to_linear.comp"  compute)

# ── Ambient probe SH projection ──────────────────────────────────────────────
# Projects a captured cubemap into the first 9 spherical-harmonic basis
# functions and stores the cosine-convolved coefficients in one slot of
# the probe SSBO.  See scene_rendering/ambient_probe_system.h for the
# orchestrating class.
add_spirv(SPIRV_FILES "sh_project_comp.spv"           "sh_project.comp"       compute)

# ── Ambient probe debug-draw ─────────────────────────────────────────────────
# Renders one icosphere per probe at its world-space position; the
# fragment shader evaluates the probe's SH coefficients in the surface
# normal direction so each sphere visually shows the irradiance it
# contributes.  Geometry is generated procedurally in the vertex
# shader (no vertex/index buffers needed).
add_spirv(SPIRV_FILES "probe_debug_vert.spv"          "probe_debug.vert"      vertex)
add_spirv(SPIRV_FILES "probe_debug_frag.spv"          "probe_debug.frag"      fragment)

# ── Cluster debug draw ────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "cluster_debug_vert.spv" "cluster_debug.vert" vertex)
add_spirv(SPIRV_FILES "cluster_debug_frag.spv" "cluster_debug.frag" fragment)

# ── Collision debug draw ──────────────────────────────────────────────────────
# Renders static-mesh collision triangles flat-shaded with a hash of the
# triangle id so adjacent faces look obviously different. Toggled in-game
# with F1; consumed by helper::CollisionDebugDraw.
add_spirv(SPIRV_FILES "collision_debug_vert.spv" "collision_debug.vert" vertex)
add_spirv(SPIRV_FILES "collision_debug_frag.spv" "collision_debug.frag" fragment)

# ── Simple geometry passes ────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "base_shape_draw_vert.spv" "base_shape_draw.vert" vertex)
add_spirv(SPIRV_FILES "base_shape_draw_frag.spv" "base_shape_draw.frag" fragment)
add_spirv(SPIRV_FILES "full_screen_vert.spv"     "full_screen.vert"     vertex)
add_spirv(SPIRV_FILES "debug_draw_vert.spv"      "debug_draw.vert"      vertex)
add_spirv(SPIRV_FILES "debug_draw_frag.spv"      "debug_draw.frag"      fragment)

# ── Skybox / environment ───────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "skybox_vert.spv"  "skybox.vert"      vertex)
add_spirv(SPIRV_FILES "skybox_frag.spv"  "skybox.frag"      fragment)
# cube_skybox.spv: output name deliberately has no _frag suffix (matches runtime).
add_spirv(SPIRV_FILES "cube_skybox.spv"  "cube_skybox.frag" fragment)
# Dithered "mini-buffer" sky update: renders a (cube_size/8)^2 cubemap at
# full atmospheric quality and scatters it into the full-res envmap mip 0
# with a per-frame 8x8 dither offset (cycles 64 unique positions in 64
# frames).  Net per-frame sky cost: ~1/64 of the full-res draw.
add_spirv(SPIRV_FILES "cube_skybox_mini_comp.spv" "cube_skybox_mini.comp" compute)
# Fullscreen envmap background pass: draws the live sky envmap into all pixels
# not covered by geometry (depth == far-plane 1.0) using a push-constant
# inv_view_proj_relative matrix to reconstruct the world-space view direction.
add_spirv(SPIRV_FILES "skybox_envmap_vert.spv" "skybox_envmap.vert" vertex)
add_spirv(SPIRV_FILES "skybox_envmap_frag.spv" "skybox_envmap.frag" fragment)

# ── IBL pre-filter (cube_ibl.frag compiled 4 ways) ───────────────────────────
# NUM_SAMPLES must be defined for all variants: getSampleVector/filterColor are
# outside any #ifdef so the compiler parses them even for PANORAMA_TO_CUBEMAP.
# 1024 matches the Khronos glTF-Sample-Viewer reference value.
add_spirv(SPIRV_FILES "panorama_to_cubemap_frag.spv" "cube_ibl.frag" fragment  PANORAMA_TO_CUBEMAP  NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_labertian_frag.spv"       "cube_ibl.frag" fragment  LAMBERTIAN_FILTER    NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_ggx_frag.spv"             "cube_ibl.frag" fragment  GGX_FILTER           NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_charlie_frag.spv"         "cube_ibl.frag" fragment  CHARLIE_FILTER       NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_smooth_comp.spv"          "ibl_smooth.comp"         compute)

# Dithered "mini-buffer" IBL convolution: same Lambertian / GGX / Charlie
# filters as cube_ibl.frag, but written as a compute shader that updates
# only 1/(dither_stride^2) of the destination IBL cubemap mip's texels per
# frame.  NUM_SAMPLES is the SAMPLE BANK size; per-frame, each touched
# texel reads NUM_SAMPLES_PER_FRAME = NUM_SAMPLES / dither_count samples.
#
# Why 16384 (= 16 × dither_count for dither_stride = 32):
#   - The dither_count for the mini path is 32×32 = 1024 frames.  With
#     NUM_SAMPLES = 1024 we'd get NUM_SAMPLES_PER_FRAME = 1, and the
#     stride-coprime trick degenerates because NUM_SAMPLES == dither_count
#     ((stride * dither_count) mod NUM_SAMPLES == 0 for every stride).
#     Each pixel ends up locked to its single random importance sample
#     forever → permanent dot-pattern noise (visible in the IBL diffuse
#     debug viewer as a regular pixelated grid).
#   - Bumping NUM_SAMPLES to 16384 gives NUM_SAMPLES_PER_FRAME = 16, the
#     same per-touch quality the shader had with the original 8×8 / 64-
#     frame setup.  gcd(NUM_SAMPLES_PER_FRAME + 1, 1024) = gcd(17, 1024)
#     = 1 and gcd(17, 16384) = 1, so consecutive touches of the same pixel
#     advance through disjoint sample subsets.  Combined with the EMA
#     (kIblTemporalAlpha = 0.25, ~4-touch effective window) each diffuse
#     pixel converges to a ~64-effective-sample estimate -- smooth blob
#     instead of pixelated noise.
#   - Shader cost is unchanged from the old setup: the per-touch loop runs
#     16 iterations per texel (same as before), and we still touch only
#     1/1024 of the mip per frame.
add_spirv(SPIRV_FILES "ibl_lambertian_mini_comp.spv" "cube_ibl_mini.comp" compute  LAMBERTIAN_FILTER  NUM_SAMPLES=16384)
add_spirv(SPIRV_FILES "ibl_ggx_mini_comp.spv"        "cube_ibl_mini.comp" compute  GGX_FILTER         NUM_SAMPLES=16384)
add_spirv(SPIRV_FILES "ibl_charlie_mini_comp.spv"    "cube_ibl_mini.comp" compute  CHARLIE_FILTER     NUM_SAMPLES=16384)

# ── CSM debug ─────────────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "csm_debug_frag.spv" "csm_debug.frag" fragment)

# ── Camera / GPU update computes ──────────────────────────────────────────────
add_spirv(SPIRV_FILES "update_camera_comp.spv"              "update_camera.comp"              compute)
add_spirv(SPIRV_FILES "update_game_objects_comp.spv"        "update_game_objects.comp"        compute)
add_spirv(SPIRV_FILES "update_gltf_indirect_draw_comp.spv"  "update_gltf_indirect_draw.comp"  compute)
add_spirv(SPIRV_FILES "update_instance_buffer_comp.spv"     "update_instance_buffer.comp"     compute)

# ── Depth / SSAO ──────────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "gen_minmax_depth_comp.spv" "gen_minmax_depth.comp" compute)
add_spirv(SPIRV_FILES "ssao_compute_comp.spv"     "ssao_compute.comp"     compute)
add_spirv(SPIRV_FILES "ssao_blur_comp.spv"        "ssao_blur.comp"        compute)
add_spirv(SPIRV_FILES "ssao_apply_comp.spv"       "ssao_apply.comp"       compute)

# ── Blur ──────────────────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "blur_image_x_comp.spv"        "blur_image_x.comp"        compute)
add_spirv(SPIRV_FILES "blur_image_y_merge_comp.spv"  "blur_image_y_merge.comp"  compute)

# ── Sky scattering LUT ────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "sky_scattering_lut_first_pass_comp.spv" "sky_scattering_lut_first_pass.comp" compute)
add_spirv(SPIRV_FILES "sky_scattering_lut_sum_pass_comp.spv"   "sky_scattering_lut_sum_pass.comp"   compute)
add_spirv(SPIRV_FILES "sky_scattering_lut_final_pass_comp.spv" "sky_scattering_lut_final_pass.comp" compute)

# ── Conemap ───────────────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "conemap_gen_comp.spv"       "conemap_gen.comp"       compute)
add_spirv(SPIRV_FILES "conemap_gen_init_comp.spv"  "conemap_gen_init.comp"  compute)
add_spirv(SPIRV_FILES "conemap_pack_comp.spv"      "conemap_pack.comp"      compute)
add_spirv(SPIRV_FILES "conemap_test_vert.spv"      "conemap_test.vert"      vertex    HAS_UV_SET0 HAS_TANGENT)
add_spirv(SPIRV_FILES "conemap_test_frag.spv"      "conemap_test.frag"      fragment  HAS_UV_SET0 HAS_TANGENT)

# ── PRT (Precomputed Radiance Transfer) ───────────────────────────────────────
add_spirv(SPIRV_FILES "gen_prt_pack_info_comp.spv"       "gen_prt_pack_info.comp"       compute)
add_spirv(SPIRV_FILES "pack_prt_comp.spv"                "pack_prt.comp"                compute)
add_spirv(SPIRV_FILES "prt_minmax_ds_comp.spv"           "prt_minmax_ds.comp"           compute)
add_spirv(SPIRV_FILES "prt_shadow_gen_comp.spv"          "prt_shadow_gen.comp"          compute)
add_spirv(SPIRV_FILES "prt_shadow_gen_with_cache_comp.spv" "prt_shadow_gen_with_cache.comp" compute)
add_spirv(SPIRV_FILES "prt_shadow_cache_init_comp.spv"   "prt_shadow_cache_init.comp"   compute)
add_spirv(SPIRV_FILES "prt_shadow_cache_update_comp.spv" "prt_shadow_cache_update.comp" compute)
add_spirv(SPIRV_FILES "perlin_noise_init_comp.spv"       "perlin_noise_init.comp"       compute)

# ── Hair / LBM demos ──────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "hair_patch_comp.spv" "hair_patch.comp" compute)
add_spirv(SPIRV_FILES "hair_test_vert.spv"  "hair_test.vert"  vertex)
add_spirv(SPIRV_FILES "hair_test_frag.spv"  "hair_test.frag"  fragment)
add_spirv(SPIRV_FILES "lbm_patch_comp.spv"  "lbm_patch.comp"  compute)
add_spirv(SPIRV_FILES "lbm_test_vert.spv"   "lbm_test.vert"   vertex)
add_spirv(SPIRV_FILES "lbm_test_frag.spv"   "lbm_test.frag"   fragment)

# ── Grass (subdir) ────────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "grass/grass_vert.spv" "grass/grass.vert" vertex)
add_spirv(SPIRV_FILES "grass/grass_frag.spv" "grass/grass.frag" fragment)
add_spirv(SPIRV_FILES "grass/grass_geom.spv" "grass/grass.geom" geometry)
# Mesh shader requires Vulkan 1.2+ NV or EXT mesh shader extension.
# glslc supports it via -fshader-stage=mesh with --target-env=vulkan1.3.
add_spirv(SPIRV_FILES "grass/grass_mesh.spv" "grass/grass.mesh" mesh)

# ── Terrain (subdir) ──────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "terrain/tile_vert.spv"             "terrain/tile.vert"         vertex)
add_spirv(SPIRV_FILES "terrain/tile_frag.spv"             "terrain/tile.frag"         fragment)
# tile_soil_vert.spv: referenced by terrain_scene_view.cpp for the soil-type
# terrain pass. Compiled from tile.vert; adjust defines if the shader uses
# material-type flags (currently tile.vert has no material-type ifdefs).
add_spirv(SPIRV_FILES "terrain/tile_soil_vert.spv"        "terrain/tile.vert"         vertex)
add_spirv(SPIRV_FILES "terrain/tile_water_vert.spv"       "terrain/tile.vert"         vertex)
add_spirv(SPIRV_FILES "terrain/tile_water_frag.spv"       "terrain/tile_water.frag"   fragment)
add_spirv(SPIRV_FILES "terrain/tile_creator_comp.spv"     "terrain/tile_creator.comp" compute)
add_spirv(SPIRV_FILES "terrain/tile_update_comp.spv"      "terrain/tile_update.comp"  compute)
add_spirv(SPIRV_FILES "terrain/tile_flow_update_comp.spv" "terrain/tile_flow_update.comp" compute)

# ── Weather (subdir) ──────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "weather/airflow_update_comp.spv"    "weather/airflow_update.comp"    compute)
add_spirv(SPIRV_FILES "weather/cloud_shadow_init_comp.spv" "weather/cloud_shadow_init.comp" compute)
add_spirv(SPIRV_FILES "weather/cloud_shadow_merge_comp.spv"  "weather/cloud_shadow_merge.comp"  compute)
add_spirv(SPIRV_FILES "weather/render_volume_cloud_comp.spv" "weather/render_volume_cloud.comp" compute)
add_spirv(SPIRV_FILES "weather/temperature_init_comp.spv"  "weather/temperature_init.comp"  compute)

# ── Ray tracing (subdir) ──────────────────────────────────────────────────────
set(RT_CALLABLE "ray_tracing/callable_test")
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_raygen_rgen.spv"     "ray_tracing/callable_test/rt_raygen.rgen"     rgen)
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_miss_rmiss.spv"      "ray_tracing/callable_test/rt_miss.rmiss"      rmiss)
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_closesthit_rchit.spv" "ray_tracing/callable_test/rt_closesthit.rchit" rchit)
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_callable1_rcall.spv" "ray_tracing/callable_test/rt_callable1.rcall" rcall)
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_callable2_rcall.spv" "ray_tracing/callable_test/rt_callable2.rcall" rcall)
add_spirv(SPIRV_FILES "${RT_CALLABLE}/rt_callable3_rcall.spv" "ray_tracing/callable_test/rt_callable3.rcall" rcall)

set(RT_SHADOW "ray_tracing/raytracing_shadow")
add_spirv(SPIRV_FILES "${RT_SHADOW}/rt_raygen_rgen.spv"      "ray_tracing/raytracing_shadow/rt_raygen.rgen"      rgen)
add_spirv(SPIRV_FILES "${RT_SHADOW}/rt_miss_rmiss.spv"       "ray_tracing/raytracing_shadow/rt_miss.rmiss"       rmiss)
add_spirv(SPIRV_FILES "${RT_SHADOW}/rt_shadow_rmiss.spv"     "ray_tracing/raytracing_shadow/rt_shadow.rmiss"     rmiss)
add_spirv(SPIRV_FILES "${RT_SHADOW}/rt_closesthit_rchit.spv" "ray_tracing/raytracing_shadow/rt_closesthit.rchit" rchit)

# ── 5. Aggregate target ───────────────────────────────────────────────────────
add_custom_target(Shaders ALL DEPENDS ${SPIRV_FILES})
