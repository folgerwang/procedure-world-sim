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
#                raygen miss closesthit callable mesh
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

# ── base_depthonly.frag variants ─────────────────────────────────────────────
add_spirv(SPIRV_FILES "base_depthonly_frag.spv"       "base_depthonly.frag" fragment)
add_spirv(SPIRV_FILES "base_depthonly_frag_TEX.spv"   "base_depthonly.frag" fragment  HAS_UV_SET0)
add_spirv(SPIRV_FILES "base_depthonly_frag_NOMTL.spv" "base_depthonly.frag" fragment  NO_MTL)

# ── base_depthonly CSM geometry shader ───────────────────────────────────────
add_spirv(SPIRV_FILES "base_depthonly_csm_geom.spv" "base_depthonly_csm.geom" geometry)

# ── Cluster bindless pass ─────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "cluster_bindless_vert.spv" "cluster_bindless.vert" vertex)
add_spirv(SPIRV_FILES "cluster_bindless_frag.spv" "cluster_bindless.frag" fragment)
add_spirv(SPIRV_FILES "cluster_cull_comp.spv"     "cluster_cull.comp"     compute)

# ── Cluster debug draw ────────────────────────────────────────────────────────
add_spirv(SPIRV_FILES "cluster_debug_vert.spv" "cluster_debug.vert" vertex)
add_spirv(SPIRV_FILES "cluster_debug_frag.spv" "cluster_debug.frag" fragment)

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

# ── IBL pre-filter (cube_ibl.frag compiled 4 ways) ───────────────────────────
# NUM_SAMPLES must be defined for all variants: getSampleVector/filterColor are
# outside any #ifdef so the compiler parses them even for PANORAMA_TO_CUBEMAP.
# 1024 matches the Khronos glTF-Sample-Viewer reference value.
add_spirv(SPIRV_FILES "panorama_to_cubemap_frag.spv" "cube_ibl.frag" fragment  PANORAMA_TO_CUBEMAP  NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_labertian_frag.spv"       "cube_ibl.frag" fragment  LAMBERTIAN_FILTER    NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_ggx_frag.spv"             "cube_ibl.frag" fragment  GGX_FILTER           NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_charlie_frag.spv"         "cube_ibl.frag" fragment  CHARLIE_FILTER       NUM_SAMPLES=1024)
add_spirv(SPIRV_FILES "ibl_smooth_comp.spv"          "ibl_smooth.comp"         compute)

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
