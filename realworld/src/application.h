#pragma once
#include <array>
#include <chrono>
#include <memory>
#include <vector>
#include "renderer/renderer.h"
#include "renderer/renderer_structs.h"
#include "shaders/global_definition.glsl.h"
#include "game_object/drawable_object.h"
#include "game_object/mesh_load_task_manager.h"
#include "game_object/terrain.h"
#include "game_object/debug_draw.h"
#include "game_object/conemap_obj.h"
#include "game_object/conemap_test.h"
#include "game_object/hair_patch.h"
#include "game_object/hair_test.h"
#include "game_object/lbm_patch.h"
#include "game_object/lbm_test.h"
#include "game_object/camera.h"
#include "game_object/camera_object.h"
#include "game_object/player_controller.h"
#include "game_object/sphere.h"
#include "scene_rendering/skydome.h"
#include "scene_rendering/weather_system.h"
#include "scene_rendering/ibl_creator.h"
#include "scene_rendering/volume_cloud.h"
#include "scene_rendering/volume_noise.h"
#include "scene_rendering/ssao.h"
#include "scene_rendering/cluster_renderer.h"
#include "scene_rendering/dynamic_cubemap.h"
#include "scene_rendering/virtual_texture.h"
#include "scene_rendering/ambient_probe_system.h"
#include "scene_rendering/conemap.h"
#include "scene_rendering/prt_shadow.h"
#include "scene_rendering/terrain_scene_view.h"
#include "scene_rendering/object_scene_view.h"
#include "ray_tracing/raytracing_base.h"
#include "helper/engine_helper.h"
#include "helper/gpu_profiler.h"
#include "helper/collision_mesh.h"
#include "ui/menu.h"
#include "plugins/plugin_manager.h"
#include "plugins/auto_rig/auto_rig_plugin.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;
namespace es = engine::scene_rendering;
namespace eh = engine::helper;

struct GLFWwindow;
namespace work {
namespace app {

const int kMaxFramesInFlight = 2;
const int kCubemapSize = 512;
const int kDifuseCubemapSize = 256;

class RealWorldApplication {
public:
    void run();
    void setFrameBufferResized(bool resized) { framebuffer_resized_ = resized; }

    // Screen-space velocity buffer (RG16F, NDC delta).  Populated by
    // the cluster G-buffer pass when deferred rendering is enabled;
    // empty (zero-cleared, in COLOR_ATTACHMENT_OPTIMAL layout) when
    // forward rendering is selected.  Future passes that want to
    // consume it (TAA, motion blur, temporal reprojection) should
    // (a) check deferred_rendering_enabled_, (b) insert the
    // appropriate barrier to SHADER_READ_ONLY before sampling, and
    // (c) restore COLOR_ATTACHMENT_OPTIMAL afterwards so the next
    // frame's G-buffer pass can re-clear into it.
    const er::TextureInfo& getVelocityBuffer() const { return gbuf_velocity_; }

private:
    void initWindow();
    void initVulkan();
    void createImageViews();
    void createRenderPasses();
    void createFramebuffers(const glm::uvec2& display_size);
    void createCommandPool();
    void createDepthResources(const glm::uvec2& display_size);
    void createHdrColorBuffer(const glm::uvec2& display_size);
    void createColorBufferCopy(const glm::uvec2& display_size);
    // ── Deferred-rendering G-buffer (3 RTs + reused depth) ────────────────
    // First step toward clustered shading: opaque cluster geometry rasterises
    // material attributes into these targets instead of running PBR in the
    // fragment shader.  A compute resolve (initDeferredResolve) then runs
    // PBR once per pixel and writes the lit colour into hdr_color_buffer_.
    // Other forward passes (terrain / grass / hair / sky / OIT) keep running
    // ON TOP of that resolved HDR target — see drawScene() for the order.
    // Lifetime hooks: createGBuffer is called from recreateRenderBuffer so
    // the targets always match the swap-chain size.
    void createGBuffer(const glm::uvec2& display_size);
    // Allocate Hi-Z pyramid sized for `display_size`, build per-mip image
    // views, and (re)write the per-mip Hi-Z build descriptor sets.  Called
    // from createGBuffer so the pyramid always tracks swap-chain extent.
    void createHiZPyramid(const glm::uvec2& display_size);
    void initHiZBuild();   // one-time pipeline/layout/sampler creation
    void writeHiZBuildDescriptors(
        const std::shared_ptr<er::ImageView>& scene_depth_view);
    // Dispatches the per-mip Hi-Z build chain: mip 0 reduces scene depth
    // (passed as `scene_depth_size`) into the half-res pyramid mip 0;
    // each subsequent mip 2:1-max-reduces the previous.  Caller must
    // already have transitioned scene depth to SHADER_READ_ONLY_OPTIMAL.
    // Pyramid stays in GENERAL throughout — the descriptor sets bind it
    // that way for both source and dest, with an internal memory barrier
    // between adjacent mip dispatches.
    void buildHiZPyramid(
        const std::shared_ptr<er::CommandBuffer>& cmd_buf,
        const glm::uvec2& scene_depth_size);
    // Per-frame Hi-Z pyramid generation.  Dispatches the build compute
    // once per mip with appropriate barriers, leaving every mip in
    // SHADER_READ_ONLY_OPTIMAL so the cluster cull pass can sample it.
    // Mip 0 sources from depth_buffer_copy_; higher mips source the
    // previous mip via the per-mip descriptor sets initialised by
    // writeHiZBuildDescriptors().
    void dispatchHiZBuild(
        const std::shared_ptr<er::CommandBuffer>& cmd_buf);
    void initDeferredResolve();
    // The resolve writes into a caller-supplied storage image — by default
    // object_scene_view_'s color buffer, since that's where the rest of the
    // scene already targets.  Pass an alternate view to override.
    void writeDeferredResolveDescriptors(
        const std::shared_ptr<er::ImageView>& output_color_view = nullptr);
    void recreateRenderBuffer(const glm::uvec2& display_size);
    void createTextureSampler();
    void createDescriptorSets();
    void setupCsmDebugDraw();
    void registerCsmDebugImTextureIds();
    // Registers ImGui texture IDs for the IBL/sky cubemap debug window.
    // Must be called after IblCreator and Skydome have been constructed
    // and ImGui has been initialised by the Menu constructor (i.e. same
    // ordering as registerCsmDebugImTextureIds, plus a re-register on
    // swap-chain rebuild since ImGui state is reset).
    void registerIblDebugImTextureIds();
    // Registers ImGui texture IDs for the VT pool viewer.  Same
    // ordering constraint as the others: must run after Menu's
    // constructor has initialised ImGui.
    void registerVtPoolImTextureIds();
    void createCommandBuffers();
    void createSyncObjects();
    er::WriteDescriptorList addGlobalTextures(
        const std::shared_ptr<er::DescriptorSet>& description_set,
        const std::shared_ptr<er::TextureInfo>& direct_shadow_tex);
    void mainLoop();
    void drawScene(
        std::shared_ptr<er::CommandBuffer> command_buffer,
        const er::SwapChainInfo& swap_chain_info,
        const std::shared_ptr<er::DescriptorSet>& view_desc_set,
        const glm::uvec2& screen_size,
        uint32_t image_index,
        float delta_t,
        float current_time);
    void initDrawFrame();
    void drawFrame();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

private:
    GLFWwindow* window_ = nullptr;

    er::PipelineRenderbufferFormats
        renderbuffer_formats_[int(er::RenderPasses::kNumRenderPasses)];

    er::QueueFamilyList queue_list_;
    er::GraphicPipelineInfo graphic_pipeline_info_;
    er::GraphicPipelineInfo graphic_double_face_pipeline_info_;
    er::GraphicPipelineInfo graphic_no_depth_write_pipeline_info_;
    er::GraphicPipelineInfo graphic_fs_pipeline_info_;
    er::GraphicPipelineInfo graphic_fs_blend_pipeline_info_;
    er::GraphicPipelineInfo graphic_cubemap_pipeline_info_;

    std::shared_ptr<er::Instance> instance_;
    er::PhysicalDeviceList physical_devices_;
    std::shared_ptr<er::PhysicalDevice> physical_device_;
    std::shared_ptr<er::Device> device_;
    std::shared_ptr<er::Queue> graphics_queue_;
    std::shared_ptr<er::Queue> present_queue_;
    std::shared_ptr<er::Surface> surface_;
    er::SwapChainInfo swap_chain_info_;
    std::shared_ptr<er::DescriptorPool> descriptor_pool_;
    std::shared_ptr<er::DescriptorSetLayout> pbr_lighting_desc_set_layout_;
    std::shared_ptr<er::DescriptorSetLayout> runtime_lights_desc_set_layout_;
    std::shared_ptr<er::DescriptorSet> pbr_lighting_desc_set_;
    std::shared_ptr<er::DescriptorSet> runtime_lights_desc_set_;
    er::TextureInfo ibl_diffuse_tex_;
    er::TextureInfo ibl_specular_tex_;
    er::TextureInfo ibl_sheen_tex_;
    std::shared_ptr<er::RenderPass> final_render_pass_;
    std::shared_ptr<er::RenderPass> cubemap_render_pass_;
    std::shared_ptr<er::CommandPool> command_pool_;

    er::PhysicalDeviceRayTracingPipelineProperties rt_pipeline_properties_;
    er::PhysicalDeviceAccelerationStructureFeatures as_features_;

    er::Format hdr_format_ = er::Format::B10G11R11_UFLOAT_PACK32;// E5B9G9R9_UFLOAT_PACK32; this format not supported here.
    er::Format depth_format_ = er::Format::D24_UNORM_S8_UINT;
    er::TextureInfo hdr_color_buffer_;
    er::TextureInfo hdr_color_buffer_copy_;
    er::TextureInfo depth_buffer_;
    er::TextureInfo depth_buffer_copy_;
    std::shared_ptr<er::Framebuffer> hdr_frame_buffer_;
    std::shared_ptr<er::Framebuffer> hdr_water_frame_buffer_;
    std::shared_ptr<er::RenderPass> hdr_render_pass_;
    std::shared_ptr<er::RenderPass> hdr_water_render_pass_;

    // ── Deferred renderer (Phase 1: cluster_bindless only) ────────────────
    // Master toggle.  When false, drawScene() runs the legacy forward path
    // and none of the gbuf_/deferred_ resources need to be valid.  When
    // true, the cluster opaque pass writes the G-buffer instead of lit
    // colour; deferred_resolve.comp reads it back and lights the scene
    // into hdr_color_buffer_; everything else (terrain / sky / OIT) is
    // overlaid forward as before.  Defaults to OFF so this commit is
    // visually identical until the user opts in via the menu / hotkey.
    bool deferred_rendering_enabled_ = true;
    // RT0  — RGBA8       albedo.rgb + ao.a
    // RT1  — RGBA8_UNORM octahedral-encoded normal.xy + roughness.z + flags.w
    //                    (RGBA8 keeps each channel symmetric and fits the
    //                    bandwidth budget; world normals are reconstructed
    //                    via octDecode in deferred_resolve.comp.)
    // RT2  — RGBA8       emissive.rgb + metallic.a
    //                    (emissive is tone-mapped/clamped at fragment time
    //                    so an LDR target is sufficient for opaque assets;
    //                    upgrade to R11G11B10F if HDR emissives become an
    //                    issue.)
    er::Format gbuf_albedo_ao_format_      = er::Format::R8G8B8A8_UNORM;
    er::Format gbuf_normal_rough_format_   = er::Format::R8G8B8A8_UNORM;
    er::Format gbuf_emissive_metal_format_ = er::Format::R8G8B8A8_UNORM;
    // RT3 — RG16F  screen-space NDC-delta velocity (curNDC - prevNDC).
    //              Written by cluster_bindless.frag GBUFFER_OUTPUT branch;
    //              deferred resolve does NOT consume it.  Exposed via
    //              getVelocityBuffer() so future passes (TAA, motion
    //              blur, temporal reprojection / upscaling) can sample
    //              it.  Half-float because NDC delta is small and signed
    //              and we want sub-pixel precision; R8G8 would clip and
    //              quantise badly at sub-pixel scales.
    er::Format gbuf_velocity_format_       = er::Format::R16G16_SFLOAT;
    er::TextureInfo gbuf_albedo_ao_;
    er::TextureInfo gbuf_normal_rough_;
    er::TextureInfo gbuf_emissive_metal_;
    er::TextureInfo gbuf_velocity_;

    // ── Hi-Z occlusion pyramid (Nanite-style two-pass culling) ───────────
    // R32_SFLOAT mip-chained texture, log2(maxDim)+1 levels.  Mip 0 holds
    // the post-Phase-A depth (max-reduced from the actual depth buffer),
    // each higher mip stores the max of its 2x2 source — conservative-far
    // Z so a cluster is only rejected if its near-plane depth is greater
    // than the Hi-Z's far plane at that screen footprint.  Built each
    // frame between Phase A render and Phase B cull.
    //
    // Lifetime: allocated alongside the G-buffer (createGBuffer) and torn
    // down with it (cleanupSwapChain).  hiz_pyramid_size_ is the size of
    // mip 0 (next-power-of-two-up of screen extent so 2x2 reductions don't
    // lose half a pixel at odd sizes).
    er::Format     hiz_format_ = er::Format::R32_SFLOAT;
    er::TextureInfo hiz_pyramid_;
    glm::uvec2     hiz_pyramid_size_   = glm::uvec2(0);
    uint32_t       hiz_pyramid_mips_   = 0;
    // Per-mip image views — one per mip, used as the writeonly storage
    // image binding when building that mip.  Mip-0 is special: it's
    // populated by sampling the scene depth buffer directly, while mips
    // 1..N-1 are populated by sampling the previous mip.  hiz_build.comp
    // takes the source mip as a sampled image and the dest mip as a
    // storage image.
    std::vector<std::shared_ptr<er::ImageView>> hiz_mip_views_;
    // Full-pyramid sampled view (all mips).  Bound as a single sampler2D
    // by downstream consumers so they can pick mips with textureLod():
    // cluster cull (Phase B) for occlusion testing, and the deferred
    // resolve in DEBUG_RENDER_MODE_HIZ for the visualisation.
    std::shared_ptr<er::ImageView> hiz_pyramid_full_view_;

    std::shared_ptr<er::Sampler>             hiz_sampler_;
    std::shared_ptr<er::DescriptorSetLayout> hiz_build_desc_set_layout_;
    std::shared_ptr<er::PipelineLayout>      hiz_build_pipeline_layout_;
    std::shared_ptr<er::Pipeline>            hiz_build_pipeline_;
    // One descriptor set per mip we're building (mips 0..N-1).  Each
    // binds source = previous mip (or scene depth for mip 0) and dest =
    // this mip's storage view.
    std::vector<std::shared_ptr<er::DescriptorSet>> hiz_build_desc_sets_;

    // Last frame's view-projection matrix.  Passed into the cluster cull
    // dispatch so the Hi-Z occlusion test can reproject cluster bounds
    // into yesterday's screen space (matches the depth pyramid that was
    // built from yesterday's depth buffer).  Initialised to identity;
    // refreshed at the END of each frame after rendering uses it.
    glm::mat4 last_view_proj_ = glm::mat4(1.0f);
    // Pipeline format descriptor used when (re)creating the cluster G-buffer
    // pipeline.  Built once in initVulkan and shared with cluster_renderer_->
    // initBindlessGBufferPipeline().
    er::PipelineRenderbufferFormats gbuffer_renderbuffer_format_;

    // Compute resolve resources.
    std::shared_ptr<er::Sampler>             gbuf_sampler_;
    std::shared_ptr<er::DescriptorSetLayout> deferred_resolve_desc_set_layout_;
    std::shared_ptr<er::DescriptorSet>       deferred_resolve_desc_set_;
    std::shared_ptr<er::PipelineLayout>      deferred_resolve_pipeline_layout_;
    std::shared_ptr<er::Pipeline>            deferred_resolve_pipeline_;

    er::TextureInfo sample_tex_;
    er::TextureInfo ggx_lut_tex_;
    er::TextureInfo brdf_lut_tex_;
    er::TextureInfo charlie_lut_tex_;
    er::TextureInfo thin_film_lut_tex_;
    er::TextureInfo heightmap_tex_;
    er::TextureInfo map_mask_tex_;
    er::TextureInfo prt_base_tex_;
    er::TextureInfo prt_height_tex_;
    er::TextureInfo prt_normal_tex_;
    er::TextureInfo prt_orh_tex_;
    std::vector<uint32_t> binding_list_;
    std::shared_ptr<er::Sampler> texture_sampler_;
    std::shared_ptr<er::Sampler> repeat_texture_sampler_;
    std::shared_ptr<er::Sampler> mirror_repeat_sampler_;
    std::shared_ptr<er::Sampler> texture_point_sampler_;
    std::vector<std::shared_ptr<er::CommandBuffer>> command_buffers_;
    std::vector<std::shared_ptr<er::Semaphore>> image_available_semaphores_;
    std::vector<std::shared_ptr<er::Semaphore>> render_finished_semaphores_;
    std::vector<std::shared_ptr<er::Fence>> in_flight_fences_;
    std::vector<std::shared_ptr<er::Fence>> images_in_flight_;
    std::shared_ptr<er::Semaphore> init_semaphore_;

    std::vector<std::shared_ptr<ego::DrawableObject>> drawable_objects_;
    std::shared_ptr<ego::DrawableObject> player_object_;
    // Drives the scene-skinned.gltf rig from WASD input each frame —
    // procedural pose (idle bob + walk cycle) plus terrain clamp and
    // capsule collision against the level.
    std::unique_ptr<ego::PlayerController> player_controller_;

    // ── Debug: one small red cube per joint, pinned to the player rig's
    // bones each frame ────────────────────────────────────────────────
    // Drawn through the regular DrawableObject pipeline (debug_force_red
    // + tiny debug_scale) so the markers obey depth / fog / lighting like
    // the rest of the scene.  Asset is assets/debug_cube.gltf — loaded
    // exactly ONCE; every marker shares that DrawableData via the in-
    // flight load dedup in DrawableObject::createAsync, and each cube is
    // placed at its bone via setInstanceRootTransform (which composes a
    // per-instance world into model_mat without touching the shared
    // node TRS, so siblings don't clobber each other).  Together the 19
    // cubes trace the live skeleton independent of the skinned mesh: if
    // the markers animate while the mesh stays in T-pose the skin path is
    // bypassed; if neither animates the bone chain itself isn't receiving
    // our writes.  Order matches kBoneNames in application.cpp (which
    // mirrors the auto-rig's getStandardJointNames).
    std::vector<std::shared_ptr<ego::DrawableObject>> bone_markers_;

    // ── Bone-link sticks ──────────────────────────────────────────────
    // One stretched debug-cube per non-root joint, drawn between its
    // parent joint and itself so the skeleton reads as connected sticks
    // instead of a sparse cloud of cubes.  Same shared mesh as
    // bone_markers_ (deduped by createAsync's in-flight cache) but with a
    // non-uniform per-instance scale: long along the bone axis, thin in
    // the other two axes.  Index lines up with bone_markers_; entry 0
    // (the root "hips") has no parent and is left null and skipped.
    std::vector<std::shared_ptr<ego::DrawableObject>> bone_links_;

    // Triangle-mesh collision world. Populated once both bistro scenes
    // are isReady(); PlayerController consults it every frame.
    engine::helper::CollisionWorld collision_world_;
    bool collision_world_built_ = false;
    std::shared_ptr<ego::DrawableObject> bistro_exterior_scene_;
    std::shared_ptr<ego::DrawableObject> bistro_interior_scene_;

    // Ground probe used by PlayerController's foot IK.  Given a world XZ
    // column and an expected foot level (y_hint), returns the ground
    // height (out_y) + unit surface normal (out_normal) beneath it.  Tries
    // the walkable collision world first, then the raw rendered scene
    // geometry, so feet are grounded even before / without the LLM
    // walkable-surface classification.  Returns false when nothing hit.
    bool queryGroundAt(float x, float z, float y_hint,
                       float& out_y, glm::vec3& out_normal);

    // Async mesh loader. Lives for the whole application run; its
    // worker thread is joined in the destructor. See
    // game_object/mesh_load_task_manager.{h,cpp} for the three-phase
    // model and startup/menu wiring below.
    std::unique_ptr<ego::MeshLoadTaskManager> mesh_load_task_manager_;
    std::shared_ptr<es::Skydome> skydome_;
    std::shared_ptr<es::WeatherSystem> weather_system_;
    std::shared_ptr<es::VolumeCloud> volume_cloud_;
    std::shared_ptr<es::SSAO> ssao_;
    std::shared_ptr<es::ClusterRenderer> cluster_renderer_;
    // Runtime Virtual Texture manager — owns the per-layer 4096²
    // pool textures and the global page table SSBO.  Materials
    // register their textures via vt_manager_->registerTextureFromImage()
    // during mesh upload; the returned VirtualTextureId is stored in
    // BindlessMaterialParams and consumed by the sampling shaders'
    // vtSample* helpers.  See virtual_texture.h for full architecture.
    std::shared_ptr<es::VirtualTextureManager> vt_manager_;
    // Monotonic frame counter passed to both vt_manager_->tick() (at
    // frame start, consumes previous frame's compact slot) and
    // vt_manager_->compactFeedback() (after cluster bindless draw,
    // produces this frame's compact slot).  Both calls must see the
    // SAME counter value within one frame so the per-FIF slot indexing
    // (idx % kVtCompactSlots) lines up — separate static counters at
    // the two call sites would drift if either call were skipped.
    uint64_t vt_frame_index_ = 0;
    // Real-time reflection cubemap centred at the active ambient probe.
    // Captures one face per frame; depth-aware reprojection keeps the
    // other 5 faces consistent as the probe origin moves.  See
    // scene_rendering/dynamic_cubemap.h.
    std::shared_ptr<es::DynamicCubemap> dynamic_cubemap_;
    // Auto-placed grid of SH ambient probes; lit by moving the dynamic
    // cubemap between probe positions and projecting each captured cube
    // into 9 SH coefficients per probe.  See
    // scene_rendering/ambient_probe_system.h.
    std::shared_ptr<es::AmbientProbeSystem> ambient_probe_system_;
    std::shared_ptr<es::VolumeNoise> volume_noise_;
    std::shared_ptr<es::IblCreator> ibl_creator_;
    std::shared_ptr<ego::ConemapObj> conemap_obj_;
    std::shared_ptr<ego::Plane> unit_plane_;
    std::shared_ptr<ego::Box> unit_box_;
    std::shared_ptr<ego::Sphere> unit_sphere_;
    std::shared_ptr<ego::ConemapTest> conemap_test_;
    std::shared_ptr<ego::HairPatch> hair_patch_;
    std::shared_ptr<ego::HairTest> hair_test_;
    std::shared_ptr<ego::LbmPatch> lbm_patch_;
    std::shared_ptr<ego::LbmTest> lbm_test_;
    std::shared_ptr<es::Conemap> conemap_gen_;
    std::shared_ptr<es::PrtShadow> prt_shadow_gen_;
    std::shared_ptr<engine::ray_tracing::RayTracingBase> ray_tracing_test_;
    std::shared_ptr<engine::ui::Menu> menu_;

    std::shared_ptr<ego::ObjectViewCameraObject> main_camera_object_;
    std::shared_ptr<ego::ShadowViewCameraObject> shadow_camera_object_;
    std::shared_ptr<es::ObjectSceneView> object_scene_view_;
    std::shared_ptr<es::ObjectSceneView> shadow_object_scene_view_;
    std::shared_ptr<es::TerrainSceneView> terrain_scene_view_;
    std::shared_ptr<er::BufferInfo> runtime_lights_buffer_;

    // CSM: 2048x2048x4 depth array texture, one layer per cascade.
    std::shared_ptr<er::TextureInfo> csm_shadow_tex_;
    std::array<std::shared_ptr<er::ImageView>, CSM_CASCADE_COUNT> csm_layer_views_;

    // CSM debug visualisation: small per-cascade R8G8B8A8 colour targets.
    static constexpr uint32_t kCsmDebugSize = 512;
    std::array<er::TextureInfo, CSM_CASCADE_COUNT> csm_debug_color_;
    std::shared_ptr<er::DescriptorSetLayout>  csm_debug_desc_set_layout_;
    std::shared_ptr<er::DescriptorSet>        csm_debug_desc_set_;
    std::shared_ptr<er::PipelineLayout>       csm_debug_pipeline_layout_;
    std::shared_ptr<er::Pipeline>             csm_debug_pipeline_;

    std::vector<er::ClearValue> clear_values_;

    glsl::ViewParams view_params_{};
    std::chrono::high_resolution_clock::time_point last_frame_time_point_;
    float current_time_ = 0;
    float delta_t_ = 0;

    // menu
    bool show_gltf_selection_ = false;
    bool dump_volume_noise_ = false;

    uint64_t current_frame_ = 0;
    bool framebuffer_resized_ = false;

    // GPU profiler (timestamp-based, frame-by-frame hierarchy).
    engine::helper::GpuProfiler gpu_profiler_;
    bool gpu_profiler_initialized_ = false;

    // Wall-clock timestamp of the last GPU-profile log dump.
    // Used to throttle log output to once per second.
    std::chrono::steady_clock::time_point gpu_profile_last_dump_{};

    // Plugin system.
    plugins::PluginManager plugin_manager_;
};

}// namespace app
}// namespace work
