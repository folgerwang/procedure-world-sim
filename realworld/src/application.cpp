#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <array>
#include <map>
#include <limits>
#include <chrono>
#include <cmath>
#include <future>
#include <string>
#include <filesystem>
#include "Windows.h"

// glm/gtc helpers used by the player pivot-offset compensation in the
// spawn + per-frame follow blocks below (glm::angleAxis, glm::mat4_cast,
// glm::translate, glm::scale).  Core glm.hpp transit comes through the
// engine headers but the gtc extensions don't, so include them here.
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "ray_tracing/raytracing_callable.h"
#include "ray_tracing/raytracing_shadow.h"
#include "helper/engine_helper.h"
#include "helper/cluster_mesh.h"
#include "helper/material_classifier.h"
#include "application.h"

namespace er = engine::renderer;
namespace ego = engine::game_object;

namespace {
constexpr int kWindowSizeX = 2560;
constexpr int kWindowSizeY = 1440;
static int s_update_frame_count = -1;
static bool s_render_prt_test = false;
static bool s_render_hair_test = false;
static bool s_render_lbm_test = false;
static bool s_bistro_scene_test = true;

// global pbr texture descriptor set layout.
std::shared_ptr<er::DescriptorSetLayout> createPbrLightingDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(6);

    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        GGX_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        CHARLIE_LUT_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        DIRECT_SHADOW_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        LAMBERTIAN_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        GGX_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));
    bindings.push_back(er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        CHARLIE_ENV_TEX_INDEX,
        SET_2_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT)));

    return device->createDescriptorSetLayout(bindings);
}

// global pbr texture descriptor set layout.
std::shared_ptr<er::DescriptorSetLayout> createRuntimeLightsDescriptorSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(1);

    bindings.push_back(er::helper::getBufferDescriptionSetLayoutBinding(
        RUNTIME_LIGHTS_CONSTANT_INDEX,
        // FRAGMENT: forward-pass lighting math reads RuntimeLightsParams.
        // COMPUTE:  cluster_cull.comp reads the translucent flag.
        // GEOMETRY: base_depthonly_csm.geom (non-cluster shadow path)
        //           reads light_view_proj[] for layered broadcast.
        // TASK:     cluster_bindless_shadow.task — per-cluster cull
        //           against all CSM_CASCADE_COUNT cascade frustums.
        // MESH:     cluster_bindless_shadow.mesh — mesh-shader CSM path
        //           reads light_view_proj[cascade] to project cluster
        //           vertices into the matching cascade's depth layer.
        // VERTEX:   base_depthonly.vert CSM_PER_CASCADE permutation
        //           (DrawMode::kCsmPerCascade — "Regular" option on the
        //           shadow draw-mode menu) reads
        //           light_view_proj[model_params.cascade_idx] to
        //           per-cascade-loop without a GS or mesh shader.
        SET_6_FLAG_BITS(ShaderStage, FRAGMENT_BIT, COMPUTE_BIT,
                        GEOMETRY_BIT, MESH_BIT_EXT, TASK_BIT_EXT,
                        VERTEX_BIT),
        er::DescriptorType::UNIFORM_BUFFER));

    return device->createDescriptorSetLayout(bindings);
}
}

namespace work {
namespace app {

void RealWorldApplication::run() {
    auto error_strings =
        eh::initCompileGlobalShaders(
            "src\\sim_engine\\shaders",
            "lib\\shaders",
            "src\\sim_engine\\third_parties\\vulkan_lib");
    if (error_strings.length() > 0) {
        MessageBoxA(NULL, error_strings.c_str(), "Shader Error!", MB_OK);
    }
    initWindow();
    initVulkan();
    initDrawFrame();
    mainLoop();
    cleanup();
}

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<work::app::RealWorldApplication*>(glfwGetWindowUserPointer(window));
    app->setFrameBufferResized(true);
}

static bool s_exit_game = false;
static bool s_game_paused = false;
static bool s_camera_paused = false;
static bool s_mouse_init = false;
static bool s_mouse_right_button_pressed = false;
static glm::vec2 s_last_mouse_pos;
static float s_mouse_wheel_offset = 0.0f;
static int s_key = 0;
static bool s_toggle_profiler_pause = false;
// Edge-trigger for the F1 keypress. Consumed in drawFrame() and
// forwarded to Menu::toggleCollisionDebug() so the menu's checkbox
// and the F1 hotkey share one canonical flag.
static bool s_request_toggle_collision_debug = false;
// Edge-trigger for Left/Right arrow keys: step the isolate-debug
// collision-mesh index by -1 / +1.  Consumed in drawFrame() and routed
// to Menu::stepCollisionIsolate so the slider and the hotkeys stay in
// sync.  The camera uses W/A/S/D, so the arrow keys are free.
static int s_collision_isolate_step = 0;

static void keyInputCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    s_key = 0;
    if (action != GLFW_RELEASE && !s_camera_paused) {
        s_key = key;
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        s_exit_game = true;
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
        s_toggle_profiler_pause = true;
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_F1) {
        // F1 is a shortcut for the "Game Objects -> Debug Collision Mesh"
        // menu item.  We can't touch the menu directly from a GLFW
        // callback, so set an edge-trigger that drawFrame() consumes.
        s_request_toggle_collision_debug = true;
    }
    // Left/Right arrows step the isolate-debug collision mesh index.
    // Accept REPEAT too, so holding the key scrubs quickly through the
    // mesh list.  drawFrame() consumes the step (and auto-enables
    // isolate mode), so this works even when the menu/slider is hidden
    // behind the on-screen clock.
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_LEFT)  s_collision_isolate_step = -1;
        if (key == GLFW_KEY_RIGHT) s_collision_isolate_step = +1;
    }
}

void mouseInputCallback(GLFWwindow* window, double xpos, double ypos)
{
    glm::vec2 cur_mouse_pos = glm::vec2(xpos, ypos);
    if (!s_mouse_init)
    {
        s_last_mouse_pos = cur_mouse_pos;
        s_mouse_init = true;
    }

    s_last_mouse_pos = cur_mouse_pos;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS) {
            s_mouse_right_button_pressed = true;
        }
        else if (action == GLFW_RELEASE) {
            s_mouse_right_button_pressed = false;
        }
    }
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset) {
    s_mouse_wheel_offset = static_cast<float>(yoffset);
}

void RealWorldApplication::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(kWindowSizeX, kWindowSizeY, "Real World", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    glfwSetKeyCallback(window_, keyInputCallback);
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window_, mouseInputCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetScrollCallback(window_, mouseWheelCallback);
}

void RealWorldApplication::createDepthResources(const glm::uvec2& display_size) {
    auto depth_format =
        er::Helper::findDepthFormat(device_);

    er::Helper::createDepthResources(
        device_,
        depth_format,
        display_size,
        depth_buffer_,
        std::source_location::current());

    // Format MUST match the blit source — object_scene_view_'s depth
    // buffer is D24_UNORM_S8_UINT (see initVulkan kForward.depth_format).
    // vkCmdBlitImage between mismatched depth formats is undefined per
    // the Vulkan spec; the previous D32_SFLOAT here meant the blitted
    // bits got reinterpreted as floats and produced regular-looking
    // garbage values (visible as moiré stripes in the SSAO debug
    // visualisation, with sky pixels reading as ~0.5–0.9 instead of
    // 1.0 so the early-out for sky never fired).
    // The view created from a depth format auto-uses DEPTH_BIT aspect
    // (renderer.cpp ~line 560), so shader sampling still returns a
    // float in [0,1] — no consumer change needed.
    er::Helper::create2DTextureImage(
        device_,
        er::Format::D24_UNORM_S8_UINT,
        display_size,
        1,
        depth_buffer_copy_,
        SET_3_FLAG_BITS(ImageUsage, SAMPLED_BIT, DEPTH_STENCIL_ATTACHMENT_BIT, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());
}

void RealWorldApplication::createHdrColorBuffer(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_,
        hdr_format_,
        display_size,
        1,
        hdr_color_buffer_,
        SET_4_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT, COLOR_ATTACHMENT_BIT, TRANSFER_SRC_BIT),
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());
}

void RealWorldApplication::createColorBufferCopy(const glm::uvec2& display_size) {
    er::Helper::create2DTextureImage(
        device_,
        hdr_format_,
        display_size,
        1,
        hdr_color_buffer_copy_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, TRANSFER_DST_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());
}

// ─── Deferred G-buffer + compute resolve ─────────────────────────────────
// Phase 1 of the deferred-rendering migration.  See application.h for the
// G-buffer layout and the rationale for the (small) RGBA8 formats; the
// compute resolve runs PBR once per pixel and writes hdr_color_buffer_.
//
// Lifetime: createGBuffer is called from recreateRenderBuffer so the
// G-buffer always matches the swap-chain extent.  initDeferredResolve runs
// once at startup (after descriptor pool + IBL textures are ready).
// writeDeferredResolveDescriptors is called from createGBuffer so the
// (re-allocated) image views are re-bound on every resize.
void RealWorldApplication::createGBuffer(const glm::uvec2& display_size) {
    auto usage = SET_3_FLAG_BITS(
        ImageUsage,
        SAMPLED_BIT,
        COLOR_ATTACHMENT_BIT,
        TRANSFER_DST_BIT);

    er::Helper::create2DTextureImage(
        device_,
        gbuf_albedo_ao_format_,
        display_size,
        1,
        gbuf_albedo_ao_,
        usage,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());

    er::Helper::create2DTextureImage(
        device_,
        gbuf_normal_rough_format_,
        display_size,
        1,
        gbuf_normal_rough_,
        usage,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());

    er::Helper::create2DTextureImage(
        device_,
        gbuf_emissive_metal_format_,
        display_size,
        1,
        gbuf_emissive_metal_,
        usage,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());

    // RT3 — screen-space NDC-delta velocity (curNDC - prevNDC).
    // Allocated alongside the rest of the G-buffer; sampled by future
    // TAA / motion blur passes via getVelocityBuffer().
    er::Helper::create2DTextureImage(
        device_,
        gbuf_velocity_format_,
        display_size,
        1,
        gbuf_velocity_,
        usage,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        std::source_location::current());

    // Re-bind the freshly-allocated image views to the resolve descriptor
    // set.  No-op until initDeferredResolve has run.
    if (deferred_resolve_desc_set_) {
        writeDeferredResolveDescriptors();
    }

    // Hi-Z pyramid scales with the swap-chain too.
    createHiZPyramid(display_size);
}

void RealWorldApplication::createHiZPyramid(const glm::uvec2& display_size) {
    // Hi-Z mip 0 is HALF the scene-depth resolution — the standard layout
    // for a Hi-Z pyramid (Frostbite / Nanite / Doom Eternal all do this).
    // It keeps the build shader's 2×2 max reduction natural at every
    // level: mip 0 samples scene depth (full res) at a 2:1 ratio to
    // produce the half-res mip 0; mips 1..N-1 are 2:1 reductions of the
    // previous mip.  Higher mips ceil-divide so the last column/row is
    // never dropped: size[k+1] = (size[k] + 1) / 2.  Mip count runs until
    // the bigger dimension reaches 1.  E.g. 1280×720 →
    //   mip 0 640×360 → 320×180 → 160×90 → 80×45 → 40×23 → 20×12 →
    //   10×6 → 5×3 → 3×2 → 2×1 → 1×1   = 11 mips.
    glm::uvec2 size = {
        (display_size.x + 1u) / 2u,
        (display_size.y + 1u) / 2u };
    if (size.x == 0) size.x = 1;
    if (size.y == 0) size.y = 1;
    uint32_t max_dim = std::max(size.x, size.y);
    // Mip count must match Vulkan's own validation rule for VkImageCreateInfo,
    // which is floor(log2(maxDim)) + 1.  Vulkan's per-mip extent is computed
    // with bit-shift (floor) halving: mip[i] = max(1, baseDim >> i).  Using
    // ceil halving `(d + 1) / 2` here overcounts by 1 for non-power-of-2
    // dimensions — e.g. 1280 ceil-halves through 1280→640→320→160→80→40→
    // 20→10→5→3→2→1 (12 steps) but Vulkan only permits 11 mips for that
    // width.  Use floor halving to stay in sync with the spec's max.
    uint32_t mips = 1;
    {
        uint32_t d = max_dim;
        while (d > 1) { d >>= 1u; ++mips; }
    }

    // Tear down any previous allocation.
    hiz_mip_views_.clear();
    hiz_pyramid_full_view_.reset();
    if (hiz_pyramid_.image) {
        hiz_pyramid_.destroy(device_);
    }

    hiz_pyramid_size_ = size;
    hiz_pyramid_mips_ = mips;

    // STORAGE_BIT — written by hiz_build.comp via image2D.
    // SAMPLED_BIT — read by the next mip's source binding AND by the Phase B
    //               cluster cull shader once we wire that up.
    er::Helper::create2DTextureImage(
        device_,
        hiz_format_,
        size,
        mips,
        hiz_pyramid_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        er::ImageLayout::GENERAL,
        std::source_location::current());

    // Per-mip image views — needed because hiz_build.comp binds dst as a
    // mip-0 storage image at each dispatch (a single-mip view), not the
    // full chain.
    hiz_mip_views_.reserve(mips);
    for (uint32_t m = 0; m < mips; ++m) {
        hiz_mip_views_.push_back(device_->createImageView(
            hiz_pyramid_.image,
            er::ImageViewType::VIEW_2D,
            hiz_format_,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current(),
            /*base_mip*/  m,
            /*mip_count*/ 1));
    }

    // Full-pyramid sampled view — covers all mips so the cull / debug
    // shaders can pick a level with textureLod().  Separate from
    // hiz_mip_views_ which are single-mip storage views.
    hiz_pyramid_full_view_ = device_->createImageView(
        hiz_pyramid_.image,
        er::ImageViewType::VIEW_2D,
        hiz_format_,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current(),
        /*base_mip*/  0,
        /*mip_count*/ mips);

    // Full-pyramid view — covers ALL mips so the cluster cull shader's
    // textureLod(samp, uv, mip) call can pick the right mip for the
    // cluster's projected footprint.  The default view created by
    // create2DTextureImage only covers mip 0.
    hiz_pyramid_full_view_ = device_->createImageView(
        hiz_pyramid_.image,
        er::ImageViewType::VIEW_2D,
        hiz_format_,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current(),
        /*base_mip*/  0,
        /*mip_count*/ mips);

    // Push the new pyramid handles into the cluster renderer so binding
    // 11 of the cull descriptor set sees them next dispatch.  No-op if
    // the renderer hasn't been constructed yet (very-first-init case);
    // initVulkan calls createHiZPyramid again after cluster_renderer_
    // exists.
    if (cluster_renderer_ && hiz_sampler_) {
        cluster_renderer_->setHiZTexture(
            hiz_sampler_, hiz_pyramid_full_view_,
            hiz_pyramid_size_, hiz_pyramid_mips_);
    }

    // Refresh the per-mip Hi-Z build descriptor sets so each set's source
    // binding points at the previous mip (or the scene depth view, for
    // mip 0).  Mip 0 reads object_scene_view_'s LIVE depth attachment —
    // the same image Phase A's render pass just wrote into.  This is
    // the true Nanite-style two-pass occlusion path: Phase A draws
    // previously-visible clusters, the Hi-Z pyramid summarises THAT
    // depth, Phase B tests every cluster against the pyramid.  Because
    // the pyramid + the cull shader's projection both refer to THIS
    // frame's coordinate system, no last_view_proj reprojection is
    // needed.  An extra layout transition (DEPTH_ATTACHMENT →
    // SHADER_READ_ONLY → DEPTH_ATTACHMENT) sits around the Hi-Z build
    // dispatch in drawScene; cheap relative to the cull/render cost.
    if (hiz_build_pipeline_) {
        std::shared_ptr<er::ImageView> phase_a_depth_view;
        if (object_scene_view_ && object_scene_view_->getDepthBuffer()) {
            // Bind depth_buffer_copy_ (held in SHADER_READ_ONLY_OPTIMAL by
            // the end-of-frame blit) rather than the LIVE object_scene_view_
            // depth attachment.  writeHiZBuildDescriptors declares the
            // descriptor layout as SHADER_READ_ONLY_OPTIMAL; pointing it at
            // the live depth target trips VUID-vkCmdDraw-None-09600 every
            // frame because that image is in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            // when the dispatch consumes the descriptor.  The dispatch's
            // mip-0 push-constant size already references depth_buffer_copy_,
            // so this also aligns the descriptor with the size the shader
            // assumes.  Functionally identical: depth_buffer_copy_ holds
            // last-frame's depth that the Hi-Z build was already designed to
            // consume (see the comment at the Hi-Z dispatch site).
            phase_a_depth_view = depth_buffer_copy_.view;
        }
        writeHiZBuildDescriptors(phase_a_depth_view);
    }
}

void RealWorldApplication::recreateRenderBuffer(const glm::uvec2& display_size) {
    createDepthResources(display_size);
    createHdrColorBuffer(display_size);
    createColorBufferCopy(display_size);
    createGBuffer(display_size);
    createFramebuffers(display_size);
}

void RealWorldApplication::initHiZBuild() {
    if (hiz_build_pipeline_) return;

    if (!hiz_sampler_) {
        // NEAREST min/mag is required for cluster_cull.comp's Hi-Z
        // occlusion test to stay conservative.  The cull shader samples
        // the pyramid at the four corners of each cluster's projected
        // AABB (not at texel centres) — under LINEAR filtering each tap
        // returns a bilinear blend of up to 4 mip texels, which can
        // produce a value smaller than every individual texel's stored
        // depth.  Because the pyramid is a max-of-far reduction, that
        // underestimate makes "ndc_z_near > pyramid_max_z" trigger on
        // visible geometry — false-positive occlusions.  In the
        // two-pass path those false rejections in Phase B are sticky:
        // the cluster never gets its visibility bit set, so Phase A
        // doesn't redraw it next frame, Hi-Z stays the same, and the
        // false rejection repeats — visually, geometry disappears and
        // pops back only when the camera moves enough to perturb the
        // sampling.
        //
        // hiz_build.comp samples at exact pixel centres (uv = (px+0.5)
        // / src_size), so for it LINEAR and NEAREST are equivalent —
        // changing this filter is a no-op for the build pass.  The
        // ImGui debug visualisation that also uses this sampler just
        // looks crisper, no correctness change.
        hiz_sampler_ = device_->createSampler(
            er::Filter::NEAREST,
            er::SamplerAddressMode::CLAMP_TO_EDGE,
            er::SamplerMipmapMode::NEAREST,
            /*anisotropy*/ 0.0f,
            std::source_location::current());
    }

    // Layout:
    //   binding 0 — sampler2D src  (previous mip OR scene depth)
    //   binding 1 — image2D dst    (this mip, r32f writeonly)
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        hiz_build_desc_set_layout_ =
            device_->createDescriptorSetLayout(bindings);
    }

    // Push constant: dst_size + src_size (uvec2 each) = 16 bytes.
    hiz_build_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device_, { hiz_build_desc_set_layout_ }, /*push_const_size*/ 16);

    hiz_build_pipeline_ = er::helper::createComputePipeline(
        device_, hiz_build_pipeline_layout_,
        "hiz_build_comp.spv",
        std::source_location::current());
}

void RealWorldApplication::writeHiZBuildDescriptors(
    const std::shared_ptr<er::ImageView>& scene_depth_view) {
    if (!hiz_build_desc_set_layout_ || hiz_pyramid_mips_ == 0) return;

    // Allocate one descriptor set per destination mip, freed implicitly
    // when the pool is destroyed.  Recreated on resize.
    hiz_build_desc_sets_.clear();
    hiz_build_desc_sets_.reserve(hiz_pyramid_mips_);
    for (uint32_t m = 0; m < hiz_pyramid_mips_; ++m) {
        hiz_build_desc_sets_.push_back(
            device_->createDescriptorSets(
                descriptor_pool_, hiz_build_desc_set_layout_, 1)[0]);
    }

    // Mip 0's source is the scene depth (read via depth-aspect view).
    // Higher mips source the Hi-Z mip directly above.  The pyramid stays
    // in GENERAL layout throughout the build chain (storage-image writes
    // need GENERAL, and sampler reads tolerate it), so mip-1+ source
    // bindings must declare GENERAL too.  Mip 0's source is the scene
    // depth which is in SHADER_READ_ONLY_OPTIMAL by the time the Hi-Z
    // build dispatches (caller transitions it).
    for (uint32_t m = 0; m < hiz_pyramid_mips_; ++m) {
        std::shared_ptr<er::ImageView> src_view =
            (m == 0) ? scene_depth_view : hiz_mip_views_[m - 1];
        const er::ImageLayout src_layout =
            (m == 0) ? er::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                     : er::ImageLayout::GENERAL;
        if (!src_view || !hiz_mip_views_[m]) continue;

        er::WriteDescriptorList writes;
        writes.reserve(2);
        er::Helper::addOneTexture(writes, hiz_build_desc_sets_[m],
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
            hiz_sampler_, src_view, src_layout);
        er::Helper::addOneTexture(writes, hiz_build_desc_sets_[m],
            er::DescriptorType::STORAGE_IMAGE, 1,
            nullptr, hiz_mip_views_[m],
            er::ImageLayout::GENERAL);
        device_->updateDescriptorSets(writes);
    }
}

void RealWorldApplication::buildHiZPyramid(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const glm::uvec2& scene_depth_size) {
    if (!hiz_build_pipeline_ || hiz_pyramid_mips_ == 0) return;
    if (hiz_build_desc_sets_.size() < hiz_pyramid_mips_) return;

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE, hiz_build_pipeline_);

    // Each iteration's src_size = previous iteration's dst_size, except
    // the very first iteration (mip 0) whose src is the scene depth.
    glm::uvec2 src_size = scene_depth_size;
    glm::uvec2 dst_size = hiz_pyramid_size_;   // pyramid mip 0

    // Reused barrier states — between mip writes and the next mip's reads
    // we only need an execution + memory dependency, no layout change
    // (pyramid stays in GENERAL).
    er::ImageResourceInfo from_write = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::ImageResourceInfo to_read = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };

    struct PushC {
        uint32_t dst_x, dst_y;
        uint32_t src_x, src_y;
    };

    for (uint32_t m = 0; m < hiz_pyramid_mips_; ++m) {
        if (m > 0) {
            // Make mip (m-1)'s writes visible to mip m's reads.
            cmd_buf->addImageBarrier(
                hiz_pyramid_.image, from_write, to_read,
                /*base_mip*/ m - 1, /*mip_count*/ 1, 0, 1);
        }

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            hiz_build_pipeline_layout_,
            { hiz_build_desc_sets_[m] });

        PushC pc = { dst_size.x, dst_size.y, src_size.x, src_size.y };
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            hiz_build_pipeline_layout_,
            &pc, sizeof(pc));

        const uint32_t G = 8u;
        uint32_t gx = (dst_size.x + G - 1u) / G;
        uint32_t gy = (dst_size.y + G - 1u) / G;
        cmd_buf->dispatch(gx, gy, 1);

        src_size = dst_size;
        dst_size = { std::max(1u, (dst_size.x + 1u) / 2u),
                     std::max(1u, (dst_size.y + 1u) / 2u) };
    }

    // Make the entire pyramid's writes visible to whatever samples it
    // next (the deferred resolve compute in DEBUG_RENDER_MODE_HIZ, the
    // Phase B cluster cull when wired up).
    cmd_buf->addImageBarrier(
        hiz_pyramid_.image, from_write, to_read,
        /*base_mip*/ 0, /*mip_count*/ hiz_pyramid_mips_, 0, 1);
}

// ─── Hi-Z pyramid build (per-frame) ─────────────────────────────────────────
// One dispatch per mip:
//   mip 0 reads scene depth (depth_buffer_copy_, in SHADER_READ_ONLY)
//   mip N reads mip N-1
// Each dispatch writes its destination mip in GENERAL layout and is
// followed by a SHADER_WRITE → SHADER_READ barrier so the next mip's
// source read sees the just-written values.  After all mips are built
// the entire pyramid sits in SHADER_READ_ONLY_OPTIMAL — exactly the
// layout the cluster cull pass binds at descriptor binding 11.
//
// Push constants per dispatch: dst_size (2 uints) + src_size (2 uints).
//   src_size for mip 0 = depth_buffer_copy_ size (== swap chain extent
//                        when matching, but we already know hi-z mip 0
//                        is sized to next-pow2-of-extent so use that).
void RealWorldApplication::dispatchHiZBuild(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf) {

    if (!hiz_build_pipeline_ ||
        hiz_pyramid_mips_ == 0 ||
        hiz_build_desc_sets_.empty()) {
        return;
    }

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE, hiz_build_pipeline_);

    // Source dimensions for mip 0 reads.  depth_buffer_copy_ matches the
    // swap-chain extent; mip 0 is sized to next-pow2 of that, so we
    // store the actual depth size for this push constant and use the
    // pyramid mip dimensions for subsequent mips.
    glm::uvec2 mip0_src_size(
        depth_buffer_copy_.size.x, depth_buffer_copy_.size.y);

    er::ImageResourceInfo prev_info = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::ImageResourceInfo write_info = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    // Inter-mip barrier destination: GENERAL layout (image stays in
    // GENERAL between dispatches) but with SHADER_READ access bits set
    // so the next mip's `textureLod` sampler read can see the previous
    // mip's writes.  Without SHADER_READ in the destination access
    // mask, the barrier only synchronises write→write — symptom in
    // testing: cluster cull alternates between 0% and 100% Hi-Z hits
    // because reads of the pyramid sometimes catch the previous-mip
    // writes and sometimes don't.
    er::ImageResourceInfo gen_read_write = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT) |
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::ImageResourceInfo read_info = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };

    for (uint32_t m = 0; m < hiz_pyramid_mips_; ++m) {
        // Destination mip dimensions = floor(mip0 / 2^m), clamped to 1.
        glm::uvec2 dst_size(
            std::max(1u, hiz_pyramid_size_.x >> m),
            std::max(1u, hiz_pyramid_size_.y >> m));
        glm::uvec2 src_size = (m == 0)
            ? mip0_src_size
            : glm::uvec2(
                std::max(1u, hiz_pyramid_size_.x >> (m - 1)),
                std::max(1u, hiz_pyramid_size_.y >> (m - 1)));

        // Transition this mip subresource to GENERAL for the write.
        // er::Helper::transitMapTextureToStoreImage handles the whole
        // image — fine here because each mip is a distinct subresource
        // of the same image and we only ever write one at a time.
        // For stricter per-mip transitions a custom barrier with mip
        // ranges would be needed; sufficient for now.
        if (m == 0) {
            er::helper::transitMapTextureToStoreImage(
                cmd_buf, { hiz_pyramid_.image });
        }

        // Push constants: dst_size + src_size (16 bytes total).
        struct HiZPC {
            glm::uvec2 dst_size;
            glm::uvec2 src_size;
        } pc;
        pc.dst_size = dst_size;
        pc.src_size = src_size;
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            hiz_build_pipeline_layout_, &pc, sizeof(pc));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::COMPUTE,
            hiz_build_pipeline_layout_,
            { hiz_build_desc_sets_[m] });

        // 8×8 workgroups.
        const uint32_t gx = (dst_size.x + 7u) / 8u;
        const uint32_t gy = (dst_size.y + 7u) / 8u;
        cmd_buf->dispatch(gx, gy, 1);

        // Insert a SHADER_WRITE → SHADER_READ|WRITE barrier between
        // mips so the next dispatch's textureLod read of the previous
        // mip is correctly synchronized AND it can also write its own
        // mip in GENERAL layout.  Skip after the last mip — caller
        // transitions the whole pyramid back to SHADER_READ_ONLY below.
        //
        // The addImageBarrier signature is (image, src, dst, base_mip=0,
        // mip_count=1, base_layer=0, layer_count=1) — no aspect-mask
        // parameter.  An earlier version of this call incorrectly passed
        // SET_FLAG_BIT(ImageAspect, COLOR_BIT) as base_mip and shifted
        // everything else right, which produced base_mip = 1, mip_count
        // = 0, base_layer = hiz_pyramid_mips_ (e.g. 11) — three back-to-
        // back VUID violations per Hi-Z build per frame (levelCount = 0,
        // baseArrayLayer ≥ arrayLayers, base+layerCount > arrayLayers).
        // Aspect is auto-derived from layout (COLOR for GENERAL).
        if (m + 1 < hiz_pyramid_mips_) {
            cmd_buf->addImageBarrier(
                hiz_pyramid_.image,
                write_info, gen_read_write,
                /*base_mip*/ 0, /*mip_count*/ hiz_pyramid_mips_,
                /*base_layer*/ 0, /*layer_count*/ 1);
        }
    }

    // Final pyramid sync.  Keep layout in GENERAL — cluster cull's
    // binding 11 declares GENERAL (cluster_renderer.cpp:1268) and the
    // hiz_build descriptor sets for mip-1+ sources declare GENERAL too
    // (application.cpp:512).  Transitioning to SHADER_READ_ONLY_OPTIMAL
    // here was the bug: next frame's dispatch would try to use the
    // image at GENERAL via descriptors written assuming GENERAL, and
    // the validation layer flagged it (VUID-VkDescriptorImageInfo-
    // imageLayout-00344) every frame.  We only need a memory
    // dependency from compute-write to subsequent sampler reads,
    // without changing the layout.
    er::ImageResourceInfo from_write_all = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
    er::ImageResourceInfo to_read_all = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_2_FLAG_BITS(PipelineStage,
            COMPUTE_SHADER_BIT, FRAGMENT_SHADER_BIT) };
    cmd_buf->addImageBarrier(
        hiz_pyramid_.image, from_write_all, to_read_all,
        /*base_mip*/ 0, /*mip_count*/ hiz_pyramid_mips_, 0, 1);
}

void RealWorldApplication::initDeferredResolve() {
    // Lazily init only once.  Called from initVulkan after descriptor pool,
    // PBR/runtime-light layouts, and IBL textures are ready (so the global
    // sets bound at dispatch time already have valid contents).
    if (deferred_resolve_pipeline_) {
        return;
    }

    // ── Sampler shared by all G-buffer reads in the resolve compute ──────
    // Linear filter is fine — the resolve runs at native screen resolution,
    // one texel per pixel, so the only effect of LINEAR vs NEAREST would be
    // half-texel shifts at attribute discontinuities, which we want to
    // smooth out anyway.  CLAMP_TO_EDGE because we never sample outside
    // the screen rect.
    if (!gbuf_sampler_) {
        gbuf_sampler_ = device_->createSampler(
            er::Filter::LINEAR,
            er::SamplerAddressMode::CLAMP_TO_EDGE,
            er::SamplerMipmapMode::NEAREST,
            /*anisotropy*/ 0.0f,
            std::source_location::current());
    }

    // ── Resolve descriptor set layout ─────────────────────────────────────
    //   binding 0   sampler2D  G-buffer RT0 (albedo + ao)
    //   binding 1   sampler2D  G-buffer RT1 (octahedral normal + roughness)
    //   binding 2   sampler2D  G-buffer RT2 (emissive + metallic)
    //   binding 3   sampler2D  depth (single-channel, used for world-pos
    //                          reconstruction via inv-VP)
    //   binding 4   image2D    HDR colour output (writeonly storage image)
    //   binding 5   sampler2D  G-buffer RT3 (velocity, NDC delta).  Not
    //                          consumed by the regular lighting math —
    //                          only sampled when render-debug mode is
    //                          DEBUG_RENDER_MODE_VELOCITY for the
    //                          motion-vector visualisation.
    //
    // Lighting data (sun/CSM and IBL) reuses the existing global descriptor
    // sets at PBR_GLOBAL_PARAMS_SET / VIEW_PARAMS_SET / RUNTIME_LIGHTS_PARAMS_SET,
    // so the resolve compute simply binds them at the same indices it would
    // in the forward fragment shader.  See deferred_resolve.comp.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(7);
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
                i, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        }
        bindings[4] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            4, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        bindings[5] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            5, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        // Binding 6 — full-pyramid Hi-Z sampler.  Only consumed by the
        // DEBUG_RENDER_MODE_HIZ visualisation; the regular lighting
        // path doesn't sample it.  Bound to hiz_pyramid_full_view_.
        bindings[6] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            6, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        deferred_resolve_desc_set_layout_ =
            device_->createDescriptorSetLayout(bindings);
    }

    // ── Pipeline layout ──
    // Mirror the forward cluster pipeline's set indices so deferred_resolve
    // .comp can use the same `set = N, binding = M` pattern as the forward
    // shaders for IBL / camera / runtime lights.  An empty layout is dropped
    // at SKIN_PARAMS_SET (3) since the resolve doesn't skin anything.
    auto empty_layout = device_->createDescriptorSetLayout({});
    er::DescriptorSetLayoutList all_layouts(RUNTIME_LIGHTS_PARAMS_SET + 1);
    all_layouts[PBR_GLOBAL_PARAMS_SET]     = pbr_lighting_desc_set_layout_;
    all_layouts[VIEW_PARAMS_SET]           =
        ego::CameraObject::getViewCameraDescriptorSetLayout();
    all_layouts[PBR_MATERIAL_PARAMS_SET]   = deferred_resolve_desc_set_layout_;
    all_layouts[SKIN_PARAMS_SET]           = empty_layout;
    all_layouts[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_layout_;
    for (auto& l : all_layouts) if (!l) l = empty_layout;

    deferred_resolve_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device_, all_layouts, /*push_const_range_size*/ 0);

    deferred_resolve_pipeline_ = er::helper::createComputePipeline(
        device_, deferred_resolve_pipeline_layout_,
        "deferred_resolve_comp.spv",
        std::source_location::current());

    // ── Descriptor set ──
    // Allocated once; image-view bindings are refreshed by
    // writeDeferredResolveDescriptors on every swap-chain resize.
    deferred_resolve_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, deferred_resolve_desc_set_layout_, 1)[0];

    writeDeferredResolveDescriptors();
}

void RealWorldApplication::writeDeferredResolveDescriptors(
    const std::shared_ptr<er::ImageView>& output_color_view) {
    if (!deferred_resolve_desc_set_ || !gbuf_sampler_) return;
    if (!gbuf_albedo_ao_.view || !gbuf_normal_rough_.view ||
        !gbuf_emissive_metal_.view) {
        return;
    }

    // Output target: explicit override → caller view; else fall back to
    // object_scene_view_'s color buffer (the engine's main scene target);
    // else hdr_color_buffer_ as a last-resort fallback (used at startup
    // before object_scene_view_ is constructed — in that window deferred
    // mode can't be on yet anyway).
    std::shared_ptr<er::ImageView> out_view = output_color_view;
    if (!out_view && object_scene_view_) {
        auto cb = object_scene_view_->getColorBuffer();
        if (cb) out_view = cb->view;
    }
    if (!out_view) out_view = hdr_color_buffer_.view;
    if (!out_view) return;

    // Depth source — must be the SAME depth target the cluster G-buffer
    // pass writes (= object_scene_view_'s depth buffer when present, since
    // that's the buffer the rest of the scene uses too).  Falls back to
    // the application-level depth_buffer_ when object_scene_view_ hasn't
    // been built yet (initVulkan startup ordering).
    std::shared_ptr<er::ImageView> depth_view;
    if (object_scene_view_) {
        auto db = object_scene_view_->getDepthBuffer();
        if (db) depth_view = db->view;
    }
    if (!depth_view) depth_view = depth_buffer_.view;
    if (!depth_view) return;

    er::WriteDescriptorList writes;
    writes.reserve(7);
    er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
        gbuf_sampler_, gbuf_albedo_ao_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
        gbuf_sampler_, gbuf_normal_rough_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 2,
        gbuf_sampler_, gbuf_emissive_metal_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 3,
        gbuf_sampler_, depth_view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
        er::DescriptorType::STORAGE_IMAGE, 4,
        nullptr, out_view,
        er::ImageLayout::GENERAL);
    if (gbuf_velocity_.view) {
        er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 5,
            gbuf_sampler_, gbuf_velocity_.view,
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
    // Full-pyramid Hi-Z sampler — consumed only by the
    // DEBUG_RENDER_MODE_HIZ visualisation.  Stays in GENERAL throughout
    // the build chain; the descriptor layout matches.  Both the view AND
    // the sampler must exist; this function is also called from inside
    // initDeferredResolve, which runs BEFORE initHiZBuild creates
    // hiz_sampler_, so on first call the sampler is null and we have to
    // skip the binding (a fresh writeDeferredResolveDescriptors after
    // initHiZBuild fills it in — see initVulkan ordering).
    if (hiz_pyramid_full_view_ && hiz_sampler_) {
        er::Helper::addOneTexture(writes, deferred_resolve_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER, 6,
            hiz_sampler_, hiz_pyramid_full_view_,
            er::ImageLayout::GENERAL);
    }
    device_->updateDescriptorSets(writes);
}

void RealWorldApplication::createRenderPasses() {
    assert(device_);
    final_render_pass_ = er::helper::createRenderPass(
        device_,
        swap_chain_info_.format,
        depth_format_,
        std::source_location::current(),
        false,
        er::SampleCountFlagBits::SC_1_BIT,
        er::ImageLayout::PRESENT_SRC_KHR);
    hdr_render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            true);
    hdr_water_render_pass_ =
        er::helper::createRenderPass(
            device_,
            hdr_format_,
            depth_format_,
            std::source_location::current(),
            false);
}

void RealWorldApplication::initVulkan() {
    renderbuffer_formats_[int(er::RenderPasses::kDepthOnly)].depth_format =
        renderbuffer_formats_[int(er::RenderPasses::kForward)].depth_format =
        er::Format::D24_UNORM_S8_UINT;
    renderbuffer_formats_[int(er::RenderPasses::kForward)].color_formats.
        push_back(er::Format::B10G11R11_UFLOAT_PACK32);
    renderbuffer_formats_[int(er::RenderPasses::kShadow)].depth_format =
        er::Format::D16_UNORM;

    // ── Deferred G-buffer pipeline format descriptor ──────────────────────
    // 3 colour RTs + reused main depth.  Used when (re)creating the cluster
    // bindless GBUFFER variant pipeline; see ClusterRenderer::initBindless
    // GBufferPipeline.  Order matches the layout(location=N) outputs in
    // cluster_bindless.frag's GBUFFER_OUTPUT branch:
    //   loc 0  RT0  albedo+ao
    //   loc 1  RT1  octahedral normal + roughness
    //   loc 2  RT2  emissive + metallic
    gbuffer_renderbuffer_format_.color_formats = {
        gbuf_albedo_ao_format_,
        gbuf_normal_rough_format_,
        gbuf_emissive_metal_format_,
        gbuf_velocity_format_,
    };
    gbuffer_renderbuffer_format_.depth_format = er::Format::D24_UNORM_S8_UINT;

    static auto color_no_blend_attachment = er::helper::fillPipelineColorBlendAttachmentState();
    static auto color_blend_attachment =
        er::helper::fillPipelineColorBlendAttachmentState(
            SET_FLAG_BIT(ColorComponent, ALL_BITS),
            true,
            er::BlendFactor::ONE,
            er::BlendFactor::SRC_ALPHA,
            er::BlendOp::ADD,
            er::BlendFactor::ONE,
            er::BlendFactor::ZERO,
            er::BlendOp::ADD);
    static std::vector<er::PipelineColorBlendAttachmentState> color_no_blend_attachments(1, color_no_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> color_blend_attachments(1, color_blend_attachment);
    static std::vector<er::PipelineColorBlendAttachmentState> cube_color_no_blend_attachments(6, color_no_blend_attachment);

    auto single_no_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_no_blend_attachments));

    auto single_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(color_blend_attachments));

    auto cube_no_blend_state_info =
        std::make_shared<er::PipelineColorBlendStateCreateInfo>(
            er::helper::fillPipelineColorBlendStateCreateInfo(cube_color_no_blend_attachments));

    auto cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo());

    auto no_cull_rasterization_info =
        std::make_shared<er::PipelineRasterizationStateCreateInfo>(
            er::helper::fillPipelineRasterizationStateCreateInfo(
                false,
                false,
                er::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));

    auto ms_info = std::make_shared<er::PipelineMultisampleStateCreateInfo>(
        er::helper::fillPipelineMultisampleStateCreateInfo());

    auto depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo());

    auto depth_no_write_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                true, false));

    auto fs_depth_stencil_info =
        std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
            er::helper::fillPipelineDepthStencilStateCreateInfo(
                false,
                false,
                er::CompareOp::ALWAYS));

    graphic_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_pipeline_info_.rasterization_info = cull_rasterization_info;
    graphic_pipeline_info_.ms_info = ms_info;
    graphic_pipeline_info_.depth_stencil_info = depth_stencil_info;

    graphic_double_face_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_double_face_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_double_face_pipeline_info_.ms_info = ms_info;
    graphic_double_face_pipeline_info_.depth_stencil_info = depth_stencil_info;

    graphic_no_depth_write_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_no_depth_write_pipeline_info_.rasterization_info = cull_rasterization_info;
    graphic_no_depth_write_pipeline_info_.ms_info = ms_info;
    graphic_no_depth_write_pipeline_info_.depth_stencil_info = depth_no_write_stencil_info;
        
    graphic_fs_pipeline_info_.blend_state_info = single_no_blend_state_info;
    graphic_fs_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_fs_pipeline_info_.ms_info = ms_info;
    graphic_fs_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

    graphic_fs_blend_pipeline_info_.blend_state_info = single_blend_state_info;
    graphic_fs_blend_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_fs_blend_pipeline_info_.ms_info = ms_info;
    graphic_fs_blend_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

    graphic_cubemap_pipeline_info_.blend_state_info = cube_no_blend_state_info;
    graphic_cubemap_pipeline_info_.rasterization_info = no_cull_rasterization_info;
    graphic_cubemap_pipeline_info_.ms_info = ms_info;
    graphic_cubemap_pipeline_info_.depth_stencil_info = fs_depth_stencil_info;

    // the initialization order has to be strict.
    instance_ = er::Helper::createInstance();
    physical_devices_ = er::Helper::collectPhysicalDevices(instance_);
    surface_ = er::Helper::createSurface(instance_, window_);
    physical_device_ = er::Helper::pickPhysicalDevice(physical_devices_, surface_);
    queue_list_ = er::Helper::findQueueFamilies(physical_device_, surface_);
    device_ = er::Helper::createLogicalDevice(physical_device_, surface_, queue_list_);
    er::Helper::initRayTracingProperties(
        physical_device_,
        device_,
        rt_pipeline_properties_,
        as_features_);
    auto queue_list =
        queue_list_.getGraphicAndPresentFamilyIndex();
    assert(device_);
    depth_format_ = er::Helper::findDepthFormat(device_);
    graphics_queue_ = device_->getDeviceQueue(queue_list[0]);
    assert(graphics_queue_);
    present_queue_ = device_->getDeviceQueue(queue_list.back());
    er::Helper::createSwapChain(
        window_,
        device_,
        surface_,
        queue_list_,
        swap_chain_info_,
        SET_2_FLAG_BITS(ImageUsage, COLOR_ATTACHMENT_BIT, TRANSFER_DST_BIT));
    createRenderPasses();
    createImageViews();
    cubemap_render_pass_ =
        er::helper::createCubemapRenderPass(
            device_,
            std::source_location::current());

    pbr_lighting_desc_set_layout_ =
        createPbrLightingDescriptorSetLayout(device_);

    runtime_lights_desc_set_layout_ =
        createRuntimeLightsDescriptorSetLayout(device_);

    createCommandPool();
    assert(command_pool_);
    er::Helper::init(device_);

    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/lambertian/diffuse.ktx2",
        ibl_diffuse_tex_,
        std::source_location::current());
    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/ggx/specular.ktx2",
        ibl_specular_tex_,
        std::source_location::current());
    eh::loadMtx2Texture(
        device_,
        cubemap_render_pass_,
        "assets/environments/doge2/charlie/sheen.ktx2",
        ibl_sheen_tex_,
        std::source_location::current());
    recreateRenderBuffer(swap_chain_info_.extent);
    auto default_color_format = er::Format::R8G8B8A8_UNORM;
    auto height_map_format = er::Format::R16_UNORM;
    eh::createTextureImage(device_, "assets/statue.jpg", default_color_format, false, sample_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/brdfLUT.png", default_color_format, false, brdf_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_ggx.png", default_color_format, false, ggx_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_charlie.png", default_color_format, false, charlie_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/lut_thin_film.png", default_color_format, false, thin_film_lut_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map_mask.png", default_color_format, false, map_mask_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/map.png", height_map_format, false, heightmap_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/tile1.tga", default_color_format, false, prt_bump_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_C.PNG", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_H.PNG", default_color_format, false, prt_height_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_N.PNG", default_color_format, false, prt_normal_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat4Mural_TRA.PNG", default_color_format, false, prt_orh_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_C.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
//    eh::createTextureImage(device_, "assets/T_Mat1Ground_ORH.jpg", default_color_format, false, prt_bump_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_C.jpg", default_color_format, false, prt_base_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_N.jpg", default_color_format, false, prt_normal_tex_, std::source_location::current());
    eh::createTextureImage(device_, "assets/T_Mat2Mountains_ORH.jpg", default_color_format, false, prt_orh_tex_, std::source_location::current());
    createTextureSampler();
    descriptor_pool_ = device_->createDescriptorPool();
    createCommandBuffers();
    createSyncObjects();

    ego::CameraObject::createViewCameraDescriptorSetLayout(device_);

    // Deferred resolve — needs descriptor pool + the global PBR / view /
    // runtime-lights layouts above, plus the G-buffer image views created
    // by recreateRenderBuffer (called earlier in initVulkan).  Safe to run
    // even if the runtime user never opts into deferred mode: pipeline +
    // desc set are created idle and only cost memory until first dispatch.
    initDeferredResolve();

    // Hi-Z build pipeline — one-time creation; descriptor sets are wired
    // up later (after object_scene_view_ exists, since mip-0's source is
    // the scene depth view).  See writeHiZBuildDescriptors() call below.
    initHiZBuild();

    // initDeferredResolve above ran BEFORE hiz_sampler_ existed, so its
    // internal writeDeferredResolveDescriptors call skipped binding 6
    // (Hi-Z sampler).  Now that the sampler is created, refresh the
    // resolve descriptors so the DEBUG_RENDER_MODE_HIZ visualisation
    // actually has a valid binding to sample.
    writeDeferredResolveDescriptors();

    // createHiZPyramid was called earlier inside createGBuffer (before
    // this initHiZBuild() ran), so its internal writeHiZBuildDescriptors
    // call was gated out (no pipeline yet → no descriptor set layout to
    // allocate against).  Now that the pipeline exists, populate the
    // per-mip descriptor sets so dispatchHiZBuild() actually has somewhere
    // to dispatch into — without this the per-frame Hi-Z build silently
    // early-exits because hiz_build_desc_sets_ is empty, the pyramid
    // never gets populated, and the cluster cull's plausibility check
    // rejects every sample (toggle has no effect).
    //
    // Source = object_scene_view_'s live depth attachment (Phase A's
    // output).  See createHiZPyramid() for why we no longer route
    // through depth_buffer_copy_.
    {
        std::shared_ptr<er::ImageView> phase_a_depth_view;
        if (object_scene_view_ && object_scene_view_->getDepthBuffer()) {
            // Bind depth_buffer_copy_ (held in SHADER_READ_ONLY_OPTIMAL by
            // the end-of-frame blit) rather than the LIVE object_scene_view_
            // depth attachment.  writeHiZBuildDescriptors declares the
            // descriptor layout as SHADER_READ_ONLY_OPTIMAL; pointing it at
            // the live depth target trips VUID-vkCmdDraw-None-09600 every
            // frame because that image is in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            // when the dispatch consumes the descriptor.  The dispatch's
            // mip-0 push-constant size already references depth_buffer_copy_,
            // so this also aligns the descriptor with the size the shader
            // assumes.  Functionally identical: depth_buffer_copy_ holds
            // last-frame's depth that the Hi-Z build was already designed to
            // consume (see the comment at the Hi-Z dispatch site).
            phase_a_depth_view = depth_buffer_copy_.view;
        }
        writeHiZBuildDescriptors(phase_a_depth_view);
    }

    auto desc_set_layouts = {
        pbr_lighting_desc_set_layout_,
        ego::CameraObject::getViewCameraDescriptorSetLayout() };

    ego::ShapeBase::initStaticMembers(
        device_,
        desc_set_layouts,
        graphic_pipeline_info_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    // "Nanite-lite" cluster-debug pipeline. Cheap to create unconditionally
    // -- the per-mesh GPU buffers are only allocated when the
    // --cluster-debug flag is set. Shares the same descriptor-set layout
    // list as ShapeBase since the vertex shader only needs the camera SSBO.
    ego::ClusterDebugDraw::initStaticMembers(
        device_,
        desc_set_layouts,
        graphic_pipeline_info_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    eh::CollisionDebugDraw::initStaticMembers(
        device_,
        desc_set_layouts,
        graphic_pipeline_info_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    prt_shadow_gen_ =
        std::make_shared<es::PrtShadow>(
            device_,
            descriptor_pool_,
            texture_sampler_);

    conemap_obj_ =
        std::make_shared<ego::ConemapObj>(
            device_,
            descriptor_pool_,
            texture_sampler_,
            prt_orh_tex_,//prt_height_tex_,
            prt_shadow_gen_,
            2,//0,
            true,
            0.05f,
            0.1f,
            8.0f / 256.0f);

    unit_plane_ =
        std::make_shared<ego::Plane>(
            device_,
            nullptr,
            10,
            10,
            std::source_location::current());

    unit_sphere_ =
        std::make_shared<ego::Sphere>(
            device_,
            1.0f,
            2,
            std::source_location::current());

    auto v = std::make_shared<std::array<glm::vec3, 8>>();
    (*v)[0] = glm::vec3(-1.0f, -5.0f, -1.0f);
    (*v)[1] = glm::vec3(1.0f, -5.0f, -1.0f);
    (*v)[2] = glm::vec3(1.0f, -5.0f, 1.0f);
    (*v)[3] = glm::vec3(-1.0f, -5.0f, 1.0f);
    (*v)[4] = glm::vec3(-1.0f, 0.0f, -1.0f);
    (*v)[5] = glm::vec3(1.0f, 0.0f, -1.0f);
    (*v)[6] = glm::vec3(1.0f, 0.0f, 1.0f);
    (*v)[7] = glm::vec3(-1.0f, 0.0f, 1.0f);

    unit_box_ =
        std::make_shared<ego::Box>(
            device_,
            v,
            10,
            30,
            10,
            std::source_location::current());

    conemap_test_ =
        std::make_shared<ego::ConemapTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            prt_base_tex_,
            prt_normal_tex_,
            prt_orh_tex_,
            conemap_obj_,
            swap_chain_info_.extent,
            unit_plane_);

    hair_patch_ =
        std::make_shared<ego::HairPatch>(
            device_,
            descriptor_pool_,
            desc_set_layouts,
            texture_sampler_,
            glm::uvec2(1024, 1024));

    hair_test_ =
        std::make_shared<ego::HairTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            hair_patch_->getHairPatchColorTexture(),
            hair_patch_->getHairPatchWeightTexture(),
            swap_chain_info_.extent,
            unit_plane_);

    lbm_patch_ =
        std::make_shared<ego::LbmPatch>(
            device_,
            descriptor_pool_,
            desc_set_layouts,
            texture_sampler_,
            glm::uvec2(1024, 1024));

    lbm_test_ =
        std::make_shared<ego::LbmTest>(
            device_,
            descriptor_pool_,
            hdr_render_pass_,
            graphic_pipeline_info_,
            desc_set_layouts,
            texture_sampler_,
            lbm_patch_->getLbmPatchTexture(),
            swap_chain_info_.extent,
            unit_plane_);

    clear_values_.resize(2);
    clear_values_[0].color = { 50.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f, 1.0f };
    clear_values_[1].depth_stencil = { 1.0f, 0 };

    conemap_gen_ =
        std::make_shared<es::Conemap>(
            device_,
            descriptor_pool_,
            texture_sampler_,
            prt_orh_tex_,//prt_height_tex_,
            conemap_obj_);

    ibl_creator_ = std::make_shared<es::IblCreator>(
        device_,
        descriptor_pool_,
        cubemap_render_pass_,
        graphic_cubemap_pipeline_info_,
        texture_sampler_,
        kCubemapSize);

    skydome_ = std::make_shared<es::Skydome>(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        cubemap_render_pass_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        graphic_no_depth_write_pipeline_info_,
        graphic_cubemap_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent,
        kCubemapSize);

    // Create the fullscreen envmap background pipeline (dynamic rendering path).
    skydome_->initEnvmapBackgroundPipeline(
        device_,
        descriptor_pool_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    ego::TileObject::initStaticMembers(device_);

    std::vector<std::shared_ptr<er::ImageView>> soil_water_texes(2);
    soil_water_texes[0] = ego::TileObject::getSoilWaterLayer(0).view;
    soil_water_texes[1] = ego::TileObject::getSoilWaterLayer(1).view;
    weather_system_ = std::make_shared<es::WeatherSystem>(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        mirror_repeat_sampler_,
        ego::TileObject::getRockLayer().view,
        soil_water_texes);

    ego::DrawableObject::initGameObjectBuffer(device_);
    assert(ego::DrawableObject::getGameObjectsBuffer());

    er::DescriptorSetLayoutList object_desc_set_layouts(MAX_NUM_PARAMS_SETS);
    object_desc_set_layouts[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_layout_;
    object_desc_set_layouts[VIEW_PARAMS_SET] = ego::CameraObject::getViewCameraDescriptorSetLayout();
    object_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_layout_;

    ego::DrawableObject::initStaticMembers(
        device_,
        descriptor_pool_,
        object_desc_set_layouts,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    ego::ViewCamera::initStaticMembers(
        device_,
        descriptor_pool_,
        desc_set_layouts);

    main_camera_object_ =
        std::make_shared<ego::ObjectViewCameraObject>(
            device_,
            descriptor_pool_,
            45.0f,
            float(kWindowSizeX)/float(kWindowSizeY));

    shadow_camera_object_ =
        std::make_shared<ego::ShadowViewCameraObject>(
            device_,
            descriptor_pool_,
            glm::vec3(0.3f, -0.8f, 0.0f));

    terrain_scene_view_ =
        std::make_shared<es::TerrainSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            main_camera_object_,
            desc_set_layouts,
            nullptr,
            nullptr,
            glm::uvec2(kWindowSizeX, kWindowSizeY));

    object_scene_view_ =
        std::make_shared<es::ObjectSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_[int(er::RenderPasses::kForward)],
            main_camera_object_,
            nullptr,
            nullptr,
            glm::uvec2(kWindowSizeX, kWindowSizeY));

    // Now that object_scene_view_ owns its color buffer, point the deferred
    // resolve compute's storage-image binding at it.  Idempotent — safe to
    // call again on swap-chain rebuild.
    writeDeferredResolveDescriptors();

    // Hi-Z mip 0 sources from object_scene_view_'s LIVE depth — Phase A
    // writes into it, the build dispatch reads it, Phase B tests against
    // the resulting pyramid all in the same frame.  See createHiZPyramid().
    {
        std::shared_ptr<er::ImageView> phase_a_depth_view;
        if (object_scene_view_ && object_scene_view_->getDepthBuffer()) {
            // Bind depth_buffer_copy_ (held in SHADER_READ_ONLY_OPTIMAL by
            // the end-of-frame blit) rather than the LIVE object_scene_view_
            // depth attachment.  writeHiZBuildDescriptors declares the
            // descriptor layout as SHADER_READ_ONLY_OPTIMAL; pointing it at
            // the live depth target trips VUID-vkCmdDraw-None-09600 every
            // frame because that image is in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            // when the dispatch consumes the descriptor.  The dispatch's
            // mip-0 push-constant size already references depth_buffer_copy_,
            // so this also aligns the descriptor with the size the shader
            // assumes.  Functionally identical: depth_buffer_copy_ holds
            // last-frame's depth that the Hi-Z build was already designed to
            // consume (see the comment at the Hi-Z dispatch site).
            phase_a_depth_view = depth_buffer_copy_.view;
        }
        writeHiZBuildDescriptors(phase_a_depth_view);
    }

    // ── Create CSM shadow depth array (kCsmSize × kCsmSize × CSM_CASCADE_COUNT) ──
    // 2048 px per cascade.  We took the smoothness benefit from going
    // 4 → 6 cascades (which makes per-cascade extent ratios smaller),
    // so 2048 per cascade is enough now without re-introducing visible
    // shadow-texel stairstep.  At 6 cascades × 2048² × 4 bytes = 96 MB
    // — back in line with typical AAA budgets.
    constexpr uint32_t kCsmSize = 2048u;
    {
        const er::Format csm_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kShadow)].depth_format;

        csm_shadow_tex_ = std::make_shared<er::TextureInfo>();
        csm_shadow_tex_->image = device_->createImage(
            er::ImageType::TYPE_2D,
            glm::uvec3(kCsmSize, kCsmSize, 1),
            csm_fmt,
            SET_2_FLAG_BITS(ImageUsage,
                            DEPTH_STENCIL_ATTACHMENT_BIT,
                            SAMPLED_BIT),
            er::ImageTiling::OPTIMAL,
            er::ImageLayout::UNDEFINED,
            std::source_location::current(),
            0, false, 1, 1,
            CSM_CASCADE_COUNT);

        auto mem_req = device_->getImageMemoryRequirements(csm_shadow_tex_->image);
        csm_shadow_tex_->memory = device_->allocateMemory(
            mem_req.size,
            mem_req.memory_type_bits,
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0);
        device_->bindImageMemory(csm_shadow_tex_->image, csm_shadow_tex_->memory);
        csm_shadow_tex_->size = glm::uvec3(kCsmSize, kCsmSize, 1);

        // Full-array view — used as the sampler2DArray in the fragment shader.
        csm_shadow_tex_->view = device_->createImageView(
            csm_shadow_tex_->image,
            er::ImageViewType::VIEW_2D_ARRAY,
            csm_fmt,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            std::source_location::current(),
            0, 1, 0, CSM_CASCADE_COUNT);

        // Per-layer views — used as render-target attachments.
        for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
            csm_layer_views_[k] = device_->createImageView(
                csm_shadow_tex_->image,
                er::ImageViewType::VIEW_2D,
                csm_fmt,
                SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
                std::source_location::current(),
                0, 1, k, 1);
        }

        // Prime all layers to DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        er::Helper::transitionImageLayout(
            device_,
            csm_shadow_tex_->image,
            csm_fmt,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            0, 1, 0, CSM_CASCADE_COUNT);
    }

    shadow_object_scene_view_ =
        std::make_shared<es::ObjectSceneView>(
            device_,
            descriptor_pool_,
            renderbuffer_formats_[int(er::RenderPasses::kShadow)],
            shadow_camera_object_,
            nullptr,
            csm_shadow_tex_,   // provide our array texture as the depth buffer
            glm::uvec2(kCsmSize),  // viewport must match the array texture
            true);

    main_camera_object_->createCameraDescSetWithTerrain(
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());

    ego::DebugDrawObject::initStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);

    createDescriptorSets();
    setupCsmDebugDraw();

    volume_noise_ = std::make_shared<es::VolumeNoise>(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        pbr_lighting_desc_set_layout_,
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    assert(volume_noise_);
    terrain_scene_view_->updateTileResDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        weather_system_->getTempTexes(),
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        heightmap_tex_.view);

    volume_cloud_ = std::make_shared<es::VolumeCloud>(
        device_,
        descriptor_pool_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        mirror_repeat_sampler_,
        texture_point_sampler_,
        depth_buffer_copy_.view,
        hdr_color_buffer_.view,
        weather_system_->getMoistureTexes(),
        weather_system_->getTempTexes(),
        weather_system_->getCloudLightingTex(),
        skydome_->getScatteringLutTex(),
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view,
        swap_chain_info_.extent);

    // SSAO writes into the SAME colour buffer the scene actually displays
    // — that's object_scene_view_'s colour buffer once it exists, NOT
    // hdr_color_buffer_ (which we used to bind here, but that buffer is
    // never blitted to the swap chain in the current pipeline, so SSAO's
    // output went nowhere — most visibly broken in DEBUG_RENDER_MODE_SSAO
    // where the apply step's diagnostic write produced no on-screen change).
    // Mirrors the comment around writeDeferredResolveDescriptors() above.
    auto ssao_color_view =
        (object_scene_view_ && object_scene_view_->getColorBuffer())
            ? object_scene_view_->getColorBuffer()->view
            : hdr_color_buffer_.view;
    ssao_ = std::make_shared<es::SSAO>(
        device_,
        descriptor_pool_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        texture_sampler_,
        depth_buffer_copy_.view,
        ssao_color_view,
        swap_chain_info_.extent);

    cluster_renderer_ = std::make_shared<es::ClusterRenderer>(
        device_,
        descriptor_pool_);

    // ── Runtime Virtual Texture manager ────────────────────────────
    // Allocates the 4 layer pool textures (4096² each), the page-table
    // SSBO (HOST_VISIBLE so registerTexture writes are coherent without
    // staging), and the per-VT meta SSBO.  Materials register their
    // source textures during mesh upload; the returned VirtualTextureId
    // resolves through the page table at sampling time in the shaders.
    // See scene_rendering/virtual_texture.h for the full architecture.
    vt_manager_ = std::make_shared<es::VirtualTextureManager>(
        device_,
        descriptor_pool_);

    // Re-enabled now that:
    //   (a) renderer.cpp::createTextureImage adds TRANSFER_SRC_BIT
    //       to all loaded textures, satisfying the vkCmdCopyImage /
    //       vkCmdCopyImageToBuffer prerequisite.
    //   (b) The albedo BC7 path uses CPU readback + parallel BC7
    //       encode (encode_pool_ in VirtualTextureManager) — no
    //       longer requires source format-class compatibility with
    //       the BC7 pool.  Other layers still use GPU-to-GPU copy
    //       and assume source is RGBA8 — works for stb_image-loaded
    //       assets (PNG/JPG); BC-source assets from DDS/KTX may
    //       still emit a one-off validation warning the first time
    //       they hit the copy path.  We can add a format-aware
    //       skip-or-decode pass for those in a follow-up.
    cluster_renderer_->setVtManager(vt_manager_.get());
    // NOTE: pool ImTextureID registration moved to
    // registerVtPoolImTextureIds(), called alongside
    // registerCsmDebugImTextureIds() AFTER the menu is constructed.
    // Doing it here would always no-op because menu_ doesn't exist
    // until ~200 lines further down in initVulkan.

    // Hand the just-created cluster renderer the Hi-Z handles that
    // createHiZPyramid built earlier in initVulkan.  Without this call
    // the cluster cull descriptor's binding 11 is left unbound when
    // finalizeUploads writes the rest of the set, and any cull
    // dispatch that tries to sample the Hi-Z reads garbage.
    if (hiz_sampler_ && hiz_pyramid_full_view_) {
        cluster_renderer_->setHiZTexture(
            hiz_sampler_, hiz_pyramid_full_view_,
            hiz_pyramid_size_, hiz_pyramid_mips_);
    }

    // ── Dynamic reflection cubemap + ambient probe system ────────────
    // The dynamic cubemap is the moving "scratch probe" — it captures
    // one cube face per frame at whatever world position the ambient
    // probe system currently has it parked over.  After 6 frames per
    // probe the probe system bakes the cube into 9 SH coefficients in
    // its grid SSBO.  Both feed cluster_bindless.frag's ambient term
    // via the ambient_probes.glsl.h helpers (descriptor set wired in
    // the cluster pipeline layout).  See scene_rendering/dynamic_cubemap.h
    // and scene_rendering/ambient_probe_system.h for the per-class
    // architecture.
    dynamic_cubemap_ = std::make_shared<es::DynamicCubemap>(
        device_,
        descriptor_pool_,
        texture_sampler_,
        cubemap_render_pass_);

    ambient_probe_system_ = std::make_shared<es::AmbientProbeSystem>(
        device_,
        descriptor_pool_,
        texture_sampler_);

    // Source-cube descriptor for the SH projection pass — point at the
    // dynamic cubemap's current read view.  We re-bind whenever the
    // ping-pong index swaps (every frame), but the initial binding
    // here gets us through the first frame before any swap.
    ambient_probe_system_->writeProjectDescriptorsForCube(
        device_,
        dynamic_cubemap_->getColorCubeView(),
        dynamic_cubemap_->getDepthCube().view);

    // Hook the sky envmap as the per-frame face-capture source.  The
    // dynamic cubemap will copy from this envmap into its active face
    // each frame — sky content is at infinity so a position-independent
    // copy is functionally equivalent to a per-probe sky render at a
    // fraction of the cost.  When you extend to scene-content capture
    // later, replace this with a render pass into the active face slice.
    dynamic_cubemap_->setSkyEnvmap(
        ibl_creator_->getEnvmapTexture().image,
        kCubemapSize);

    // Probe debug-draw pipeline.  Renders a small icosphere at each
    // probe's world position into the forward pass; colored by the
    // probe's SH-evaluated irradiance.  Toggleable via menu (see
    // Render Debug → Show Probes).
    ambient_probe_system_->initDebugPipeline(
        device_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        renderbuffer_formats_[int(er::RenderPasses::kForward)],
        graphic_pipeline_info_);

    // Initial probe placement.  Uses a generous default bounding box
    // centred at the world origin — covers typical scene scales (out to
    // ±50 m horizontal, ±15 m vertical).  When the level's actual
    // bounding box is known (e.g. after assets stream in) the
    // application can re-call placeProbeGrid() to retarget the grid.
    ambient_probe_system_->placeProbeGrid(
        device_,
        glm::vec3(-50.0f, -10.0f, -50.0f),
        glm::vec3( 50.0f,  30.0f,  50.0f));

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
        weather_system_->getAirflowTex());

    // ray tracing test.
    ray_tracing_test_ = std::make_shared<engine::ray_tracing::RayTracingShadowTest>();
    ray_tracing_test_->init(
        device_,
        descriptor_pool_,
        main_camera_object_->getViewCameraBuffer(),
        rt_pipeline_properties_,
        as_features_,
        glm::uvec2(1024, 768));

    // Spin up the async mesh loader before kicking off any load.
    // Its ctor probes device_->hasLoaderQueue(): on hardware with the
    // dedicated loader queue the worker thread starts here and the
    // startup assets below stream in asynchronously; on single-queue
    // hardware the manager transparently falls back to inline loads
    // so this block behaves exactly like the pre-async code path.
    mesh_load_task_manager_ =
        std::make_unique<ego::MeshLoadTaskManager>(device_);

    // Async startup loads. createAsync returns a shell DrawableObject
    // immediately; the render loop's isReady() check skips it until
    // the worker thread finishes parsing + GPU upload and the main
    // thread finalizes descriptors/pipelines on the next poll(). The
    // HUD menu spinner (see Menu::drawMeshLoadProgress) reads
    // MeshLoadTaskManager::inFlightFilenames() to tell the user
    // what's still loading.
    bistro_exterior_scene_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Bistro_v5_2/BistroExterior.fbx",
            glm::inverse(view_params_.view));

    bistro_interior_scene_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Bistro_v5_2/BistroInterior.fbx",
            glm::inverse(view_params_.view));

    // Spawn the rigged auto-rig character (19 bones, no embedded
    // animation channels — PlayerController drives the rig procedurally
    // each frame). spawnAt() is called once the level finishes loading
    // so the character appears in front of the camera.
    player_object_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/Characters/scene-skinned.gltf",
            glm::inverse(view_params_.view));

    // ── Foot debug markers (two tiny red cubes pinned to the rig's
    // left_foot / right_foot bones each frame).  Asset is the
    // 264-byte assets/debug_cube.gltf.  Configured + repositioned
    // in the "Drawable updates (CPU)" block once both markers and
    // the player rig are isReady().  Kept out of drawable_objects_
    // on purpose so PlayerController doesn't see them as collision
    // obstacles (the player would otherwise bump into its own
    // feet).
    foot_marker_left_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/debug_cube.gltf",
            glm::inverse(view_params_.view));
    foot_marker_right_ =
        ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            repeat_texture_sampler_,
            thin_film_lut_tex_,
            "assets/debug_cube.gltf",
            glm::inverse(view_params_.view));

    // The player is PlayerController-driven (procedural pose + spawnAt
    // -driven world placement).  Opt it into "identity instance + skip
    // imported animations" mode so:
    //   1. update_instance_buffer.comp writes an identity instance
    //      transform instead of the shared game_objects_buffer_'s
    //      (camera-tracking, gravity-drifting) position — otherwise the
    //      player double-transforms by adding camera_pos@frame0 on top
    //      of the node-hierarchy translation.
    //   2. Any imported glTF animation channels are NOT evaluated, so
    //      PlayerController::applyPose's setRootNodeTransform +
    //      setNodeRotationByName writes survive each frame instead of
    //      being clobbered by the asset's authored animation timeline.
    // See DrawableObject::setUseNodeTransformOnly() for the full
    // rationale.
    // NOTE — DO NOT call any flag setter on player_object_ here.
    // createAsync() returned a SHELL: object_ is nullptr until the
    // worker-thread load completes + phase 3 runs.  Every setter
    // (setUseNodeTransformOnly, setDebugForceRed, setDebugSkipSkinning,
    // setDebugScale) starts with `if (object_) ...` and silently
    // no-ops when called now — that wasted the whole previous round
    // of debugging because no override actually took effect.  All
    // flag wiring is done inside the player-spawn block below, which
    // is gated on isReady() (so object_ is guaranteed populated) and
    // only fires once via the `!isSpawned()` latch.

    player_controller_ = std::make_unique<ego::PlayerController>();
    // Feed the foot-IK solver a ground probe.  PlayerController plants each
    // foot independently via two-bone IK and needs the ground height +
    // surface normal under each foot; queryGroundAt reuses the walkable
    // collision world (when populated) and falls back to the rendered scene
    // geometry, so feet stay grounded even before / without the LLM
    // walkable classification.  The lambda captures `this`; it is only
    // invoked later (per frame) once the scene members are populated.
    player_controller_->setGroundQuery(
        [this](float gx, float gz, float yh,
               float& gy, glm::vec3& gn) -> bool {
            return this->queryGroundAt(gx, gz, yh, gy, gn);
        });

    // Bistro scenes contain a sky dome mesh that draws a purple atmospheric
    // background.  Disabled so only game objects and the player render.
    // object_scene_view_->addDrawableObject(bistro_exterior_scene_);
    // object_scene_view_->addDrawableObject(bistro_interior_scene_);
    // shadow_object_scene_view_->addDrawableObject(bistro_exterior_scene_);

    object_scene_view_->addDrawableObject(
        player_object_);

    shadow_object_scene_view_->addDrawableObject(
        player_object_);

    // ── Foot debug markers: register with the same scene views as the
    // player so they're picked up by the forward + shadow render
    // passes.  This is the bit that was missing: scene_view iteration
    // is what actually drives the visible draws; the direct draw()
    // call sites further down are redundant for this path.  Shadow
    // registration is optional (they're tiny cubes, casting shadows
    // looks fine but isn't required) — kept in for parity with the
    // player so toggling depth-test off doesn't make them flicker.
    object_scene_view_->addDrawableObject(foot_marker_left_);
    object_scene_view_->addDrawableObject(foot_marker_right_);
    shadow_object_scene_view_->addDrawableObject(foot_marker_left_);
    shadow_object_scene_view_->addDrawableObject(foot_marker_right_);

    menu_ = std::make_shared<engine::ui::Menu>(
        window_,
        device_,
        instance_,
        queue_list_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_,
        texture_sampler_,
        ray_tracing_test_->getFinalImage().view,
        nullptr);

    // Register CSM debug colour targets as ImGui textures (ImGui must be
    // initialised by the Menu constructor before this can be called).
    registerCsmDebugImTextureIds();
    registerIblDebugImTextureIds();
    registerVtPoolImTextureIds();

    // Wire the GPU profiler into the menu so it can display itself.
    menu_->setGpuProfiler(&gpu_profiler_);

    // Wire the async mesh loader so the menu can draw a HUD spinner
    // for in-flight loads (see Menu::draw). Passing a non-owning
    // pointer — the manager's lifetime is tied to the Application.
    if (mesh_load_task_manager_) {
        menu_->setMeshLoadTaskManager(mesh_load_task_manager_.get());
    }

    // ---- Plugin system --------------------------------------------------------
    plugin_manager_.registerPlugin(
        std::make_unique<plugins::auto_rig::AutoRigPlugin>());
    plugin_manager_.initAll(device_);
    menu_->setPluginManager(&plugin_manager_);

    if (ssao_) {
        menu_->setSSAO(ssao_.get());
    }
    if (cluster_renderer_) {
        menu_->setClusterRenderer(cluster_renderer_.get());
    }
    if (vt_manager_) {
        menu_->setVtManager(vt_manager_.get());
    }
}

void RealWorldApplication::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    device_->waitIdle();

    menu_->destroy();

    cleanupSwapChain();

    er::Helper::createSwapChain(
        window_,
        device_,
        surface_,
        queue_list_,
        swap_chain_info_,
        SET_2_FLAG_BITS(ImageUsage, COLOR_ATTACHMENT_BIT, TRANSFER_DST_BIT));

    createRenderPasses();
    createImageViews();

    // Two-element variant kept for the consumers below (DebugDraw,
    // ViewCamera) that only need PBR + VIEW slots.
    auto desc_set_layouts = {
        pbr_lighting_desc_set_layout_,
        ego::CameraObject::getViewCameraDescriptorSetLayout() };

    // ── Full-width layout list for DrawableObject ────────────────────
    // DrawableObject::createStaticMembers indexes the layout list at
    // RUNTIME_LIGHTS_PARAMS_SET when constructing the mesh-shader CSM
    // pipeline layout (see the mesh_shader_shadow_pipeline_layout_
    // block).  The 2-element initializer above would assert-fire on
    // any swapchain recreate (window resize, alt-tab, format change).
    // Mirror the MAX_NUM_PARAMS_SETS-sized list the initial init uses
    // at startup (search for "object_desc_set_layouts" near
    // DrawableObject::initStaticMembers) so both code paths feed the
    // same shape into createStaticMembers.
    er::DescriptorSetLayoutList drawable_desc_set_layouts(
        MAX_NUM_PARAMS_SETS);
    drawable_desc_set_layouts[PBR_GLOBAL_PARAMS_SET] =
        pbr_lighting_desc_set_layout_;
    drawable_desc_set_layouts[VIEW_PARAMS_SET] =
        ego::CameraObject::getViewCameraDescriptorSetLayout();
    drawable_desc_set_layouts[RUNTIME_LIGHTS_PARAMS_SET] =
        runtime_lights_desc_set_layout_;

    ego::TileObject::recreateStaticMembers(device_);
    ego::DrawableObject::recreateStaticMembers(
        device_,
        &renderbuffer_formats_[0],
        graphic_pipeline_info_,
        drawable_desc_set_layouts);
    ego::ViewCamera::recreateStaticMembers(
        device_);
    ego::DebugDrawObject::recreateStaticMembers(
        device_,
        hdr_render_pass_,
        graphic_pipeline_info_,
        desc_set_layouts,
        swap_chain_info_.extent);
    // ── Descriptor pool MUST be created BEFORE recreateRenderBuffer ──
    // recreateRenderBuffer → createGBuffer → createHiZPyramid issues
    // BOTH vkUpdateDescriptorSets (cluster_renderer_->setHiZTexture,
    // writeDeferredResolveDescriptors) AND vkAllocateDescriptorSets
    // (writeHiZBuildDescriptors allocates hiz_build_desc_sets_ one per
    // mip).  Both operations require a live pool.  We used to create
    // the new pool AFTER recreateRenderBuffer, which left the allocate
    // call dispatching against the just-destroyed pool — NVIDIA crashes
    // inside the driver dispatch with an nvoglv64.dll callstack.
    // The pre-pool nullouts in cleanupSwapChain (deferred_resolve_desc_
    // set_ = nullptr + cluster_renderer_->onDescriptorPoolDestroyed)
    // are now belt-and-braces: those handles are still dead between
    // cleanupSwapChain and this createDescriptorPool, but createGBuffer
    // can also legitimately need to ALLOCATE — the early-return
    // guards don't help there.  Moving the pool earlier is the right
    // fix; the nullouts are still useful in case any other code path
    // we don't see is touching a stale handle in this window.
    descriptor_pool_ = device_->createDescriptorPool();

    // Re-allocate the deferred-resolve descriptor set from the fresh
    // pool.  The set itself was destroyed alongside the previous pool
    // in cleanupSwapChain (and the field nulled there); the pipeline
    // layout it binds to is size-independent and survives.  Re-write
    // happens later in this function via writeDeferredResolveDescriptors
    // once object_scene_view_ has its new G-buffer / depth views.
    if (deferred_resolve_desc_set_layout_) {
        deferred_resolve_desc_set_ = device_->createDescriptorSets(
            descriptor_pool_, deferred_resolve_desc_set_layout_, 1)[0];
    }

    // Same treatment for AmbientProbeSystem: its probe_desc_set_ and
    // project_desc_set_ were nulled in cleanupSwapChain alongside the
    // pool destruction; reallocate them from the fresh pool here so
    // drawScene's per-frame writeProjectDescriptorsForCube call has a
    // live handle to write through.  The probe_buffer_ binding is
    // re-issued inside recreateDescriptorSets; the cube source/depth
    // bindings are refreshed every frame by writeProjectDescriptorsForCube.
    if (ambient_probe_system_) {
        ambient_probe_system_->recreateDescriptorSets(
            device_, descriptor_pool_);
    }

    // And same for VirtualTextureManager — its compact_desc_set_ was
    // nulled in cleanupSwapChain; reallocate from the fresh pool and
    // re-write the (stable) feedback-buffer bindings before the next
    // tick / compactFeedback call fires this frame.
    if (vt_manager_) {
        vt_manager_->recreateDescriptorSets(descriptor_pool_);
    }

    // And DynamicCubemap — reallocate all three descriptor-set
    // families (face_view, depth_to_linear, reproject) from the
    // fresh pool.  AmbientProbeSystem::update binds these every
    // frame; without this reallocation the release build crashes
    // on the first post-resize frame.
    if (dynamic_cubemap_) {
        dynamic_cubemap_->recreateDescriptorSets(
            device_, descriptor_pool_);
    }

    recreateRenderBuffer(swap_chain_info_.extent);
    createCommandBuffers();

    skydome_->recreate(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        graphic_no_depth_write_pipeline_info_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        swap_chain_info_.extent);

    // Recreate the envmap background pipeline for the new descriptor pool /
    // swap-chain format (the dynamic rendering pipeline must match the current
    // forward pass color + depth formats).
    skydome_->initEnvmapBackgroundPipeline(
        device_,
        descriptor_pool_,
        ibl_creator_->getEnvmapTexture(),
        texture_sampler_,
        renderbuffer_formats_[int(er::RenderPasses::kForward)]);

    ibl_creator_->recreate(
        device_,
        descriptor_pool_,
        texture_sampler_);

    weather_system_->recreate(
        device_,
        descriptor_pool_,
        desc_set_layouts,
        mirror_repeat_sampler_,
        ego::TileObject::getRockLayer().view,
        { ego::TileObject::getSoilWaterLayer(0).view,
          ego::TileObject::getSoilWaterLayer(1).view });

    createDescriptorSets();

    // ---- Scene views: re-allocate descriptor sets + resize buffers ----
    {
        const auto& fwd_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kForward)];
        object_scene_view_->recreate(fwd_fmt, swap_chain_info_.extent);
        terrain_scene_view_->recreate(fwd_fmt, swap_chain_info_.extent);
        // Shadow view keeps its fixed CSM size (set in initVulkan from
        // kCsmSize).  Window resize doesn't change the shadow map, but
        // the desc sets need rebinding.  Read the size back from the
        // existing texture rather than redeclaring kCsmSize so any
        // future change to the constant only needs touching initVulkan.
        const auto& shd_fmt =
            renderbuffer_formats_[int(er::RenderPasses::kShadow)];
        const glm::uvec2 csm_extent(
            csm_shadow_tex_->size.x, csm_shadow_tex_->size.y);
        shadow_object_scene_view_->recreate(shd_fmt, csm_extent);

        // Re-point the deferred resolve compute at the (newly re-allocated)
        // object_scene_view_ color buffer + G-buffer views.  No-op if the
        // resolve hasn't been initialised.
        writeDeferredResolveDescriptors();

        // Hi-Z build descriptors point at object_scene_view_'s freshly
        // re-allocated depth attachment (Phase A's render target).
        {
            std::shared_ptr<er::ImageView> phase_a_depth_view;
            if (object_scene_view_ && object_scene_view_->getDepthBuffer()) {
                // Bind depth_buffer_copy_ (held in SHADER_READ_ONLY_OPTIMAL by
            // the end-of-frame blit) rather than the LIVE object_scene_view_
            // depth attachment.  writeHiZBuildDescriptors declares the
            // descriptor layout as SHADER_READ_ONLY_OPTIMAL; pointing it at
            // the live depth target trips VUID-vkCmdDraw-None-09600 every
            // frame because that image is in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            // when the dispatch consumes the descriptor.  The dispatch's
            // mip-0 push-constant size already references depth_buffer_copy_,
            // so this also aligns the descriptor with the size the shader
            // assumes.  Functionally identical: depth_buffer_copy_ holds
            // last-frame's depth that the Hi-Z build was already designed to
            // consume (see the comment at the Hi-Z dispatch site).
            phase_a_depth_view = depth_buffer_copy_.view;
            }
            writeHiZBuildDescriptors(phase_a_depth_view);
        }
    }

    // ---- Camera descriptor sets ----
    main_camera_object_->recreateDescriptorSet();
    shadow_camera_object_->recreateDescriptorSet();

    // re-create the terrain camera descriptor (binds terrain textures).
    main_camera_object_->createCameraDescSetWithTerrain(
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());

    // ---- Ray tracing ----
    ray_tracing_test_->recreate(
        device_,
        descriptor_pool_,
        main_camera_object_->getViewCameraBuffer(),
        rt_pipeline_properties_,
        as_features_,
        glm::uvec2(1024, 768));

    // ---- LBM test ----
    lbm_test_->recreate(
        device_,
        descriptor_pool_,
        texture_sampler_,
        lbm_patch_->getLbmPatchTexture());

    // ---- CSM debug descriptor set (re-alloc from new pool) ----
    if (csm_debug_desc_set_layout_) {
        csm_debug_desc_set_ = device_->createDescriptorSets(
            descriptor_pool_, csm_debug_desc_set_layout_, 1)[0];
        er::WriteDescriptorList writes;
        er::Helper::addOneTexture(
            writes,
            csm_debug_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            texture_sampler_,
            csm_shadow_tex_->view,
            er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        device_->updateDescriptorSets(writes);
    }

    for (auto& image : swap_chain_info_.images) {
        er::Helper::transitionImageLayout(
            device_,
            image,
            swap_chain_info_.format,
            er::ImageLayout::UNDEFINED,
            er::ImageLayout::PRESENT_SRC_KHR);
    }

    ego::TileObject::generateAllDescriptorSets(
        device_,
        descriptor_pool_,
        texture_sampler_,
        heightmap_tex_.view);

    ego::DebugDrawObject::generateAllDescriptorSets(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
        weather_system_->getAirflowTex());

    // ── Provide the view-camera buffer to the game-objects update path ──
    // update_game_objects.comp declares a CameraInfoBuffer at binding
    // CAMERA_OBJECT_BUFFER_INDEX (= 3) and reads camera_info.position to
    // place newly-spawned game objects relative to the player.  The
    // helper that populates the set (addGameObjectsInfoBuffer) has the
    // CAMERA write commented out, so the slot is otherwise left
    // unbound — validation flags VUID-vkCmdDispatch-None-08114 every
    // frame.  Setting the static here BEFORE generateDescriptorSet ensures
    // createGameObjectUpdateDescSet picks it up at descset-create time
    // and writes the binding in the same batch as the other slots, so the
    // descset is fully valid the moment it lands.
    ego::DrawableObject::setViewCameraBufferForUpdate(
        main_camera_object_->getViewCameraBuffer());

    ego::DrawableObject::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
//        ego::ViewCamera::getViewCameraBuffer(),
        thin_film_lut_tex_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        ego::TileObject::getWaterFlow(),
        weather_system_->getAirflowTex());

    // Late-arrival safety net: even if the static was unset at descset-
    // create time (e.g. a recreate path runs before the static is set),
    // this call patches the binding after the fact.
    ego::DrawableObject::updateGameObjectsCameraBuffer(
        device_, main_camera_object_->getViewCameraBuffer());

/*    assert(ego::DrawableObject::getGameObjectsBuffer());
    ego::ViewCamera::generateDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        ego::TileObject::getRockLayer(),
        ego::TileObject::getSoilWaterLayer(0),
        ego::TileObject::getSoilWaterLayer(1),
        *ego::DrawableObject::getGameObjectsBuffer());*/

    ego::TileObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        heightmap_tex_.view);

    ego::DebugDrawObject::updateStaticDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        weather_system_->getTempTex(0),
        weather_system_->getMoistureTex(0),
        weather_system_->getAirflowTex());

    volume_noise_->recreate(
        device_,
        descriptor_pool_,
        hdr_render_pass_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        graphic_pipeline_info_,
        texture_sampler_,
        texture_point_sampler_,
        swap_chain_info_.extent);

    // Terrain tile descriptors need the noise textures just recreated above.
    terrain_scene_view_->updateTileResDescriptorSet(
        device_,
        descriptor_pool_,
        texture_sampler_,
        repeat_texture_sampler_,
        weather_system_->getTempTexes(),
        map_mask_tex_.view,
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view);

    volume_cloud_->recreate(
        device_,
        descriptor_pool_,
        ego::CameraObject::getViewCameraDescriptorSetLayout(),
        mirror_repeat_sampler_,
        texture_point_sampler_,
        depth_buffer_copy_.view,
        hdr_color_buffer_.view,
        weather_system_->getMoistureTexes(),
        weather_system_->getTempTexes(),
        weather_system_->getCloudLightingTex(),
        skydome_->getScatteringLutTex(),
        volume_noise_->getDetailNoiseTexture().view,
        volume_noise_->getRoughNoiseTexture().view,
        swap_chain_info_.extent);

    if (ssao_) {
        // Re-bind to the live scene colour view (see matching note at
        // SSAO construction).  On swap-chain rebuild object_scene_view_'s
        // colour buffer is recreated, so the SSAO apply descriptor needs
        // to follow it — otherwise writes land on a dead image.
        auto ssao_color_view =
            (object_scene_view_ && object_scene_view_->getColorBuffer())
                ? object_scene_view_->getColorBuffer()->view
                : hdr_color_buffer_.view;
        ssao_->recreate(
            device_,
            descriptor_pool_,
            ego::CameraObject::getViewCameraDescriptorSetLayout(),
            texture_sampler_,
            depth_buffer_copy_.view,
            ssao_color_view,
            swap_chain_info_.extent);
    }

    if (cluster_renderer_) {
        cluster_renderer_->recreate(descriptor_pool_);
    }

    menu_->init(
        window_,
        device_,
        instance_,
        queue_list_,
        swap_chain_info_,
        graphics_queue_,
        descriptor_pool_,
        final_render_pass_);

    // Re-register CSM debug textures — ImGui was re-initialized above.
    registerCsmDebugImTextureIds();
    registerIblDebugImTextureIds();
    registerVtPoolImTextureIds();

    // Re-wire profiler after menu re-init.
    menu_->setGpuProfiler(&gpu_profiler_);
}

void RealWorldApplication::createImageViews() {
    swap_chain_info_.image_views.resize(swap_chain_info_.images.size());
    for (uint64_t i_img = 0; i_img < swap_chain_info_.images.size(); i_img++) {
        swap_chain_info_.image_views[i_img] = device_->createImageView(
            swap_chain_info_.images[i_img],
            er::ImageViewType::VIEW_2D,
            swap_chain_info_.format,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            std::source_location::current());
    }
}

void RealWorldApplication::createFramebuffers(const glm::uvec2& display_size) {
    swap_chain_info_.framebuffers.resize(swap_chain_info_.image_views.size());
    assert(depth_buffer_.view);
    assert(final_render_pass_);
    for (uint64_t i = 0; i < swap_chain_info_.image_views.size(); i++) {
        assert(swap_chain_info_.image_views[i]);
        std::vector<std::shared_ptr<er::ImageView>> attachments(2);
        attachments[0] = swap_chain_info_.image_views[i];
        attachments[1] = depth_buffer_.view;

        swap_chain_info_.framebuffers[i] =
            device_->createFrameBuffer(
                final_render_pass_,
                attachments,
                display_size,
                std::source_location::current());
    }

    assert(hdr_render_pass_);
    assert(hdr_color_buffer_.view);
    std::vector<std::shared_ptr<er::ImageView>> attachments(2);
    attachments[0] = hdr_color_buffer_.view;
    attachments[1] = depth_buffer_.view;

    hdr_frame_buffer_ =
        device_->createFrameBuffer(
            hdr_render_pass_,
            attachments,
            display_size,
            std::source_location::current());

    assert(hdr_water_render_pass_);
    hdr_water_frame_buffer_ =
        device_->createFrameBuffer(
            hdr_water_render_pass_,
            attachments,
            display_size,
            std::source_location::current());
}

void RealWorldApplication::createCommandPool() {
    command_pool_ = device_->createCommandPool(
        queue_list_.getGraphicAndPresentFamilyIndex()[0],
        SET_FLAG_BIT(CommandPoolCreate, RESET_COMMAND_BUFFER_BIT));
}

void RealWorldApplication::createCommandBuffers() {
    // One command buffer per FRAMES-IN-FLIGHT slot — NOT per swapchain
    // image.  Pairs 1:1 with in_flight_fences_[current_frame_], so the
    // per-FIF fence wait at the top of drawFrame() guarantees the GPU
    // is done with command_buffers_[current_frame_] before we reset
    // and re-record it.  This is the invariant the rest of drawFrame
    // assumes; sizing the array to swapchain image count and indexing
    // by image_index (the previous approach) could let the CPU reset
    // a CB still executing on the GPU under MAILBOX or out-of-order
    // acquireNextImage.
    command_buffers_ =
        device_->allocateCommandBuffers(
            command_pool_,
            static_cast<uint32_t>(kMaxFramesInFlight));
}

void RealWorldApplication::createSyncObjects() {
    image_available_semaphores_.resize(kMaxFramesInFlight);
    render_finished_semaphores_.resize(kMaxFramesInFlight);
    in_flight_fences_.resize(kMaxFramesInFlight);
    images_in_flight_.resize(swap_chain_info_.images.size(), VK_NULL_HANDLE);
    init_semaphore_ =
        device_->createSemaphore(
            std::source_location::current());

    assert(device_);
    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        image_available_semaphores_[i] =
            device_->createSemaphore(
                std::source_location::current());
        render_finished_semaphores_[i] =
            device_->createSemaphore(
                std::source_location::current());
        in_flight_fences_[i] =
            device_->createFence(
                std::source_location::current(),
                true);
    }
}

er::WriteDescriptorList RealWorldApplication::addGlobalTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::TextureInfo>& direct_shadow_tex )
{
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(6);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GGX_LUT_INDEX,
        texture_sampler_,
        ggx_lut_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        CHARLIE_LUT_INDEX,
        texture_sampler_,
        charlie_lut_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DIRECT_SHADOW_INDEX,
        texture_sampler_,
        direct_shadow_tex->view,
        er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    ibl_creator_->addToGlobalTextures(
        descriptor_writes,
        description_set,
        texture_sampler_);

    return descriptor_writes;
}

void RealWorldApplication::createDescriptorSets() {
    auto buffer_count = swap_chain_info_.images.size();

    pbr_lighting_desc_set_ =
        device_->createDescriptorSets(
            descriptor_pool_,
            pbr_lighting_desc_set_layout_,
            1)[0];

    // create a global ibl texture descriptor set.
    auto pbr_lighting_descs = 
        addGlobalTextures(
            pbr_lighting_desc_set_,
            shadow_object_scene_view_->getDepthBuffer());
    device_->updateDescriptorSets(pbr_lighting_descs);

    runtime_lights_desc_set_ =
        device_->createDescriptorSets(
        descriptor_pool_,
        runtime_lights_desc_set_layout_,
        1)[0];

    if (!runtime_lights_buffer_) {
        runtime_lights_buffer_ = std::make_shared<er::BufferInfo>();
        device_->createBuffer(
            sizeof(glsl::RuntimeLightsParams),
            SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
            SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
            0,
            runtime_lights_buffer_->buffer,
            runtime_lights_buffer_->memory,
            std::source_location::current());
    }

    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);
    er::Helper::addOneBuffer(
        descriptor_writes,
        runtime_lights_desc_set_,
        engine::renderer::DescriptorType::UNIFORM_BUFFER,
        RUNTIME_LIGHTS_CONSTANT_INDEX,
        runtime_lights_buffer_->buffer,
        runtime_lights_buffer_->buffer->getSize());
    device_->updateDescriptorSets(descriptor_writes);
}

void RealWorldApplication::setupCsmDebugDraw() {
    // Idempotent: skip if already set up (e.g. called twice during init).
    if (csm_debug_pipeline_) return;

    // ---- 4 small R8G8B8A8_UNORM colour render targets (one per cascade) ----
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        er::Helper::create2DTextureImage(
            device_,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec2(kCsmDebugSize),
            1,
            csm_debug_color_[k],
            SET_3_FLAG_BITS(ImageUsage,
                COLOR_ATTACHMENT_BIT,
                SAMPLED_BIT,
                TRANSFER_SRC_BIT),
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    // ---- Descriptor set layout: binding 0 = sampler2DArray (frag only) ----
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            0,
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT));
        csm_debug_desc_set_layout_ = device_->createDescriptorSetLayout(bindings);
    }

    // ---- Descriptor set: bind the CSM depth array ----
    csm_debug_desc_set_ = device_->createDescriptorSets(
        descriptor_pool_, csm_debug_desc_set_layout_, 1)[0];
    {
        er::WriteDescriptorList writes;
        er::Helper::addOneTexture(
            writes,
            csm_debug_desc_set_,
            er::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0,
            texture_sampler_,
            csm_shadow_tex_->view,
            er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        device_->updateDescriptorSets(writes);
    }

    // ---- Pipeline layout: our desc set + push constant (cascade index) ----
    {
        er::PushConstantRange push_const{};
        push_const.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
        push_const.offset = 0;
        push_const.size = sizeof(int32_t);
        csm_debug_pipeline_layout_ = device_->createPipelineLayout(
            { csm_debug_desc_set_layout_ },
            { push_const },
            std::source_location::current());
    }

    // ---- Pipeline: full_screen.vert + csm_debug.frag, no depth buffer ----
    {
        er::ShaderModuleList shaders(2);
        shaders[0] = er::helper::loadShaderModule(
            device_,
            "full_screen_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
        shaders[1] = er::helper::loadShaderModule(
            device_,
            "csm_debug_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

        er::PipelineRenderbufferFormats fmt;
        fmt.color_formats = { er::Format::R8G8B8A8_UNORM };
        fmt.depth_format   = er::Format::UNDEFINED;   // no depth attachment

        er::PipelineInputAssemblyStateCreateInfo ia;
        ia.topology = er::PrimitiveTopology::TRIANGLE_LIST;
        ia.restart_enable = false;

        // graphic_fs_pipeline_info_ = no cull, no depth test/write, single colour.
        csm_debug_pipeline_ = device_->createPipeline(
            csm_debug_pipeline_layout_,
            {},   // no vertex bindings
            {},   // no vertex attributes
            ia,
            graphic_fs_pipeline_info_,
            shaders,
            fmt,
            {},   // default RasterizationStateOverride
            std::source_location::current());
    }
}

void RealWorldApplication::registerCsmDebugImTextureIds() {
    if (!csm_debug_pipeline_) return;  // not set up yet
    std::array<ImTextureID, CSM_CASCADE_COUNT> ids;
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        ids[k] = er::Helper::addImTextureID(
            texture_sampler_,
            csm_debug_color_[k].view);
    }
    menu_->setCsmDebugTextureIds(ids);
}

void RealWorldApplication::registerVtPoolImTextureIds() {
    if (!menu_ || !vt_manager_) return;
    // Each ImTextureID is a long-lived (sampler, view) descriptor — safe
    // to register once at init and reuse for the application's lifetime.
    // The pool textures are created in the VirtualTextureManager
    // constructor with SAMPLED_BIT usage, so this binding is well-formed
    // even before any pages have been registered into them.
    //
    // Use the DEBUG view + sampler (single-mip-0 view, NEAREST mipmap):
    // ImGui draws the pool image as a small thumbnail, and its
    // implicit LOD calculation lands well past the regular trilinear
    // view's mip range (only mip 0 + mip 1).  Some drivers return
    // black instead of clamping in that case; binding ImGui to a
    // single-mip view side-steps the issue.
    std::array<ImTextureID, 4> vt_ids{};
    for (int k = 0; k < 4; ++k) {
        auto view = vt_manager_->getPoolDebugView(
            static_cast<es::VtLayer>(k));
        if (view) {
            vt_ids[k] = er::Helper::addImTextureID(
                vt_manager_->getPoolDebugSampler(), view);
        }
    }
    menu_->setVtPoolTextureIds(vt_ids);
}

void RealWorldApplication::registerIblDebugImTextureIds() {
    if (!menu_ || !ibl_creator_ || !skydome_) return;

    using MipFaceArray = engine::ui::Menu::IblDebugMipFaceArray;
    constexpr int kMaxMips = engine::ui::Menu::kIblDebugMaxMips;

    // Walk every mip [0, num_mips) of one TextureInfo and wrap each cube
    // face's pre-existing 2D view (TextureInfo::surface_views[mip][face],
    // populated by createCubemapTexture for use_as_framebuffer cubemaps)
    // as an ImTextureID via the standard linear sampler.  Returns the
    // count actually registered (may be < tex.surface_views.size() if it
    // exceeds kMaxMips).
    auto register_all_mips = [&](
        const er::TextureInfo& tex,
        MipFaceArray& out) -> int {
        out = {};
        const int actual = static_cast<int>(tex.surface_views.size());
        const int n = std::min(actual, kMaxMips);
        for (int m = 0; m < n; ++m) {
            const auto& faces = tex.surface_views[m];
            const int nf = std::min<int>(6, static_cast<int>(faces.size()));
            for (int f = 0; f < nf; ++f) {
                if (faces[f]) {
                    out[m][f] = er::Helper::addImTextureID(
                        texture_sampler_, faces[f]);
                }
            }
        }
        return n;
    };

    // Sky envmap: full mip chain (let user inspect mipgen-smoothed
    // partial updates).
    {
        MipFaceArray ids{};
        const int n = register_all_mips(ibl_creator_->getEnvmapTexture(), ids);
        menu_->setEnvmapFaceMipTextureIds(ids, n);
    }
    // Sky mini-buffer: single mip but exposed as the same shape so the
    // menu's per-row mip slider is uniform.
    {
        MipFaceArray ids{};
        const int n = register_all_mips(skydome_->getMiniEnvmapTexture(), ids);
        menu_->setMiniEnvmapFaceMipTextureIds(ids, n);
    }
    // IBL diffuse: single mip in the current pipeline.
    {
        MipFaceArray ids{};
        const int n = register_all_mips(ibl_creator_->getIblDiffuseTexture(), ids);
        menu_->setIblDiffuseFaceMipTextureIds(ids, n);
    }
    // IBL specular: full mip chain.  Mip 0 is the GGX convolution; mips
    // 1..N-1 are box-filter mipgen of mip 0 - the slider lets the user
    // step through the blur progression.
    {
        MipFaceArray ids{};
        const int n = register_all_mips(ibl_creator_->getIblSpecularTexture(), ids);
        menu_->setIblSpecularFaceMipTextureIds(ids, n);
    }
    // IBL sheen: full mip chain (Charlie convolution at mip 0 + mipgen).
    {
        MipFaceArray ids{};
        const int n = register_all_mips(ibl_creator_->getIblSheenTexture(), ids);
        menu_->setIblSheenFaceMipTextureIds(ids, n);
    }

    // ── Hi-Z pyramid mip thumbnails for the Render Debug viewer ──
    // One ImTextureID per mip wrapped via Helper::addImTextureID using
    // the same Hi-Z sampler the build / cull passes use.  Lets the user
    // visually verify each frame whether the pyramid actually contains
    // depth data — black/grey thumbnails = real depth, uniform white =
    // build dispatch never wrote, etc.  See menu.cpp's "Hi-Z Pyramid
    // Debug" window.
    if (menu_ && hiz_sampler_ && !hiz_mip_views_.empty()) {
        std::vector<ImTextureID> hiz_ids;
        hiz_ids.reserve(hiz_mip_views_.size());
        for (const auto& mip_view : hiz_mip_views_) {
            if (mip_view) {
                hiz_ids.push_back(
                    er::Helper::addImTextureID(hiz_sampler_, mip_view));
            } else {
                hiz_ids.push_back(ImTextureID(0));
            }
        }
        menu_->setHiZDebugTextureIds(std::move(hiz_ids));
    }

    // ── Dynamic cubemap face thumbnails for the Render Debug viewer ──
    // Both ping-pong buffers' six per-face 2D layer views are wrapped
    // as ImTextureIDs.  The viewer picks the right ping-pong slice each
    // frame using getCurrentReadIdx() (forwarded via setDynamicCubeFrameInfo
    // each frame in renderFrame).
    if (menu_ && dynamic_cubemap_) {
        std::array<engine::ui::Menu::IblDebugFaceArray, 2> dyn_ids{};
        for (int p = 0; p < 2; ++p) {
            for (uint32_t f = 0; f < es::DynamicCubemap::kNumFaces; ++f) {
                const auto& view = dynamic_cubemap_->getColorFaceView(p, f);
                if (view) {
                    dyn_ids[p][f] = er::Helper::addImTextureID(
                        texture_sampler_, view);
                }
            }
        }
        menu_->setDynamicCubeFaceTextureIds(dyn_ids);
    }
}

void RealWorldApplication::createTextureSampler() {
    texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::CLAMP_TO_EDGE,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    texture_point_sampler_ = device_->createSampler(
        er::Filter::NEAREST,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    repeat_texture_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());

    mirror_repeat_sampler_ = device_->createSampler(
        er::Filter::LINEAR,
        er::SamplerAddressMode::MIRRORED_REPEAT,
        er::SamplerMipmapMode::LINEAR, 16.0f,
        std::source_location::current());
}

void RealWorldApplication::mainLoop() {
    while (!glfwWindowShouldClose(window_) && !s_exit_game) {
        glfwPollEvents();
        drawFrame();
    }

    device_->waitIdle();
}

void RealWorldApplication::drawScene(
    std::shared_ptr<er::CommandBuffer> command_buffer,
    const er::SwapChainInfo& swap_chain_info,
    const std::shared_ptr<er::DescriptorSet>& view_desc_set,
    const glm::uvec2& screen_size,
    uint32_t image_index,
    float delta_t,
    float current_time) {

    auto frame_buffer = swap_chain_info.framebuffers[image_index];
    auto src_color = swap_chain_info.image_views[image_index];

    auto& cmd_buf = command_buffer;

    static int s_dbuf_idx = 0;

    // Sync the deferred-vs-forward toggle from the Render Debug menu.
    // The menu owns the canonical state; drawScene reads it each frame
    // so flipping the menu item takes effect on the very next draw.
    //
    // EXCEPTION: a couple of debug-render modes depend on per-material
    // flags (BindlessMaterialParams.flags) that the deferred G-buffer
    // intentionally drops at write time -- the deferred_resolve.comp
    // never sees them, so the mode would render as the regular shaded
    // path with no visible difference and the user (rightly) reports
    // "nothing shows up".  Force-flip back to forward for those modes:
    //
    //   9  DEBUG_RENDER_MODE_TRANSLUCENT — needs BINDLESS_MAT_TRANSLUCENT
    //   13 DEBUG_RENDER_MODE_CATEGORY    — needs BINDLESS_MAT_CATEGORY_*
    //
    // The menu's deferred_rendering_ state is left untouched so flipping
    // the debug mode back to 0 restores the user's chosen pipeline.
    if (menu_) {
        deferred_rendering_enabled_ = menu_->isDeferredRendering();
        const int dbg_mode = menu_->getDebugRenderMode();
        if (dbg_mode == DEBUG_RENDER_MODE_TRANSLUCENT ||
            dbg_mode == DEBUG_RENDER_MODE_CATEGORY ||
            dbg_mode == DEBUG_RENDER_MODE_OBJECT_ID) {
            if (deferred_rendering_enabled_) {
                static bool s_logged = false;
                if (!s_logged) {
                    std::cout
                        << "[render.debug] mode " << dbg_mode
                        << " requires per-material flags; forcing "
                           "forward rendering for this frame "
                           "(deferred drops the bits in the G-buffer)"
                        << std::endl;
                    s_logged = true;
                }
                deferred_rendering_enabled_ = false;
            }
        }
    }

    // ----- GPU Profiler: begin frame -----------------------------------------
    if (!gpu_profiler_initialized_) {
        gpu_profiler_.init(device_, kMaxFramesInFlight);
        gpu_profiler_initialized_ = true;
    }
    // Consume Space-key pause toggle.
    if (s_toggle_profiler_pause) {
        gpu_profiler_.togglePause();
        s_toggle_profiler_pause = false;
    }
    // Consume F1-key edge-trigger for collision-mesh debug. Routes
    // through the menu so the menu checkbox and the F1 hotkey are
    // perfectly in sync.
    if (s_request_toggle_collision_debug) {
        if (menu_) {
            menu_->toggleCollisionDebug();
            std::cout << "[collision-debug] "
                      << (menu_->isCollisionDebugOn() ? "ON" : "OFF")
                      << "  meshes=" << collision_world_.meshCount()
                      << "  pipeline_ready="
                      << (eh::CollisionDebugDraw::ready() ? "yes" : "no")
                      << "  collision_world_built=" << collision_world_built_
                      << "  game_state="
                      << static_cast<int>(menu_->getGameState())
                      << std::endl;
            if (menu_->isCollisionDebugOn() &&
                collision_world_.meshCount() == 0) {
                std::cout << "[collision-debug] WARNING: collision world "
                             "is empty. The build retries every frame "
                             "while in InGame state -- if you keep "
                             "seeing 0, buildFromDrawable() is failing "
                             "(vertex_position_ or vertex_indices_ not "
                             "populated)." << std::endl;
            }
        }
        s_request_toggle_collision_debug = false;
    }
    // Consume Left/Right arrow edge-trigger: step the isolated collision
    // mesh.  Routed through the menu (auto-enables isolate + clamps) so
    // the slider and the hotkeys share one index.
    if (s_collision_isolate_step != 0) {
        if (menu_) menu_->stepCollisionIsolate(s_collision_isolate_step);
        s_collision_isolate_step = 0;
    }
    gpu_profiler_.beginFrame(cmd_buf, static_cast<uint32_t>(current_frame_ % kMaxFramesInFlight));
    // CPU frame is anchored at the top of drawFrame() (well before
    // here) so its timeline covers the whole host work for this
    // frame, not just the command-recording portion.  See drawFrame.

    {
        // GpuProfiler::Scope = paired CPU + GPU scope (RAII).  The CPU
        // bar measures host-side recording time, the GPU bar measures
        // execution time; they share a name and align horizontally in
        // the profiler timeline.
        engine::helper::GpuProfiler::Scope _scope(
            gpu_profiler_, cmd_buf, "IBL / Skydome");
        // ORDER MATTERS: the sky scattering LUT must be (re)generated
        // BEFORE the cubemap consumes it.  The LUT is gated by scale-
        // height changes (cheap no-op once converged), but on frame 0
        // its contents are uninitialised - if we let updateCubeSkyBoxMini
        // run first it samples zeros, computes garbage atmospheric
        // values, and the block-fill bootstrap broadcasts that garbage
        // across every envmap texel, leaving the EMA to slowly correct
        // it over many seconds.  Doing the LUT first guarantees frame 0
        // sees correct LUT data and the first-touch fill is valid.
        skydome_->updateSkyScatteringLut(cmd_buf);

        // Sky envmap + IBL convolution: pure dithered compute path.
        //
        // Both Skydome::updateCubeSkyBoxMini and IblCreator::updateIbl*MapMini
        // self-bootstrap.  On the very first call (mini_frame_index_ == 0)
        // the compute shader runs in "block fill" mode: each thread
        // evaluates one sample at its dither position and broadcasts it
        // to every texel in the corresponding 8x8 / stride^2 block, so
        // the entire output is initialized in one cheap dispatch.  From
        // frame 1 onward each thread updates only the dither-selected
        // texel and EMA-blends with the existing value, integrating
        // additional samples over time.
        //
        // This removes the old expensive bootstrap (drawCubeSkyBox +
        // createIbl*Map) entirely.  Steady-state per-frame cost is the
        // same as before; the first frame is now ~4000x cheaper than
        // it used to be.
        skydome_->updateCubeSkyBoxMini(
            cmd_buf,
            ibl_creator_->getEnvmapTexture(),
            kCubemapSize);

        // updateIblDiffuseMapMini now (a) convolves into the 4× / 16×-
        // area accumulator rt_ibl_diffuse_tex_ for higher dither
        // coverage per frame, then (b) LINEAR-filter blits down to the
        // 1× consumer-facing tmp_ibl_diffuse_tex_ — the 4×4 → 1
        // downsample is a free 16-tap box filter that absorbs the
        // residual Monte-Carlo speckle without an explicit Gaussian
        // blur pass.  See IblCreator::updateIblDiffuseMapMini for the
        // size choice and blit setup.
        ibl_creator_->updateIblDiffuseMapMini(cmd_buf, kCubemapSize);
        ibl_creator_->updateIblSpecularMapMini(cmd_buf, kCubemapSize);
        // Sheen runs last - it advances the shared mini_frame_index_ so
        // all three IBL filters use the same per-frame dither offset and
        // sample stratum.
        ibl_creator_->updateIblSheenMapMini(cmd_buf, kCubemapSize);
        // _scope auto-closes via RAII.
    }

    // NOTE: ambient probe update USED to live here (right after IBL
    // / Skydome update), but that was BEFORE cluster_renderer_->cull()
    // runs — and cull's first action is a host-write to zero
    // draw_count_buffer_.  Since the count buffer is HOST_COHERENT,
    // the GPU sees the zeroed value at the moment it executes the
    // probe's drawIndexedIndirectCount, which then draws nothing.
    // The probe-pass code is therefore moved to AFTER the cluster
    // cull dispatch (search for the "Ambient probe system" block
    // below).

    er::DescriptorSetList desc_sets{
        pbr_lighting_desc_set_,
        view_desc_set };

    {
        engine::helper::GpuProfiler::Scope _scope_terrain(
            gpu_profiler_, cmd_buf, "Terrain / Weather Update");
        // only init one time.
        static bool s_tile_buffer_inited = false;
        if (!s_tile_buffer_inited) {
            ego::TileObject::generateTileBuffers(cmd_buf);
            weather_system_->initTemperatureBuffer(cmd_buf);
            volume_noise_->initNoiseTexture(cmd_buf);
            s_tile_buffer_inited = true;
        }
        else {
            ego::TileObject::updateTileFlowBuffers(cmd_buf, current_time_, s_dbuf_idx);
            ego::TileObject::updateTileBuffers(cmd_buf, current_time_, s_dbuf_idx);
            weather_system_->updateAirflowBuffer(
                cmd_buf,
                menu_->getWeatherControls(),
                menu_->getGloalFlowDir(),
                menu_->getGlobalFlowSpeed(),
                menu_->getCloudMoistToPressureRatio(),
                s_dbuf_idx,
                current_time);
        }

        skydome_->updateSkyScatteringLut(cmd_buf);

        weather_system_->updateCloudShadow(
            cmd_buf,
            skydome_->getSunDir(),
            menu_->getLightExtFactor(),
            s_dbuf_idx,
            current_time);
        // _scope_terrain auto-closes via RAII.
    }

    // Declared here so both the update block and the render block can access them.
    // 6 view-space far depths spanning ~0–300 m with smooth ~2.5× ratios
    // (was 4 splits with ~4× ratios — that produced visible cascade
    // steps at boundaries because per-cascade extent doubled too fast).
    const std::array<float, CSM_CASCADE_COUNT> csm_cascade_splits = {
        3.0f, 8.0f, 20.0f, 50.0f, 120.0f, 300.0f
    };
    std::array<glm::mat4, CSM_CASCADE_COUNT> csm_cascade_vps;

    // Light-space orthographic projection fitted to the FULL view frustum
    // (z_near → last cascade's far).  No longer used by the cluster
    // shadow path (Option B switched to per-cascade culling, each
    // dispatch with its own cascade VP).  computeCascadeMatrices's
    // out_union_vp param is left in place for any future caller that
    // wants a single "union of all cascades" frustum, but here we pass
    // nullptr to skip the fit work.

    // this has to be happened after tile update, or you wont get the right height info.
    {
        engine::helper::GpuProfiler::Scope _scope_go(
            gpu_profiler_, cmd_buf, "Game Object Updates");
        static std::chrono::time_point s_last_time = std::chrono::steady_clock::now();
        auto cur_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = cur_time - s_last_time;
        auto delta_t = static_cast<float>(elapsed_seconds.count());
        s_last_time = cur_time;

        ego::DrawableObject::updateGameObjectsBuffer(
            cmd_buf,
            ego::TileObject::getWorldMin(),
            ego::TileObject::getWorldRange(),
            main_camera_object_->getCameraPosition(),
            menu_->getAirFlowStrength(),
            menu_->getWaterFlowStrength(),
            0,//s_update_frame_count,
            s_dbuf_idx,
            delta_t,
            menu_->isAirfowOn());

        main_camera_object_->updateCamera(
            cmd_buf,
            s_dbuf_idx,
            s_key,
            s_update_frame_count,
            delta_t,
            s_last_mouse_pos,
            s_mouse_wheel_offset,
            !s_camera_paused && s_mouse_right_button_pressed);

        // Push per-frame feature flags into the camera buffer so shaders can
        // conditionally skip expensive passes (e.g. shadow sampling) and
        // dispatch on the active "Render Debug" visualisation mode.
        {
            uint32_t input_flags = 0u;
            if (menu_->isShadowPassTurnOff())
                input_flags |= FEATURE_INPUT_SHADOW_DISABLED;
            // Pack the menu's render-debug mode (0..255) into bits 16..23.
            // base.frag and cluster_bindless.frag mask + branch on this.
            input_flags |= (static_cast<uint32_t>(menu_->getDebugRenderMode())
                            << FEATURE_INPUT_DEBUG_MODE_SHIFT)
                           & FEATURE_INPUT_DEBUG_MODE_MASK;
            // Pack the Hi-Z debug mip selector into bits 24..27, clamped
            // to the actual pyramid mip count so the shader's textureLod
            // never reads off the top of the chain when a high level was
            // remembered from a different swap-chain size.
            {
                int mip = menu_->getHiZDebugMip();
                if (mip < 0) mip = 0;
                int max_mip = hiz_pyramid_mips_ > 0
                    ? static_cast<int>(hiz_pyramid_mips_) - 1 : 0;
                if (mip > max_mip) mip = max_mip;
                input_flags |= (static_cast<uint32_t>(mip)
                                << FEATURE_INPUT_HIZ_MIP_SHIFT)
                               & FEATURE_INPUT_HIZ_MIP_MASK;
            }
            // Collision-LOD debug background.  Two mutually-exclusive
            // modes, both gated on the overlay being active with a built
            // (post-classification) collision world:
            //
            //   * ISOLATE — when the Left/Right-arrow isolate slider is
            //     enabled, make the textured background follow the SAME
            //     debug index as the collision overlay: resolve the
            //     isolated collision mesh back to its source primitive
            //     (drawable, mesh, prim), map that to the mesh's cluster
            //     object id (MeshInfo::cluster_global_mesh_idx_) and then
            //     to the globally-unique cluster material_idx of that
            //     primitive, and restrict the background to JUST that
            //     material via FEATURE_INPUT_ISOLATE_MESH.  Every other
            //     mesh is discarded in cluster_bindless.frag.  Falls back
            //     to FLOOR_ONLY when the source can't be resolved (the
            //     aggregate build paths don't track a single primitive,
            //     the mesh was never cluster-uploaded, or its geometry was
            //     released).
            //
            //   * FLOOR_ONLY (default, unchanged) — restrict the textured
            //     scene to Floor-category geometry so it matches the
            //     floor-only collision overlay.  Gated on
            //     categoriesApplied() so the background is NOT blanked
            //     before classification finishes (pre-ML the full scene
            //     shows under the fallback floor overlay).
            if (menu_->isCollisionDebugOn() && !collision_world_.empty()) {
                int isolate_material = -1;
                if (menu_->collisionIsolateEnabled() && cluster_renderer_) {
                    const auto* cm = collision_world_.meshAt(
                        static_cast<size_t>(menu_->collisionIsolateIndex()));
                    if (cm) {
                        const auto* src = cm->sourceDrawable();
                        const size_t smi = cm->sourceMeshIdx();
                        const size_t spi = cm->sourcePrimIdx();
                        if (src && smi < src->getMeshes().size()) {
                            const int32_t gmi =
                                src->getMeshes()[smi].cluster_global_mesh_idx_;
                            if (gmi >= 0) {
                                isolate_material =
                                    cluster_renderer_->materialIdxForPrimitive(
                                        static_cast<uint32_t>(gmi),
                                        static_cast<uint32_t>(spi));
                            }
                        }
                    }
                }
                if (isolate_material >= 0) {
                    // setDebugIsolateMaterial stages the value; the
                    // setInputFeatureFlags call below re-uploads the whole
                    // camera struct, so it must come first.
                    main_camera_object_->setDebugIsolateMaterial(
                        static_cast<uint32_t>(isolate_material));
                    input_flags |= FEATURE_INPUT_ISOLATE_MESH;
                } else if (cluster_renderer_ &&
                           cluster_renderer_->categoriesApplied()) {
                    input_flags |= FEATURE_INPUT_FLOOR_ONLY;
                }
            }
            main_camera_object_->setInputFeatureFlags(input_flags);
        }

        // Tell the forward pass whether to skip clustered meshes.
        // When the cluster indirect draw is active the cluster renderer owns
        // those meshes; drawing them in the forward pass as well causes
        // double-rendering with z-fighting.
        // Require getTotalVisible() > 0 so the forward pass acts as a fallback
        // for the first frame (before the first readback arrives) and for any
        // frame where the GPU cull produces zero draws — preventing black holes
        // where neither path renders the mesh.
        engine::helper::clusterIndirectActive() =
            cluster_renderer_ &&
            cluster_renderer_->isEnabled() &&
            !engine::helper::clusterRenderingEnabled() &&  // not debug-draw mode
            cluster_renderer_->getTotalVisible() > 0;      // cluster pass has output

        // Update the base shadow camera position (facing dir, up vector).
        shadow_camera_object_->updateCamera(
            cmd_buf,
            main_camera_object_->getCameraPos());

        s_mouse_wheel_offset = 0;

        // ── Sync shadow light direction with the live sun position ────────────
        // getSunDir() returns a ground→sun unit vector (positive Y = above
        // horizon).  The shadow camera expects sun→ground (negate it).
        // Clamp to a minimum elevation of ~5° so cascades don't blow up
        // when the sun skims the horizon or dips below it.
        {
            glm::vec3 sun = skydome_->getSunDir();
            sun.y = std::max(sun.y, 0.087f);  // sin(5°)
            shadow_camera_object_->setLightDir(-glm::normalize(sun));
        }

        // ── Compute CSM cascade matrices ──────────────────────────────────────
        main_camera_object_->readGpuCameraInfo();
        const auto& main_cam = main_camera_object_->getCameraViewInfo();

        // View-space far depths for each cascade (tune to scene scale).
        // out_union_vp = nullptr — Option B per-cascade culling uses the
        // individual cascade VPs directly, no union needed.
        std::array<glm::vec3, CSM_CASCADE_COUNT * 8> csm_slab_corners_ws;
        shadow_camera_object_->computeCascadeMatrices(
            main_cam.view,
            main_cam.proj,
            csm_cascade_splits,
            0.1f,  // main camera z_near
            csm_cascade_vps,
            /*out_union_vp*/ nullptr,
            &csm_slab_corners_ws);

        glsl::RuntimeLightsParams runtime_lights_params = {};
        {
            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                runtime_lights_params.light_view_proj[k] = csm_cascade_vps[k];
            }
            for (int i = 0; i < CSM_CASCADE_COUNT * 8; ++i) {
                runtime_lights_params.cascade_slab_corners_ws[i] =
                    glm::vec4(csm_slab_corners_ws[i], 1.0f);
            }
            // Pack the 6 view-space far depths into vec4[2].  Slots 6
            // and 7 are unused but must be initialised — leave them at
            // a value larger than any cascade so misindexing falls
            // through correctly to the last cascade in the shader.
            runtime_lights_params.cascade_splits[0] = glm::vec4(
                csm_cascade_splits[0], csm_cascade_splits[1],
                csm_cascade_splits[2], csm_cascade_splits[3]);
            runtime_lights_params.cascade_splits[1] = glm::vec4(
                csm_cascade_splits[4], csm_cascade_splits[5],
                1e9f, 1e9f);
            for (int l = 0; l < LIGHT_COUNT; l++) {
                runtime_lights_params.lights[l].type = glsl::LightType_Directional;
                runtime_lights_params.lights[l].color = glm::vec3(255.0f, 250.0f, 240.0f) / 255.0f;
                runtime_lights_params.lights[l].direction = shadow_camera_object_->getLightDir();
                runtime_lights_params.lights[l].intensity = 20.0f;
                runtime_lights_params.lights[l].position = glm::vec3(0, 0, 0);
            }
        }

        device_->updateBufferMemory(
            runtime_lights_buffer_->memory,
            sizeof(runtime_lights_params),
            &runtime_lights_params);

        if (player_object_) {
            player_object_->updateBuffers(cmd_buf);
        }

        // Foot debug markers — same buffer-update step as the player
        // so their instance / indirect-draw / joint-matrix uploads ride
        // the same command-buffer recording window.
        if (foot_marker_left_  && foot_marker_left_->isReady())  foot_marker_left_->updateBuffers(cmd_buf);
        if (foot_marker_right_ && foot_marker_right_->isReady()) foot_marker_right_->updateBuffers(cmd_buf);

        // Bistro scene buffer updates — enabled once game has started.
        if (bistro_exterior_scene_ && bistro_exterior_scene_->isReady()) {
            bistro_exterior_scene_->updateBuffers(cmd_buf);
        }
        if (bistro_interior_scene_ && bistro_interior_scene_->isReady()) {
            bistro_interior_scene_->updateBuffers(cmd_buf);
        }

        for (auto& drawable_obj : drawable_objects_) {
            drawable_obj->updateBuffers(cmd_buf);
        }

        if (s_update_frame_count >= 0) {
            s_update_frame_count++;
        }
        // _scope_go auto-closes via RAII.
    }

    {
        if (s_render_hair_test) {
            hair_patch_->update(
                device_,
                cmd_buf,
                desc_sets
            );
        }
        else if (s_render_lbm_test) {
            lbm_patch_->update(
                device_,
                cmd_buf,
                desc_sets
            );
        }

        er::DescriptorSetList desc_sets(MAX_NUM_PARAMS_SETS);
        desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
        desc_sets[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_;

        // ── Single-pass CSM shadow render ─────────────────────────────────────
        // The geometry shader (base_depthonly_csm.geom) reads all 4 VP matrices
        // from RuntimeLightsParams and broadcasts each triangle to all 4 depth
        // array layers in one draw.  Vertex transform runs once per vertex
        // instead of 4×, giving roughly 4× the throughput of separate passes.
        // The full 4-layer array view is used so the GS can write to all layers.
        if (!menu_->isShadowPassTurnOff()) {
            engine::helper::GpuProfiler::Scope _scope_shadow(
                gpu_profiler_, cmd_buf, "CSM Shadow");

            // ── Periodic shadow-pass material classification stats ───
            // Prints one line every ~10 s of wall-clock time showing how
            // many engine-wide materials are on the slow alpha-cutoff
            // frag-shader path vs the fast no-frag path.  Counters live
            // on ego::DrawableObject and are bumped by
            // computeEffectiveOpaqueForMaterials at mesh-load time
            // (sync constructor + async Phase2Fn).  Per-frame printing
            // was too spammy; the totals only move when new meshes
            // stream in, so a 10-second cadence still catches every
            // change without flooding the log.
            {
                static float s_last_print_time = -1e30f;
                if (current_time - s_last_print_time >= 10.0f) {
                    const int total = ego::DrawableObject::
                        s_total_materials_count_.load(
                            std::memory_order_relaxed);
                    const int cutoff = ego::DrawableObject::
                        s_alpha_cutoff_materials_count_.load(
                            std::memory_order_relaxed);
                    const double pct =
                        (total > 0)
                            ? (100.0 * double(cutoff) / double(total))
                            : 0.0;
                    std::printf(
                        "[shadow] frame=%llu alpha_cutoff=%d/%d (%.1f%%)\n",
                        static_cast<unsigned long long>(current_frame_),
                        cutoff, total, pct);
                    s_last_print_time = current_time;
                }
            }

            // ── CSM silhouette prepass ───────────────────────────────────
            // Runs in its OWN dynamic-rendering scope BEFORE the legacy
            // shadow draw.  Clears the depth buffer to 0.0 and fills each
            // cascade's main-camera-frustum interior with depth=1.0.
            // Subsequent shadow draws (legacy + cluster) use LOAD_OP_LOAD
            // so their writes layer on top of this pre-fill.  Out-of-
            // silhouette texels remain at 0.0 → reject every caster via
            // LESS_OR_EQUAL → Hi-Z propagates to per-tile primitive culling
            // at the PD.  Only runs when the cluster pipeline is up (it
            // owns the prepass pipeline + shader).
            bool silhouette_prefilled = false;
            if (cluster_renderer_ &&
                cluster_renderer_->isEnabled() &&
                engine::helper::clusterIndirectActive() &&
                menu_->isCsmSilhouettePrepassEnabled()) {
                er::RenderingAttachmentInfo prepass_depth_attach;
                prepass_depth_attach.image_view   = csm_shadow_tex_->view;
                prepass_depth_attach.image_layout =
                    er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                prepass_depth_attach.load_op  = er::AttachmentLoadOp::CLEAR;
                prepass_depth_attach.store_op = er::AttachmentStoreOp::STORE;
                prepass_depth_attach.clear_value.depth_stencil = { 0.0f, 0 };

                er::RenderingInfo prepass_ri = {};
                prepass_ri.render_area_offset = { 0, 0 };
                prepass_ri.render_area_extent = {
                    csm_shadow_tex_->size.x, csm_shadow_tex_->size.y };
                prepass_ri.layer_count        = CSM_CASCADE_COUNT;
                prepass_ri.view_mask          = 0;
                prepass_ri.color_attachments  = {};
                prepass_ri.depth_attachments  = { prepass_depth_attach };
                prepass_ri.stencil_attachments = {};
                cmd_buf->beginDynamicRendering(prepass_ri);

                er::Viewport vp;
                vp.x = 0; vp.y = 0;
                vp.width  = float(csm_shadow_tex_->size.x);
                vp.height = float(csm_shadow_tex_->size.y);
                vp.min_depth = 0.0f;
                vp.max_depth = 1.0f;
                er::Scissor sc;
                sc.offset = glm::ivec2(0);
                sc.extent = glm::uvec2(
                    csm_shadow_tex_->size.x, csm_shadow_tex_->size.y);

                cluster_renderer_->drawCsmSilhouettePrepass(
                    cmd_buf,
                    runtime_lights_desc_set_,
                    { vp }, { sc });

                cmd_buf->endDynamicRendering();
                silhouette_prefilled = true;
            }

            // ── Drawable-shadow draw-mode dispatch ────────────────────
            // Menu exposes three mutually-exclusive paths for CSM:
            //   kRegular        — N=CSM_CASCADE_COUNT single-layer passes
            //                      driven by a host-side cascade loop.
            //                      Each iteration draws into
            //                      csm_layer_views_[k] with cascade_idx=k
            //                      pushed via ModelParams; the _CSMCASC
            //                      VS permutation reads
            //                      lights_params.light_view_proj[cascade_idx].
            //                      No GS, no mesh shader.
            //   kGeometryShader — Single layered draw,
            //                      base_depthonly_csm.geom broadcasts each
            //                      triangle to every cascade layer.
            //   kMeshShader     — Single layered draw routed through
            //                      DrawMode::kCsmMeshShader.  Eligible
            //                      primitives (opaque, non-skinned, UINT32
            //                      indices, ≤256 verts/tris) dispatch via
            //                      base_depthonly_csm.task + .mesh and
            //                      per-primitive SSBO bindings.  Ineligible
            //                      primitives fall back to the GS pipeline
            //                      inside drawMesh.
            const auto csm_mode = menu_->getCsmDrawMode();
            using CsmDrawMode = engine::ui::Menu::CsmDrawMode;

            // ── Periodic CSM-mode trace ──────────────────────────────
            // Confirms the menu toggle is actually reaching the dispatch.
            // Prints once on first frame after a mode change, and every
            // ~10 s thereafter while that mode is active.  Lets the user
            // verify "I picked Regular and the loop is running" without
            // having to attach a debugger.
            {
                static CsmDrawMode s_last_mode  = CsmDrawMode::kRegular;
                static float       s_last_print = -1e30f;
                static bool        s_first      = true;
                const bool mode_changed = s_first || (s_last_mode != csm_mode);
                if (mode_changed ||
                    (current_time - s_last_print >= 10.0f)) {
                    const char* mode_name =
                        (csm_mode == CsmDrawMode::kRegular)    ? "Regular (per-cascade)" :
                        (csm_mode == CsmDrawMode::kMeshShader) ? "MeshShader (task+mesh, GS fallback for ineligible)" :
                                                                 "GeometryShader (layered)";
                    std::printf("[shadow] csm_draw_mode=%s\n", mode_name);
                    s_last_mode  = csm_mode;
                    s_last_print = current_time;
                    s_first      = false;
                }
            }

            if (csm_mode == CsmDrawMode::kRegular) {
                // ── Regular mode: host loops cascades, no GS, no mesh.
                // Each iteration opens a single-layer dynamic-rendering
                // scope against csm_layer_views_[k] and dispatches the
                // _CSMCASC pipeline via DrawMode::kCsmPerCascade.  Each
                // cascade's depth layer is independent, so preserve_depth
                // matches the silhouette-prepass intent (LOAD if pre-
                // filled, CLEAR otherwise) for every iteration.
                for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
                    shadow_object_scene_view_->draw(
                        cmd_buf,
                        desc_sets,
                        nullptr,
                        s_dbuf_idx,
                        delta_t,
                        current_time,
                        /*depth_only*/ true,
                        csm_layer_views_[k],      // single-layer view of cascade k
                        /*layer_count*/ 1,        // 1 = single-cascade draw
                        /*preserve_depth*/ silhouette_prefilled,
                        /*csm_cascade_idx*/ int32_t(k));
                }
            } else {
                // kGeometryShader or kMeshShader: single layered draw.
                // When kMeshShader, ObjectSceneView::draw routes through
                // DrawMode::kCsmMeshShader (task+mesh for eligible
                // primitives, GS fallback inside drawMesh for the rest).
                // Otherwise the GS path is used directly.
                const bool use_mesh_shader =
                    (csm_mode == CsmDrawMode::kMeshShader);
                shadow_object_scene_view_->draw(
                    cmd_buf,
                    desc_sets,
                    nullptr,
                    s_dbuf_idx,
                    delta_t,
                    current_time,
                    true,
                    csm_shadow_tex_->view,    // full CSM_CASCADE_COUNT-layer array view
                    CSM_CASCADE_COUNT,        // layer_count triggers layered path
                    silhouette_prefilled,     // preserve_depth — keep prepass output
                    /*csm_cascade_idx*/ -1,
                    /*csm_use_mesh_shader*/ use_mesh_shader);
            }

            // ── Cluster CSM shadow draw — 3-way menu dispatch ───────────
            // The cluster path now mirrors the drawable path's three modes
            // so the menu's "Drawable shadow draw mode" toggle drives the
            // bulk of shadow work too (cluster meshes own ~95% of Bistro
            // triangles; without this the toggle only changed the tiny
            // legacy path).  Cluster meshes are excluded from the previous
            // shadow_object_scene_view_->draw via the gate in
            // drawable_object.cpp.
            if (cluster_renderer_ &&
                cluster_renderer_->isEnabled() &&
                engine::helper::clusterIndirectActive()) {

                er::Viewport vp;
                vp.x = 0; vp.y = 0;
                vp.width  = float(csm_shadow_tex_->size.x);
                vp.height = float(csm_shadow_tex_->size.y);
                vp.min_depth = 0.0f;
                vp.max_depth = 1.0f;
                er::Scissor sc;
                sc.offset = glm::ivec2(0);
                sc.extent = glm::uvec2(
                    csm_shadow_tex_->size.x, csm_shadow_tex_->size.y);

                er::DescriptorSetList csm_cluster_desc(
                    RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                csm_cluster_desc[VIEW_PARAMS_SET] =
                    shadow_camera_object_->getViewCameraDescriptorSet();
                csm_cluster_desc[RUNTIME_LIGHTS_PARAMS_SET] =
                    runtime_lights_desc_set_;

                if (csm_mode == CsmDrawMode::kRegular) {
                    // Per-cascade: 6 single-layer dispatches, each its own
                    // dynamic-rendering scope against csm_layer_views_[k].
                    // Push cascade_idx via the VS's push constant; no GS,
                    // no mesh shader.  preserve_depth carries the
                    // silhouette-prepass output through to LOAD on every
                    // iteration.
                    for (uint32_t k = 0; k < CSM_CASCADE_COUNT; ++k) {
                        er::RenderingAttachmentInfo depth_attach;
                        depth_attach.image_view   = csm_layer_views_[k];
                        depth_attach.image_layout =
                            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                        depth_attach.load_op = er::AttachmentLoadOp::LOAD;
                        depth_attach.store_op = er::AttachmentStoreOp::STORE;

                        er::RenderingInfo per_cascade_ri = {};
                        per_cascade_ri.render_area_offset = { 0, 0 };
                        per_cascade_ri.render_area_extent = {
                            csm_shadow_tex_->size.x,
                            csm_shadow_tex_->size.y };
                        per_cascade_ri.layer_count        = 1;
                        per_cascade_ri.view_mask          = 0;
                        per_cascade_ri.color_attachments  = {};
                        per_cascade_ri.depth_attachments  = { depth_attach };
                        per_cascade_ri.stencil_attachments = {};
                        cmd_buf->beginDynamicRendering(per_cascade_ri);

                        cluster_renderer_->drawClusterShadowPerCascade(
                            cmd_buf, csm_cluster_desc, { vp }, { sc }, k);

                        cmd_buf->endDynamicRendering();
                    }
                } else {
                    // kGeometryShader / kMeshShader: single layered scope
                    // against the full CSM_CASCADE_COUNT-layer array view.
                    er::RenderingAttachmentInfo depth_attach;
                    depth_attach.image_view   = csm_shadow_tex_->view;
                    depth_attach.image_layout =
                        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    depth_attach.load_op  = er::AttachmentLoadOp::LOAD;
                    depth_attach.store_op = er::AttachmentStoreOp::STORE;

                    er::RenderingInfo cluster_csm_ri = {};
                    cluster_csm_ri.render_area_offset = { 0, 0 };
                    cluster_csm_ri.render_area_extent = {
                        csm_shadow_tex_->size.x,
                        csm_shadow_tex_->size.y };
                    cluster_csm_ri.layer_count        = CSM_CASCADE_COUNT;
                    cluster_csm_ri.view_mask          = 0;
                    cluster_csm_ri.color_attachments  = {};
                    cluster_csm_ri.depth_attachments  = { depth_attach };
                    cluster_csm_ri.stencil_attachments = {};
                    cmd_buf->beginDynamicRendering(cluster_csm_ri);

                    if (csm_mode == CsmDrawMode::kMeshShader) {
                        cluster_renderer_->drawClusterShadow(
                            cmd_buf, csm_cluster_desc, { vp }, { sc });
                    } else {
                        cluster_renderer_->drawClusterShadowGs(
                            cmd_buf, csm_cluster_desc, { vp }, { sc });
                    }

                    cmd_buf->endDynamicRendering();
                }
            }
        }

        // Custom resource infos matching the shadow pass attachment layout.
        // The shadow depth buffer uses DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        // (not DEPTH_ATTACHMENT_OPTIMAL) because object_scene_view uses
        // a combined depth/stencil format.
        er::ImageResourceInfo csm_as_depth_attachment = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_2_FLAG_BITS(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, DEPTH_STENCIL_ATTACHMENT_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage, EARLY_FRAGMENT_TESTS_BIT, LATE_FRAGMENT_TESTS_BIT) };
        // Use DEPTH_STENCIL_READ_ONLY_OPTIMAL (combined-aspect "both read-
        // only") instead of DEPTH_READ_ONLY_OPTIMAL.  The CSM depth array
        // is D24_UNORM_S8_UINT, and without the separateDepthStencilLayouts
        // device feature enabled, DEPTH_READ_ONLY_OPTIMAL applies only to
        // the depth aspect — validation tracks the stencil aspect's layout
        // independently and treats DEPTH_READ_ONLY as effectively
        // SHADER_READ_ONLY for stencil.  Next frame the begin-rendering
        // pass treats stencil as still in DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        // and our addImageBarrier's auto-aspect mask (DEPTH|STENCIL for
        // D24_S8) hits a mismatch.  DEPTH_STENCIL_READ_ONLY_OPTIMAL applies
        // cleanly to both aspects on a combined format, eliminating the
        // VUID-VkImageMemoryBarrier-oldLayout-01197 cascade.
        er::ImageResourceInfo csm_as_shader_sampler = {
            er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(PipelineStage, FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT) };

        // Transition ALL cascade layers from depth-attachment to shader-read.
        cmd_buf->addImageBarrier(
            shadow_object_scene_view_->getDepthBuffer()->image,
            csm_as_depth_attachment,
            csm_as_shader_sampler,
            0, 1, 0, CSM_CASCADE_COUNT);

        // ---- CSM debug visualisation ----------------------------------------
        // While the depth array is in DEPTH_READ_ONLY_OPTIMAL, blit each
        // cascade to its small R8G8B8A8 colour target so the ImGui window can
        // display them.
        if (menu_->showCsmDebug() && csm_debug_pipeline_) {
            er::ImageResourceInfo dbg_color_attach = {
                er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
                SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
            er::ImageResourceInfo dbg_shader_read = {
                er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                SET_FLAG_BIT(Access, SHADER_READ_BIT),
                SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

            for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
                // SHADER_READ_ONLY → COLOR_ATTACHMENT
                cmd_buf->addImageBarrier(
                    csm_debug_color_[k].image,
                    dbg_shader_read, dbg_color_attach,
                    0, 1, 0, 1);

                // Render cascade k depth → colour target.
                er::RenderingAttachmentInfo color_attach;
                color_attach.image_view   = csm_debug_color_[k].view;
                color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                color_attach.load_op      = er::AttachmentLoadOp::CLEAR;
                color_attach.store_op     = er::AttachmentStoreOp::STORE;
                color_attach.clear_value.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

                er::RenderingInfo ri = {};
                ri.render_area_offset = { 0, 0 };
                ri.render_area_extent = { kCsmDebugSize, kCsmDebugSize };
                ri.layer_count        = 1;
                ri.view_mask          = 0;
                ri.color_attachments  = { color_attach };
                ri.depth_attachments  = {};
                ri.stencil_attachments = {};
                cmd_buf->beginDynamicRendering(ri);

                er::Viewport vp;
                vp.x = 0; vp.y = 0;
                vp.width  = float(kCsmDebugSize);
                vp.height = float(kCsmDebugSize);
                vp.min_depth = 0.0f;
                vp.max_depth = 1.0f;
                er::Scissor sc;
                sc.offset = glm::ivec2(0);
                sc.extent = glm::uvec2(kCsmDebugSize);
                cmd_buf->setViewports({ vp });
                cmd_buf->setScissors({ sc });

                cmd_buf->bindPipeline(
                    er::PipelineBindPoint::GRAPHICS, csm_debug_pipeline_);

                int32_t cascade_idx = k;
                cmd_buf->pushConstants(
                    SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                    csm_debug_pipeline_layout_,
                    &cascade_idx,
                    sizeof(int32_t));
                cmd_buf->bindDescriptorSets(
                    er::PipelineBindPoint::GRAPHICS,
                    csm_debug_pipeline_layout_,
                    { csm_debug_desc_set_ });

                cmd_buf->draw(3); // fullscreen triangle (no vertex buffer needed)

                cmd_buf->endDynamicRendering();

                // COLOR_ATTACHMENT → SHADER_READ_ONLY (for ImGui)
                cmd_buf->addImageBarrier(
                    csm_debug_color_[k].image,
                    dbg_color_attach, dbg_shader_read,
                    0, 1, 0, 1);
            }
        }
        // ---- End CSM debug --------------------------------------------------

        // ── GPU cluster culling (Nanite-lite) ──
        // Originally this was the only cluster cull dispatch per frame.
        // After adding the dynamic cubemap probe pass we now run the
        // cull TWICE per frame:
        //   1. INSIDE DynamicCubemap::update — once per face capture
        //      with the FACE's view-proj, so each cube face sees its
        //      proper frustum (not just the main camera's).
        //   2. AFTER the probe pass — with the main camera's view-proj
        //      so the main forward pass below reads the right set.
        // The second of those is in a separate dispatch a few blocks
        // down (search "Cluster Cull (main, post-probe)").  This
        // first one is kept for the case where the probe path is
        // disabled — when ambient_probe_system_ isn't initialized,
        // we still need the main cull to populate the indirect
        // buffer for the forward pass.
        // Build the Hi-Z pyramid from LAST frame's depth (still sitting
        // in depth_buffer_copy_ from the previous frame's end-of-frame
        // blit) BEFORE the cluster cull samples it.  Doing this within
        // the same command buffer guarantees the build's writes happen
        // before the cull's reads via natural in-CB execution ordering,
        // which side-steps the cross-command-buffer race that was
        // producing 0% / 100% Hi-Z cull flickering when the build was
        // at end-of-frame.
        // Run the Hi-Z build whenever the cull-side toggle is on OR the
        // viewer is open, so a user can inspect the pyramid contents in
        // the Render Debug menu without having to also enable the
        // cull-side switch.
        const bool hiz_debug_open = menu_ && menu_->isHiZDebugOn();
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            (cluster_renderer_->getUseHiZOcclusionCull() ||
             hiz_debug_open)) {
            engine::helper::GpuProfiler::Scope _scope_hiz(
                gpu_profiler_, cmd_buf, "Hi-Z Build");
            dispatchHiZBuild(cmd_buf);
            // _scope_hiz auto-closes via RAII.
        }

        // ── Legacy main-camera cluster cull (forward path only) ──────
        // In deferred mode this dispatch is dead work — Phase B's cull
        // (run later inside the deferred block) re-emits the canonical
        // opaque + translucent indirect buffers from scratch, so any
        // counts/indirects we'd write here are about to be wiped.  Skip
        // the dispatch, but still call pollDebugReadback() so the Smart
        // Mesh "visible clusters / triangles" stats stay live — that
        // readback is a host-only map of LAST frame's already-complete
        // buffers and doesn't depend on the cull running this frame.
        //
        // Forward mode keeps the full dispatch: its cluster draw later
        // in this command buffer consumes indirect_draw_buffer_ and has
        // no Phase B to repopulate it.
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            !ambient_probe_system_) {
            if (deferred_rendering_enabled_) {
                cluster_renderer_->pollDebugReadback();
            } else {
                engine::helper::GpuProfiler::Scope _scope_cull(
                    gpu_profiler_, cmd_buf, "Cluster Cull");
                cluster_renderer_->cull(
                    cmd_buf,
                    main_camera_object_->getViewProjMatrix(),
                    main_camera_object_->getCameraPosition(),
                    last_view_proj_);
            }
        }

        // ── Ambient probe system ──────────────────────────────────────
        // Drives the dynamic cubemap to the active grid probe each frame
        // and bakes the resulting cubemap into 9 SH coefficients per
        // probe every 6 frames.  Runs AFTER cluster_renderer_->cull()
        // because cull's host-side counter zeroing (HOST_COHERENT) is
        // visible to the GPU immediately at submission time — placing
        // the probe draw before cull would make the indirect draw read
        // a count of 0 (cull's zeroed state) and render nothing.
        // See scene_rendering/ambient_probe_system.h.
        if (ambient_probe_system_ && dynamic_cubemap_) {
            engine::helper::GpuProfiler::Scope _scope_probes(
                gpu_profiler_, cmd_buf, "Ambient Probes");

            ambient_probe_system_->writeProjectDescriptorsForCube(
                device_,
                dynamic_cubemap_->getColorCubeView(),
                dynamic_cubemap_->getDepthCube().view);

            // Lock the dynamic cubemap origin to the main camera —
            // this is now the production architecture, not a debug
            // override.  The cubemap captures the camera's view; each
            // probe's SH integration uses parallax-aware sampling
            // (see sh_project.comp) to reconstruct what THAT PROBE
            // would see from its grid position, using camera depth as
            // the world-geometry approximation.
            ambient_probe_system_->setLockedOrigin(
                /*enable*/ true,
                main_camera_object_->getCameraPosition());

            er::DescriptorSetList probe_shared_sets(
                RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
            probe_shared_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
            probe_shared_sets[VIEW_PARAMS_SET]       = view_desc_set;
            probe_shared_sets[RUNTIME_LIGHTS_PARAMS_SET] =
                runtime_lights_desc_set_;
            ambient_probe_system_->update(
                device_,
                cmd_buf,
                dynamic_cubemap_,
                cluster_renderer_,
                skydome_,
                probe_shared_sets,
                main_camera_object_->getCameraPosition());

            if (menu_) {
                glm::vec3 face_pos[6];
                for (int f = 0; f < 6; ++f) {
                    face_pos[f] = dynamic_cubemap_->getFaceCapturePos(f);
                }
                menu_->setDynamicCubeFrameInfo(
                    dynamic_cubemap_->getCurrentReadIdx(),
                    static_cast<int>(dynamic_cubemap_->getCurrentFace()),
                    dynamic_cubemap_->getFrameIndex(),
                    face_pos);
            }

            // _scope_probes auto-closes via RAII.
        }

        // ── Re-cull cluster set for the MAIN camera ───────────────────
        // The probe pass internally re-culls cluster_renderer_'s
        // indirect_draw_buffer_ with each face's frustum (see
        // DynamicCubemap::update — the per-face cull picks up clusters
        // off the main camera's view that probes still need to see).
        // After it returns, the indirect buffer holds the LAST face's
        // cull results; we rebuild it for the main camera's frustum
        // here so the main render pass below draws the correct set.
        //
        // DEFERRED MODE: skip this dispatch entirely.  Phase B (run
        // inside the deferred block below) overwrites indirect_draw_
        // buffer_ + trans_indirect_draw_buffer_ from scratch using
        // freshly culled survivors against this frame's Hi-Z, so
        // anything we'd write here is dead work.  The probe path's
        // own per-face cull() calls already triggered the debug
        // readback prologue, so stats stay live without an extra call.
        //
        // FORWARD MODE: still essential — the cluster forward draw
        // immediately below reads indirect_draw_buffer_, which the
        // probe pass left holding the last cube face's set.
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            !deferred_rendering_enabled_) {
            engine::helper::GpuProfiler::Scope _scope_recull(
                gpu_profiler_, cmd_buf, "Cluster Cull (main, post-probe)");
            cluster_renderer_->cull(
                cmd_buf,
                main_camera_object_->getViewProjMatrix(),
                main_camera_object_->getCameraPosition(),
                last_view_proj_);
            // _scope_recull auto-closes via RAII.
        }

        // ── Per-mesh frustum culling ──────────────────────────────────
        // When cluster rendering is enabled, extract world-space frustum
        // planes and pass them to DrawableObject. drawMesh() will transform
        // each mesh's local-space bounding sphere by its model matrix and
        // cull against these planes — correct regardless of node transform.
        if (cluster_renderer_ && cluster_renderer_->isEnabled()) {
            const auto& vp = main_camera_object_->getViewProjMatrix();
            glm::vec4 fplanes[6];
            fplanes[0] = glm::vec4(vp[0][3]+vp[0][0], vp[1][3]+vp[1][0],
                                    vp[2][3]+vp[2][0], vp[3][3]+vp[3][0]);
            fplanes[1] = glm::vec4(vp[0][3]-vp[0][0], vp[1][3]-vp[1][0],
                                    vp[2][3]-vp[2][0], vp[3][3]-vp[3][0]);
            fplanes[2] = glm::vec4(vp[0][3]+vp[0][1], vp[1][3]+vp[1][1],
                                    vp[2][3]+vp[2][1], vp[3][3]+vp[3][1]);
            fplanes[3] = glm::vec4(vp[0][3]-vp[0][1], vp[1][3]-vp[1][1],
                                    vp[2][3]-vp[2][1], vp[3][3]-vp[3][1]);
            fplanes[4] = glm::vec4(vp[0][3]+vp[0][2], vp[1][3]+vp[1][2],
                                    vp[2][3]+vp[2][2], vp[3][3]+vp[3][2]);
            fplanes[5] = glm::vec4(vp[0][3]-vp[0][2], vp[1][3]-vp[1][2],
                                    vp[2][3]-vp[2][2], vp[3][3]-vp[3][2]);
            for (int i = 0; i < 6; ++i) {
                float len = glm::length(glm::vec3(fplanes[i]));
                if (len > 0.0001f) fplanes[i] /= len;
            }
            ego::DrawableObject::setFrustumCullPlanes(fplanes);
        }

        // Hoisted so the cluster / sky passes below can see it and
        // skip themselves -- collision debug is meant to be a clean
        // segmentation view, no cluster geometry layered on top.
        const bool show_collision_dbg =
            menu_ && menu_->isCollisionDebugOn() && !collision_world_.empty();

        {
            engine::helper::GpuProfiler::Scope _scope_forward(
                gpu_profiler_, cmd_buf, "Forward Pass");

            object_scene_view_->draw(
                cmd_buf,
                desc_sets,
                nullptr,
                s_dbuf_idx,
                delta_t,
                current_time);

            // _scope_forward auto-closes via RAII.
        }

        // Clear frustum cull state so depth-only / shadow passes are not culled.
        ego::DrawableObject::clearFrustumCull();

        // ── Bindless cluster draw (single drawIndexedIndirectCount) ──
        // Rendered in its own dynamic rendering pass, on top of the
        // forward pass output (LOAD, not CLEAR).
        // Skipped when cluster debug draw is active — the forward pass already
        // drew the cluster visualisation and the indirect draw would overwrite it.
        // Also skipped when collision-mesh debug is on -- the forward pass
        // drew per-mesh segmentation colors and we want them to remain the
        // only thing on screen.
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            !engine::helper::clusterRenderingEnabled()) {
            // CPU+GPU combined scope (records on CPU + executes on GPU).
            engine::helper::GpuProfiler::Scope _scope_cluster(
                gpu_profiler_, cmd_buf, "Cluster Bindless Draw");

            auto color_buf = object_scene_view_->getColorBuffer();
            auto depth_buf = object_scene_view_->getDepthBuffer();

            // ── Deferred path (Phase 1 — cluster bindless only) ──────────
            // Forward base above already cleared + lit terrain/grass/hair/
            // etc into color_buf and stamped depth_buf.  Here we route the
            // cluster opaque indirect draw through the 3-RT G-buffer
            // pipeline, then run deferred_resolve.comp to light those
            // pixels (only).  The resolve compute samples color_buf via
            // the storage-image binding and skips pixels where the cluster
            // pass didn't write (albedo_ao.a == 0), preserving the forward
            // base content.  Translucent clusters and glass still go
            // through the existing WBOIT path, called below.
            if (deferred_rendering_enabled_ &&
                deferred_resolve_pipeline_ &&
                deferred_resolve_desc_set_ &&
                gbuf_albedo_ao_.image) {
                // CPU+GPU combined scope.
                engine::helper::GpuProfiler::Scope _scope_def(
                    gpu_profiler_, cmd_buf, "Deferred (G-buf + Resolve)");

                // ── Two-pass occlusion culling (Nanite-style) ─────────
                // Phase A renders only the clusters that were visible
                // last frame (gated on the persistent visibility-bit
                // buffer) — they're the trusted depth pre-pass.  Hi-Z
                // is built from THAT depth, then Phase B tests every
                // cluster against the Hi-Z and renders any newly
                // disoccluded ones into the same G-buffer.  The union
                // of A's draws + B's draws forms next frame's "visible
                // last frame" set via Phase B's atomicOr writes.

                // Helper for G-buffer attachments parameterised on load_op.
                auto make_gbuf_attach = [](
                        const std::shared_ptr<er::ImageView>& v,
                        er::AttachmentLoadOp load_op) {
                    er::RenderingAttachmentInfo a;
                    a.image_view   = v;
                    a.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                    a.load_op      = load_op;
                    a.store_op     = er::AttachmentStoreOp::STORE;
                    a.clear_value.color = { {0.0f, 0.0f, 0.0f, 0.0f} };
                    return a;
                };

                // Same descriptor set list both phases use — the
                // G-buffer pipeline's layout is identical to the forward
                // bindless one (only the fragment SPV + RT formats differ).
                er::DescriptorSetList cluster_desc_sets(
                    RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                cluster_desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
                cluster_desc_sets[VIEW_PARAMS_SET] =
                    main_camera_object_->getViewCameraDescriptorSet();
                cluster_desc_sets[RUNTIME_LIGHTS_PARAMS_SET] =
                    runtime_lights_desc_set_;

                er::Viewport vp_g;
                vp_g.x = 0; vp_g.y = 0;
                vp_g.width  = float(screen_size.x);
                vp_g.height = float(screen_size.y);
                vp_g.min_depth = 0.0f;
                vp_g.max_depth = 1.0f;
                er::Scissor sc_g;
                sc_g.offset = glm::ivec2(0);
                sc_g.extent = screen_size;

                // ── Phase A: cull → render visible-last-frame set ──
                // Cull dispatch reads visibility_bit_buffer_ and emits
                // opaque draws to indirect_draw_buffer_phase_a_.  The
                // method internally barriers the compute writes so the
                // immediately-following indirect draw sees them.
                {
                    engine::helper::GpuProfiler::Scope _scope_cullA(
                        gpu_profiler_, cmd_buf, "Cluster Cull (Phase A)");
                    cluster_renderer_->cullPhaseA(
                        cmd_buf,
                        main_camera_object_->getViewProjMatrix(),
                        main_camera_object_->getCameraPosition());
                    // _scope_cullA auto-closes via RAII.
                }

                {
                    engine::helper::GpuProfiler::Scope _scope_drawA(
                        gpu_profiler_, cmd_buf, "Cluster Draw (Phase A)");
                    er::RenderingAttachmentInfo gbuf0 =
                        make_gbuf_attach(gbuf_albedo_ao_.view,
                                         er::AttachmentLoadOp::CLEAR);
                    er::RenderingAttachmentInfo gbuf1 =
                        make_gbuf_attach(gbuf_normal_rough_.view,
                                         er::AttachmentLoadOp::CLEAR);
                    er::RenderingAttachmentInfo gbuf2 =
                        make_gbuf_attach(gbuf_emissive_metal_.view,
                                         er::AttachmentLoadOp::CLEAR);
                    er::RenderingAttachmentInfo gbuf3 =
                        make_gbuf_attach(gbuf_velocity_.view,
                                         er::AttachmentLoadOp::CLEAR);

                    er::RenderingAttachmentInfo gbuf_depth;
                    gbuf_depth.image_view   = depth_buf->view;
                    gbuf_depth.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    gbuf_depth.load_op      = er::AttachmentLoadOp::LOAD;
                    gbuf_depth.store_op     = er::AttachmentStoreOp::STORE;

                    er::RenderingInfo gri = {};
                    gri.render_area_offset = { 0, 0 };
                    gri.render_area_extent = screen_size;
                    gri.layer_count        = 1;
                    gri.view_mask          = 0;
                    gri.color_attachments  = { gbuf0, gbuf1, gbuf2, gbuf3 };
                    gri.depth_attachments  = { gbuf_depth };
                    cmd_buf->beginDynamicRendering(gri);
                    cluster_renderer_->drawOpaqueGBufferPhaseA(
                        cmd_buf, cluster_desc_sets, { vp_g }, { sc_g });
                    cmd_buf->endDynamicRendering();
                    // _scope_drawA auto-closes via RAII.
                }

                // ── Hi-Z build sandwiched between the two render passes ──
                // Depth must briefly leave attachment layout for the build
                // compute to sample.  Phase B's render needs it back as
                // a depth attachment immediately after, so we transition
                // it twice — cheap relative to the cull / render cost.
                {
                    er::ImageResourceInfo depth_attach_to_read = {
                        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        SET_FLAG_BIT(Access, SHADER_READ_BIT),
                        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
                    er::ImageResourceInfo from_depth_att = {
                        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                        SET_FLAG_BIT(PipelineStage, LATE_FRAGMENT_TESTS_BIT) };
                    cmd_buf->addImageBarrier(depth_buf->image,
                        from_depth_att, depth_attach_to_read, 0, 1, 0, 1);

                    {
                        engine::helper::GpuProfiler::Scope _scope_hiz(
                            gpu_profiler_, cmd_buf, "Hi-Z Build");
                        buildHiZPyramid(cmd_buf, screen_size);
                        // _scope_hiz auto-closes via RAII.
                    }

                    // Restore depth as an attachment for Phase B render.
                    er::ImageResourceInfo read_to_depth_att = {
                        er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        SET_2_FLAG_BITS(Access,
                            DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                            DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                        SET_2_FLAG_BITS(PipelineStage,
                            EARLY_FRAGMENT_TESTS_BIT,
                            LATE_FRAGMENT_TESTS_BIT) };
                    cmd_buf->addImageBarrier(depth_buf->image,
                        depth_attach_to_read, read_to_depth_att, 0, 1, 0, 1);
                }

                // Clear last frame's visibility bits.  Phase A already
                // consumed them; Phase B is about to atomicOr in this
                // frame's true visible set.
                {
                    engine::helper::GpuProfiler::Scope _scope_visclear(
                        gpu_profiler_, cmd_buf, "Visibility Bits Clear");
                    cluster_renderer_->clearVisibilityBuffer(cmd_buf);
                    // _scope_visclear auto-closes via RAII.
                }

                // ── Phase B: cull (with Hi-Z) → render newly-disoccluded ──
                {
                    engine::helper::GpuProfiler::Scope _scope_cullB(
                        gpu_profiler_, cmd_buf, "Cluster Cull (Phase B)");
                    cluster_renderer_->cullPhaseB(
                        cmd_buf,
                        main_camera_object_->getViewProjMatrix(),
                        main_camera_object_->getCameraPosition());
                    // _scope_cullB auto-closes via RAII.
                }

                {
                    engine::helper::GpuProfiler::Scope _scope_drawB(
                        gpu_profiler_, cmd_buf, "Cluster Draw (Phase B)");
                    // LOAD all four colour RTs so Phase A's contributions
                    // are preserved; Phase B writes additional pixels on
                    // top wherever its clusters survive the Hi-Z test.
                    er::RenderingAttachmentInfo gbuf0 =
                        make_gbuf_attach(gbuf_albedo_ao_.view,
                                         er::AttachmentLoadOp::LOAD);
                    er::RenderingAttachmentInfo gbuf1 =
                        make_gbuf_attach(gbuf_normal_rough_.view,
                                         er::AttachmentLoadOp::LOAD);
                    er::RenderingAttachmentInfo gbuf2 =
                        make_gbuf_attach(gbuf_emissive_metal_.view,
                                         er::AttachmentLoadOp::LOAD);
                    er::RenderingAttachmentInfo gbuf3 =
                        make_gbuf_attach(gbuf_velocity_.view,
                                         er::AttachmentLoadOp::LOAD);

                    er::RenderingAttachmentInfo gbuf_depth;
                    gbuf_depth.image_view   = depth_buf->view;
                    gbuf_depth.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    gbuf_depth.load_op      = er::AttachmentLoadOp::LOAD;
                    gbuf_depth.store_op     = er::AttachmentStoreOp::STORE;

                    er::RenderingInfo gri = {};
                    gri.render_area_offset = { 0, 0 };
                    gri.render_area_extent = screen_size;
                    gri.layer_count        = 1;
                    gri.view_mask          = 0;
                    gri.color_attachments  = { gbuf0, gbuf1, gbuf2, gbuf3 };
                    gri.depth_attachments  = { gbuf_depth };
                    cmd_buf->beginDynamicRendering(gri);
                    cluster_renderer_->drawOpaqueGBufferPhaseB(
                        cmd_buf, cluster_desc_sets, { vp_g }, { sc_g });
                    cmd_buf->endDynamicRendering();
                    // _scope_drawB auto-closes via RAII.
                }

                // 2. Barriers for the resolve dispatch.
                //    G-buffer + depth: COLOR/DEPTH attachment → SHADER_READ
                //    color_buf:        COLOR_ATTACHMENT → GENERAL (storage write)
                er::ImageResourceInfo from_color_att = {
                    er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
                    SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
                er::ImageResourceInfo to_shader_read = {
                    er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    SET_FLAG_BIT(Access, SHADER_READ_BIT),
                    SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
                cmd_buf->addImageBarrier(gbuf_albedo_ao_.image,
                    from_color_att, to_shader_read, 0, 1, 0, 1);
                cmd_buf->addImageBarrier(gbuf_normal_rough_.image,
                    from_color_att, to_shader_read, 0, 1, 0, 1);
                cmd_buf->addImageBarrier(gbuf_emissive_metal_.image,
                    from_color_att, to_shader_read, 0, 1, 0, 1);
                // Velocity isn't read by the resolve compute, but transition
                // it now alongside the others so it ends the frame in a
                // sample-friendly layout for any TAA/motion-blur pass that
                // the application may add in the future.  The post-resolve
                // restore below puts it back to COLOR_ATTACHMENT_OPTIMAL
                // for next frame's clear.
                cmd_buf->addImageBarrier(gbuf_velocity_.image,
                    from_color_att, to_shader_read, 0, 1, 0, 1);

                er::ImageResourceInfo from_depth_att = {
                    er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                    SET_FLAG_BIT(PipelineStage, LATE_FRAGMENT_TESTS_BIT) };
                cmd_buf->addImageBarrier(depth_buf->image,
                    from_depth_att, to_shader_read, 0, 1, 0, 1);

                // ── Hi-Z pyramid build ──────────────────────────────────
                // (Hi-Z was already built once between Phase A and Phase B
                // above — that's the pyramid the resolve's debug mode
                // samples.  Building it a second time after Phase B would
                // give a slightly more complete pyramid but cost another
                // ~0.2 ms; not worth it for the debug viz, and the
                // cull-side consumer is Phase B which already ran.)

                er::ImageResourceInfo color_att_to_general = {
                    er::ImageLayout::GENERAL,
                    SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
                    SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };
                cmd_buf->addImageBarrier(color_buf->image,
                    from_color_att, color_att_to_general, 0, 1, 0, 1);

                // 3. Bind compute resolve and dispatch.  Set indices match
                //    the forward path — see initDeferredResolve.
                {
                    engine::helper::GpuProfiler::Scope _scope_resolve(
                        gpu_profiler_, cmd_buf, "Deferred Resolve Compute");
                    er::DescriptorSetList resolve_sets(
                        RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                    resolve_sets[PBR_GLOBAL_PARAMS_SET]     = pbr_lighting_desc_set_;
                    resolve_sets[VIEW_PARAMS_SET]           =
                        main_camera_object_->getViewCameraDescriptorSet();
                    resolve_sets[PBR_MATERIAL_PARAMS_SET]   = deferred_resolve_desc_set_;
                    resolve_sets[RUNTIME_LIGHTS_PARAMS_SET] = runtime_lights_desc_set_;
                    cmd_buf->bindPipeline(
                        er::PipelineBindPoint::COMPUTE, deferred_resolve_pipeline_);
                    cmd_buf->bindDescriptorSets(
                        er::PipelineBindPoint::COMPUTE,
                        deferred_resolve_pipeline_layout_,
                        resolve_sets);
                    const uint32_t kGroupSize = 8;
                    uint32_t gx = (screen_size.x + kGroupSize - 1) / kGroupSize;
                    uint32_t gy = (screen_size.y + kGroupSize - 1) / kGroupSize;
                    cmd_buf->dispatch(gx, gy, 1);
                    // _scope_resolve auto-closes via RAII.
                }

                // 4. Barriers back to attachment-friendly layouts so the
                //    upcoming translucent / sky / debug passes can attach
                //    color and depth.  G-buffer images ride next-frame in
                //    SHADER_READ_ONLY → COLOR_ATTACHMENT (re-cleared).
                er::ImageResourceInfo general_to_color_att = {
                    er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
                    SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
                cmd_buf->addImageBarrier(color_buf->image,
                    color_att_to_general, general_to_color_att, 0, 1, 0, 1);

                er::ImageResourceInfo shader_read_to_depth_att = {
                    er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    SET_2_FLAG_BITS(Access,
                        DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                        DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                    SET_2_FLAG_BITS(PipelineStage,
                        EARLY_FRAGMENT_TESTS_BIT,
                        LATE_FRAGMENT_TESTS_BIT) };
                cmd_buf->addImageBarrier(depth_buf->image,
                    to_shader_read, shader_read_to_depth_att, 0, 1, 0, 1);

                er::ImageResourceInfo shader_read_to_color_att = {
                    er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
                    SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };
                cmd_buf->addImageBarrier(gbuf_albedo_ao_.image,
                    to_shader_read, shader_read_to_color_att, 0, 1, 0, 1);
                cmd_buf->addImageBarrier(gbuf_normal_rough_.image,
                    to_shader_read, shader_read_to_color_att, 0, 1, 0, 1);
                cmd_buf->addImageBarrier(gbuf_emissive_metal_.image,
                    to_shader_read, shader_read_to_color_att, 0, 1, 0, 1);
                cmd_buf->addImageBarrier(gbuf_velocity_.image,
                    to_shader_read, shader_read_to_color_att, 0, 1, 0, 1);

                // _scope_def auto-closes via RAII at end of this if-block.
            }

            er::RenderingAttachmentInfo color_attach;
            color_attach.image_view   = color_buf->view;
            color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            color_attach.load_op      = er::AttachmentLoadOp::LOAD;
            color_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingAttachmentInfo depth_attach;
            depth_attach.image_view   = depth_buf->view;
            depth_attach.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attach.load_op      = er::AttachmentLoadOp::LOAD;
            depth_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingInfo ri = {};
            ri.render_area_offset  = { 0, 0 };
            ri.render_area_extent  = screen_size;
            ri.layer_count         = 1;
            ri.view_mask           = 0;
            ri.color_attachments   = { color_attach };
            ri.depth_attachments   = { depth_attach };
            ri.stencil_attachments = {};

            cmd_buf->beginDynamicRendering(ri);

            // Build descriptor set list matching the bindless pipeline layout:
            //   set 0 = PBR global (IBL + shadow sampler)
            //   set 1 = view camera (VP matrices)
            //   set 2 = cluster bindless (injected by draw())
            //   set 3 = nullptr (SKIN — unused by cluster pass)
            //   set 4 = runtime lights (sun direction/color + CSM data)
            er::DescriptorSetList cluster_desc_sets(
                RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
            cluster_desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
            cluster_desc_sets[VIEW_PARAMS_SET] =
                main_camera_object_->getViewCameraDescriptorSet();
            cluster_desc_sets[RUNTIME_LIGHTS_PARAMS_SET] =
                runtime_lights_desc_set_;

            // Viewport and scissor for the full screen.
            er::Viewport vp;
            vp.x = 0; vp.y = 0;
            vp.width  = static_cast<float>(screen_size.x);
            vp.height = static_cast<float>(screen_size.y);
            vp.min_depth = 0.0f;
            vp.max_depth = 1.0f;
            er::Scissor sc;
            sc.offset = glm::ivec2(0);
            sc.extent = screen_size;

            // Opaque cluster draw only.  The translucent (alpha-blended
            // glass) draw is issued LATER, AFTER the sky envmap pass —
            // see the "Glass / Translucent Forward" block below.
            //
            // Reason for the order: glass uses depth_write OFF on its
            // pipeline (so multi-layer glass can blend without depth-
            // rejecting itself), which means pixels with NO opaque
            // behind the glass keep the cleared depth (1.0).  If we
            // drew glass before the sky envmap pass — which uses
            // LESS_OR_EQUAL vs depth=1.0 — sky would then paint over
            // those glass pixels, "deleting" any window panes whose
            // backdrop is sky.  Drawing glass AFTER sky alpha-blends
            // the glass colour over either opaque or sky content
            // correctly.
            //
            // Deferred mode: the deferred G-buffer + resolve already
            // ran above and wrote opaque content into color_buf.  No
            // additional opaque cluster work needed here — the block
            // is intentionally empty for the deferred path.
            //
            // Forward mode: cluster_renderer_->draw() does opaque
            // (alpha-mask included) into the currently-open color/depth
            // pass.  It used to also do translucent in the same call;
            // that part was split out so both rendering modes follow
            // the same opaque-then-sky-then-glass order.
            if (deferred_rendering_enabled_ &&
                deferred_resolve_pipeline_ &&
                gbuf_albedo_ao_.image) {
                // Deferred opaque already handled by the G-buf + resolve
                // path above.  Nothing to draw here for the opaque pass.
            } else {
                cluster_renderer_->draw(
                    cmd_buf,
                    cluster_desc_sets,
                    { vp },
                    { sc },
                    color_buf->view,
                    depth_buf->view,
                    screen_size);
            }

            // Probe debug-draw — small icospheres at every probe
            // position, colored by SH-evaluated irradiance.  Toggled
            // via the Render Debug menu.  Drawn inside the cluster
            // pass so it shares depth state with the main scene.
            if (ambient_probe_system_ && menu_ &&
                menu_->showProbeDebug()) {
                engine::helper::GpuProfiler::Scope _scope_probe_dbg(
                    gpu_profiler_, cmd_buf, "Probe Debug Draw");
                ambient_probe_system_->drawDebug(
                    cmd_buf,
                    cluster_desc_sets,
                    { vp },
                    { sc });
                // _scope_probe_dbg auto-closes via RAII.
            }

            cmd_buf->endDynamicRendering();

            // _scope_cluster auto-closes via RAII at end of this if-block.
        }

        // ── VT feedback compaction (GPU dedupe → small host-readable list) ─────
        // After the cluster bindless draw has written its tile-key feedback,
        // dispatch a tiny compute pass that scans the 1 MB feedback buffer on
        // the GPU and writes a compact (typically few-KB) list to the per-FIF
        // host-visible slot the next-next frame's tick() reads.  This replaces
        // the previous CPU-side mapMemory+linear scan over 262144 entries
        // (which was costing ~10 ms per frame on its own) with a pointer
        // dereference at frame start.  The dispatch also clears
        // feedback_buffer_ for next frame.
        //
        // Always run when vt_manager_ exists (even when the user has VT
        // toggled off) — the dispatch is microseconds and keeps the
        // compact buffer + feedback buffer in a clean state for instant
        // re-enable.
        if (vt_manager_) {
            engine::helper::GpuProfiler::Scope _scope_vt_compact(
                gpu_profiler_, cmd_buf, "VT Feedback Compact");
            // Same counter as the start-of-frame tick() — produces the
            // slot tick() will read on the next-next frame.  Counter
            // advances ONCE per drawFrame, here at the end (after
            // both calls have used the same value) so the increment
            // doesn't desync the two halves.
            vt_manager_->compactFeedback(cmd_buf, vt_frame_index_);
            ++vt_frame_index_;
        }

        // ── Sky envmap background pass ─────────────────────────────────────────
        // Draws the sky envmap into all pixels that were NOT covered by geometry
        // (i.e. depth buffer still holds the cleared value 1.0 = far plane).
        // Uses LESS_OR_EQUAL depth test + no depth write so it is a cheap
        // background fill that automatically respects the forward pass results.
        if (skydome_) {
            engine::helper::GpuProfiler::Scope _scope_sky(
                gpu_profiler_, cmd_buf, "Sky Envmap");

            auto color_buf = object_scene_view_->getColorBuffer();
            auto depth_buf = object_scene_view_->getDepthBuffer();

            er::RenderingAttachmentInfo sky_color_attach;
            sky_color_attach.image_view   = color_buf->view;
            sky_color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            sky_color_attach.load_op      = er::AttachmentLoadOp::LOAD;
            sky_color_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingAttachmentInfo sky_depth_attach;
            sky_depth_attach.image_view   = depth_buf->view;
            sky_depth_attach.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            sky_depth_attach.load_op      = er::AttachmentLoadOp::LOAD;
            sky_depth_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingInfo sky_ri = {};
            sky_ri.render_area_offset  = { 0, 0 };
            sky_ri.render_area_extent  = screen_size;
            sky_ri.layer_count         = 1;
            sky_ri.view_mask           = 0;
            sky_ri.color_attachments   = { sky_color_attach };
            sky_ri.depth_attachments   = { sky_depth_attach };
            sky_ri.stencil_attachments = {};

            cmd_buf->beginDynamicRendering(sky_ri);

            er::Viewport sky_vp;
            sky_vp.x = 0; sky_vp.y = 0;
            sky_vp.width      = float(screen_size.x);
            sky_vp.height     = float(screen_size.y);
            sky_vp.min_depth  = 0.0f;
            sky_vp.max_depth  = 1.0f;
            er::Scissor sky_sc;
            sky_sc.offset = glm::ivec2(0);
            sky_sc.extent = screen_size;
            cmd_buf->setViewports({ sky_vp });
            cmd_buf->setScissors({ sky_sc });

            skydome_->drawEnvmap(
                cmd_buf,
                main_camera_object_->getCameraViewInfo().inv_view_proj_relative);

            cmd_buf->endDynamicRendering();

            // _scope_sky auto-closes via RAII.
        }

        // ── Glass / Translucent draw — alpha-blend OR WBOIT ────────────────────
        // Runs AFTER the sky envmap pass so glass renders correctly over
        // either opaque geometry or sky background.  Two paths, picked at
        // dispatch time by reading cluster_renderer_->getTranslucentMode():
        //
        //   ALPHA_BLEND (default):  open a colour/depth render pass with
        //     LOAD/LOAD; call drawTranslucentForward (which writes into
        //     the open pass with hardware src_alpha / one_minus_src_alpha
        //     blending); close the pass.
        //
        //   WBOIT:  NO outer render pass — drawTranslucentOit manages every
        //     pass it needs internally (the WBOIT accum/reveal draw and
        //     the fullscreen composite resolve are on different render
        //     targets, so the API can't share a single outer pass).
        //
        // The split entry-point design (rather than dispatching inside a
        // single draw function) is deliberate — see the comment block
        // at the top of drawTranslucentOit for the rationale.
        if (cluster_renderer_ && cluster_renderer_->isEnabled() &&
            !engine::helper::clusterRenderingEnabled()) {
            engine::helper::GpuProfiler::Scope _scope_glass(
                gpu_profiler_, cmd_buf, "Glass / Translucent");

            auto color_buf = object_scene_view_->getColorBuffer();
            auto depth_buf = object_scene_view_->getDepthBuffer();

            // Shared descriptor-set list: glass uses the same bindless
            // pipeline layout as the opaque cluster pass, so set 0
            // (PBR / IBL), set 1 (camera VP), set 4 (runtime lights) are
            // the same.  set 2 (PBR_MATERIAL_PARAMS_SET) is injected by
            // the cluster_renderer's draw functions.
            er::DescriptorSetList glass_desc_sets(
                RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
            glass_desc_sets[PBR_GLOBAL_PARAMS_SET] = pbr_lighting_desc_set_;
            glass_desc_sets[VIEW_PARAMS_SET] =
                main_camera_object_->getViewCameraDescriptorSet();
            glass_desc_sets[RUNTIME_LIGHTS_PARAMS_SET] =
                runtime_lights_desc_set_;

            er::Viewport glass_vp;
            glass_vp.x = 0; glass_vp.y = 0;
            glass_vp.width  = static_cast<float>(screen_size.x);
            glass_vp.height = static_cast<float>(screen_size.y);
            glass_vp.min_depth = 0.0f;
            glass_vp.max_depth = 1.0f;
            er::Scissor glass_sc;
            glass_sc.offset = glm::ivec2(0);
            glass_sc.extent = screen_size;

            using TMode = engine::scene_rendering::
                ClusterRenderer::TranslucentMode;
            const TMode mode = cluster_renderer_->getTranslucentMode();

            if (mode == TMode::ALPHA_BLEND) {
                // Open a single colour/depth pass and let the alpha-blend
                // path draw straight into it.
                er::RenderingAttachmentInfo glass_color_attach;
                glass_color_attach.image_view   = color_buf->view;
                glass_color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                glass_color_attach.load_op      = er::AttachmentLoadOp::LOAD;
                glass_color_attach.store_op     = er::AttachmentStoreOp::STORE;

                er::RenderingAttachmentInfo glass_depth_attach;
                glass_depth_attach.image_view   = depth_buf->view;
                glass_depth_attach.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                glass_depth_attach.load_op      = er::AttachmentLoadOp::LOAD;
                glass_depth_attach.store_op     = er::AttachmentStoreOp::STORE;

                er::RenderingInfo glass_ri = {};
                glass_ri.render_area_offset  = { 0, 0 };
                glass_ri.render_area_extent  = screen_size;
                glass_ri.layer_count         = 1;
                glass_ri.view_mask           = 0;
                glass_ri.color_attachments   = { glass_color_attach };
                glass_ri.depth_attachments   = { glass_depth_attach };
                glass_ri.stencil_attachments = {};
                cmd_buf->beginDynamicRendering(glass_ri);

                cluster_renderer_->drawTranslucentForward(
                    cmd_buf,
                    glass_desc_sets,
                    { glass_vp },
                    { glass_sc },
                    color_buf->view,
                    depth_buf->view,
                    screen_size);

                cmd_buf->endDynamicRendering();
            } else {
                // WBOIT path: caller MUST NOT have a pass open — the
                // function opens its own accum/reveal pass, draws, then
                // opens a composite pass and resolves onto color_buf.
                cluster_renderer_->drawTranslucentOit(
                    cmd_buf,
                    glass_desc_sets,
                    { glass_vp },
                    { glass_sc },
                    color_buf->view,
                    depth_buf->view,
                    screen_size);
            }
            // _scope_glass auto-closes via RAII.
        }

        // ── Collision LOD overlay (debug) ────────────
        // Draw the simplified collision meshes translucently (per-
        // category colour, ~80% opacity) on TOP of the fully-rendered,
        // textured scene so the LOD proxy can be compared against the
        // original geometry.  LOAD the scene colour + depth so the
        // overlay blends over the scene and is occluded by it; the
        // collision solid pipeline is alpha-blended + biased toward the
        // camera so it wins over the coincident original surface it
        // approximates.  Replaces the old clear-to-blue segmentation
        // view (same isolate-debug controls, now drawn over the scene).
        if (show_collision_dbg) {
            engine::helper::GpuProfiler::Scope _scope_coll_overlay(
                gpu_profiler_, cmd_buf, "Collision LOD overlay");

            auto color_buf = object_scene_view_->getColorBuffer();
            auto depth_buf = object_scene_view_->getDepthBuffer();

            er::RenderingAttachmentInfo color_attach;
            color_attach.image_view   = color_buf->view;
            color_attach.image_layout = er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            color_attach.load_op      = er::AttachmentLoadOp::LOAD;
            color_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingAttachmentInfo depth_attach;
            depth_attach.image_view   = depth_buf->view;
            depth_attach.image_layout = er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attach.load_op      = er::AttachmentLoadOp::LOAD;
            depth_attach.store_op     = er::AttachmentStoreOp::STORE;

            er::RenderingInfo ri = {};
            ri.render_area_offset  = { 0, 0 };
            ri.render_area_extent  = screen_size;
            ri.layer_count         = 1;
            ri.view_mask           = 0;
            ri.color_attachments   = { color_attach };
            ri.depth_attachments   = { depth_attach };
            ri.stencil_attachments = {};
            cmd_buf->beginDynamicRendering(ri);

            er::Viewport vp;
            vp.x = 0.0f; vp.y = 0.0f;
            vp.width  = static_cast<float>(screen_size.x);
            vp.height = static_cast<float>(screen_size.y);
            vp.min_depth = 0.0f; vp.max_depth = 1.0f;
            er::Scissor sc;
            sc.offset = glm::ivec2(0);
            sc.extent = screen_size;
            std::vector<er::Viewport> viewports = { vp };
            std::vector<er::Scissor>  scissors  = { sc };

            // Collision pipeline layout = exactly two descriptor sets:
            // set 0 PBR-global (part of the layout, not sampled by the
            // collision shaders) and set 1 the camera view-proj SSBO
            // consumed by collision_debug.vert.
            er::DescriptorSetList collision_desc_sets = {
                pbr_lighting_desc_set_,
                main_camera_object_->getViewCameraDescriptorSet(),
            };

                int collision_isolate = -1;
                if (menu_) {
                    menu_->setCollisionMeshCount(collision_world_.meshCount());
                    if (menu_->collisionIsolateEnabled()) {
                        collision_isolate = menu_->collisionIsolateIndex();
                        const auto* cm = collision_world_.meshAt(
                            static_cast<size_t>(collision_isolate));
                        if (cm) {
                            // Simplified (current proxy) vs original source
                            // primitive polygon counts, plus the kept ratio.
                            const size_t simp_tris = cm->triangleCount();
                            const size_t orig_tris =
                                cm->originalTriangleCount();
                            const int kept_pct = orig_tris > 0
                                ? static_cast<int>(
                                      (static_cast<uint64_t>(simp_tris) * 100 +
                                       orig_tris / 2) / orig_tris)
                                : 0;
                            std::string info =
                                "index " +
                                std::to_string(collision_isolate) + " / " +
                                std::to_string(collision_world_.meshCount()) +
                                "\ncategory = " +
                                engine::helper::meshCategoryTag(cm->category()) +
                                "\ntris = " +
                                std::to_string(simp_tris) +
                                " / " + std::to_string(orig_tris) +
                                " orig (" + std::to_string(kept_pct) + "%)" +
                                "\nmaterial = '" + cm->materialName() + "'" +
                                "\nobject = '" + cm->objectName() + "'";
                            menu_->setCollisionIsolateInfo(info);
                            // On index change: log the identity AND dump the
                            // mesh's exact triangles to an OBJ in the
                            // workspace folder so the geometry can be
                            // inspected off-line (interior holes /
                            // disconnected pieces a screenshot can't show).
                            // The path is hard-coded to the project root for
                            // debugging; remove this dump when done.
                            static int s_last_iso = -1;
                            if (collision_isolate != s_last_iso) {
                                s_last_iso = collision_isolate;
                                std::cout << "[collision.isolate] " << info
                                          << std::endl;
                                std::ofstream obj(
                                    "G:/work/procedure-world-sim/"
                                    "debug_isolated_mesh.obj");
                                if (obj.is_open()) {
                                    obj << "# collision mesh index "
                                        << collision_isolate << "  cat="
                                        << engine::helper::meshCategoryTag(
                                               cm->category())
                                        << "  tris=" << cm->triangleCount()
                                        << "  mat=" << cm->materialName()
                                        << "  obj=" << cm->objectName()
                                        << "\n";
                                    const auto& dv = cm->debugVertices();
                                    const auto& di = cm->debugIndices();
                                    for (const auto& v : dv)
                                        obj << "v " << v.x << " " << v.y
                                            << " " << v.z << "\n";
                                    for (size_t t = 0; t + 2 < di.size();
                                         t += 3)
                                        obj << "f " << (di[t] + 1) << " "
                                            << (di[t + 1] + 1) << " "
                                            << (di[t + 2] + 1) << "\n";
                                    std::cout << "[collision.isolate] dumped "
                                                 "OBJ ("
                                              << dv.size() << " verts, "
                                              << di.size() / 3 << " tris) -> "
                                                 "debug_isolated_mesh.obj"
                                              << std::endl;
                                }
                            }
                        } else {
                            menu_->setCollisionIsolateInfo(
                                "index out of range");
                        }
                    }
                }

                collision_world_.drawDebug(
                    device_, cmd_buf, collision_desc_sets, viewports,
                    scissors, collision_isolate);
            cmd_buf->endDynamicRendering();
        }

        // Transition ALL cascade layers back to depth-attachment for next frame.
        cmd_buf->addImageBarrier(
            shadow_object_scene_view_->getDepthBuffer()->image,
            csm_as_shader_sampler,
            csm_as_depth_attachment,
            0, 1, 0, CSM_CASCADE_COUNT);

#if 0
        cmd_buf->beginRenderPass(
            hdr_render_pass_,
            hdr_frame_buffer_,
            screen_size,
            clear_values_);

        if (s_render_prt_test) {
            conemap_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                conemap_obj_);
        }
        else if (s_render_hair_test) {
            hair_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                unit_box_);
        }
        else if (s_render_lbm_test) {
            lbm_test_->draw(
                device_,
                cmd_buf,
                desc_sets,
                unit_plane_,
                unit_box_);
        }
        else if (s_bistro_scene_test) {
            if (bistro_exterior_scene_) {
                bistro_exterior_scene_->draw(cmd_buf, desc_sets);
            }
        }
        else {
            // render drawable.
            {
                for (auto& drawable_obj : drawable_objects_) {
                    drawable_obj->draw(cmd_buf, desc_sets);
                }
            }

            if (player_object_) {
                player_object_->draw(cmd_buf, desc_sets);
            }

            // Foot debug markers — drawn through the regular forward
            // path with setDebugForceRed so the tint pops against any
            // backdrop.  Depth-tested like everything else so a marker
            // hidden behind geometry will be occluded; if you want
            // always-on-top, switch to an ImGui-overlay path instead.
            if (foot_marker_left_  && foot_marker_left_->isReady())  foot_marker_left_->draw(cmd_buf, desc_sets);
            if (foot_marker_right_ && foot_marker_right_->isReady()) foot_marker_right_->draw(cmd_buf, desc_sets);

            // render debug draw.
            if (menu_->getDebugDrawType() != NO_DEBUG_DRAW) {
                ego::DebugDrawObject::draw(
                    cmd_buf,
                    desc_sets,
                    focus_scene_view->getCameraPosition(),
                    menu_->getDebugDrawType());
            }

            // render skybox.
            {
                skydome_->draw(cmd_buf, view_desc_set);
            }
        }

        cmd_buf->endRenderPass();
#endif
        er::ImageResourceInfo depth_src_info = {
            er::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            SET_FLAG_BIT(Access, DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, EARLY_FRAGMENT_TESTS_BIT) };

        er::ImageResourceInfo depth_dst_info = {
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT) };

        // Source the blit from object_scene_view_'s depth buffer when it
        // exists — that's the depth target the cluster pipeline + sky
        // envmap pass actually write into.  The application-level
        // depth_buffer_ is a leftover used only by some legacy render
        // passes and is empty when the cluster/object_scene_view_ path
        // runs, so blitting from it would feed SSAO (and the volume
        // cloud, which also samples depth_buffer_copy_) garbage depth.
        // Symptom of the old wiring: SSAO depth ≈ 0 everywhere → every
        // pixel hits the "sky" early-out at ssao_compute.comp:70 and
        // returns ao = 1.0, producing the all-red diagnostic output.
        auto depth_blit_src =
            (object_scene_view_ && object_scene_view_->getDepthBuffer())
                ? object_scene_view_->getDepthBuffer()->image
                : depth_buffer_.image;
        auto depth_blit_src_size =
            (object_scene_view_ && object_scene_view_->getDepthBuffer())
                ? object_scene_view_->getDepthBuffer()->size
                : depth_buffer_.size;
        er::Helper::blitImage(
            cmd_buf,
            depth_blit_src,
            depth_buffer_copy_.image,
            depth_src_info,
            depth_src_info,
            depth_dst_info,
            depth_dst_info,
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            SET_FLAG_BIT(ImageAspect, DEPTH_BIT),
            depth_blit_src_size,
            depth_buffer_copy_.size);

        // (Hi-Z build was here; moved earlier in the frame so it runs
        // BEFORE the cluster cull's pyramid sample, eliminating the
        // cross-command-buffer race that was producing 0%/100% flickery
        // cull counts.  Sources from depth_buffer_copy_ which holds
        // last-frame's depth at that point in the frame.)

        // ── SSAO: generate, blur, apply to HDR color ──
        // Run when SSAO is enabled OR the user has selected the SSAO
        // render-debug mode — that mode wants the apply step to fire
        // so the screen gets overwritten with the raw AO factor, even
        // if the SSAO toggle in the menu is off.  See SSAO::render's
        // force_run handling.
        const int dbg_mode =
            menu_ ? menu_->getDebugRenderMode() : 0;
        // Only run the SSAO pipeline for the FINAL (=0) and SSAO (=11)
        // debug modes.  Every other diagnostic (Hi-Z, velocity, albedo,
        // normals, etc.) writes its visualisation directly into the
        // colour buffer and should not be modulated by AO — running the
        // gen/blur/apply chain is also wasted GPU work in those modes.
        const bool ssao_mode_allowed =
            dbg_mode == DEBUG_RENDER_MODE_FINAL ||
            dbg_mode == DEBUG_RENDER_MODE_SSAO;
        if (ssao_ && ssao_mode_allowed &&
            (ssao_->enabled || dbg_mode == DEBUG_RENDER_MODE_SSAO)) {
            engine::helper::GpuProfiler::Scope _scope_ssao(
                gpu_profiler_, cmd_buf, "SSAO");
            // Pass the SAME image that's bound into the apply descriptor
            // (see SSAO creation site).  Using hdr_color_buffer_ here
            // worked when SSAO was bound to that view, but now its apply
            // pass writes to object_scene_view_'s colour buffer, so the
            // barrier transition has to target that image.
            auto ssao_target_image =
                (object_scene_view_ && object_scene_view_->getColorBuffer())
                    ? object_scene_view_->getColorBuffer()->image
                    : hdr_color_buffer_.image;
            ssao_->render(cmd_buf, view_desc_set,
                          ssao_target_image, screen_size,
                          dbg_mode);
            // _scope_ssao auto-closes via RAII.
        }

        if (!menu_->isVolumeMoistTurnOff()) {
            engine::helper::GpuProfiler::Scope _scope_cloud(
                gpu_profiler_, cmd_buf, "Volume Cloud");
            volume_cloud_->renderVolumeCloud(
                cmd_buf,
                view_desc_set,
                hdr_color_buffer_.image,
                skydome_,
                menu_->getViewExtFactor(),
                menu_->getViewExtExponent(),
                menu_->getCloudAmbientIntensity(),
                menu_->getCloudPhaseIntensity(),
                menu_->getCloudMoistToPressureRatio(),
                menu_->getCloudNoiseWeight(0),
                menu_->getCloudNoiseWeight(1),
                menu_->getCloudNoiseThresold(),
                menu_->getCloudNoiseScrollingSpeed(),
                menu_->getCloudNoiseScale(),
                screen_size,
                s_dbuf_idx,
                current_time);
            // _scope_cloud auto-closes via RAII.
        }
    }

    er::ImageResourceInfo src_info = {
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT)};

    er::ImageResourceInfo dst_info = {
        er::ImageLayout::PRESENT_SRC_KHR,
        SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

    {
        engine::helper::GpuProfiler::Scope _scope_blit(
            gpu_profiler_, cmd_buf, "Final Blit");
        er::Helper::blitImage(
            cmd_buf,
            object_scene_view_->getColorBuffer()->image,
            swap_chain_info.images[image_index],
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            object_scene_view_->getColorBuffer()->size,
            glm::ivec3(screen_size.x, screen_size.y, 1));
        // _scope_blit auto-closes via RAII.
    }

    // ----- GPU Profiler: end frame -------------------------------------------
    gpu_profiler_.endFrame(cmd_buf, static_cast<uint32_t>(current_frame_ % kMaxFramesInFlight));

    // Snapshot this frame's view-projection so the NEXT frame's cluster
    // cull can reproject cluster bounds into yesterday's screen space
    // and sample the Hi-Z pyramid built from yesterday's depth.
    if (main_camera_object_) {
        last_view_proj_ = main_camera_object_->getViewProjMatrix();
    }

    s_dbuf_idx = 1 - s_dbuf_idx;
}

bool RealWorldApplication::queryGroundAt(
    float x, float z, float y_hint,
    float& out_y, glm::vec3& out_normal) {
    // Tiers 1-2: the walkable collision world (BVH) when built & non-empty.
    // Tier 1 starts just above the expected foot level (won't grab an upper
    // storey); tier 2 starts high to recover a foot that sits below the floor.
    if (collision_world_built_ && !collision_world_.empty()) {
        glm::vec3 hit, nrm;
        if (collision_world_.raycastDown(glm::vec3(x, y_hint + 2.0f, z),
                                         200.0f, hit, nrm) ||
            collision_world_.raycastDown(glm::vec3(x, y_hint + 80.0f, z),
                                         200.0f, hit, nrm)) {
            out_y = hit.y;
            out_normal = (glm::length(nrm) > 0.1f)
                ? glm::normalize(nrm) : glm::vec3(0.0f, 1.0f, 0.0f);
            if (out_normal.y < 0.0f) out_normal = -out_normal;
            return true;
        }
    }

    // Tier 3: vertical ray through the ACTUAL rendered scene geometry,
    // classification-independent (works even when nothing was tagged
    // WALKABLE_SURFACE).  Picks the surface whose Y is nearest y_hint and
    // returns its triangle normal.  Per-triangle work is budgeted; a
    // world-AABB column test rejects most meshes first.
    const glm::vec3 ro(x, y_hint + 100.0f, z);
    bool  found  = false;
    float best_y = 0.0f;
    float best_d = std::numeric_limits<float>::max();
    glm::vec3 best_n(0.0f, 1.0f, 0.0f);
    int   budget = 120000;
    auto probe = [&](const std::shared_ptr<ego::DrawableObject>& obj) {
        if (!obj || !obj->isReady()) return;
        const auto& d = obj->getDrawableData();
        for (const auto& nd : d.nodes_) {
            if (nd.mesh_idx_ < 0 || nd.mesh_idx_ >= (int)d.meshes_.size())
                continue;
            const auto& msh = d.meshes_[nd.mesh_idx_];
            if (!msh.vertex_position_ || msh.bbox_min_.x > msh.bbox_max_.x)
                continue;
            const glm::mat4& M = nd.cached_matrix_;
            glm::vec3 wmn(0.0f), wmx(0.0f);
            for (int cc = 0; cc < 8; ++cc) {
                const glm::vec3 lc(
                    (cc & 1) ? msh.bbox_max_.x : msh.bbox_min_.x,
                    (cc & 2) ? msh.bbox_max_.y : msh.bbox_min_.y,
                    (cc & 4) ? msh.bbox_max_.z : msh.bbox_min_.z);
                const glm::vec3 w = glm::vec3(M * glm::vec4(lc, 1.0f));
                if (cc == 0) { wmn = wmx = w; }
                else { wmn = glm::min(wmn, w); wmx = glm::max(wmx, w); }
            }
            if (x < wmn.x || x > wmx.x || z < wmn.z || z > wmx.z) continue;
            if (ro.y < wmn.y) continue;
            if (budget <= 0) return;
            const glm::mat4 invM = glm::inverse(M);
            const glm::vec3 rol = glm::vec3(invM * glm::vec4(ro, 1.0f));
            const glm::vec3 rdl =
                glm::vec3(invM * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
            const auto& Pv = *msh.vertex_position_;
            for (const auto& prim : msh.primitives_) {
                if (!prim.vertex_indices_) continue;
                const auto& I = *prim.vertex_indices_;
                for (size_t k = 0; k + 2 < I.size(); k += 3) {
                    if (--budget <= 0) break;
                    const size_t ia = (size_t)I[k + 0];
                    const size_t ib = (size_t)I[k + 1];
                    const size_t ic = (size_t)I[k + 2];
                    if (ia >= Pv.size() || ib >= Pv.size() ||
                        ic >= Pv.size()) continue;
                    const glm::vec3 e1 = Pv[ib] - Pv[ia];
                    const glm::vec3 e2 = Pv[ic] - Pv[ia];
                    const glm::vec3 pv = glm::cross(rdl, e2);
                    const float det = glm::dot(e1, pv);
                    const float ad = det < 0.0f ? -det : det;
                    if (ad < 1e-12f) continue;
                    const float inv = 1.0f / det;
                    const glm::vec3 tv = rol - Pv[ia];
                    const float u = glm::dot(tv, pv) * inv;
                    if (u < 0.0f || u > 1.0f) continue;
                    const glm::vec3 qv = glm::cross(tv, e1);
                    const float vb = glm::dot(rdl, qv) * inv;
                    if (vb < 0.0f || u + vb > 1.0f) continue;
                    const float tt = glm::dot(e2, qv) * inv;
                    if (tt <= 0.0f) continue;
                    const float wy = ro.y - tt;
                    const float dd = wy > y_hint ? wy - y_hint : y_hint - wy;
                    if (dd < best_d) {
                        best_d = dd;
                        best_y = wy;
                        found  = true;
                        glm::vec3 nl = glm::cross(e1, e2);
                        glm::vec3 nw = glm::vec3(
                            glm::transpose(invM) * glm::vec4(nl, 0.0f));
                        if (glm::length(nw) > 1e-8f) nw = glm::normalize(nw);
                        else nw = glm::vec3(0.0f, 1.0f, 0.0f);
                        if (nw.y < 0.0f) nw = -nw;
                        best_n = nw;
                    }
                }
            }
        }
    };
    probe(bistro_exterior_scene_);
    probe(bistro_interior_scene_);
    for (auto& dr : drawable_objects_) probe(dr);
    if (found) { out_y = best_y; out_normal = best_n; }
    return found;
}

void RealWorldApplication::initDrawFrame() {
    if (s_render_prt_test) {
        const auto full_buffer_size =
            conemap_obj_->getConemapTexture()->size;

        // generate minmax depth buffer.
        {
            const auto& cmd_buf =
                device_->setupTransientCommandBuffer();
            conemap_obj_->update(
                cmd_buf,
                conemap_obj_->getConemapTexture()->size);
            device_->submitAndWaitTransientCommandBuffer();
        }

        // conemap generation.
        {
            auto conemap_start_point_ =
                std::chrono::high_resolution_clock::now();

            auto dispatch_block_size =
                glm::uvec2(kConemapGenBlockSizeX, kConemapGenBlockSizeY);

            auto dispatch_block_count =
                (glm::uvec2(full_buffer_size) + dispatch_block_size - glm::uvec2(1)) / dispatch_block_size;

            auto num_passes =
                dispatch_block_count.x * dispatch_block_count.y;

            const uint32_t pass_step = 8;
            for (uint32_t i_pass = 0; i_pass < num_passes; i_pass+= pass_step) {
                auto pass_end = std::min(i_pass + pass_step, num_passes);
/*                std::cout <<
                    "conemap generation pass: " <<
                    i_pass <<
                    ", " <<
                    pass_end <<
                    std::endl;*/
                const auto& cmd_buf =
                    device_->setupTransientCommandBuffer();
                conemap_gen_->update(
                    cmd_buf,
                    conemap_obj_,
                    i_pass,
                    pass_end);
                device_->submitAndWaitTransientCommandBuffer();
            }

            auto conemap_end_point_ =
                std::chrono::high_resolution_clock::now();
            float delta_t_ =
                std::chrono::duration<float, std::chrono::seconds::period>(
                    conemap_end_point_ - conemap_start_point_).count();
            std::cout << "conemap generation time: " << delta_t_ << "s" << std::endl;
        }

        // prt shadow generation.
        {
            auto prt_start_point_ =
                std::chrono::high_resolution_clock::now();
            const auto& prt_gen_cmd_buf =
                device_->setupTransientCommandBuffer();
            prt_shadow_gen_->update(
                prt_gen_cmd_buf,
                conemap_obj_);
            device_->submitAndWaitTransientCommandBuffer();
            auto prt_end_point_ =
                std::chrono::high_resolution_clock::now();
            delta_t_ =
                std::chrono::duration<float, std::chrono::seconds::period>(
                    prt_end_point_ - prt_start_point_).count();
            std::cout << "prt generation time: " << delta_t_ << "s" << std::endl;
        }
    }

    prt_shadow_gen_->destroy(device_);
}

void RealWorldApplication::drawFrame() {
    // ── CPU frame anchor ──
    // Capture host-side time relative to the same frame-in-flight slot
    // that the GPU profiler will record into.  Doing this at the very
    // top of drawFrame means CPU scopes cover the WHOLE host frame
    // (fence wait, image acquire, all the per-frame state updates,
    // command recording, submit, present).  endCpuFrame at the
    // function bottom closes the recording.  drawScene runs inside
    // this frame and adds its own beginScope/endScope pairs against
    // the same slot.
    const uint32_t cpu_frame_idx =
        static_cast<uint32_t>(current_frame_ % kMaxFramesInFlight);
    gpu_profiler_.beginCpuFrame(cpu_frame_idx);
    auto _cpu_frame = gpu_profiler_.beginCpuScope("drawFrame");

    // Outer "Fence Wait + Acquire" scope kept for at-a-glance timing
    // of all the frame-start blocking calls.  Three inner sub-scopes
    // pinpoint WHICH call is blocking when this region grows large
    // (typical scenarios: GPU-bound → "Wait Frame Fence" dominates;
    // VSync / compositor pacing → "Acquire Image" dominates;
    // swapchain image starvation → "Wait Image Fence" dominates).
    auto _cpu_wait = gpu_profiler_.beginCpuScope("Fence Wait + Acquire");

    // (1) Wait for the GPU to be done with the work we submitted in
    //     this same FIF slot one frame-cycle ago.  If the GPU work
    //     for that frame is longer than our CPU work, the CPU has
    //     nothing to do but block here — the canonical "GPU-bound"
    //     stall.  Conversely, if this stays tiny, the CPU is the
    //     bottleneck and the GPU is sitting idle on submit.
    {
        auto _cpu_wait_fence = gpu_profiler_.beginCpuScope("Wait Frame Fence");
        device_->waitForFences({ in_flight_fences_[current_frame_] });
        device_->resetFences({ in_flight_fences_[current_frame_] });
        gpu_profiler_.endCpuScope(_cpu_wait_fence);
    }

    // NOTE: prev-frame fence wait sits below, just before
    // command_buffer->reset() / drawScene().  CPU pre-work above this
    // point runs concurrently with frame N-1's GPU work because it
    // doesn't touch per-frame host-coherent buffers.

    // ── Collect GPU-profiler query results for the FIF slot we just
    //    drained ────────────────────────────────────────────────────────
    // We just waited on in_flight_fences_[current_frame_], so the GPU
    // work submitted FIF cycles ago against THIS slot's query pool is
    // guaranteed complete.  Read its timestamps now — vkGetQueryPoolResults
    // will succeed without VK_NOT_READY.
    //
    // This used to live at the END of drawFrame, reading slot
    // (current_frame_ - 1) % FIF.  That worked only because the legacy
    // per-image fence wait artificially slowed the CPU enough for the
    // GPU to catch up by the time of the read.  With that wait gone,
    // the CPU outruns the GPU, the read keeps returning VK_NOT_READY,
    // collectResults silently drops every frame, and the profiler
    // appears frozen on the last successfully-ingested frame.
    //
    // Doing it RIGHT after the fence wait is the standard pattern:
    // the slot is provably complete, no stall, FIF-frame latency
    // (acceptable for a profile display).
    if (gpu_profiler_initialized_) {
        gpu_profiler_.collectResults(
            device_,
            static_cast<uint32_t>(current_frame_));
    }

    // (2) Get the next swapchain image.  With FIFO present mode this
    //     blocks for the next vblank if the prior frame's present is
    //     still queued.  With MAILBOX it blocks only when the swapchain
    //     ring is empty (all images presenting / queued for present).
    //     If your refresh rate × frame time = 1.0 (e.g., 16.67 ms at
    //     60 Hz), this scope is where the VSync wait shows up.
    uint32_t image_index = 0;
    bool need_recreate_swap_chain = false;
    {
        auto _cpu_acquire = gpu_profiler_.beginCpuScope("Acquire Image");
        need_recreate_swap_chain = er::Helper::acquireNextImage(
            device_,
            swap_chain_info_.swap_chain,
            image_available_semaphores_[current_frame_],
            image_index);
        gpu_profiler_.endCpuScope(_cpu_acquire);
    }

    if (need_recreate_swap_chain) {
        gpu_profiler_.endCpuScope(_cpu_wait);
        gpu_profiler_.endCpuScope(_cpu_frame);
        gpu_profiler_.endCpuFrame(cpu_frame_idx);
        recreateSwapChain();
        return;
    }

    // (3) Per-IMAGE fence wait — REMOVED (was a tutorial-grade copy-
    //     paste hazard, dominated frame time at 11+ ms).
    //
    // The Khronos Vulkan tutorial includes this wait to handle two
    // edge cases generically:
    //   (a) MAX_FRAMES_IN_FLIGHT > swapchain image count, so the
    //       per-FIF fence in (1) doesn't necessarily drain the
    //       previous use of this specific swapchain image.
    //   (b) vkAcquireNextImageKHR returning images out-of-order,
    //       so the image we just acquired might still be in use
    //       by a frame between the "oldest in flight" (drained by
    //       (1)) and "right now".
    //
    // Neither applies in this engine, AND the wait is actively
    // harmful when swapchain_image_count > FIF (our typical case:
    // 3 swapchain images + FIF=2):
    //
    //   Frame N+2 acquires image (N+1)%3 — an image the compositor
    //   handed back early.  images_in_flight_[that] = fence from
    //   frame N+1, which is STILL EXECUTING ON THE GPU.  The wait
    //   blocks the CPU for the remainder of frame N+1's GPU work.
    //   Result: even though we have 3 images and FIF=2 (i.e., room
    //   for the CPU to run 2 frames ahead of the GPU), the per-image
    //   wait silently throttles us to "CPU at most 1 frame ahead",
    //   which destroys the FIF-2 pipelining we actually configured.
    //
    // What we have instead:
    //   * One command buffer per FIF slot (not per image), reset
    //     safely after the per-FIF fence in (1).
    //   * The image_available_semaphores_[current_frame_] semaphore
    //     bound as a wait on submit, with TRANSFER_BIT dst stage —
    //     guarantees the GPU won't blit-write to the image before
    //     the compositor releases it.
    //   * The render_finished_semaphores_[current_frame_] signal
    //     bound on submit and waited on by present — guarantees
    //     present doesn't read the image before our blit completes.
    //
    // Together those three give complete swapchain-image safety.
    // The per-image fence wait was redundant safety belt.
    //
    // (Kept the images_in_flight_ vector itself + the assignment
    // below for now in case we want to bring back a DIAGNOSTIC
    // version that times the wait but doesn't actually wait — easier
    // to spot regressions if FIF or swapchain count changes.  The
    // assignment is essentially free.)
    images_in_flight_[image_index] = in_flight_fences_[current_frame_];
    gpu_profiler_.endCpuScope(_cpu_wait);

    time_t now = time(0);
    tm localtm;
    gmtime_s(&localtm, &now);

    float latitude   = 37.4419f;
    float longtitude = -122.1430f; // west (negative = west of prime meridian).

    // Time-of-day driven by the Menu slider (see Skydome window).  When
    // auto-advance is on, the Menu ticks tod_hours_ forward by real time
    // each frame.  consumeTodJump() returns true once when the user has
    // yanked the slider far enough to call it a hard skip - we use that
    // as a cue to reset the sky / IBL mini-buffer accumulators so they
    // re-bootstrap cleanly instead of EMA-blending toward the new sun
    // position over many seconds.
    // ── CPU profile: "TOD + Skydome math" ────────────────────────────
    // Lumps three CPU-only blocks together: menu time-of-day stepping,
    // local-solar-time computation, and skydome_->update (which evaluates
    // the analytic sun position and refreshes a handful of scalar
    // params).  Expected to stay sub-millisecond — if it grows, look at
    // Skydome::update's transmittance/multiple-scattering recomputes.
    auto _cpu_tod = gpu_profiler_.beginCpuScope("TOD + Skydome math");
    menu_->advanceTimeOfDay(delta_t_);

    // Convert the menu's UTC-based time to local solar time for the hardcoded
    // longitude.  Each 15° of longitude = 1 hour; west is negative.
    // This ensures the sun position is correct regardless of the timezone
    // the machine running the app is set to.
    const float utc_h =
        static_cast<float>(localtm.tm_hour) +
        static_cast<float>(localtm.tm_min)  / 60.0f +
        static_cast<float>(localtm.tm_sec)  / 3600.0f;
    const float solar_h = std::fmod(utc_h + longtitude / 15.0f + 24.0f, 24.0f);

    // Sync the menu slider to solar time once on startup.
    // Use solar noon (12.0) as the fallback if the computed solar time is
    // outside daytime hours (below-horizon sun gives a pitch-black sky and
    // kills IBL, so always start in daytime regardless of the user's timezone).
    static bool s_tod_synced = false;
    if (!s_tod_synced) {
        const bool is_daytime = (solar_h >= 6.0f && solar_h <= 20.0f);
        menu_->setTimeOfDayHours(is_daytime ? solar_h : 12.0f);
        s_tod_synced = true;
    }

    const float tod_h = menu_->getTimeOfDayHours();
    const int   tod_hour = static_cast<int>(tod_h);
    const int   tod_min  = static_cast<int>((tod_h - tod_hour) * 60.0f);
    const int   tod_sec  = static_cast<int>(((tod_h - tod_hour) * 60.0f - tod_min) * 60.0f);

    menu_->consumeTodJump(); // consume the flag; dither cycle runs continuously, no reset needed

    skydome_->update(
        latitude,
        longtitude,
        localtm.tm_yday,
        tod_hour,
        tod_min,
        tod_sec);
    gpu_profiler_.endCpuScope(_cpu_tod);

    // ── CPU profile: "readGpuCameraInfo" ─────────────────────────────
    // Dumps the camera UBO back to host memory so the CPU can read the
    // newly-computed camera matrices (for shadow-cascade math etc.).
    // This is a host-coherent buffer read — costs only the memcpy on
    // sane drivers, but if vkInvalidateMappedMemoryRanges is implicit
    // and the buffer is on a non-coherent heap it could stall.  Keep
    // an eye on this one when the wait lane shrinks.
    {
        auto _cpu_read = gpu_profiler_.beginCpuScope("readGpuCameraInfo");
        main_camera_object_->readGpuCameraInfo();
        gpu_profiler_.endCpuScope(_cpu_read);
    }

    // ── CPU profile: "Terrain updateAllTiles" ────────────────────────
    // Walks the tile cache against the camera position: evicts out-of-
    // range tiles, allocates new ones inside the visibility radius, and
    // does a host-coherent updateBufferMemory write per visible tile to
    // refresh its grass indirect-draw-cmd buffer.  Cost scales with
    // visible-tile count (typically 9–25) and how many tiles cross the
    // cache boundary this frame.
    std::vector<std::shared_ptr<ego::TileObject>> visible_tiles;
    {
        auto _cpu_tiles = gpu_profiler_.beginCpuScope("Terrain updateAllTiles");
        visible_tiles =
            ego::TileObject::updateAllTiles(
                device_,
                descriptor_pool_,
                128,
                glm::vec2(main_camera_object_->getCameraPosition()));
        gpu_profiler_.endCpuScope(_cpu_tiles);
    }

    if (dump_volume_noise_) {
        const auto noise_texture_size = kDetailNoiseTextureSize;
#if 1
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_,
            volume_noise_->getDetailNoiseTexture().image,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            4,
            temp_buffer.data(),
            std::source_location::current());
        
        eh::saveDdsTexture(
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            temp_buffer.data(),
            "volume_noise.dds");
#else
        uint32_t pixel_count = noise_texture_size * noise_texture_size * noise_texture_size;
        std::vector<uint32_t> temp_buffer;
        temp_buffer.resize(pixel_count);
        er::Helper::dumpTextureImage(
            device_,
            volume_noise_->getDetailNoiseTexture().image,
            er::Format::R8G8B8A8_UNORM,
            glm::uvec3(noise_texture_size, noise_texture_size, noise_texture_size),
            4,
            temp_buffer.data());

        // flatten 3d texture to 2d texture.
        uint32_t h = 1 << (31 - eh::clz(static_cast<uint32_t>(std::sqrt(noise_texture_size))));
        uint32_t w = noise_texture_size / h;
        std::vector<uint32_t> flattened_buffer;
        flattened_buffer.resize(pixel_count);
        for (uint32_t y = 0; y < h * noise_texture_size; y++) {
            for (uint32_t x = 0; x < w * noise_texture_size; x++) {
                uint32_t tile_y = y / noise_texture_size;
                uint32_t tile_x = x / noise_texture_size;
                uint32_t depth = tile_y * w + tile_x;
                uint32_t dst_idx = y * (w * noise_texture_size) + x;
                uint32_t src_idx = depth * noise_texture_size * noise_texture_size +
                    (y % noise_texture_size) * noise_texture_size + (x % noise_texture_size);
                flattened_buffer[dst_idx] = temp_buffer[src_idx];
            }
        }
        eh::saveDdsTexture(
            glm::uvec3(w * noise_texture_size, h * noise_texture_size, 1),
            flattened_buffer.data(),
            "volume_noise.dds");
#endif
        dump_volume_noise_ = false;
    }

    // Pick the command buffer by FIF slot — image_index is still used
    // below for the swapchain image we render INTO (framebuffer / blit
    // destination) but NOT for selecting which CB to record.  See
    // createCommandBuffers() for the rationale.
    auto command_buffer = command_buffers_[current_frame_];
    std::vector<std::shared_ptr<er::CommandBuffer>>command_buffers(1, command_buffer);

    if (current_time_ == 0) {
        last_frame_time_point_ = std::chrono::high_resolution_clock::now();
    }

    auto current_time_point = std::chrono::high_resolution_clock::now();
    delta_t_ = std::chrono::duration<float, std::chrono::seconds::period>(
                    current_time_point - last_frame_time_point_).count();

    current_time_ += delta_t_;
    last_frame_time_point_ = current_time_point;

    // ── CPU profile: "Mesh load poll" ────────────────────────────────
    // Cheap when nothing is ready: peeks at the worker-thread's
    // completion queue.  Spikes whenever a streaming load finishes —
    // phase 3 (descriptor-set + pipeline creation) runs synchronously on
    // the main thread inside poll(), so a single asset finalisation
    // here can cost a couple of ms on first-frame populate.
    auto _cpu_mesh_poll = gpu_profiler_.beginCpuScope("Mesh load poll");
    // Drain any in-flight async mesh loads whose GPU fence has signaled:
    // this runs phase 3 (descriptor-set + pipeline creation) on the main
    // thread. Cheap when nothing is ready. Must run before we touch
    // drawable_objects_ for draw so that objects finalized this frame
    // pop in immediately instead of waiting one extra frame.
    if (mesh_load_task_manager_) {
        mesh_load_task_manager_->poll();
    }
    gpu_profiler_.endCpuScope(_cpu_mesh_poll);

    // ---- "New Game" handler: add the bistro scenes (already created at
    // startup and streaming in the background) into the scene views so
    // they start rendering.  The XML mesh list is available for any
    // additional meshes that weren't pre-loaded.
    if (menu_->consumeNewGameRequest()) {
        // Restore the bistro exterior/interior into the scene views.
        // They were created at startup but NOT added to the views so the
        // title screen stays clean.
        if (bistro_exterior_scene_) {
            object_scene_view_->addDrawableObject(bistro_exterior_scene_);
            shadow_object_scene_view_->addDrawableObject(bistro_exterior_scene_);
        }
        if (bistro_interior_scene_) {
            object_scene_view_->addDrawableObject(bistro_interior_scene_);
        }

        s_update_frame_count = -1;
    }

    // Transition Loading → InGame once bistro scenes have finished streaming.
    if (menu_->getGameState() == engine::ui::GameState::Loading) {
        bool all_ready = true;
        if (bistro_exterior_scene_ && !bistro_exterior_scene_->isReady())
            all_ready = false;
        if (bistro_interior_scene_ && !bistro_interior_scene_->isReady())
            all_ready = false;
        if (all_ready) {
            menu_->setGameState(engine::ui::GameState::InGame);
            menu_->setBackgroundEnabled(false);

            // Upload cluster data to the GPU cluster renderer now that
            // meshes are fully loaded. Iterates every mesh in every
            // drawable and uploads its ClusterMesh sidecar.
            if (cluster_renderer_) {
                // Cluster bounds from buildClusterMesh() are already in
                // the mesh's local space, which for FBX scenes IS world
                // space (the FBX loader bakes node transforms into
                // vertex positions). Using identity here avoids the
                auto uploadClusters = [&](
                    const std::shared_ptr<ego::DrawableObject>& obj) {
                    if (!obj || !obj->isReady()) return;
                    auto& meshes  = obj->getMutableMeshes();
                    const auto& data = obj->getDrawableData();

                    // Build mesh_idx → first-node world transform table.
                    // node.cached_matrix_ is set by DrawableData::update()
                    // and already includes the full parent hierarchy.
                    // Defaults to identity for meshes with no owning node.
                    std::vector<glm::mat4> mesh_transforms(
                        meshes.size(), glm::mat4(1.0f));
                    for (const auto& node : data.nodes_) {
                        if (node.mesh_idx_ >= 0 &&
                            static_cast<uint32_t>(node.mesh_idx_) < meshes.size()) {
                            // First node wins — handles non-instanced meshes.
                            if (mesh_transforms[node.mesh_idx_] == glm::mat4(1.0f))
                                mesh_transforms[node.mesh_idx_] = node.cached_matrix_;
                        }
                    }

                    for (uint32_t mi = 0; mi < meshes.size(); ++mi) {
                        const auto& cm = meshes[mi].cluster_mesh_;
                        if (cm.empty()) continue;
                        meshes[mi].cluster_global_mesh_idx_ =
                            static_cast<int32_t>(
                                cluster_renderer_->getMeshCount());
                        cluster_renderer_->uploadMeshClusters(
                            cm, obj->getDrawableData(), mi,
                            meshes[mi].cluster_prim_map_,
                            mesh_transforms[mi]);
                    }
                };
                uploadClusters(bistro_exterior_scene_);
                uploadClusters(bistro_interior_scene_);
                for (auto& d : drawable_objects_) {
                    uploadClusters(d);
                }

                // Merge all staged cluster data into single flat GPU SSBOs.
                cluster_renderer_->finalizeUploads();

                // Initialize the bindless graphics pipeline for cluster
                // rendering. Needs the global descriptor set layouts
                // (PBR_GLOBAL at set 0, VIEW_PARAMS at set 1) so the
                // pipeline layout matches the forward pass.
                {
                    // Provide all sets 0..RUNTIME_LIGHTS so the cluster
                    // pipeline can read the same sun light + shadow data
                    // as the standard forward pass.
                    er::DescriptorSetLayoutList global_layouts(
                        RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                    global_layouts[PBR_GLOBAL_PARAMS_SET] =
                        pbr_lighting_desc_set_layout_;
                    global_layouts[VIEW_PARAMS_SET] =
                        ego::CameraObject::getViewCameraDescriptorSetLayout();
                    // Set 2 (PBR_MATERIAL_PARAMS_SET) is replaced by the
                    // cluster bindless set inside initBindlessPipeline.
                    // Set 3 (SKIN_PARAMS_SET) stays nullptr.
                    global_layouts[RUNTIME_LIGHTS_PARAMS_SET] =
                        runtime_lights_desc_set_layout_;
                    cluster_renderer_->initBindlessPipeline(
                        global_layouts,
                        graphic_pipeline_info_,
                        renderbuffer_formats_[
                            int(er::RenderPasses::kForward)]);
                    // Deferred G-buffer variant — paired with the forward
                    // pipeline above; reuses its bindless layout, swaps
                    // the fragment SPV for cluster_bindless_gbuf_frag.spv,
                    // and targets the application's 3-RT G-buffer format.
                    cluster_renderer_->initBindlessGBufferPipeline(
                        graphic_pipeline_info_,
                        gbuffer_renderbuffer_format_);

                    // Depth-only CSM shadow variant.  Single drawIndirect-
                    // Count broadcasts every cluster triangle to all
                    // CSM_CASCADE_COUNT cascades via a geometry shader,
                    // replacing the ~2400 individual per-mesh shadow draws
                    // that were costing ~17 ms of CPU recording time per
                    // frame.  Slim pipeline layout — only VIEW_PARAMS_SET
                    // (for layout-compat) and RUNTIME_LIGHTS_PARAMS_SET
                    // (GS reads cascade VP matrices).  Render target is
                    // the kShadow pass's depth-array format.
                    er::DescriptorSetLayoutList shadow_layouts(
                        RUNTIME_LIGHTS_PARAMS_SET + 1, nullptr);
                    shadow_layouts[VIEW_PARAMS_SET] =
                        ego::CameraObject::getViewCameraDescriptorSetLayout();
                    shadow_layouts[RUNTIME_LIGHTS_PARAMS_SET] =
                        runtime_lights_desc_set_layout_;
                    cluster_renderer_->initBindlessShadowPipeline(
                        shadow_layouts,
                        graphic_pipeline_info_,
                        renderbuffer_formats_[
                            int(er::RenderPasses::kShadow)].depth_format);

                    // Silhouette prepass pipeline — small mesh-shader
                    // pass that pre-fills each cascade's in-camera-
                    // frustum interior with depth=1 so out-of-frustum
                    // texels (cleared 0) reject every shadow caster via
                    // LESS_OR_EQUAL.  Independent of the cluster pipe-
                    // line; only needs RUNTIME_LIGHTS for cascade VPs
                    // and slab corners.
                    cluster_renderer_->initCsmSilhouettePrepassPipeline(
                        runtime_lights_desc_set_layout_,
                        graphic_pipeline_info_,
                        renderbuffer_formats_[
                            int(er::RenderPasses::kShadow)].depth_format);
                }
            }
        }
    }

    // ── Player spawn (deferred-friendly retry) ──────────────────────
    // Originally this was nested inside the Loading→InGame transition
    // block above, which ran ONCE.  If the user picked a player AFTER
    // bistro finished loading (i.e. after we'd already transitioned to
    // InGame), the spawn block had already executed with player_object_
    // null or not-yet-isReady, the !isSpawned() guard latched, and the
    // character stayed at its default (0,0,0) position — invisible if
    // that point was underground or behind something.  Moving the
    // check here, gated on !isSpawned(), keeps it idempotent (only
    // spawns once) but lets the check run every frame until the player
    // genuinely is ready and spawnable.
    //
    // The user can also force a re-spawn (snap to current camera) by
    // picking "Reset player position" from the Game Objects menu,
    // which sets a flag the menu surfaces via consumeResetPlayerPosition.
    // We OR that flag with the !isSpawned() condition so the same
    // code path serves both first-time spawning and on-demand recovery.
    const bool reset_pos_req =
        menu_ ? menu_->consumeResetPlayerPosition() : false;
    // First-time spawn waits until the level is fully loaded AND the
    // (simplified) collision mesh has finished building, so the very first
    // placement can land her on the ground (the per-frame follow block
    // grounds her via the collision / real-floor raycast, which needs the
    // collision world ready).  The manual "Reset player position" path
    // (reset_pos_req) still fires anytime.
    if (player_object_ && player_object_->isReady() &&
        player_controller_ &&
        ((!player_controller_->isSpawned() && collision_world_built_) ||
         reset_pos_req)) {
        // ── Apply configuration flags on the player ───────────────────
        // object_ is now guaranteed non-null (isReady true), so the
        // setters can land.  IMPORTANT: do NOT move this back to the
        // post-createAsync init site — the shell has object_=nullptr
        // and the setters silently no-op there, which is the bug that
        // caused the whole "player invisible" investigation.
        //
        // setUseNodeTransformOnly(true) is the production fix:
        //   • identity instance buffer (no double-transform from the
        //     shared game_objects_buffer_'s camera-tracking position)
        //   • skip imported animations (PlayerController owns the rig
        //     pose; the gltf timeline would otherwise clobber it)
        //
        // The debug overrides (force_red, skip_skinning, scale=30) are
        // turned off — kept as setters in case we need to re-enable
        // any of them for future visibility debugging without having
        // to chase down the async-load gotcha again.
        // Production configuration.  setUseNodeTransformOnly is the
        // ONLY non-default flag the player needs — without it the
        // shared game_objects_buffer_'s instance position double-
        // transforms the character.  The debug overrides are all
        // turned off; the bisect proved the character renders
        // correctly through the skinned forward path at authored
        // scale, with real materials.
        player_object_->setUseNodeTransformOnly(true);
        player_object_->setDebugForceRed(false);
        player_object_->setDebugSkipSkinning(false);
        player_object_->setDebugScale(0.0f);
        // Decoupled-from-visuals diagnostic: enables the per-second
        // [player.draw] indirect-draws-issued tally without tinting
        // the character.  Useful for confirming the forward draw is
        // actually running while we investigate the marker-vs-body
        // offset.
        player_object_->setDebugLogDraws(true);

        const auto& cam = main_camera_object_->getCameraViewInfo();
        glm::vec3 cam_pos = main_camera_object_->getCameraPosition();
        glm::vec2 fwd_xz(cam.facing_dir.x, cam.facing_dir.z);
        float fxz_len = glm::length(fwd_xz);
        if (fxz_len > 1e-3f) {
            fwd_xz /= fxz_len;
        } else {
            float yaw_rad = glm::radians(-cam.yaw);
            fwd_xz = glm::vec2(
                std::cos(yaw_rad), std::sin(yaw_rad));
        }
        // ── Spawn placement ───────────────────────────────────────────
        // Put the character 2 m in front of the camera at "feet height"
        // (cam_pos.y - eye offset).  Earlier this used 3 m + a terrain-Y
        // heuristic that compared the heightmap to the camera Y and
        // picked one or the other; on bistro that produced unstable
        // spawn positions because the heightmap and the bistro mesh
        // sit in different coordinate systems.  Dropping the heuristic
        // and always using cam_pos.y - kPlayerEyeOffset gives a
        // predictable "right where you're looking" placement — exactly
        // what someone testing the character expects.  PlayerController
        // does its own terrain snap in update() once WASD movement
        // starts, so walking onto real terrain still lands correctly.
        const float kSpawnAhead      = 2.0f;
        // Camera-Y → player-feet vertical offset.  ~1.7m drops the
        // character's feet to a plausible standing position when the
        // camera is at adult eye level.  Was temporarily +1.5m extra
        // during the giant-red-debug phase to keep the inflated mesh
        // out of the camera's near volume — restored to 1.7m now
        // that the character renders normally at authored scale.
        const float kPlayerEyeOffset = 1.7f;

        // ── Asset pivot-offset compensation (override-aware) ────────
        // The glTF root's authored T,R,S placed the mesh somewhere
        // in world space.  But PlayerController::applyPose overrides
        // root.translation AND root.rotation each frame:
        //   • root.translation = spawn_pos
        //   • root.rotation    = quat(Y-axis, yaw_deg)
        //   • root.scale       = preserved (asset-authored)
        //
        // The asset's authored rotation (for scene-skinned.gltf this
        // is a 90° X-axis "Y-up → Z-up" swap baked into the root)
        // gets wiped out by that override.  So measuring world Y
        // through the CURRENT cached_matrix (which still holds the
        // authored rotation) gives a feet-offset that's correct for
        // the OLD orientation but wrong for the orientation the
        // character will actually render in.  The character then
        // either floats or — in scene-skinned's case — sinks.
        //
        // Fix: simulate the post-override world transform.  For each
        // mesh-owning node, we have:
        //   mesh.cached = root_cached_old * descendant_chain
        //              = (T_old * R_old * S) * descendant_chain
        // We want to swap the leading (T_old*R_old*S) for the
        // post-override (T_new*R_new*S).  Since S is unchanged we
        // can factor it out:
        //   simulated_mesh = (T_new * R_new * S) * inverse(T_old * R_old * S) * mesh.cached
        // Or equivalently:
        //   simulated_mesh = (T_new * R_new) * inverse(T_old * R_old) * mesh.cached
        // because the inverse-then-multiply cancels the scale.
        //
        // We can apply this to mesh AABB corners and find world Y.
        // World Y is linear in T_new.y (R_new is a Y-axis rotation,
        // which doesn't touch Y), so once we measure with T_new.y=0
        // the result is a constant `f` and the equation to solve is
        //   T_new.y + f = desired_feet_y
        // giving T_new.y = desired_feet_y - f.
        const auto& data_for_pivot = player_object_->getDrawableData();
        int rn_for_pivot = -1;
        if (!data_for_pivot.scenes_.empty() &&
            data_for_pivot.default_scene_ >= 0 &&
            data_for_pivot.default_scene_ <
                (int)data_for_pivot.scenes_.size() &&
            !data_for_pivot.scenes_[data_for_pivot.default_scene_]
                .nodes_.empty()) {
            rn_for_pivot = data_for_pivot
                .scenes_[data_for_pivot.default_scene_].nodes_[0];
        }

        glm::vec2 spawn_xz =
            glm::vec2(cam_pos.x, cam_pos.z) + fwd_xz * kSpawnAhead;
        // ── Pick desired feet Y ─────────────────────────────────────
        // Prefer the level's actual floor over "camera height minus
        // eye offset".  The Bistro scene's local bbox_min.y, when
        // transformed by its world location matrix, gives the
        // lowest-point of the interior — which for Bistro is the
        // floor (or basement level, close enough — the difference is
        // a couple of metres).  Using cam.y - 1.7 only works when
        // the camera happens to be at standing-eye height; if the
        // user has flown the camera up to a balcony / ceiling angle
        // the character pops into the sky.  Snapping to the bistro
        // floor instead anchors her to a stable world reference
        // regardless of where the camera ends up.
        float bistro_floor_y = -std::numeric_limits<float>::max();
        bool  have_floor = false;
        // Sanity gate: any "floor" outside this range is treated as
        // garbage (uninitialized bbox sentinel, degenerate matrix,
        // NaN/Inf, etc.).  ±1e5 covers any plausible world unit;
        // FLT_MAX / FLT_LOWEST / NaN are rejected.  Without this
        // an FBX whose bbox or location matrix isn't fully populated
        // the first ready frame propagates FLT_MAX through the spawn
        // → setRootNodeTransform → cached_matrix chain and the
        // character teleports to infinity (HUD shows the player Y
        // and bbox Y as 3.4e38).
        const float kFloorSanityLo = -1e5f;
        const float kFloorSanityHi =  1e5f;
        auto ingestFloorY = [&](const std::shared_ptr<ego::DrawableObject>& obj) {
            if (!obj || !obj->isReady()) return;
            glm::vec3 lo = obj->getModelBboxMin();
            glm::vec3 hi = obj->getModelBboxMax();
            if (lo.x > hi.x) return; // empty bbox sentinel
            // 8 local corners → 8 world Ys; the min is the world
            // floor of this drawable.  Cheaper than walking nodes.
            const glm::mat4& M = obj->getLocation();
            const glm::vec3 c[8] = {
                {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z},
                {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z},
                {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z},
                {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z},
            };
            float mn = std::numeric_limits<float>::max();
            for (int k = 0; k < 8; ++k) {
                glm::vec4 w = M * glm::vec4(c[k], 1.0f);
                if (w.y < mn) mn = w.y;
            }
            // Reject garbage values: NaN, ±Inf, FLT_MAX-ish, or any
            // sentinel that survived the empty-bbox check above.
            if (!std::isfinite(mn) ||
                mn < kFloorSanityLo || mn > kFloorSanityHi) {
                std::cout << "[player.floor] rejected: mn=" << mn
                          << " (bbox lo.y=" << lo.y << " hi.y=" << hi.y
                          << " M[3].y=" << M[3][1] << ")" << std::endl;
                return;
            }
            if (!have_floor || mn > bistro_floor_y) {
                // Pick the HIGHER of available floors — Bistro
                // interior + exterior both qualify; if interior is
                // present we want it (it's where the character is
                // standing) and it sits above any exterior basement.
                bistro_floor_y = mn;
                have_floor = true;
            }
        };
        ingestFloorY(bistro_interior_scene_);
        if (!have_floor) ingestFloorY(bistro_exterior_scene_);

        float desired_feet_y_cam   = cam_pos.y - kPlayerEyeOffset;
        // Manual nudge: drops the character below the auto-computed
        // floor by this fixed delta.  Added because the bistro
        // bbox_min.y picks up prop/basement geometry that sits well
        // above the visible interior floor; the character ends up
        // floating until we crank her down.  Tune by hand or expose
        // via menu once we have a feel for the right value.
        const float kManualFloorYAdjust = -6.0f;
        float desired_feet_y       = (have_floor
            ? bistro_floor_y
            : desired_feet_y_cam) + kManualFloorYAdjust;
        float spawn_y = desired_feet_y;        // fallback if no mesh
        float pivot_feet_offset = 0.0f;        // for logging
        bool  have_pivot_offset = false;

        if (rn_for_pivot >= 0 &&
            rn_for_pivot < (int)data_for_pivot.nodes_.size()) {
            const auto& root_node = data_for_pivot.nodes_[rn_for_pivot];
            const glm::mat4& old_root_cached = root_node.cached_matrix_;

            // Build post-override root transform with spawn_xz set
            // and Y temporarily 0 (we solve for Y below).
            glm::quat yaw_quat = glm::angleAxis(
                glm::radians(cam.yaw), glm::vec3(0, 1, 0));
            glm::mat4 trial_root =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(spawn_xz.x, 0.0f, spawn_xz.y)) *
                glm::mat4_cast(yaw_quat) *
                glm::scale(glm::mat4(1.0f), root_node.scale_);

            // root_old_to_new converts world coords expressed under
            // the OLD root frame to world coords under the NEW root
            // frame, with NEW T.y = 0.
            glm::mat4 root_old_to_new =
                trial_root * glm::inverse(old_root_cached);

            float min_world_y_at_zero = std::numeric_limits<float>::max();
            for (size_t ni = 0; ni < data_for_pivot.nodes_.size(); ++ni) {
                const auto& n = data_for_pivot.nodes_[ni];
                if (n.mesh_idx_ < 0 ||
                    n.mesh_idx_ >= (int)data_for_pivot.meshes_.size())
                    continue;
                const auto& mb = data_for_pivot.meshes_[n.mesh_idx_];
                const glm::vec3 c[8] = {
                    {mb.bbox_min_.x, mb.bbox_min_.y, mb.bbox_min_.z},
                    {mb.bbox_max_.x, mb.bbox_min_.y, mb.bbox_min_.z},
                    {mb.bbox_min_.x, mb.bbox_max_.y, mb.bbox_min_.z},
                    {mb.bbox_max_.x, mb.bbox_max_.y, mb.bbox_min_.z},
                    {mb.bbox_min_.x, mb.bbox_min_.y, mb.bbox_max_.z},
                    {mb.bbox_max_.x, mb.bbox_min_.y, mb.bbox_max_.z},
                    {mb.bbox_min_.x, mb.bbox_max_.y, mb.bbox_max_.z},
                    {mb.bbox_max_.x, mb.bbox_max_.y, mb.bbox_max_.z},
                };
                // Simulated cached: take the mesh node's existing
                // cached_matrix, peel the OLD root off the front,
                // then apply the NEW root (with Y=0).
                glm::mat4 simulated_cached = root_old_to_new * n.cached_matrix_;
                for (int k = 0; k < 8; ++k) {
                    glm::vec4 w =
                        simulated_cached * glm::vec4(c[k], 1.0f);
                    if (w.y < min_world_y_at_zero) {
                        min_world_y_at_zero = w.y;
                    }
                }
                have_pivot_offset = true;
            }

            if (have_pivot_offset) {
                // World Y at NEW T.y = 0 is min_world_y_at_zero.
                // For feet at desired_feet_y, set T.y to make
                //   T.y + min_world_y_at_zero = desired_feet_y.
                spawn_y = desired_feet_y - min_world_y_at_zero;
                pivot_feet_offset = -min_world_y_at_zero;
            }
        }

        // Final guard: if any input was garbage (Inf, NaN, FLT_MAX-ish)
        // and propagated, the spawn would teleport the character to
        // infinity and every subsequent frame would render her bbox
        // at FLT_MAX (HUD shows 3.4e38).  Reject and fall back to a
        // neutral spawn at origin so the user can still see something.
        if (!std::isfinite(spawn_y) ||
            std::abs(spawn_y) > 1e5f) {
            std::cout << "[player] spawn_y rejected as garbage ("
                      << spawn_y << "), falling back to 0"
                      << std::endl;
            spawn_y = 0.0f;
        }

        glm::vec3 spawn_pos(spawn_xz.x, spawn_y, spawn_xz.y);
        player_controller_->spawnAt(spawn_pos, cam.yaw);
        std::cout << "[player] "
                  << (reset_pos_req ? "Reset to " : "Spawned at ")
                  << "(" << spawn_pos.x << ", " << spawn_pos.y << ", "
                  << spawn_pos.z << ") yaw=" << cam.yaw
                  << "  pivot_feet_offset=" << pivot_feet_offset
                  << "  desired_feet_y=" << desired_feet_y
                  << "  (have_pivot_offset="
                  << (have_pivot_offset ? 1 : 0) << ")"
                  << std::endl;

        // ── One-shot diagnostic dump ──────────────────────────────────
        // Print the loaded gltf's structure at spawn time so we can
        // see node hierarchy, mesh count, per-primitive bbox + index
        // count, and the root cached_matrix translation/basis.  Useful
        // for diagnosing position / scale anomalies.  Fires once each
        // time the spawn block runs (initial spawn + every Reset).
        const auto& data = player_object_->getDrawableData();
        std::cout << "[player.diag] nodes=" << data.nodes_.size()
                  << " meshes=" << data.meshes_.size()
                  << " scenes=" << data.scenes_.size()
                  << " default_scene=" << data.default_scene_
                  << std::endl;
        if (!data.scenes_.empty() && data.default_scene_ >= 0 &&
            data.default_scene_ < (int)data.scenes_.size()) {
            const auto& sc = data.scenes_[data.default_scene_];
            int rn = sc.nodes_.empty() ? -1 : sc.nodes_[0];
            std::cout << "[player.diag] scene[" << data.default_scene_
                      << "] root_node=" << rn
                      << " bbox=(" << sc.bbox_min_.x << ","
                      << sc.bbox_min_.y << "," << sc.bbox_min_.z
                      << ")..(" << sc.bbox_max_.x << ","
                      << sc.bbox_max_.y << "," << sc.bbox_max_.z << ")"
                      << std::endl;
            if (rn >= 0 && rn < (int)data.nodes_.size()) {
                const auto& n = data.nodes_[rn];
                std::cout << "[player.diag] root.name='" << n.name_
                          << "' mesh_idx=" << n.mesh_idx_
                          << " skin_idx=" << n.skin_idx_
                          << " children=" << n.child_idx_.size()
                          << " T=(" << n.translation_.x << ","
                          << n.translation_.y << "," << n.translation_.z
                          << ") S=(" << n.scale_.x << "," << n.scale_.y
                          << "," << n.scale_.z << ")"
                          << " R(wxyz)=(" << n.rotation_.w << ","
                          << n.rotation_.x << "," << n.rotation_.y
                          << "," << n.rotation_.z << ")" << std::endl;
                // cached_matrix is what model_params.model_mat
                // ultimately reads.  Decompose its translation column
                // so we can see where the engine THINKS the root is in
                // world space.
                const glm::mat4& cm = n.cached_matrix_;
                std::cout << "[player.diag] root.cached_matrix translation=("
                          << cm[3][0] << "," << cm[3][1] << ","
                          << cm[3][2] << ")  basis lengths=("
                          << glm::length(glm::vec3(cm[0])) << ","
                          << glm::length(glm::vec3(cm[1])) << ","
                          << glm::length(glm::vec3(cm[2])) << ")"
                          << std::endl;
            }
        }
        // Deep mesh-data dump — checks that the loader actually
        // populated geometry (not just an empty asset shell) and that
        // each primitive has non-zero indices + valid vertex/attribute
        // bindings.  If a primitive prints index_count=0 or
        // attribute_count=0 the asset failed to load that primitive
        // and the renderer has nothing to rasterise — that's what
        // "missing" really means.
        for (size_t mi = 0; mi < data.meshes_.size(); ++mi) {
            const auto& m = data.meshes_[mi];
            std::cout << "[player.diag] mesh[" << mi << "] bbox=("
                      << m.bbox_min_.x << "," << m.bbox_min_.y << ","
                      << m.bbox_min_.z << ")..(" << m.bbox_max_.x
                      << "," << m.bbox_max_.y << "," << m.bbox_max_.z
                      << ") prims=" << m.primitives_.size()
                      << " cluster_global_mesh_idx="
                      << m.cluster_global_mesh_idx_ << std::endl;
            for (size_t pi = 0; pi < m.primitives_.size(); ++pi) {
                const auto& p = m.primitives_[pi];
                uint64_t idx_count = 0;
                if (!p.index_desc_.empty()) {
                    idx_count = p.index_desc_[0].index_count;
                }
                std::cout << "[player.diag]   prim[" << pi << "]"
                          << " material_idx=" << p.material_idx_
                          << " idx_count=" << idx_count
                          << " attribs=" << p.attribute_descs_.size()
                          << " bindings=" << p.binding_descs_.size()
                          << " bbox=(" << p.bbox_min_.x << ","
                          << p.bbox_min_.y << "," << p.bbox_min_.z
                          << ")..(" << p.bbox_max_.x << ","
                          << p.bbox_max_.y << "," << p.bbox_max_.z
                          << ")  tri_count_eligible="
                          << p.mesh_shader_tri_count_
                          << std::endl;
            }
        }

        // Also list any node that owns a mesh — this confirms the
        // skinned mesh's owning node and that the node hierarchy
        // surfaced it under the scene root we wrote spawn_pos to.
        for (size_t ni = 0; ni < data.nodes_.size(); ++ni) {
            const auto& n = data.nodes_[ni];
            if (n.mesh_idx_ < 0) continue;
            std::cout << "[player.diag] node[" << ni << "] '"
                      << n.name_ << "' mesh=" << n.mesh_idx_
                      << " skin=" << n.skin_idx_
                      << " parent=" << n.parent_idx_
                      << " children=" << n.child_idx_.size()
                      << std::endl;
        }

        // ── Skin / skeleton dump ───────────────────────────────────
        // glTF skinning bypasses the mesh node's transform: rendered
        // verts come from joint.cached_matrix * inverse_bind_matrix.
        // If the skeleton joints are NOT descendants of the node
        // setRootNodeTransform is writing to, only the mesh node moves
        // (which is what our cyan bbox follows) while the actual
        // rendered geometry stays wherever the bones live.  Dump:
        //   • how many scene roots exist (we currently only translate
        //     the first one — if the skeleton is a sibling root, that
        //     explains "bbox high up, body on the floor"),
        //   • skin count and the skeleton_root node index for each,
        //   • each scene root's name + child count + has-mesh flag.
        for (size_t si = 0; si < data.scenes_.size(); ++si) {
            const auto& sc = data.scenes_[si];
            std::cout << "[player.diag] scene[" << si << "] root_nodes={";
            for (size_t k = 0; k < sc.nodes_.size(); ++k) {
                std::cout << sc.nodes_[k];
                if (k + 1 < sc.nodes_.size()) std::cout << ",";
            }
            std::cout << "}" << std::endl;
            for (auto rn_idx : sc.nodes_) {
                if (rn_idx < 0 || rn_idx >= (int)data.nodes_.size()) continue;
                const auto& rn = data.nodes_[rn_idx];
                std::cout << "[player.diag]   root_node[" << rn_idx
                          << "] '" << rn.name_ << "'"
                          << " mesh=" << rn.mesh_idx_
                          << " skin=" << rn.skin_idx_
                          << " children=" << rn.child_idx_.size()
                          << " T=(" << rn.translation_.x << ","
                          << rn.translation_.y << ","
                          << rn.translation_.z << ")"
                          << std::endl;
            }
        }
        std::cout << "[player.diag] skins=" << data.skins_.size() << std::endl;
        for (size_t si = 0; si < data.skins_.size(); ++si) {
            const auto& sk = data.skins_[si];
            std::cout << "[player.diag] skin[" << si << "] name='" << sk.name_
                      << "' skeleton_root=" << sk.skeleton_root_
                      << " joints=" << sk.joints_.size();
            // Print where the skeleton root currently sits in world
            // space — if it's at origin / far from the mesh node's
            // cached translation, that's the bug.
            if (sk.skeleton_root_ >= 0 &&
                sk.skeleton_root_ < (int)data.nodes_.size()) {
                const auto& srn = data.nodes_[sk.skeleton_root_];
                std::cout << " skeleton_root.T=(" << srn.translation_.x
                          << "," << srn.translation_.y << ","
                          << srn.translation_.z << ")"
                          << " skeleton_root.cached_T=("
                          << srn.cached_matrix_[3][0] << ","
                          << srn.cached_matrix_[3][1] << ","
                          << srn.cached_matrix_[3][2] << ")";
            }
            std::cout << std::endl;
        }

        // ── Materials & textures dump ──────────────────────────────
        // Diagnoses the "gray / untextured body" symptom by exposing
        // (a) how many materials the asset shipped with, (b) what
        // textures each material references, and (c) whether the
        // texture slots actually resolved to non-null Vulkan handles.
        // If all base_color_idx_ are -1 the asset has no albedo
        // textures (untextured rig) — that explains the gray body
        // entirely and the fix is asset-side.  If base_color_idx_ >= 0
        // but the underlying TextureInfo has no .view, the loader
        // failed to read/decode the image.
        std::cout << "[player.diag] textures=" << data.textures_.size()
                  << " materials=" << data.materials_.size() << std::endl;
        for (size_t ti = 0; ti < data.textures_.size(); ++ti) {
            const auto& tex = data.textures_[ti];
            std::cout << "[player.diag] tex[" << ti << "] size=("
                      << tex.size.x << "," << tex.size.y << "x"
                      << tex.size.z << ")"
                      << " mips=" << tex.mip_levels
                      << " has_image=" << (tex.image ? 1 : 0)
                      << " has_view=" << (tex.view  ? 1 : 0)
                      << " has_memory=" << (tex.memory ? 1 : 0)
                      << std::endl;
        }
        for (size_t mi = 0; mi < data.materials_.size(); ++mi) {
            const auto& m = data.materials_[mi];
            std::cout << "[player.diag] mat[" << mi << "] '"
                      << m.name_ << "'"
                      << " base_color_idx="     << m.base_color_idx_
                      << " normal_idx="         << m.normal_idx_
                      << " mr_idx="             << m.metallic_roughness_idx_
                      << " emissive_idx="       << m.emissive_idx_
                      << " occlusion_idx="      << m.occlusion_idx_
                      << " specular_color_idx=" << m.specular_color_idx_
                      << " alpha_mode="         << int(m.alpha_mode_)
                      << " alpha_cutoff="       << m.alpha_cutoff_
                      << " has_uniform_buffer="
                      << (m.uniform_buffer_.buffer ? 1 : 0)
                      << " has_desc_set="
                      << (m.desc_set_ ? 1 : 0)
                      << std::endl;
        }
    }

    // ── Per-frame "follow the camera" ──────────────────────────────
    // Updates the controller's world position + yaw EVERY frame so the
    // character stays anchored 2 m in front of the camera at "feet
    // height" (cam.y - kPlayerEyeOffset).  Without this block the
    // character only ever gets one shot at being placed — spawnAt fires
    // once when isReady() flips, then update() runs in stationary mode
    // and never re-derives position from the camera.  The block was
    // temporarily wrapped in `#if 0` while we chased a release-only
    // resize crash that turned out to be unrelated (hiz_pyramid
    // double-free in TextureInfo::destroy + ImGui clock_tex_id_
    // dangling-set hazards — both fixed).  Re-enabled now.
#if 1
    if (player_object_ && player_object_->isReady() &&
        player_controller_ && player_controller_->isSpawned()) {
        const auto& cam = main_camera_object_->getCameraViewInfo();
        glm::vec3 cam_pos = main_camera_object_->getCameraPosition();
        glm::vec2 fwd_xz(cam.facing_dir.x, cam.facing_dir.z);
        float fxz_len = glm::length(fwd_xz);
        if (fxz_len > 1e-3f) {
            fwd_xz /= fxz_len;
        } else {
            float yaw_rad = glm::radians(-cam.yaw);
            fwd_xz = glm::vec2(std::cos(yaw_rad), std::sin(yaw_rad));
        }
        const float kFollowAhead       = 2.0f;
        const float kPlayerEyeOffset   = 1.7f;
        glm::vec2 follow_xz =
            glm::vec2(cam_pos.x, cam_pos.z) + fwd_xz * kFollowAhead;

        // ── Pivot compensation (override-aware) ─────────────────────
        // Same approach as the spawn block: peel the OLD root frame
        // off each mesh node's cached_matrix and replace it with a
        // simulated NEW root (T=(spawn_xz.x, 0, spawn_xz.y), R=yaw,
        // S=preserved) so the measurement matches what the GPU will
        // actually render, not what the asset's authored rotation
        // currently looks like.  Without this the feet sink (or
        // float) by an amount equal to how much the authored R+T
        // moves the mesh's lowest point off the world Y axis.
        const auto& data_for_follow = player_object_->getDrawableData();
        int rn_for_follow = -1;
        if (!data_for_follow.scenes_.empty() &&
            data_for_follow.default_scene_ >= 0 &&
            data_for_follow.default_scene_ <
                (int)data_for_follow.scenes_.size() &&
            !data_for_follow.scenes_[data_for_follow.default_scene_]
                .nodes_.empty()) {
            rn_for_follow = data_for_follow
                .scenes_[data_for_follow.default_scene_].nodes_[0];
        }

        // ── Snap desired feet Y to bistro floor when available ─────
        // Same approach as the spawn block: prefer the level's
        // actual floor over the camera-eye-height heuristic.  Without
        // this the per-frame follow re-derives a feet_y purely from
        // cam.y - 1.7, which means flying the camera up reels the
        // character into the air on the very next frame even after
        // the spawn block placed her on the floor.
        float bistro_floor_y_f = -std::numeric_limits<float>::max();
        bool  have_floor_f = false;
        // Same sanity gate as the spawn block — see comment there.
        const float kFloorSanityLoF = -1e5f;
        const float kFloorSanityHiF =  1e5f;
        auto ingestFloorYFollow =
            [&](const std::shared_ptr<ego::DrawableObject>& obj) {
            if (!obj || !obj->isReady()) return;
            glm::vec3 lo = obj->getModelBboxMin();
            glm::vec3 hi = obj->getModelBboxMax();
            if (lo.x > hi.x) return;
            const glm::mat4& M = obj->getLocation();
            const glm::vec3 c[8] = {
                {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z},
                {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z},
                {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z},
                {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z},
            };
            float mn = std::numeric_limits<float>::max();
            for (int k = 0; k < 8; ++k) {
                glm::vec4 w = M * glm::vec4(c[k], 1.0f);
                if (w.y < mn) mn = w.y;
            }
            if (!std::isfinite(mn) ||
                mn < kFloorSanityLoF || mn > kFloorSanityHiF) {
                return;
            }
            if (!have_floor_f || mn > bistro_floor_y_f) {
                bistro_floor_y_f = mn;
                have_floor_f = true;
            }
        };
        ingestFloorYFollow(bistro_interior_scene_);
        if (!have_floor_f) ingestFloorYFollow(bistro_exterior_scene_);

        // Same manual nudge as the spawn block — keep in sync
        // (kManualFloorYAdjust constant defined inside the spawn
        // block scope, mirrored here so the follow path matches).
        const float kManualFloorYAdjustF = -6.0f;
        float desired_feet_y = (have_floor_f
            ? bistro_floor_y_f
            : (cam_pos.y - kPlayerEyeOffset)) + kManualFloorYAdjustF;

        // ── Attach to ground: snap the body-center feet height to the
        // walkable surface directly under the player via a downward
        // collision raycast, overriding the coarse scene-bbox / eye-height
        // estimate above.  Falls back to the estimate on miss.
        if (collision_world_built_ && !collision_world_.empty()) {
            const glm::vec3 gc_from(follow_xz.x, cam_pos.y + 0.5f,
                                    follow_xz.y);
            glm::vec3 gc_hit, gc_nrm;
            if (collision_world_.raycastDown(
                    gc_from, /*max_distance=*/200.0f, gc_hit, gc_nrm)) {
                desired_feet_y = gc_hit.y;
            }
        }

        // Default Y: camera-relative feet height (eye - kPlayerEyeOffset),
        // NOT desired_feet_y (which carries kManualFloorYAdjustF = -6 and
        // dropped her metres underground when no real ground was found).
        float follow_y = cam_pos.y - kPlayerEyeOffset;
        bool  have_follow_offset = false;

        if (rn_for_follow >= 0 &&
            rn_for_follow < (int)data_for_follow.nodes_.size()) {
            const auto& root_node =
                data_for_follow.nodes_[rn_for_follow];
            const glm::mat4& old_root_cached = root_node.cached_matrix_;

            glm::quat yaw_quat = glm::angleAxis(
                glm::radians(cam.yaw), glm::vec3(0, 1, 0));
            glm::mat4 trial_root =
                glm::translate(glm::mat4(1.0f),
                               glm::vec3(follow_xz.x, 0.0f, follow_xz.y)) *
                glm::mat4_cast(yaw_quat) *
                glm::scale(glm::mat4(1.0f), root_node.scale_);

            glm::mat4 root_old_to_new =
                trial_root * glm::inverse(old_root_cached);

            // Real-floor probe (independent of the LLM walkable
            // classification): casts a vertical ray DOWN through the ACTUAL
            // rendered scene geometry at (wx, wz) and returns the surface
            // whose Y is closest to ref_y (the expected feet level).  Used
            // only when the walkable collision world has nothing under the
            // player (the classifier missed that floor).  Per-triangle work
            // is budgeted; a world-AABB column test rejects most meshes.
            auto sceneFloorNearest =
                [&](float wx, float wz, float ref_y, float& out_y) -> bool {
                const glm::vec3 ro(wx, ref_y + 100.0f, wz);
                bool  found  = false;
                float best_y = 0.0f;
                float best_d = std::numeric_limits<float>::max();
                int   budget = 120000;
                auto probe =
                    [&](const std::shared_ptr<ego::DrawableObject>& obj) {
                    if (!obj || !obj->isReady()) return;
                    const auto& d = obj->getDrawableData();
                    for (const auto& nd : d.nodes_) {
                        if (nd.mesh_idx_ < 0 ||
                            nd.mesh_idx_ >= (int)d.meshes_.size()) continue;
                        const auto& msh = d.meshes_[nd.mesh_idx_];
                        if (!msh.vertex_position_ ||
                            msh.bbox_min_.x > msh.bbox_max_.x) continue;
                        const glm::mat4& M = nd.cached_matrix_;
                        glm::vec3 wmn(0.0f), wmx(0.0f);
                        for (int cc = 0; cc < 8; ++cc) {
                            const glm::vec3 lc(
                                (cc & 1) ? msh.bbox_max_.x : msh.bbox_min_.x,
                                (cc & 2) ? msh.bbox_max_.y : msh.bbox_min_.y,
                                (cc & 4) ? msh.bbox_max_.z : msh.bbox_min_.z);
                            const glm::vec3 w =
                                glm::vec3(M * glm::vec4(lc, 1.0f));
                            if (cc == 0) { wmn = wmx = w; }
                            else { wmn = glm::min(wmn, w); wmx = glm::max(wmx, w); }
                        }
                        if (wx < wmn.x || wx > wmx.x ||
                            wz < wmn.z || wz > wmx.z) continue;
                        if (ro.y < wmn.y) continue;
                        if (budget <= 0) return;
                        const glm::mat4 invM = glm::inverse(M);
                        const glm::vec3 rol =
                            glm::vec3(invM * glm::vec4(ro, 1.0f));
                        const glm::vec3 rdl =
                            glm::vec3(invM * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
                        const auto& P = *msh.vertex_position_;
                        for (const auto& prim : msh.primitives_) {
                            if (!prim.vertex_indices_) continue;
                            const auto& I = *prim.vertex_indices_;
                            for (size_t k = 0; k + 2 < I.size(); k += 3) {
                                if (--budget <= 0) break;
                                const size_t a  = (size_t)I[k + 0];
                                const size_t b  = (size_t)I[k + 1];
                                const size_t ci = (size_t)I[k + 2];
                                if (a >= P.size() || b >= P.size() ||
                                    ci >= P.size()) continue;
                                const glm::vec3 e1 = P[b] - P[a];
                                const glm::vec3 e2 = P[ci] - P[a];
                                const glm::vec3 pv = glm::cross(rdl, e2);
                                const float det = glm::dot(e1, pv);
                                const float ad = det < 0.0f ? -det : det;
                                if (ad < 1e-12f) continue;
                                const float inv = 1.0f / det;
                                const glm::vec3 tv = rol - P[a];
                                const float u = glm::dot(tv, pv) * inv;
                                if (u < 0.0f || u > 1.0f) continue;
                                const glm::vec3 qv = glm::cross(tv, e1);
                                const float vb = glm::dot(rdl, qv) * inv;
                                if (vb < 0.0f || u + vb > 1.0f) continue;
                                const float t = glm::dot(e2, qv) * inv;
                                if (t <= 0.0f) continue;
                                const float wy = ro.y - t;     // world hit Y
                                const float dd =
                                    wy > ref_y ? wy - ref_y : ref_y - wy;
                                if (dd < best_d) {
                                    best_d = dd; best_y = wy; found = true;
                                }
                            }
                        }
                    }
                };
                probe(bistro_exterior_scene_);
                probe(bistro_interior_scene_);
                for (auto& dr : drawable_objects_) probe(dr);
                if (found) out_y = best_y;
                return found;
            };

            // ── Plant the feet on the ground (per-foot) ──────────────
            // For each foot bone, find its world XZ in the follow frame and
            // raycast the walkable collision world straight DOWN under it
            // (the SAME per-foot ground the red foot markers snap to), then
            // compute the root height that puts that foot's SOLE on its own
            // ground.  Take the MAX so the LOWER foot plants and neither
            // foot sinks.  Priority: (1) per-foot raycast hit -> real ground;
            // (2) no hit -> camera-relative feet height (eye - 1.7 m) via the
            // lower foot -- NO scene-bbox + (-6) fudge (that put her under-
            // ground).  kSoleDrop = ankle-bone -> sole distance.
            constexpr float kSoleDrop = 0.08f;
            float plant_follow_y   = -std::numeric_limits<float>::max();
            float min_sole_at_zero =  std::numeric_limits<float>::max();
            int   feet_found = 0, feet_planted = 0;
            for (size_t ni = 0; ni < data_for_follow.nodes_.size(); ++ni) {
                const auto& n = data_for_follow.nodes_[ni];
                if (n.name_ != "left_foot" && n.name_ != "right_foot")
                    continue;
                ++feet_found;
                const glm::mat4 sim = root_old_to_new * n.cached_matrix_;
                const float sole_at_zero = sim[3].y - kSoleDrop;
                if (sole_at_zero < min_sole_at_zero)
                    min_sole_at_zero = sole_at_zero;
                glm::vec3 gh, gn;
                bool hit = false;
                // Tiers 1-2: walkable collision world (fast, BVH) when ready
                // AND non-empty.  Tier 1 starts just above the camera (finds
                // the floor she is near, won't grab an upper storey indoors);
                // tier 2 starts HIGH and only runs if tier 1 misses, to
                // recover her when she's below the floor.
                if (collision_world_built_ && !collision_world_.empty()) {
                    hit = collision_world_.raycastDown(
                        glm::vec3(sim[3].x, cam_pos.y + 2.0f, sim[3].z),
                        200.0f, gh, gn);
                    if (!hit) {
                        hit = collision_world_.raycastDown(
                            glm::vec3(sim[3].x, cam_pos.y + 80.0f, sim[3].z),
                            200.0f, gh, gn);
                    }
                }
                // Tier 3: the ACTUAL rendered floor -- ALWAYS tried when the
                // collision world found nothing.  Crucially this now runs even
                // when the walkable collision world is EMPTY (the LLM-only
                // classifier may tag nothing WALKABLE_SURFACE) -- that gate
                // was previously skipping this fallback and leaving the feet
                // ungrounded.
                if (!hit) {
                    float sy;
                    if (sceneFloorNearest(sim[3].x, sim[3].z,
                                          cam_pos.y - kPlayerEyeOffset, sy)) {
                        gh.y = sy; hit = true;
                    }
                }
                if (hit) {
                    const float req = gh.y - sole_at_zero;
                    if (req > plant_follow_y) plant_follow_y = req;
                    ++feet_planted;
                }
            }

            if (feet_planted > 0) {
                // A foot found real walkable ground -> plant the lower
                // foot's sole on it.
                follow_y = plant_follow_y;
                have_follow_offset = true;
            } else if (feet_found > 0) {
                // No walkable collision surface under the feet (that floor
                // isn't classified WALKABLE_SURFACE, or she's over an edge).
                // Fall back to the CAMERA-relative feet height referenced
                // off the lower foot -- NEVER the scene-bbox + (-6) estimate
                // that placed her metres underground.
                const float cam_feet_y = cam_pos.y - kPlayerEyeOffset;
                follow_y = cam_feet_y - min_sole_at_zero;
                have_follow_offset = true;
            }
            // (No foot bones -> follow_y keeps the camera-relative default.)

            // Diagnostic (once/sec): planted=0 with feet>0 means NO walkable
            // collision surface under the feet (LLM didn't classify that
            // floor, or she's over an edge) -> feet use the coarse estimate.
            {
                static uint64_t s_plant_log = 0;
                if ((s_plant_log++ % 60u) == 0u) {
                    std::cout << "[foot_plant] feet=" << feet_found
                              << " planted=" << feet_planted
                              << " follow_y=" << follow_y
                              << " desired_feet_y=" << desired_feet_y
                              << " cw_built=" << (collision_world_built_ ? 1 : 0)
                              << " cw_meshes=" << collision_world_.meshCount()
                              << std::endl;
                }
            }
        }

        // Last-resort guard: don't propagate Inf/NaN/FLT_MAX into the
        // controller — once it lands there, setRootNodeTransform
        // bakes it into every cached_matrix and the bbox blows up to
        // FLT_MAX for every subsequent frame.  Hold the previous
        // position instead (no-op the controller write this frame).
        const bool follow_y_ok =
            std::isfinite(follow_y) && std::abs(follow_y) <= 1e5f;
        glm::vec3 follow_pos(follow_xz.x, follow_y, follow_xz.y);
        if (follow_y_ok) {
            player_controller_->setPositionAndYaw(follow_pos, cam.yaw);
        } else {
            std::cout << "[follow] follow_y rejected as garbage ("
                      << follow_y << "), holding previous position"
                      << std::endl;
        }

        // Per-second diagnostic for the follow block.  Compare these
        // numbers against [player.live] root_T (printed in the same
        // frame just after player_object_->update).  If cam_pos
        // changes when you move the camera but root_T stays still,
        // setPositionAndYaw isn't reaching applyPose (or applyPose
        // isn't reaching cached_matrix).  If cam_pos itself doesn't
        // change, the camera input isn't being read on the host side
        // and we're chasing a CPU-vs-GPU camera-sync issue.
        static uint64_t s_follow_frame = 0;
        if ((s_follow_frame++ % 60u) == 0u) {
            std::cout << "[follow] cam_pos=(" << cam_pos.x << ","
                      << cam_pos.y << "," << cam_pos.z
                      << ")  fwd=(" << cam.facing_dir.x << ","
                      << cam.facing_dir.y << "," << cam.facing_dir.z
                      << ")  follow_pos=(" << follow_pos.x << ","
                      << follow_pos.y << "," << follow_pos.z
                      << ")  yaw=" << cam.yaw
                      << "  applied=" << (follow_y_ok ? 1 : 0)
                      << std::endl;
        }
    }
#endif // 1 — follow block re-enabled after resize crash root-cause fixed

    // Build the collision world once both bistro scenes have actually
    // populated their CPU-side mesh data. We retry every frame in
    // InGame state until at least one mesh is added so that any
    // transient "isReady() == true but vertex_position_ not yet
    // populated" race resolves itself instead of leaving the world
    // permanently empty.
    // If the user changed the collision-shape selector in the menu,
    // tear down the existing CollisionWorld so the build retry
    // below rebuilds with the new shape. waitIdle is needed because
    // CollisionDebugMeshBuffers (the GPU-side companion) may still
    // be referenced by an in-flight command buffer; destroying its
    // buffers without synchronisation is a Vulkan validation error.
    // ── Incremental collision-build state (persists across frames) ───
    // The collision world is generated a batch of primitives PER FRAME
    // (not all at once) so the render thread never hitches and the
    // second progress bar can animate.  These statics carry the build
    // across drawScene calls; they are RESET in the dirty block below
    // (and re-enumerated lazily) so a shape-change rebuild starts clean.
    //   s_coll_work            — flattened (drawable, mesh, prim) list.
    //   s_coll_cursor          — index of the next work item to build.
    //   s_coll_work_enumerated — true once the list has been populated.
    //   s_coll_cat_counts / _unknown_sample — classification histogram,
    //                            accumulated across the WHOLE build.
    //   s_coll_attempted / _built / _total_tris — build coverage stats.
    struct CollisionBuildItem {
        std::shared_ptr<ego::DrawableObject> obj;
        size_t mesh_idx;
        size_t prim_idx;
    };
    static std::vector<CollisionBuildItem> s_coll_work;
    static size_t                          s_coll_cursor = 0;
    static bool                            s_coll_work_enumerated = false;
    static std::array<int, 11>             s_coll_cat_counts{};
    static std::vector<std::string>        s_coll_unknown_sample;
    static size_t                          s_coll_attempted  = 0;
    static size_t                          s_coll_built      = 0;
    static size_t                          s_coll_total_tris = 0;
    // Clears the work list + accumulators so the next build starts from
    // scratch.  Referenced by the dirty block below.
    auto resetCollisionBuild = [&]() {
        s_coll_work.clear();
        s_coll_cursor = 0;
        s_coll_work_enumerated = false;
        s_coll_cat_counts = {};
        s_coll_unknown_sample.clear();
        s_coll_attempted = s_coll_built = s_coll_total_tris = 0;
        if (menu_) menu_->setCollisionBuildStatus(
            engine::ui::Menu::CollisionBuildStatus::Idle, 0, 0, 0);
    };

    if (menu_->collisionWorldDirty()) {
        device_->waitIdle();
        collision_world_.destroyDebugBuffers(device_);
        collision_world_.clear();
        collision_world_built_ = false;
        resetCollisionBuild();   // restart the incremental build clean
        menu_->clearCollisionWorldDirty();
    }

    // ── Async LLM material classifier — function-scope state ─────────
    // These statics live across the one-shot collision-world build
    // block AND the per-frame poll block below.  Declared here at
    // drawScene function scope so both blocks see the same storage
    // (a `static` inside one `{}` wouldn't be visible to the other).
    //
    //   s_mat_classifier         — owns the collected name set and the
    //                              eventual classified map.
    //   s_mat_classifier_future  — the std::async handle for the
    //                              background classifyAll() call.
    //   s_mat_classifier_kicked  — gates re-entry into the kick-off
    //                              block; once true the worker has
    //                              been started (or already finished).
    static engine::helper::MaterialClassifier s_mat_classifier;
    static std::future<bool>                  s_mat_classifier_future;
    static bool                               s_mat_classifier_kicked  = false;
    //   s_mat_classifier_applied — latched true by the poll block the
    //                              first frame classifyAll()'s future is
    //                              ready (and consumed via .get()).  The
    //                              collision-world build is GATED on this
    //                              so each primitive's Floor verdict
    //                              reflects the finished AI classifier
    //                              instead of the substring fallback.
    //                              Hoisted here (was poll-block-local) so
    //                              the build block above can read it too.
    static bool                               s_mat_classifier_applied = false;

    // ── CPU profile: "Collision world build" ────────────────────────
    // INCREMENTAL CPU work — once the classifier has landed this lane
    // shows up for a handful of consecutive frames, each capped at the
    // ~6ms time budget below (kBuildBudgetMs), as the world is built a
    // batch of primitives per frame.  The lane goes quiet for good once
    // collision_world_built_ latches.  If it keeps spiking forever the
    // finalise branch (build_done) isn't being reached — check that the
    // work list enumerated (drawables ready) and the cursor advances.
    if (!collision_world_built_ &&
        menu_->getGameState() == engine::ui::GameState::InGame) {
        auto _cpu_coll = gpu_profiler_.beginCpuScope("Collision world build");

        // Map the menu's user-facing CollisionDebugShape onto the
        // engine's CollisionShape -- the menu deliberately exposes
        // a curated 3-option subset (Original / Simplified / Volume)
        // instead of the full enum so the UI stays uncluttered.
        engine::helper::CollisionShape shape =
            engine::helper::CollisionShape::Decimate;
        const char* shape_label = "gap-filled simplified";
        switch (menu_->collisionDebugShape()) {
        case engine::ui::Menu::CollisionDebugShape::Original:
            shape = engine::helper::CollisionShape::None;
            shape_label = "original mesh";
            break;
        case engine::ui::Menu::CollisionDebugShape::Simplified:
            shape = engine::helper::CollisionShape::Decimate;
            shape_label = "gap-filled simplified";
            break;
        case engine::ui::Menu::CollisionDebugShape::Volume:
            shape = engine::helper::CollisionShape::VoxelCube;
            shape_label = "5cm voxel cubes";
            break;
        }

        // ── Pass 1: AI-backed classifier (materials + objects) ────
        // KICK OFF only — actual HTTP call runs on a background
        // thread.  On a 3B model with bistro's ~500 names the
        // classify takes minutes on CPU, which is way too long to
        // block the render thread.  We snapshot every (material,
        // albedo, object) name into the classifier synchronously
        // (fast — pure walk of in-memory data), then std::async the
        // classifyAll() and let the game keep rendering.
        //
        // TEST CHANGE: the collision build below is now DEFERRED until
        // the async classifier has finished — it is gated on
        // s_mat_classifier_applied, which the poll block latches the
        // first frame classifyAll()'s future is ready.  We wait so each
        // primitive's Floor verdict comes from the finished AI
        // classifier rather than the substring fallback.  Consequence:
        // there is NO collision world (no foot IK / capsule physics /
        // collision overlay) until the classifier lands.  That's
        // acceptable for this floor-only test.  If classifyAll() fails
        // (daemon down / model not pulled), the poll block STILL latches
        // applied, so the build proceeds using the substring + geometric
        // categories instead of blocking forever.
        //
        // s_mat_classifier / s_mat_classifier_future / s_mat_classifier_kicked
        // are declared at drawScene function scope above so the poll
        // block further down (which lives in a sibling brace scope)
        // can see the same storage.

        auto collectFromDrawable = [&](
            const std::shared_ptr<ego::DrawableObject>& obj) {
            if (!obj || !obj->isReady()) return;
            const auto& data = obj->getDrawableData();

            // Materials side: walk the material table once.
            for (const auto& m : data.materials_) {
                std::string albedo;
                if (m.base_color_idx_ >= 0 &&
                    static_cast<size_t>(m.base_color_idx_) <
                        data.textures_.size()) {
                    albedo = data.textures_[m.base_color_idx_]
                                .source_filename_;
                }
                // Use the three-arg overload with an empty object
                // name for the material-only side; the classifier
                // accepts empty fields and skips them.
                s_mat_classifier.collect(m.name_, albedo, std::string());
            }
            // Objects side: walk the node table once.  Only nodes
            // that reference a real mesh produce CollisionMeshes,
            // so we filter by mesh_idx_ >= 0 to keep skeleton /
            // group nodes out of the classifier payload.
            for (const auto& node : data.nodes_) {
                if (node.mesh_idx_ < 0 || node.name_.empty()) continue;
                s_mat_classifier.collect(
                    std::string(), std::string(), node.name_);
            }
        };
        if (!s_mat_classifier_kicked) {
            collectFromDrawable(bistro_exterior_scene_);
            collectFromDrawable(bistro_interior_scene_);
            std::cout << "[collision] kicking off async LLM classifier "
                      << "(mats=" << s_mat_classifier.collectedMaterialCount()
                      << " objs=" << s_mat_classifier.collectedObjectCount()
                      << "); collision build is DEFERRED until the "
                         "classifier finishes (floor-only test)"
                      << std::endl;
            s_mat_classifier_future = std::async(
                std::launch::async, []() {
                    return s_mat_classifier.classifyAll("Bistro");
                });
            s_mat_classifier_kicked = true;
        }

        // One CollisionMesh PER (mesh, primitive) pair so each
        // material part paints in its own segmentation colour AND
        // carries its own MaterialInfo::name_ for gameplay surface
        // lookups (footstep sounds, friction, etc.). build_bvh is
        // false here -- the synchronous BVH builder runs an
        // O(N log N) multithreaded pass that on Bistro-sized input
        // would freeze the render thread on the first in-game
        // frame. Instead we kick off CollisionWorld::buildBVHsAsync
        // below, which builds every per-mesh BVH on a background
        // thread; foot raycasts / capsule physics fall back to
        // brute-force (raycastDown) or no-op (resolveCapsule) per
        // mesh whose BVH isn't ready yet, then transparently switch
        // to BVH-accelerated descent once each mesh's build lands.
        // Per-category histogram + a short sample of "Unknown" material
        // names live in the s_coll_* statics (declared above) so they
        // accumulate across the whole incremental build and the final
        // [collision.cat] summary covers exterior + interior together.
        // The category enum has 11 values (Unknown..Ladder) -- see
        // MeshCategory in collision_mesh.h.
        constexpr size_t kUnknownSampleMax = 8;

        // GATED on the ML classifier finishing: the simplified collision
        // meshes are generated only AFTER the material categories have been
        // computed, so every primitive's Floor verdict reflects the finished
        // ML classifier (the floor name-rule in buildOne still WINS on top of
        // it).  s_mat_classifier_applied is latched true by the poll block
        // the first frame classifyAll()'s future is ready (and stays false
        // until then), so the enumerate + build below simply doesn't run
        // while the classifier is still working its way through the 511
        // material/object names.  (If classifyAll() fails outright the poll
        // block STILL latches applied, so the build proceeds on the geometric
        // + name categories instead of blocking forever.)
        //
        // NOTE: this deferral is intentional -- it is NOT what blanked the
        // overlay earlier; that was a truncated cleanup() tail that stopped
        // the whole file compiling.  With the file fixed, the overlay simply
        // populates once the ML pass (orange progress bar) completes.
        if (s_mat_classifier_applied) {
            // ── Enumerate the (drawable, mesh, primitive) work list
            //    ONCE — the first frame the classifier is done AND the
            //    drawables are ready.  While the scenes are still
            //    streaming in the list comes back empty, so we leave
            //    s_coll_work_enumerated false and retry next frame
            //    instead of latching an empty collision world (this is
            //    the incremental-build replacement for the old
            //    any_added retry guard).
            if (!s_coll_work_enumerated) {
                s_coll_work.clear();
                auto enumerateDrawable =
                    [&](const std::shared_ptr<ego::DrawableObject>& obj) {
                        if (!obj || !obj->isReady()) return;
                        const auto& data = obj->getDrawableData();
                        for (size_t mi = 0; mi < data.meshes_.size(); ++mi) {
                            const auto& mesh = data.meshes_[mi];
                            for (size_t pi = 0;
                                 pi < mesh.primitives_.size(); ++pi) {
                                s_coll_work.push_back({obj, mi, pi});
                            }
                        }
                    };
                enumerateDrawable(bistro_exterior_scene_);
                enumerateDrawable(bistro_interior_scene_);
                if (!s_coll_work.empty()) {
                    s_coll_cursor = 0;
                    s_coll_cat_counts = {};
                    s_coll_unknown_sample.clear();
                    s_coll_attempted = s_coll_built = s_coll_total_tris = 0;
                    s_coll_work_enumerated = true;
                    std::cout << "[collision] incremental build start: "
                              << s_coll_work.size()
                              << " primitive(s) queued (" << shape_label
                              << ")" << std::endl;
                }
            }

            // ── Build ONE work item.  Same per-primitive logic as the
            //    old single-frame loop body, factored out so the
            //    time-budgeted driver below can call it one item at a
            //    time across frames.
            auto buildOne = [&](const CollisionBuildItem& it) {
                const auto& obj = it.obj;
                if (!obj || !obj->isReady()) return;
                ++s_coll_attempted;

                // ── Resolve material + object NAME first, WITHOUT building ──
                // "Simplify AFTER material category": the kept-category
                // decision is the LLM classifier's verdict on the (material,
                // object) names, so we make it BEFORE paying for the
                // expensive weld + QEM decimation.  Only meshes we keep
                // (walkable surface) get simplified; every other primitive is
                // tallied for the [collision.cat] histogram and skipped.
                const auto& data = obj->getDrawableData();
                if (it.mesh_idx >= data.meshes_.size()) return;
                const auto& mesh = data.meshes_[it.mesh_idx];
                if (it.prim_idx >= mesh.primitives_.size()) return;
                const auto& prim = mesh.primitives_[it.prim_idx];

                std::string mat_name;
                if (prim.material_idx_ >= 0 &&
                    static_cast<size_t>(prim.material_idx_) <
                        data.materials_.size()) {
                    mat_name = data.materials_[prim.material_idx_].name_;
                }
                std::string obj_name;
                for (const auto& node : data.nodes_) {
                    if (node.mesh_idx_ == static_cast<int32_t>(it.mesh_idx)) {
                        obj_name = node.name_;
                        break;
                    }
                }

                // ── Decide the category via the LLM classifier ──
                engine::helper::MeshCategory cat =
                    engine::helper::MeshCategory::Unknown;
                // LLM verdict — GUARDED on s_mat_classifier_applied so we only
                // READ the classifier map once classifyAll() has finished on
                // its background thread (never race the writer).
                if (s_mat_classifier_applied) {
                    const auto llm_cat =
                        s_mat_classifier.lookup(mat_name, obj_name);
                    if (llm_cat != engine::helper::MeshCategory::Unknown)
                        cat = llm_cat;
                }
                // The walkable-surface verdict is the LLM classifier's
                // ALONE.  The old deterministic material-name keyword rule
                // (Pavement / Road / Floor / Ground / ...) was removed on
                // request so the category is decided by the model, never by
                // string analysis.  Trade-off: if the LLM mislabels a ground
                // material, that floor piece is dropped (watch
                // [collision.excluded]) -- fix it by improving the prompt or
                // re-running the classifier, NOT by re-adding keyword matching.
                // If the classifier never ran (Ollama down), cat stays
                // Unknown and nothing is kept.

                // Tally EVERY primitive's category into the histogram BEFORE
                // the floor-only filter so [collision.cat] still reflects the
                // full classification even though only Floor meshes are built.
                {
                    const auto idx = static_cast<size_t>(cat);
                    if (idx < s_coll_cat_counts.size())
                        ++s_coll_cat_counts[idx];
                    if (cat == engine::helper::MeshCategory::Unknown &&
                        s_coll_unknown_sample.size() < kUnknownSampleMax &&
                        !mat_name.empty()) {
                        s_coll_unknown_sample.push_back(mat_name);
                    }
                }

                // Non-Floor: do NOT simplify.  Log the excluded material once
                // (worklist for the name-rule) and skip the build entirely.
                if (cat != engine::helper::MeshCategory::Floor) {
                    static std::vector<std::string> s_excluded_seen;
                    if (!mat_name.empty()) {
                        bool seen = false;
                        for (const auto& s : s_excluded_seen) {
                            if (s == mat_name) { seen = true; break; }
                        }
                        if (!seen) {
                            s_excluded_seen.push_back(mat_name);
                            std::cout << "[collision.excluded] cat="
                                      << static_cast<int>(cat)
                                      << " mat='" << mat_name
                                      << "' obj='" << obj_name << "'"
                                      << std::endl;
                        }
                    }
                    return;
                }

                // ── Floor: NOW build + simplify (weld + QEM decimate) ──
                // Simplification happens only AFTER the material category is
                // settled, and only for the meshes we keep.
                //
                // INDEX PARITY: a Floor primitive must occupy the SAME index
                // in the collision world no matter which debug shape is
                // selected, so the isolate-index slider points at the SAME
                // floor piece whether you are viewing Original or Simplified.
                // The candidate set is already shape-independent (category is
                // decided by material NAME above, not by geometry).  To make
                // the KEPT set shape-independent too: if the requested shape's
                // build drops this primitive (e.g. decimation degenerates a
                // sliver to <3 tris), fall back to the un-simplified (None)
                // geometry so the entry still EXISTS at the same position.
                // Net: Simplified enumerates exactly the same floor pieces as
                // Original -- only their triangle density differs.
                auto m = std::make_shared<engine::helper::CollisionMesh>();
                bool built = m->buildFromDrawablePrimitive(
                    *obj, it.mesh_idx, it.prim_idx,
                    /*build_bvh=*/false,
                    shape,
                    // Sub-mm weld (NOT the 0.1m default, NOT zero): merges
                    // only genuinely-coincident vertices so the QEM pass can
                    // lock the true outer edge and simplify the interior
                    // without holes.  10cm deleted real triangles (holes); 0
                    // over-locked split-vertex floors so they never simplified.
                    /*weld_eps=*/1.0e-3f);
                if (!built &&
                    shape != engine::helper::CollisionShape::None) {
                    // Requested shape dropped it -- keep the raw welded mesh so
                    // Simplified and Original stay index-aligned.
                    m = std::make_shared<engine::helper::CollisionMesh>();
                    built = m->buildFromDrawablePrimitive(
                        *obj, it.mesh_idx, it.prim_idx,
                        /*build_bvh=*/false,
                        engine::helper::CollisionShape::None,
                        /*weld_eps=*/1.0e-3f);
                }
                if (built) {
                    s_coll_total_tris += m->triangleCount();
                    ++s_coll_built;
                    // Enforce the name/LLM Floor verdict over whatever the
                    // geometric classifier voted inside the build (floor
                    // primitives with reversed/mixed winding can otherwise
                    // come back mis-voted as Wall/Ceiling).
                    m->setCategory(engine::helper::MeshCategory::Floor);
                    collision_world_.addMesh(std::move(m));
                }
            };

            // ── Time-budgeted slice: build work items until ~6ms of
            //    wall clock has elapsed this frame, then resume next
            //    frame.  Always advances at least one item so a single
            //    expensive primitive can't stall the build.  Runs only
            //    once the work list has actually been enumerated.
            if (s_coll_work_enumerated) {
                const auto _build_t0 = std::chrono::steady_clock::now();
                constexpr double kBuildBudgetMs = 6.0;
                while (s_coll_cursor < s_coll_work.size()) {
                    buildOne(s_coll_work[s_coll_cursor]);
                    ++s_coll_cursor;
                    const double ms =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - _build_t0)
                        .count();
                    if (ms >= kBuildBudgetMs) break;
                }

                // Push progress to the second (collision) bar every frame.
                const bool build_done =
                    (s_coll_cursor >= s_coll_work.size());
                menu_->setCollisionBuildStatus(
                    build_done
                        ? engine::ui::Menu::CollisionBuildStatus::Done
                        : engine::ui::Menu::CollisionBuildStatus::Building,
                    s_coll_cursor,
                    s_coll_work.size(),
                    collision_world_.meshCount());

                // Finalise once every queued primitive has been processed.
                if (build_done) {
                    collision_world_built_ = true;
                    std::cout << "[collision] World ready: "
                              << collision_world_.meshCount()
                              << " mesh(es) total (" << s_coll_built << "/"
                              << s_coll_attempted << " prims built, "
                              << s_coll_total_tris << " tris)" << std::endl;
                    // Category histogram + Unknown sample (colour key:
                    // Floor green, Wall red, Door amber, Object purple,
                    // Glass cyan, Unknown grey).
                    std::cout << "[collision.cat] Unknown=" << s_coll_cat_counts[0]
                              << " Floor="      << s_coll_cat_counts[1]
                              << " Wall="       << s_coll_cat_counts[2]
                              << " Door="       << s_coll_cat_counts[3]
                              << " Object="     << s_coll_cat_counts[4]
                              << " Glass="      << s_coll_cat_counts[5]
                              << " Ceiling="    << s_coll_cat_counts[6]
                              << " Stairs="     << s_coll_cat_counts[7]
                              << " Vegetation=" << s_coll_cat_counts[8]
                              << " Elevator="   << s_coll_cat_counts[9]
                              << " Ladder="     << s_coll_cat_counts[10]
                              << std::endl;
                    if (!s_coll_unknown_sample.empty()) {
                        std::cout << "[collision.cat] Unknown sample materials:";
                        for (const auto& s : s_coll_unknown_sample) {
                            std::cout << " \"" << s << "\"";
                        }
                        std::cout << std::endl;
                    }
                    // Kick off per-mesh SAH BVH construction on a
                    // background thread (non-blocking — foot raycasts use
                    // brute force per mesh until that mesh's BVH lands).
                    collision_world_.buildBVHsAsync();
                    std::cout << "[collision] BVH async build started"
                              << std::endl;
                }
            }
        }
        gpu_profiler_.endCpuScope(_cpu_coll);
    }

    // ── Async LLM classifier polling ─────────────────────────────────
    // Runs EVERY frame (NOT gated on collision_world_built_) so it
    // can fire after the one-shot collision build is done.  Reads
    // the s_mat_classifier* statics declared at drawScene function
    // scope above; both blocks see the same storage.
    //
    // First time the future is ready we pop the result, call
    // applyMaterialCategories on the cluster renderer (patches the
    // per-material flag bits so DEBUG_RENDER_MODE_CATEGORY paints in
    // colour instead of grey), and latch s_mat_classifier_applied so
    // the block becomes a single byte-load for the rest of the
    // session.
    {
        // s_mat_classifier_applied is now declared at drawScene function
        // scope (above) so the collision-build block can gate on it.
        // Wall-clock anchor for the progress heartbeat below.  Set on
        // the first frame the worker is detected pending, then used
        // to throttle status prints to one every 5 s — enough cadence
        // for the user to confirm the daemon's still cranking,
        // infrequent enough to not flood the console.
        static std::chrono::steady_clock::time_point s_mat_classifier_started{};
        static std::chrono::steady_clock::time_point s_mat_classifier_last_log{};
        // Push the always-on banner state every frame so the menu
        // doesn't have to walk the classifier itself.  Idle until
        // kicked; Pending while the worker runs; Ready / Failed
        // latched once the verdict is in.  Using `menu_` directly
        // here is safe — the banner setter is a single inline assign,
        // no Vulkan / no descriptor churn.
        if (menu_) {
            using S = engine::ui::Menu::ClassifierStatus;
            S status = S::Idle;
            float elapsed_s = 0.0f;
            int mats = 0, objs = 0;
            size_t bytes_received = 0;
            size_t bytes_total    = 0;
            if (s_mat_classifier_kicked) {
                // Use the kick-off anchor (set when status was first
                // observed pending below) if it's been initialised;
                // otherwise the just-kicked frame shows 0s elapsed.
                if (s_mat_classifier_started.time_since_epoch().count() != 0) {
                    elapsed_s = std::chrono::duration<float>(
                        std::chrono::steady_clock::now() -
                        s_mat_classifier_started).count();
                }
                if (s_mat_classifier_applied) {
                    // Ready vs Failed isn't preserved across frames
                    // by the existing logic, so use the simple
                    // heuristic: any classifications present = ready,
                    // none = failed.  classifyAll() clears both maps
                    // on failure so this is unambiguous.
                    mats = static_cast<int>(
                        s_mat_classifier.classifiedMaterialCount());
                    objs = static_cast<int>(
                        s_mat_classifier.classifiedObjectCount());
                    status = (mats + objs > 0) ? S::Ready : S::Failed;
                } else {
                    mats = static_cast<int>(
                        s_mat_classifier.collectedMaterialCount());
                    objs = static_cast<int>(
                        s_mat_classifier.collectedObjectCount());
                    status = S::Pending;
                }

                // Live item-count progress — with batching we KNOW
                // both the total names submitted and how many have
                // been classified so far, so the bar is true
                // progress rather than a byte estimate.  The
                // `bytes_*` parameters retain their names for
                // backward compat but now carry (classified / total
                // items) — the menu overlay text was updated to
                // say "items" instead of "bytes".
                bytes_received =
                    s_mat_classifier.classifiedMaterialCount() +
                    s_mat_classifier.classifiedObjectCount();
                bytes_total =
                    s_mat_classifier.collectedMaterialCount() +
                    s_mat_classifier.collectedObjectCount();
                if (bytes_total == 0) bytes_total = 1;  // avoid /0
            }
            menu_->setClassifierStatus(
                status, elapsed_s, mats, objs,
                bytes_received, bytes_total);
        }

        // ── Incremental snapshot push ────────────────────────────
        // The worker classifies in batches (default 10/req) and
        // bumps mat_/obj_classified_count_ after each batch lands.
        // Every frame here we check whether the totals grew; if
        // so, copy the classifier's classified maps under its
        // mutex (snapshotClassified) and push the result to the
        // menu so the Mesh-Category Inspector window populates
        // progressively instead of waiting for the entire run.
        // No-op while the totals are flat — no allocations, no
        // sort, no menu push.
        static size_t s_last_snapshot_total = 0;
        if (menu_ && s_mat_classifier_kicked) {
            const size_t cur_total =
                s_mat_classifier.classifiedMaterialCount() +
                s_mat_classifier.classifiedObjectCount();
            if (cur_total > s_last_snapshot_total) {
                engine::helper::MaterialClassifier::TagMap mats_snap;
                engine::helper::MaterialClassifier::TagMap objs_snap;
                s_mat_classifier.snapshotClassified(
                    mats_snap, objs_snap);
                std::vector<std::pair<std::string, uint32_t>> mats_vec;
                std::vector<std::pair<std::string, uint32_t>> objs_vec;
                mats_vec.reserve(mats_snap.size());
                objs_vec.reserve(objs_snap.size());
                for (const auto& [name, cat] : mats_snap) {
                    mats_vec.emplace_back(
                        name, static_cast<uint32_t>(cat));
                }
                for (const auto& [name, cat] : objs_snap) {
                    objs_vec.emplace_back(
                        name, static_cast<uint32_t>(cat));
                }
                menu_->setMaterialCategorySnapshot(
                    std::move(mats_vec), std::move(objs_vec));
                s_last_snapshot_total = cur_total;
            }
        }

        if (s_mat_classifier_kicked &&
            !s_mat_classifier_applied &&
            s_mat_classifier_future.valid()) {
            const auto status = s_mat_classifier_future.wait_for(
                std::chrono::seconds(0));
            if (status != std::future_status::ready) {
                // Pending — emit a heartbeat every 5 s of wall clock.
                const auto now = std::chrono::steady_clock::now();
                if (s_mat_classifier_started.time_since_epoch().count() == 0) {
                    s_mat_classifier_started = now;
                    s_mat_classifier_last_log = now;
                }
                if (now - s_mat_classifier_last_log >=
                    std::chrono::seconds(5)) {
                    const auto elapsed_s =
                        std::chrono::duration_cast<std::chrono::seconds>(
                            now - s_mat_classifier_started).count();
                    std::cout
                        << "[mat.cls] still waiting (" << elapsed_s
                        << "s elapsed, mats="
                        << s_mat_classifier.collectedMaterialCount()
                        << " objs="
                        << s_mat_classifier.collectedObjectCount()
                        << " pending)" << std::endl;
                    s_mat_classifier_last_log = now;
                }
            }
            if (status == std::future_status::ready) {
                const bool ok = s_mat_classifier_future.get();
                std::cout
                    << "[collision] async LLM classifier "
                    << (ok ? "succeeded" : "fell back")
                    << " (classified mats="
                    << s_mat_classifier.classifiedMaterialCount()
                    << " objs="
                    << s_mat_classifier.classifiedObjectCount()
                    << "); patching cluster renderer..." << std::endl;
                if (cluster_renderer_) {
                    cluster_renderer_->applyMaterialCategories(
                        s_mat_classifier);
                }

                // Dump the full classification table to a timestamped
                // log file so the user has a per-name verdict to grep
                // through.  511 lines is too long for stdout but
                // makes a perfect offline artefact -- pair it with
                // the histogram in [collision.cat] to see exactly
                // which materials drove each colour band in the
                // Mesh-Category overlay.
                {
                    // Always write a file — even when the future
                    // resolved to a "failed" verdict (HTTP error,
                    // empty maps).  A zero-row dump is still a useful
                    // breadcrumb: it confirms the dump code ran, and
                    // the user can grep for the timestamped filename
                    // to verify cwd.  Path is absolutised so the
                    // confirmation log line tells the user exactly
                    // where on disk to look — relative paths landing
                    // in an unexpected cwd were the main source of
                    // "where is the file?" confusion.
                    std::error_code ec;
                    std::filesystem::create_directories("logs", ec);
                    auto now_t = std::chrono::system_clock::to_time_t(
                        std::chrono::system_clock::now());
                    std::tm tm_local{};
                    localtime_s(&tm_local, &now_t);
                    char fname[256];
                    std::strftime(
                        fname, sizeof(fname),
                        "logs/material_categories_%Y-%m-%d_%H-%M-%S.log",
                        &tm_local);
                    std::filesystem::path abs_path =
                        std::filesystem::absolute(fname, ec);
                    std::ofstream dump(fname);
                    if (dump.is_open()) {
                        dump << "# Material/Object → MeshCategory dump\n"
                             << "# classifier_ok="
                             << (ok ? "true" : "false") << '\n'
                             << "# materials_total="
                             << s_mat_classifier
                                    .classifiedMaterials().size()
                             << " objects_total="
                             << s_mat_classifier
                                    .classifiedObjects().size()
                             << '\n'
                             << "# kind\tname\tcategory\n";
                        for (const auto& [name, cat] :
                                s_mat_classifier.classifiedMaterials()) {
                            dump << "material\t" << name << '\t'
                                 << engine::helper::meshCategoryTag(cat)
                                 << '\n';
                        }
                        for (const auto& [name, cat] :
                                s_mat_classifier.classifiedObjects()) {
                            dump << "object\t" << name << '\t'
                                 << engine::helper::meshCategoryTag(cat)
                                 << '\n';
                        }
                        std::cout
                            << "[mat.cls] full table dumped to "
                            << abs_path.string() << std::endl;
                    } else {
                        std::cout
                            << "[mat.cls] FAILED to open dump file "
                            << abs_path.string()
                            << " (errno=" << errno << ")" << std::endl;
                    }
                }

                // Push a snapshot to the menu so the Mesh-Category
                // Inspector ImGui window has data to display.  The
                // menu stores (name, uint32_t) pairs so menu.h stays
                // free of a collision_mesh.h include — we cast each
                // MeshCategory enum to its underlying value here.
                if (menu_) {
                    std::vector<std::pair<std::string, uint32_t>> mats;
                    std::vector<std::pair<std::string, uint32_t>> objs;
                    mats.reserve(
                        s_mat_classifier.classifiedMaterials().size());
                    objs.reserve(
                        s_mat_classifier.classifiedObjects().size());
                    for (const auto& [name, cat] :
                            s_mat_classifier.classifiedMaterials()) {
                        mats.emplace_back(
                            name, static_cast<uint32_t>(cat));
                    }
                    for (const auto& [name, cat] :
                            s_mat_classifier.classifiedObjects()) {
                        objs.emplace_back(
                            name, static_cast<uint32_t>(cat));
                    }
                    menu_->setMaterialCategorySnapshot(
                        std::move(mats), std::move(objs));
                }

                s_mat_classifier_applied = true;
            }
        }
    }

    // Legacy background-disable check (for non-title-screen mesh loads).
    if (menu_->isBackgroundEnabled() &&
        menu_->getGameState() == engine::ui::GameState::InGame) {
        menu_->setBackgroundEnabled(false);
    }

    auto to_load_drawable_names = menu_->getToLoadGltfNamesAndClear();
    for (auto& drawable_name : to_load_drawable_names) {
        auto drawable_obj = ego::DrawableObject::createAsync(
            *mesh_load_task_manager_,
            device_,
            descriptor_pool_,
            renderbuffer_formats_,
            graphic_pipeline_info_,
            texture_sampler_,
            thin_film_lut_tex_,
            drawable_name,
            glm::inverse(view_params_.view));

        s_update_frame_count = -1;
        drawable_objects_.push_back(drawable_obj);
        object_scene_view_->addDrawableObject(drawable_obj);
        shadow_object_scene_view_->addDrawableObject(drawable_obj);
    }

    // (The menu-triggered async player-load path was removed: the
    // player is eager-loaded at startup with assets/Characters/
    // scene-skinned.gltf — see the createAsync call near application
    // init.  That single source of truth removed the duplicate code
    // path that destroyed and recreated player_object_ on every
    // "Spawn player" menu click, which both forced a reload of the
    // (already-loaded) rig and risked picking up an asset whose joint
    // names PlayerController::applyPose doesn't know how to drive.)

    // ── CPU profile: "Player controller" ────────────────────────────
    // Reads input, runs movement integration, and resolves character
    // capsule-vs-scene collision against the CollisionWorld BVH.  If
    // this is slow, the BVH branch in CollisionWorld::resolveCapsule
    // is the usual suspect.
    if (player_object_ && player_controller_) {
        auto _cpu_player_ctrl = gpu_profiler_.beginCpuScope("Player controller");
        const float cam_yaw =
            main_camera_object_->getCameraViewInfo().yaw;
        // Apply live foot-IK tuning from the Physics > Foot IK menu,
        // then read back the resulting pelvis drop for the menu's
        // read-only readout.  Cheap setters; safe to call every frame.
        if (menu_) {
            const auto& ik = menu_->footIkParams();
            player_controller_->setFootIkEnabled(ik.enabled);
            player_controller_->setFootStrideAmp(ik.stride_amp);
            player_controller_->setFootLiftAmp(ik.lift_amp);
            player_controller_->setFootSoleDrop(ik.sole_drop);
            player_controller_->setFootTiltWeight(ik.tilt_weight);
            player_controller_->setPelvisDropMax(ik.pelvis_drop_max);
            menu_->setFootIkPelvisDrop(player_controller_->pelvisDrop());
        }
        player_controller_->update(
            window_,
            delta_t_,
            cam_yaw,
            player_object_,
            drawable_objects_,
            &collision_world_);
        gpu_profiler_.endCpuScope(_cpu_player_ctrl);
    }

    // ── CPU profile: "Drawable updates (CPU)" ───────────────────────
    // Walks every loaded DrawableObject and calls its CPU-side update
    // (node-transform recomputes, animation step, etc.).  Cost grows
    // ~linearly with drawable count and animation complexity.  The
    // matching GPU buffer copies live in drawScene's "Game Object
    // Updates" scope, not here.
    {
        auto _cpu_drw = gpu_profiler_.beginCpuScope("Drawable updates (CPU)");
        if (player_object_) {
            player_object_->update(device_, current_time_);
        }

        // ── Foot debug markers ─────────────────────────────────────────
        // Pin the two red-cube markers to the player rig's left_foot /
        // right_foot bones.  Must run AFTER player_object_->update()
        // (so cached_matrix_ on the foot nodes reflects the current
        // frame's pose) but BEFORE the markers' own update() (which
        // recomputes their root cached_matrix_ from setRootNodeTransform).
        //
        // Idempotent debug-flag setters are called every frame:
        //   • setDebugForceRed(true)        — fragment output forced red
        //   • setUseNodeTransformOnly(true) — bypass the shared
        //     game_objects_buffer_ camera-tracking position, otherwise
        //     the marker double-transforms and floats off to camera+foot
        //   • setDebugScale(0.1)            — 10 cm visual size
        //   • setDebugLogDraws(false)       — quiet; flip to true if a
        //     marker stops appearing and you need to confirm draws are
        //     actually being issued
        // ── Foot debug markers ─────────────────────────────────────────
        // Configuration: tiny red cubes pinned to the player rig's
        // left_foot / right_foot bones each frame.  Rendering is wired
        // through object_scene_view_ / shadow_object_scene_view_ —
        // see the addDrawableObject calls next to player_object_'s
        // scene-view registration.  No diagnostic prints in this path
        // anymore; if you need to debug visibility, flip
        // kDebugCubeSize up to 1.0 and add setDebugLogDraws(true) to
        // configure_marker.
        //
        //   • setDebugScale(0.08) — 8 cm cube, roughly foot-sized.
        //   • setDebugForceRed   — fragment forced red regardless of
        //     lighting, so it pops against any backdrop.
        //   • setUseNodeTransformOnly — bypass the shared
        //     game_objects_buffer_ camera-tracking position so
        //     setRootNodeTransform(world_pos) lands exactly there.
        //
        // Markers stay at world origin until the player spawns; once
        // spawned, each frame we read the foot bone's cached_matrix
        // and place the corresponding cube at that translation.
        constexpr float kDebugCubeSize = 0.08f;
        auto configure_marker = [](
            const std::shared_ptr<ego::DrawableObject>& m) {
            if (!m || !m->isReady()) return;
            m->setDebugForceRed(true);
            m->setUseNodeTransformOnly(true);
            m->setDebugScale(kDebugCubeSize);
        };
        configure_marker(foot_marker_left_);
        configure_marker(foot_marker_right_);

        if (player_object_ && player_object_->isReady() &&
            player_controller_ && player_controller_->isSpawned()) {
            const glm::quat identity_q(1.0f, 0.0f, 0.0f, 0.0f);

            // For each foot, take the bone's world translation, cast
            // a ray straight down from a point ~0.3 m above it (to
            // make sure we start ABOVE the floor mesh even when the
            // ankle bone is slightly buried), and snap the marker to
            // the closest triangle hit.  When the raycast misses (no
            // floor under the foot — open air, edge of world) we fall
            // back to the bone position so the marker doesn't snap to
            // (0,0,0) or vanish.
            //
            // The raycast is gated on collision_world_built_ because
            // the BVH builds asynchronously over the Bistro meshes;
            // before that we just show the bone-position fallback.
            constexpr float kFootRayStartLift = 0.3f;   // start ray 30 cm above the bone
            constexpr float kFootRayMaxDist   = 2.0f;   // search 2 m down
            // Throttled diagnostic — print once per second per foot so
            // we can tell whether the raycast hit (and the marker just
            // happens to be near the ankle because the foot bone IS
            // close to the ground) vs. the raycast missing entirely
            // and us silently falling back to the bone position.
            static int s_foot_ray_frame = 0;
            ++s_foot_ray_frame;
            const bool log_this_frame =
                (s_foot_ray_frame % 60 == 0);
            auto place_at_foot = [&](
                const std::shared_ptr<ego::DrawableObject>& marker,
                const char* bone_name) {
                if (!marker || !marker->isReady()) return;
                glm::mat4 m =
                    player_object_->getNodeWorldMatrixByName(bone_name);
                glm::vec3 bone_pos(m[3]);

                glm::vec3 target = bone_pos;
                bool      ray_hit = false;
                glm::vec3 hit_pos(0.0f);
                if (collision_world_built_) {
                    glm::vec3 ray_from = bone_pos;
                    ray_from.y += kFootRayStartLift;
                    glm::vec3 hit_normal;
                    if (collision_world_.raycastDown(
                            ray_from, kFootRayStartLift + kFootRayMaxDist,
                            hit_pos, hit_normal)) {
                        target = hit_pos;
                        ray_hit = true;
                    }
                }
                if (log_this_frame) {
                    std::cout << "[foot_ray] " << bone_name
                              << " bone=(" << bone_pos.x << "," << bone_pos.y << "," << bone_pos.z << ")"
                              << " cw_built=" << (collision_world_built_ ? 1 : 0)
                              << " hit=" << (ray_hit ? 1 : 0);
                    if (ray_hit) {
                        std::cout << " hit_pos=(" << hit_pos.x << "," << hit_pos.y << "," << hit_pos.z << ")"
                                  << " drop=" << (bone_pos.y - hit_pos.y) << "m";
                    }
                    std::cout << std::endl;
                }
                marker->setRootNodeTransform(target, identity_q);
            };

            place_at_foot(foot_marker_left_,  "left_foot");
            place_at_foot(foot_marker_right_, "right_foot");
        }
        if (foot_marker_left_)  foot_marker_left_->update(device_, current_time_);
        if (foot_marker_right_) foot_marker_right_->update(device_, current_time_);

        for (auto& drawable_obj : drawable_objects_) {
            drawable_obj->update(device_, current_time_);
        }
        gpu_profiler_.endCpuScope(_cpu_drw);
    }

    // ── Periodic player live-state diagnostic ───────────────────────
    // Once per ~60 frames, dump the root node's cached_matrix
    // translation (= what model_params.model_mat carries into base.vert
    // for the root mesh draws), the bbox WORLD center derived through
    // that matrix, the bounding radius, and the camera distance.
    // Lets us A/B the controller's spawn position vs. where the
    // skinned mesh actually ends up in world space.
    if (player_object_ && player_object_->isReady() &&
        player_controller_ && player_controller_->isSpawned()) {
        static uint64_t s_diag_frame = 0;
        if ((s_diag_frame++ % 60u) == 0u) {
            const auto& data = player_object_->getDrawableData();
            int rn = -1;
            if (!data.scenes_.empty() && data.default_scene_ >= 0 &&
                data.default_scene_ < (int)data.scenes_.size() &&
                !data.scenes_[data.default_scene_].nodes_.empty()) {
                rn = data.scenes_[data.default_scene_].nodes_[0];
            }
            if (rn >= 0 && rn < (int)data.nodes_.size() &&
                !data.meshes_.empty()) {
                const auto& cm  = data.nodes_[rn].cached_matrix_;
                const auto& mbb = data.meshes_[0];
                glm::vec3 lc =
                    (mbb.bbox_min_ + mbb.bbox_max_) * 0.5f;
                glm::vec4 wc4 = cm * glm::vec4(lc, 1.0f);
                glm::vec3 lh =
                    (mbb.bbox_max_ - mbb.bbox_min_) * 0.5f;
                float local_r = glm::length(lh);
                float sx = glm::length(glm::vec3(cm[0]));
                float sy = glm::length(glm::vec3(cm[1]));
                float sz = glm::length(glm::vec3(cm[2]));
                float world_r =
                    local_r * glm::max(sx, glm::max(sy, sz));
                glm::vec3 cam_pos =
                    main_camera_object_->getCameraPosition();
                std::cout << "[player.live] root_T=("
                          << cm[3][0] << "," << cm[3][1] << ","
                          << cm[3][2] << ")  mesh0_world_center=("
                          << wc4.x << "," << wc4.y << "," << wc4.z
                          << ")  world_r=" << world_r
                          << "  dist_to_cam=" << glm::length(
                                 glm::vec3(wc4) - cam_pos)
                          << std::endl;
            }
        }
    }

    // ── Wait Prev Frame Fence ────────────────────────────────────────
    // Drain frame N-1's submit before we touch any host-visible per-
    // frame buffer or reset the command buffer below.
    //
    // Why here, not at the top of drawFrame:
    //   FIF=2.  The per-FIF fence wait at the start drained FIF[N-2].
    //   Frame N-1's submit is still running on the GPU through this
    //   point, but everything above (input handling, player controller
    //   update, mesh-load polling, collision-world build, drawable CPU
    //   updates, skydome math) is either CPU-only or writes to GPU
    //   resources that are already multi-buffered against s_dbuf_idx
    //   or per-FIF.  Letting all of that run concurrently with frame
    //   N-1's GPU work is the actual async win — frame time drops to
    //   max(CPU_pre, GPU_prev) instead of CPU_pre + GPU_prev.
    //
    // What still needs the wait, and why this wait protects them:
    //   * command_buffer->reset() below — the CB is now per-FIF, so it
    //     was last submitted FIF cycles ago against THIS slot's fence,
    //     which the (1)-block wait already drained.  Strictly, this
    //     prev-frame wait isn't needed for CB safety — the (1) wait
    //     is.  Keeping it here grouped with the host-UBO sync just
    //     makes the "no GPU work in flight" point clear.
    //   * drawScene's host-coherent updateBufferMemory writes to
    //     view_camera_buffer_, runtime_lights_buffer_, and
    //     rt_render_info->ubo.  Those buffers are single-buffered and
    //     read by frame N-1's still-in-flight passes; overwriting
    //     them before this wait would race and produce the every-
    //     other-frame ABAB flicker we hit earlier.
    //
    // Lifecycle: we do NOT reset this fence here.  It will be reset
    // next frame by the (1) block at drawFrame entry, since
    // current_frame_ alternates between the two FIF slots.
    {
        auto _cpu_wait_prev = gpu_profiler_.beginCpuScope("Wait Prev Frame Fence");
        const uint64_t prev_frame_slot =
            (current_frame_ + 1) % kMaxFramesInFlight;
        device_->waitForFences({ in_flight_fences_[prev_frame_slot] });
        gpu_profiler_.endCpuScope(_cpu_wait_prev);
    }

    command_buffer->reset(0);
    command_buffer->beginCommandBuffer(SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

    // ── VT streaming tick (Phase A: feedback log) ─────────────────────
    // Reads previous frame's tile-key feedback (visible because the
    // previous frame's submit fence has been waited on by the
    // beginCommandBuffer reset path) and queues a fillBuffer at the
    // start of THIS frame's command buffer to clear the buffer before
    // any bindless fragment writes it again.  The bindless draw later
    // in drawScene runs after the fill, so by submit time the order
    // is: fill → bindless writes fresh keys.  Phase B will replace the
    // log inside tick() with actual page uploads.
    if (vt_manager_) {
        // CPU scope: tick() reads the GPU-compacted request list (a
        // few-KB pointer dereference, no big map+scan anymore) and
        // schedules uploads onto cmd_buf.  Used to dominate at
        // ~30 ms/frame from a synchronous transient submit-and-wait
        // plus a 1 MB cold map walk; both are gone.  Frame counter
        // is shared with the matching compactFeedback() call later
        // in the frame so the per-FIF slot indexing lines up.
        auto _cpu_vt = gpu_profiler_.beginCpuScope("VT tick (CPU)");
        vt_manager_->tick(command_buffer, vt_frame_index_);
        gpu_profiler_.endCpuScope(_cpu_vt);
    }

    std::shared_ptr<ego::ViewObject> display_scene_view = nullptr;

    display_scene_view = object_scene_view_;
    //display_scene_view = terrain_scene_view_;


    terrain_scene_view_->setVisibleTiles(visible_tiles);

    // Skip the full 3D scene (forward pass, shadows, terrain, volume
    // cloud, etc.) while on the title screen — only the ImGui title UI
    // and background image are shown. This avoids partially-loaded
    // meshes drawing garbage geometry through the background.
    if (menu_->getGameState() != engine::ui::GameState::TitleScreen) {
        auto _cpu_scene = gpu_profiler_.beginCpuScope("drawScene (record)");
        drawScene(command_buffer,
            swap_chain_info_,
            main_camera_object_->getViewCameraDescriptorSet(),
            swap_chain_info_.extent,
            image_index,
            delta_t_,
            current_time_);
        gpu_profiler_.endCpuScope(_cpu_scene);
    }

    if (!menu_->isRayTracingTurnOff()) {
        auto result_image =
            ray_tracing_test_->getFinalImage();
#define BLIT_IMAGE_TO_DISPLAY 0

        er::BarrierList raytracing_barrier_list_start;
        raytracing_barrier_list_start.image_barriers.reserve(1);

        er::helper::addTexturesToBarrierList(
            raytracing_barrier_list_start,
            { result_image.image },
            er::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_READ_BIT),
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT));

        command_buffer->addBarriers(
            raytracing_barrier_list_start,
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT),
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR));

        ray_tracing_test_->draw(
            device_,
            command_buffer,
            view_params_);

#if BLIT_IMAGE_TO_DISPLAY
        er::ImageResourceInfo src_info = {
            er::ImageLayout::GENERAL,
            SET_FLAG_BIT(Access, SHADER_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR) };

        er::ImageResourceInfo dst_info = {
            er::ImageLayout::PRESENT_SRC_KHR,
            SET_FLAG_BIT(Access, COLOR_ATTACHMENT_WRITE_BIT),
            SET_FLAG_BIT(PipelineStage, COLOR_ATTACHMENT_OUTPUT_BIT) };

        er::Helper::blitImage(
            command_buffer,
            result_image.image,
            swap_chain_info_.images[image_index],
            src_info,
            src_info,
            dst_info,
            dst_info,
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            SET_FLAG_BIT(ImageAspect, COLOR_BIT),
            result_image.size);
#else
        er::BarrierList raytracing_barrier_list_end;
        raytracing_barrier_list_end.image_barriers.reserve(1);

        er::helper::addTexturesToBarrierList(
            raytracing_barrier_list_end,
            { result_image.image },
            er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
            SET_FLAG_BIT(Access, SHADER_READ_BIT));

        command_buffer->addBarriers(
            raytracing_barrier_list_end,
            SET_FLAG_BIT(PipelineStage, RAY_TRACING_SHADER_BIT_KHR),
            SET_FLAG_BIT(PipelineStage, FRAGMENT_SHADER_BIT));
#endif
    }

    // ── Player debug overlay feed ────────────────────────────────────
    // Stash the player's world position + camera VP + world-space
    // AABB into the menu so its ImGui draw can paint:
    //   • a red on-screen marker at the controller position,
    //   • HUD text with position / distance / bbox values,
    //   • a wireframe box around the character's renderable volume.
    // The wireframe is computed from the SHIPPING cached_matrix (what
    // the GPU will actually use this frame), so it reflects every
    // applyPose / setRootNodeTransform write.  Comparing the
    // wireframe vs. the rendered body tells us whether the renderer
    // and our CPU-side bbox math agree.
    if (menu_ && main_camera_object_) {
        const bool has_player =
            player_controller_ && player_controller_->isSpawned();
        const glm::vec3 ppos = has_player
            ? player_controller_->getPosition()
            : glm::vec3(0.0f);

        bool      bbox_valid = false;
        glm::vec3 bbox_min(0.0f), bbox_max(0.0f);
        if (has_player && player_object_ && player_object_->isReady()) {
            const auto& data = player_object_->getDrawableData();
            float mn[3] = {  std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::max() };
            float mx[3] = {  std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::lowest() };
            auto accum = [&](const glm::vec3& w) {
                mn[0] = std::min(mn[0], w.x);
                mn[1] = std::min(mn[1], w.y);
                mn[2] = std::min(mn[2], w.z);
                mx[0] = std::max(mx[0], w.x);
                mx[1] = std::max(mx[1], w.y);
                mx[2] = std::max(mx[2], w.z);
            };

            // ── Skinned-mesh bbox: walk SKIN joints, not mesh nodes ─
            // glTF skinned meshes ignore the mesh node's cached_matrix
            // at render time — vertex world positions come from
            //   v_world = sum(w_i * joint.cached_matrix * inv_bind) * v_local
            // So our previous bbox (mesh.bbox * mesh_node.cached_matrix)
            // showed the BIND-POSE mesh transformed by the mesh node's
            // matrix, which is NOT where the rendered character actually
            // sits.  Instead, compute the bbox from the skeleton's joint
            // world positions: each joint contributes joint.cached_matrix
            // * (0,0,0) as a world point.  The set of joint positions
            // forms a skeleton AABB that closely approximates the
            // rendered mesh's AABB (slightly tighter than the actual
            // mesh — flesh extends a few cm beyond bones — but a much
            // better visual match than the bind-pose-on-mesh-node box).
            bool used_skin = false;
            for (const auto& sk : data.skins_) {
                for (int32_t joint_idx : sk.joints_) {
                    if (joint_idx < 0 ||
                        joint_idx >= (int)data.nodes_.size()) continue;
                    const auto& jn = data.nodes_[joint_idx];
                    glm::vec4 jw =
                        jn.cached_matrix_ * glm::vec4(0, 0, 0, 1);
                    accum(glm::vec3(jw));
                    bbox_valid = true;
                    used_skin = true;
                }
            }

            // Fallback to the mesh-node bind-pose box ONLY when the
            // asset has no skin (static meshes) — there we have no
            // joints to walk, and mesh_node.cached_matrix actually is
            // what the GPU uses to position the geometry.
            if (!used_skin) {
              for (size_t ni = 0; ni < data.nodes_.size(); ++ni) {
                const auto& n = data.nodes_[ni];
                if (n.mesh_idx_ < 0 ||
                    n.mesh_idx_ >= (int)data.meshes_.size()) continue;
                const auto& mb = data.meshes_[n.mesh_idx_];
                const glm::vec3 c[8] = {
                    {mb.bbox_min_.x, mb.bbox_min_.y, mb.bbox_min_.z},
                    {mb.bbox_max_.x, mb.bbox_min_.y, mb.bbox_min_.z},
                    {mb.bbox_min_.x, mb.bbox_max_.y, mb.bbox_min_.z},
                    {mb.bbox_max_.x, mb.bbox_max_.y, mb.bbox_min_.z},
                    {mb.bbox_min_.x, mb.bbox_min_.y, mb.bbox_max_.z},
                    {mb.bbox_max_.x, mb.bbox_min_.y, mb.bbox_max_.z},
                    {mb.bbox_min_.x, mb.bbox_max_.y, mb.bbox_max_.z},
                    {mb.bbox_max_.x, mb.bbox_max_.y, mb.bbox_max_.z},
                };
                for (int k = 0; k < 8; ++k) {
                    glm::vec4 w = n.cached_matrix_ * glm::vec4(c[k], 1.0f);
                    accum(glm::vec3(w));
                }
                bbox_valid = true;
              }
            }
            if (bbox_valid) {
                bbox_min = glm::vec3(mn[0], mn[1], mn[2]);
                bbox_max = glm::vec3(mx[0], mx[1], mx[2]);

                // Per-second console mirror so the values appear in
                // logs alongside the existing [player.live] /
                // [follow] lines.  Easier to scroll through than
                // squinting at the on-screen HUD across frames.
                static uint64_t s_bbox_log_frame = 0;
                if ((s_bbox_log_frame++ % 60u) == 0u) {
                    std::cout
                        << "[player.bbox] world_min=("
                        << bbox_min.x << "," << bbox_min.y << ","
                        << bbox_min.z << ") world_max=("
                        << bbox_max.x << "," << bbox_max.y << ","
                        << bbox_max.z << ") size=("
                        << (bbox_max.x - bbox_min.x) << ","
                        << (bbox_max.y - bbox_min.y) << ","
                        << (bbox_max.z - bbox_min.z)
                        << ")  controller_pos=(" << ppos.x << ","
                        << ppos.y << "," << ppos.z << ")"
                        << std::endl;
                }
            }
        }

        menu_->setPlayerDebugInfo(
            has_player,
            ppos,
            main_camera_object_->getViewProjMatrix(),
            main_camera_object_->getCameraPosition(),
            bbox_valid,
            bbox_min,
            bbox_max);

        // ── Per-frame player debug stream ────────────────────────────
        // Mirror the data the Player Debug HUD shows on-screen (pos,
        // dist, bbox, size) into a TSV log file so a long session can
        // be analysed offline without scroll-back-hunting through
        // stdout.  Opened lazily on the first frame the player is
        // spawned; one fresh timestamped file per run, matching the
        // naming convention in realworld/logs/.  Static local: no
        // class-field change, no global, and the destructor flushes
        // on shutdown.
        //
        // Filesystem path is relative to the working directory the
        // exe was launched from (typically realworld/), same as
        // gpu_profile.log.  We create the logs/ dir if it doesn't
        // exist so this also works in a fresh checkout.
        if (has_player) {
            static std::ofstream s_player_dbg_log;
            static bool          s_player_dbg_opened = false;
            if (!s_player_dbg_opened) {
                s_player_dbg_opened = true;
                std::error_code ec;
                std::filesystem::create_directories("logs", ec);
                auto now_t = std::chrono::system_clock::to_time_t(
                    std::chrono::system_clock::now());
                std::tm tm_local{};
                localtime_s(&tm_local, &now_t);
                char fname[256];
                std::strftime(
                    fname, sizeof(fname),
                    "logs/player_debug_%Y-%m-%d_%H-%M-%S.log",
                    &tm_local);
                s_player_dbg_log.open(fname, std::ios::out | std::ios::app);
                if (s_player_dbg_log.is_open()) {
                    // Tab-separated columns; '#' header line is
                    // skipped by awk/pandas defaults so the file
                    // tools can parse it directly.
                    s_player_dbg_log
                        << "# Player debug stream\n"
                        << "# columns: frame\ttime\tpos_x\tpos_y\tpos_z"
                           "\tdist_m\tbbox_valid\tbbox_min_x\tbbox_min_y"
                           "\tbbox_min_z\tbbox_max_x\tbbox_max_y"
                           "\tbbox_max_z\tsize_x\tsize_y\tsize_z\n";
                    std::cout << "[player.dbg] streaming to "
                              << fname << std::endl;
                }
            }
            if (s_player_dbg_log.is_open()) {
                static uint64_t s_player_dbg_frame = 0;
                // Wall-clock HH:MM:SS.mmm for human-readable timeline.
                auto now      = std::chrono::system_clock::now();
                auto now_t    = std::chrono::system_clock::to_time_t(now);
                auto now_ms   = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;
                std::tm tm_local{};
                localtime_s(&tm_local, &now_t);
                char tbuf[16];
                std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", &tm_local);

                const float dist = glm::length(
                    ppos - main_camera_object_->getCameraPosition());
                const glm::vec3 sz = bbox_valid
                    ? (bbox_max - bbox_min) : glm::vec3(0.0f);

                s_player_dbg_log
                    << s_player_dbg_frame++ << '\t'
                    << tbuf << '.'
                    << std::setw(3) << std::setfill('0') << now_ms
                    << std::setfill(' ') << '\t'
                    << std::fixed << std::setprecision(3)
                    << ppos.x << '\t' << ppos.y << '\t' << ppos.z << '\t'
                    << dist << '\t'
                    << (bbox_valid ? 1 : 0) << '\t'
                    << bbox_min.x << '\t' << bbox_min.y << '\t' << bbox_min.z << '\t'
                    << bbox_max.x << '\t' << bbox_max.y << '\t' << bbox_max.z << '\t'
                    << sz.x      << '\t' << sz.y      << '\t' << sz.z
                    << '\n';
                // Flush every 60 frames (~1 s at 60 Hz) so a crash
                // doesn't lose the last second.  Cheaper than
                // per-line flushing because the underlying syscall
                // is amortised across 60 writes.
                if ((s_player_dbg_frame % 60u) == 0u) {
                    s_player_dbg_log.flush();
                }
            }
        }
    }

    // ── CPU profile: "ImGui menu draw" ───────────────────────────────
    // Records the ImGui menu / HUD into the command buffer.  This is
    // ImGui's own NewFrame → Render → recordVulkanCommands chain — cost
    // tracks both widget count and number of draw calls emitted.  Big
    // tab-tree menus (profiler view, full Menu UI) easily hit 1–2 ms
    // here.
    auto _cpu_imgui = gpu_profiler_.beginCpuScope("ImGui menu draw");

    // ── Hover-pick: name the object + surface under the mouse cursor ─────
    // The name reflects WHAT IS DRAWN, not all scene geometry:
    //   * collision-debug on  -> ray-pick the collision-world meshes (the
    //     green floor proxies actually on screen) and report each proxy's
    //     source object + material.  Lets you audit which source objects the
    //     classifier put into the floor set, and matches the overlay rather
    //     than naming a full-res scene mesh hidden behind it.
    //   * otherwise            -> ray-pick the full scene (everything that
    //     is rendered).
    // Broad phase ray-AABB, narrow phase exact ray-triangle (Moller-
    // Trumbore).  A per-frame triangle budget caps the work.
    {
        std::string hover_name;
        const glm::uvec2 ss = swap_chain_info_.extent;
        if (main_camera_object_ && ss.x > 0 && ss.y > 0) {
            // Mouse pixel -> NDC.  Our projection already flips Y (see the
            // player-HUD un-project in menu.cpp), so map y straight through.
            const float nx = 2.0f * (s_last_mouse_pos.x / float(ss.x)) - 1.0f;
            const float ny = 2.0f * (s_last_mouse_pos.y / float(ss.y)) - 1.0f;
            const glm::mat4 inv_vp =
                glm::inverse(main_camera_object_->getViewProjMatrix());
            const glm::vec3 ro = main_camera_object_->getCameraPosition();
            glm::vec4 fp = inv_vp * glm::vec4(nx, ny, 1.0f, 1.0f);  // far plane
            glm::vec3 rd = glm::vec3(fp) / fp.w - ro;
            const float rd_len = glm::length(rd);
            if (rd_len > 1e-6f) {
                rd /= rd_len;
                const glm::vec3 inv_rd(
                    1.0f / (rd.x != 0.0f ? rd.x : 1e-20f),
                    1.0f / (rd.y != 0.0f ? rd.y : 1e-20f),
                    1.0f / (rd.z != 0.0f ? rd.z : 1e-20f));
                float best_dist = 1e30f;
                size_t tri_budget = 400000;  // cap ray-triangle work / frame

                // Moller-Trumbore (double-sided).  O/D is the ray in the
                // SAME space as v0..v2; for a node's local space, pass the
                // un-normalised M^-1*rd as D so the solved t stays a world
                // distance and is comparable across objects.
                auto rayTri = [](const glm::vec3& O, const glm::vec3& D,
                                 const glm::vec3& v0, const glm::vec3& v1,
                                 const glm::vec3& v2, float& out_t) -> bool {
                    const glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
                    const glm::vec3 pv = glm::cross(D, e2);
                    const float det = glm::dot(e1, pv);
                    const float ad = det < 0.0f ? -det : det;
                    if (ad < 1e-12f) return false;
                    const float inv = 1.0f / det;
                    const glm::vec3 tv = O - v0;
                    const float u = glm::dot(tv, pv) * inv;
                    if (u < 0.0f || u > 1.0f) return false;
                    const glm::vec3 qv = glm::cross(tv, e1);
                    const float v = glm::dot(D, qv) * inv;
                    if (v < 0.0f || u + v > 1.0f) return false;
                    const float t = glm::dot(e2, qv) * inv;
                    if (t <= 1e-3f) return false;
                    out_t = t; return true;
                };
                // Ray-AABB slab test in world space; out_tnear = entry dist.
                auto rayAabb = [&](const glm::vec3& bmin, const glm::vec3& bmax,
                                   float& out_tnear) -> bool {
                    const glm::vec3 t0 = (bmin - ro) * inv_rd;
                    const glm::vec3 t1 = (bmax - ro) * inv_rd;
                    const glm::vec3 tsm = glm::min(t0, t1);
                    const glm::vec3 tbg = glm::max(t0, t1);
                    const float tn = glm::max(glm::max(tsm.x, tsm.y), tsm.z);
                    const float tf = glm::min(glm::min(tbg.x, tbg.y), tbg.z);
                    if (tf < glm::max(tn, 0.0f)) return false;
                    out_tnear = glm::max(tn, 0.0f);
                    return true;
                };

                const bool coll_dbg = menu_ && menu_->isCollisionDebugOn() &&
                                      !collision_world_.empty();
                if (coll_dbg) {
                    // Pick among the VISIBLE collision-debug meshes only.
                    // Their vertices_ are already world-space, so the ray is
                    // tested directly.  Honour the isolate index (only the
                    // single drawn mesh) when isolate mode is on.
                    const int iso = menu_->collisionIsolateEnabled()
                        ? menu_->collisionIsolateIndex() : -1;
                    const size_t n = collision_world_.meshCount();
                    for (size_t mi = 0; mi < n; ++mi) {
                        if (iso >= 0 && mi != static_cast<size_t>(iso))
                            continue;
                        const auto* cm = collision_world_.meshAt(mi);
                        if (!cm) continue;
                        float box_t;
                        if (!rayAabb(cm->bounds().min_bounds,
                                     cm->bounds().max_bounds, box_t)) continue;
                        if (box_t > best_dist) continue;
                        if (tri_budget == 0) continue;
                        const auto& V = cm->debugVertices();  // world space
                        const auto& I = cm->debugIndices();
                        for (size_t k = 0; k + 2 < I.size(); k += 3) {
                            if (tri_budget == 0) break;
                            --tri_budget;
                            const int ia = I[k + 0], ib = I[k + 1],
                                      ic = I[k + 2];
                            if (ia < 0 || ib < 0 || ic < 0) continue;
                            if (static_cast<size_t>(ia) >= V.size() ||
                                static_cast<size_t>(ib) >= V.size() ||
                                static_cast<size_t>(ic) >= V.size()) continue;
                            float t;
                            if (rayTri(ro, rd, V[ia], V[ib], V[ic], t) &&
                                t < best_dist) {
                                best_dist = t;
                                hover_name = cm->objectName();
                                const std::string& mat = cm->materialName();
                                if (!mat.empty()) hover_name += "\n" + mat;
                            }
                        }
                    }
                } else {
                    // Full-scene pick: ray-test every drawable mesh, in its
                    // node-local space (so t is a world distance).
                    auto testDrawable =
                        [&](const std::shared_ptr<ego::DrawableObject>& obj) {
                        if (!obj || !obj->isReady()) return;
                        const auto& data = obj->getDrawableData();
                        for (const auto& node : data.nodes_) {
                            if (node.mesh_idx_ < 0 ||
                                static_cast<size_t>(node.mesh_idx_) >=
                                    data.meshes_.size())
                                continue;
                            const auto& mesh = data.meshes_[node.mesh_idx_];
                            if (!mesh.vertex_position_ ||
                                mesh.bbox_min_.x > mesh.bbox_max_.x) continue;
                            const glm::mat4& M = node.cached_matrix_;
                            glm::vec3 wmin(0.0f), wmax(0.0f);
                            for (int cc = 0; cc < 8; ++cc) {
                                const glm::vec3 lc(
                                    (cc & 1) ? mesh.bbox_max_.x
                                             : mesh.bbox_min_.x,
                                    (cc & 2) ? mesh.bbox_max_.y
                                             : mesh.bbox_min_.y,
                                    (cc & 4) ? mesh.bbox_max_.z
                                             : mesh.bbox_min_.z);
                                const glm::vec3 w =
                                    glm::vec3(M * glm::vec4(lc, 1.0f));
                                if (cc == 0) { wmin = wmax = w; }
                                else { wmin = glm::min(wmin, w);
                                       wmax = glm::max(wmax, w); }
                            }
                            float box_t;
                            if (!rayAabb(wmin, wmax, box_t)) continue;
                            if (box_t > best_dist) continue;
                            if (tri_budget == 0) continue;
                            const glm::mat4 invM = glm::inverse(M);
                            const glm::vec3 rol =
                                glm::vec3(invM * glm::vec4(ro, 1.0f));
                            const glm::vec3 rdl =
                                glm::vec3(invM * glm::vec4(rd, 0.0f));
                            const auto& P = *mesh.vertex_position_;
                            for (const auto& prim : mesh.primitives_) {
                                if (!prim.vertex_indices_) continue;
                                const auto& I = *prim.vertex_indices_;
                                for (size_t k = 0; k + 2 < I.size(); k += 3) {
                                    if (tri_budget == 0) break;
                                    --tri_budget;
                                    const size_t a =
                                        static_cast<size_t>(I[k + 0]);
                                    const size_t b =
                                        static_cast<size_t>(I[k + 1]);
                                    const size_t ci =
                                        static_cast<size_t>(I[k + 2]);
                                    if (a >= P.size() || b >= P.size() ||
                                        ci >= P.size()) continue;
                                    float t;
                                    if (rayTri(rol, rdl, P[a], P[b], P[ci],
                                               t) && t < best_dist) {
                                        best_dist = t;
                                        std::string mat;
                                        if (prim.material_idx_ >= 0 &&
                                            static_cast<size_t>(
                                                prim.material_idx_) <
                                                data.materials_.size())
                                            mat = data.materials_[
                                                prim.material_idx_].name_;
                                        hover_name = node.name_;
                                        if (!mat.empty())
                                            hover_name += "\n" + mat;
                                    }
                                }
                            }
                        }
                    };
                    testDrawable(bistro_exterior_scene_);
                    testDrawable(bistro_interior_scene_);
                    for (auto& d : drawable_objects_) testDrawable(d);
                }
            }
        }
        if (menu_) menu_->setHoverObjectName(hover_name);
    }

    s_camera_paused = menu_->draw(
        command_buffer,
        final_render_pass_,
        swap_chain_info_.framebuffers[image_index],
        swap_chain_info_.extent,
        skydome_,
        dump_volume_noise_,
        delta_t_);
    gpu_profiler_.endCpuScope(_cpu_imgui);

    // ── CPU profile: "End CB" ───────────────────────────────────────
    // vkEndCommandBuffer.  Should be sub-100µs.  If it grows the
    // driver is doing something interesting (validation layer, etc.).
    {
        auto _cpu_end_cb = gpu_profiler_.beginCpuScope("End CB");
        command_buffer->endCommandBuffer();
        gpu_profiler_.endCpuScope(_cpu_end_cb);
    }

    // ── CPU profile: "Submit" / "Present" split ─────────────────────
    // Old combined "Submit + Present" scope split into the two halves
    // so the lane tells you whether the cost is in vkQueueSubmit (the
    // driver kicking the GPU and prepping the per-frame fence) or in
    // vkQueuePresentKHR (compositor pacing / vsync delay).  Both can
    // block on Windows depending on present mode.
    {
        auto _cpu_submit = gpu_profiler_.beginCpuScope("Submit");
        er::Helper::submitQueue(
            graphics_queue_,
            in_flight_fences_[current_frame_],
            { image_available_semaphores_[current_frame_] },
            { command_buffer },
            { render_finished_semaphores_[current_frame_] },
            { });
        gpu_profiler_.endCpuScope(_cpu_submit);
    }

    {
        auto _cpu_present = gpu_profiler_.beginCpuScope("Present");
        need_recreate_swap_chain = er::Helper::presentQueue(
            present_queue_,
            { swap_chain_info_.swap_chain },
            { render_finished_semaphores_[current_frame_] },
            image_index,
            framebuffer_resized_);
        gpu_profiler_.endCpuScope(_cpu_present);
    }

    if (need_recreate_swap_chain) {
        recreateSwapChain();
    }

    // GPU timestamp collection moved to RIGHT AFTER the per-frame
    // fence wait at the head of this function — the slot that wait
    // drained is the one whose query results are guaranteed ready,
    // and reading there avoids the "VK_NOT_READY → frame dropped"
    // problem that surfaced once the CPU started outrunning the GPU.
    // The per-second log dump still lives here because it runs after
    // each successful collectResults regardless and just snapshots
    // whatever's currently the latest ingested frame.
    if (gpu_profiler_initialized_) {
        // ── Per-second GPU profile log dump ──────────────────────────────────
        // Append the latest FrameRecord to gpu_profile.log once per wall-clock
        // second.  Each entry is prefixed with a timestamp so the log is
        // easy to tail-follow during a live session.
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - gpu_profile_last_dump_).count();
        if (elapsed >= 1000) {
            gpu_profile_last_dump_ = now;
            const auto* frame = gpu_profiler_.getLatestFrame();
            if (frame) {
                // Use a wall-clock time string for the log header.
                auto wall = std::chrono::system_clock::now();
                auto wall_t = std::chrono::system_clock::to_time_t(wall);

                static const char* kLogPath = "gpu_profile.log";
                std::ofstream log(kLogPath, std::ios::app);
                if (log.is_open()) {
                    log << "── GPU frame @ " << std::put_time(std::localtime(&wall_t), "%H:%M:%S")
                        << "  total=" << std::fixed << std::setprecision(3)
                        << frame->total_ms << " ms"
                        << "  scopes=" << frame->scopes.size() << "\n";
                    for (const auto& s : frame->scopes) {
                        float dur_ms = s.end_ms - s.begin_ms;
                        // Indent by depth for readability.
                        for (int d = 0; d < s.depth; ++d) log << "  ";
                        log << std::left << std::setw(32) << s.name
                            << "  " << std::fixed << std::setprecision(3)
                            << std::setw(8) << dur_ms << " ms"
                            << "  [" << s.begin_ms << " → " << s.end_ms << "]\n";
                    }
                    log << "\n";
                }
            }
        }
    }

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
    if (s_update_frame_count < 0) {
        s_update_frame_count = 0;
    }

    // ── Close CPU frame ──
    // Mirrors beginCpuFrame at the top of drawFrame.  endCpuScope on
    // the wrapping "drawFrame" scope first so it's properly closed
    // before we tell the profiler this frame's CPU recording is done.
    gpu_profiler_.endCpuScope(_cpu_frame);
    gpu_profiler_.endCpuFrame(cpu_frame_idx);
}

void RealWorldApplication::cleanupSwapChain() {
    assert(device_);
    depth_buffer_.destroy(device_);
    depth_buffer_copy_.destroy(device_);
    hdr_color_buffer_.destroy(device_);
    hdr_color_buffer_copy_.destroy(device_);
    // Deferred G-buffer targets — paired with the swap-chain extent and so
    // recreated alongside the depth/HDR colour buffers above.  Sampler /
    // pipeline / descriptor set / layout outlive the swap chain (they are
    // size-independent) and are torn down in cleanup() instead.
    gbuf_albedo_ao_.destroy(device_);
    gbuf_normal_rough_.destroy(device_);
    gbuf_emissive_metal_.destroy(device_);
    gbuf_velocity_.destroy(device_);
    // Hi-Z pyramid + per-mip views are sized to the swap chain too.
    hiz_mip_views_.clear();
    if (hiz_pyramid_.image) hiz_pyramid_.destroy(device_);
    hiz_pyramid_size_ = glm::uvec2(0);
    hiz_pyramid_mips_ = 0;
    hiz_build_desc_sets_.clear();
    device_->destroyFramebuffer(hdr_frame_buffer_);
    device_->destroyFramebuffer(hdr_water_frame_buffer_);

    for (auto framebuffer : swap_chain_info_.framebuffers) {
        device_->destroyFramebuffer(framebuffer);
    }

    device_->freeCommandBuffers(command_pool_, command_buffers_);
    device_->destroyRenderPass(final_render_pass_);
    device_->destroyRenderPass(hdr_render_pass_);
    device_->destroyRenderPass(hdr_water_render_pass_);

    for (auto image_view : swap_chain_info_.image_views) {
        device_->destroyImageView(image_view);
    }

    device_->destroySwapchain(swap_chain_info_.swap_chain);

    device_->destroyDescriptorPool(descriptor_pool_);

    // The deferred-resolve descriptor set was allocated FROM
    // descriptor_pool_ in initDeferredResolve.  Destroying the pool
    // implicitly destroys every set allocated from it, leaving
    // deferred_resolve_desc_set_ as a dangling Vulkan handle.  Null
    // it out so writeDeferredResolveDescriptors (called from inside
    // recreateRenderBuffer → createGBuffer below) hits its early-out
    // guard instead of issuing vkUpdateDescriptorSets against the
    // dead handle (which crashes inside the NVIDIA driver — the
    // validation layer surfaces this as an UpdateDescriptorSets
    // failure with a deep nvoglv64.dll callstack).  recreateSwapChain
    // re-allocates the set from the fresh descriptor_pool_ before the
    // canonical writeDeferredResolveDescriptors() call later in the
    // recreate sequence.
    deferred_resolve_desc_set_ = nullptr;

    // Same hazard exists for ClusterRenderer's pool-owned sets — most
    // notably cull_desc_set_, which createHiZPyramid (also called
    // from inside recreateRenderBuffer → createGBuffer) tries to
    // patch via cluster_renderer_->setHiZTexture before the new pool
    // exists.  ClusterRenderer::onDescriptorPoolDestroyed nulls all
    // the relevant handles (bindless / cull / per-cascade shadow /
    // mesh-shader cluster data / OIT composite) so the existing
    // "set is null" guards inside its consumers short-circuit until
    // recreate() and the next finalizeUploads / setHiZTexture re-
    // populate them.
    if (cluster_renderer_) {
        cluster_renderer_->onDescriptorPoolDestroyed();
    }

    // Same dangling-handle hazard for AmbientProbeSystem.  Its
    // probe_desc_set_ and project_desc_set_ were allocated from
    // descriptor_pool_ in AmbientProbeSystem::create; the per-frame
    // writeProjectDescriptorsForCube call (drawScene's "Ambient
    // Probes" block) would otherwise write through dead handles
    // after this point — surfaces as the same NVIDIA / validation
    // crash in vkUpdateDescriptorSets.  recreateSwapChain reallocates
    // both sets right after the new pool is created (see
    // ambient_probe_system_->recreateDescriptorSets call below).
    if (ambient_probe_system_) {
        ambient_probe_system_->onDescriptorPoolDestroyed();
    }

    // Same hazard for VirtualTextureManager — owns compact_desc_set_
    // allocated from descriptor_pool_, used every frame by
    // compactFeedback / tick.  Without this nullout the very first
    // frame after a window resize hits the NVIDIA driver crash with
    // an EXCEPTION_ACCESS_VIOLATION_READ at a tiny offset (~0x38)
    // while binding the dangling set.  Re-allocated below in
    // recreateSwapChain right after the new pool is built.
    if (vt_manager_) {
        vt_manager_->onDescriptorPoolDestroyed();
    }

    // And DynamicCubemap — owns 3 families of pool-allocated sets
    // (face_view, depth_to_linear, reproject).  AmbientProbeSystem
    // calls into it every frame to capture a cube face, binding
    // these sets.  Skipping this nullout is the most likely cause
    // of the release-only resize crash that survived all earlier
    // descriptor-set hooks — debug builds escape because validation
    // detects the bad bind, release builds dispatch the dangling
    // handle into the NVIDIA driver and crash.
    if (dynamic_cubemap_) {
        dynamic_cubemap_->onDescriptorPoolDestroyed();
    }
}

void RealWorldApplication::cleanup() {
    // Tear down the async mesh loader first. Its dtor drains the
    // pending queue, joins the worker thread, and waitAll()'s any
    // in-flight GPU fences. Doing this before waitIdle/destroyXxx
    // means no phase 2/phase 3 work can race with the destroy
    // sequence below (which would otherwise wipe buffers/images
    // the worker still holds shared_ptrs to).
    if (mesh_load_task_manager_) {
        mesh_load_task_manager_.reset();
    }

    // Wait for the GPU to finish before tearing anything down, otherwise
    // destroying resources still referenced by in-flight command buffers
    // (query pools, images, buffers, etc.) will crash.
    if (device_) {
        device_->waitIdle();
    }

    plugin_manager_.shutdownAll();

    collision_world_.destroyDebugBuffers(device_);
    eh::CollisionDebugDraw::destroyStaticMembers(device_);

    if (gpu_profiler_initialized_) {
        gpu_profiler_.destroy(device_);
        gpu_profiler_initialized_ = false;
    }
    menu_->destroyResources();
    menu_->destroy();
    cleanupSwapChain();

    skydome_->destroy(device_);
    device_->destroyRenderPass(cubemap_render_pass_);

    assert(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    sample_tex_.destroy(device_);
    er::Helper::destroy(device_);
    ggx_lut_tex_.destroy(device_);
    brdf_lut_tex_.destroy(device_);
    charlie_lut_tex_.destroy(device_);
    thin_film_lut_tex_.destroy(device_);
    map_mask_tex_.destroy(device_);
    heightmap_tex_.destroy(device_);
    prt_base_tex_.destroy(device_);
    prt_normal_tex_.destroy(device_);
    prt_height_tex_.destroy(device_);
    prt_orh_tex_.destroy(device_);
    ibl_diffuse_tex_.destroy(device_);
    ibl_specular_tex_.destroy(device_);
    ibl_sheen_tex_.destroy(device_);
    runtime_lights_buffer_->destroy(device_);
    device_->destroySampler(texture_sampler_);
    device_->destroySampler(texture_point_sampler_);
    device_->destroySampler(repeat_texture_sampler_);
    device_->destroySampler(mirror_repeat_sampler_);
    device_->destroyDescriptorSetLayout(pbr_lighting_desc_set_layout_);
    device_->destroyDescriptorSetLayout(runtime_lights_desc_set_layout_);

    ego::TileObject::destroyAllTiles(device_);
    ego::TileObject::destroyStaticMembers(device_);
    if (player_object_) {
        player_object_->destroy(device_);
        player_object_ = nullptr;
    }

    if (bistro_exterior_scene_) {
        bistro_exterior_scene_->destroy(device_);
        bistro_exterior_scene_ = nullptr;
    }

    if (bistro_interior_scene_) {
        bistro_interior_scene_->destroy(device_);
        bistro_interior_scene_ = nullptr;
    }

    for (auto& obj : drawable_objects_) {
        obj->destroy(device_);
    }
    drawable_objects_.clear();
    if (ray_tracing_test_) {
        ray_tracing_test_->destroy(device_);
    }
    ego::DrawableObject::destroyStaticMembers(device_);
    ego::ViewCamera::destroyStaticMembers(device_);
    ego::DebugDrawObject::destroyStaticMembers(device_);
    ego::ShapeBase::destroyStaticMembers(device_);
    ibl_creator_->destroy(device_);
    weather_system_->destroy(device_);
    volume_noise_->destroy(device_);
    volume_cloud_->destroy(device_);
    if (ssao_) ssao_->destroy(device_);
    if (cluster_renderer_) cluster_renderer_->destroy();
    if (vt_manager_) vt_manager_->destroy();
    if (ambient_probe_system_) ambient_probe_system_->destroy(device_);
    if (dynamic_cubemap_) dynamic_cubemap_->destroy(device_);
    unit_plane_->destroy(device_);
    unit_box_->destroy(device_);
    unit_sphere_->destroy(device_);
    conemap_obj_->destroy(device_); 
    conemap_gen_->destroy(device_);
    conemap_test_->destroy(device_);
    hair_patch_->destroy(device_);
    hair_test_->destroy(device_);
    lbm_patch_->destroy(device_);
    lbm_test_->destroy(device_);

    main_camera_object_->destroy(device_);
    shadow_camera_object_->destroy(device_);
    // Destroy the shared static descriptor set layout exactly once (after
    // both camera instances have released their descriptor sets).
    ego::CameraObject::destroyStaticMembers(device_);
    terrain_scene_view_->destroy(device_);
    object_scene_view_->destroy(device_);

    // Destroy CSM debug colour targets and pipeline.
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (csm_debug_color_[k].image) {
            csm_debug_color_[k].destroy(device_);
        }
    }
    if (csm_debug_pipeline_) {
        device_->destroyPipeline(csm_debug_pipeline_);
        csm_debug_pipeline_ = nullptr;
    }
    if (csm_debug_pipeline_layout_) {
        device_->destroyPipelineLayout(csm_debug_pipeline_layout_);
        csm_debug_pipeline_layout_ = nullptr;
    }
    if (csm_debug_desc_set_layout_) {
        device_->destroyDescriptorSetLayout(csm_debug_desc_set_layout_);
        csm_debug_desc_set_layout_ = nullptr;
    }

    // Destroy CSM per-layer image views BEFORE destroying the scene view that
    // owns the underlying array image — Vulkan requires views are destroyed
    // before the image they reference.
    for (int k = 0; k < CSM_CASCADE_COUNT; ++k) {
        if (csm_layer_views_[k]) {
            device_->destroyImageView(csm_layer_views_[k]);
            csm_layer_views_[k] = nullptr;
        }
    }

    // Destroys the array texture (csm_shadow_tex_) via m_depth_buffer_.
    shadow_object_scene_view_->destroy(device_);

    er::helper::clearCachedShaderModules(device_);

    device_->destroySemaphore(init_semaphore_);

    for (uint64_t i = 0; i < kMaxFramesInFlight; i++) {
        device_->destroySemaphore(render_finished_semaphores_[i]);
        device_->destroySemaphore(image_available_semaphores_[i]);
        device_->destroyFence(in_flight_fences_[i]);
    }

    device_->destroyCommandPool(command_pool_);
    device_->destroy();

    instance_->destroySurface(surface_);
    instance_->destroy();

    glfwDestroyWindow(window_);
    glfwTerminate();
}

}//namespace app
}//namespace work
