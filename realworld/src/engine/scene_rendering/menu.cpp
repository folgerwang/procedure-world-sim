#include <vector>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "engine/renderer/renderer.h"
#include "engine/renderer/renderer_helper.h"

#include "menu.h"

namespace er = engine::renderer;

namespace {
    static void check_vk_result(VkResult err)
    {
        if (err == 0)
            return;
        fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
        if (err < 0)
            abort();
    }
}

namespace engine {
namespace scene_rendering {

Menu::Menu(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Instance>& instance,
    GLFWwindow* window,
    const renderer::QueueFamilyIndices& queue_family_indices,
    const renderer::SwapChainInfo& swap_chain_info,
    std::shared_ptr<renderer::Queue> graphics_queue,
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    std::shared_ptr<renderer::RenderPass> render_pass,
    std::shared_ptr<renderer::CommandBuffer> command_buffer) {
    std::string path = "assets";
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        auto path_string = entry.path();
        auto ext_string = std::filesystem::path(path_string).extension();
        if (ext_string == ".glb" || ext_string == ".gltf") {
            gltf_file_names_.push_back(path_string.filename().string());
        }
    }

    renderer::Helper::initImgui(
        device_info,
        instance,
        window,
        queue_family_indices,
        swap_chain_info,
        graphics_queue,
        descriptor_pool,
        render_pass,
        command_buffer);
}

void Menu::init(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::Instance>& instance,
    GLFWwindow* window,
    const renderer::QueueFamilyIndices& queue_family_indices,
    const renderer::SwapChainInfo& swap_chain_info,
    std::shared_ptr<renderer::Queue> graphics_queue,
    std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    std::shared_ptr<renderer::RenderPass> render_pass,
    std::shared_ptr<renderer::CommandBuffer> command_buffer) {

    renderer::Helper::initImgui(
        device_info,
        instance,
        window,
        queue_family_indices,
        swap_chain_info,
        graphics_queue,
        descriptor_pool,
        render_pass,
        command_buffer);
}

bool Menu::draw(
    std::shared_ptr<er::CommandBuffer> command_buffer,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const er::SwapChainInfo& swap_chain_info,
    const glm::uvec2& screen_size,
    uint32_t image_index) {

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    std::vector<er::ClearValue> clear_values;
    clear_values.resize(2);
    clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
    clear_values[1].depth_stencil = { 1.0f, 0 };

    command_buffer->beginRenderPass(
        render_pass,
        swap_chain_info.framebuffers[image_index],
        screen_size,
        clear_values);

    static bool s_select_load_gltf = false;
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Game Objects"))
        {
            if (ImGui::MenuItem("Load gltf", NULL)) {
                s_select_load_gltf = true;
            }

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Tools"))
        {
            if (ImGui::MenuItem("Turn off water pass", NULL, turn_off_water_pass_)) {
                turn_off_water_pass_ = !turn_off_water_pass_;
            }
            if (ImGui::MenuItem("Turn on air flow effect", NULL, turn_on_airflow_)) {
                turn_on_airflow_ = !turn_on_airflow_;
            }
            static float begin = 10, end = 90;
            ImGui::DragFloatRange2("range", &begin, &end, 0.25f, 0.0f, 100.0f, "Min: %.1f %%", "Max: %.1f %%");

            ImGui::Separator();

            const char* items[] = { "no debug draw", "debug temperature", "debug moisture" };
            if (ImGui::BeginCombo("##debugmode", items[debug_draw_type_])) // The second parameter is the label previewed before opening the combo.
            {
                for (int n = 0; n < IM_ARRAYSIZE(items); n++)
                {
                    bool is_selected = (items[debug_draw_type_] == items[n]); // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(items[n], is_selected)) {
                        debug_draw_type_ = n;
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    bool in_focus =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) ||
        ImGui::IsWindowFocused(ImGuiHoveredFlags_AnyWindow);

    if (s_select_load_gltf) {
        ImGui::OpenPopup("select gltf object");
        if (ImGui::BeginPopup("select gltf object"))
        {
            std::vector<const char*> listbox_items;
            for (const auto& name : gltf_file_names_) {
                listbox_items.push_back(name.c_str());
            }
            static int s_listbox_item_current = -1;
            ImGui::ListBox("",
                &s_listbox_item_current, listbox_items.data(),
                static_cast<int>(listbox_items.size()),
                static_cast<int>(listbox_items.size()));

            if (s_listbox_item_current >= 0) {
                auto file_name = "assets/" + gltf_file_names_[s_listbox_item_current];
                to_load_gltf_names_.push_back(file_name);
                s_listbox_item_current = -1;
                s_select_load_gltf = false;
            }

            ImGui::EndPopup();
        }

        if (!s_select_load_gltf) {
            ImGui::CloseCurrentPopup();
        }
    }

    renderer::Helper::addImGuiToCommandBuffer(command_buffer);

    command_buffer->endRenderPass();

    return in_focus;
}

void Menu::destroy() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

}//namespace scene_rendering
}//namespace engine
