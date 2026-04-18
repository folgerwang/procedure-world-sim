#include "plugin_manager.h"
#include <cassert>
#include <cstdio>
#include "imgui.h"

namespace plugins {

void PluginManager::registerPlugin(std::unique_ptr<IPlugin> plugin) {
    assert(plugin);
    const std::string name = plugin->getName();
    name_map_[name] = plugin.get();
    plugins_.push_back(std::move(plugin));
    fprintf(stderr, "[PluginManager] registered: %s\n", name.c_str());
}

void PluginManager::initAll(
    const std::shared_ptr<engine::renderer::Device>& device) {
    for (auto& p : plugins_) {
        if (p->getState() == PluginState::kUnloaded) {
            if (p->init(device)) {
                fprintf(stderr, "[PluginManager] init OK: %s v%s\n",
                    p->getName(), p->getVersion());
            } else {
                fprintf(stderr, "[PluginManager] init FAILED: %s\n",
                    p->getName());
            }
        }
    }
}

void PluginManager::shutdownAll() {
    for (auto& p : plugins_) {
        if (p->getState() != PluginState::kUnloaded) {
            p->shutdown();
        }
    }
    name_map_.clear();
    plugins_.clear();
}

IPlugin* PluginManager::findPlugin(const std::string& name) const {
    auto it = name_map_.find(name);
    return (it != name_map_.end()) ? it->second : nullptr;
}

void PluginManager::drawAllImGui() {
    ImGuiID main_vp = ImGui::GetMainViewport()->ID;
    for (auto& p : plugins_) {
        if (p->getState() != PluginState::kUnloaded) {
            ImGui::SetNextWindowViewport(main_vp);
            p->drawImGui();
        }
    }
}

} // namespace plugins
