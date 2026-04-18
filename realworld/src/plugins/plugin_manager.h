#pragma once
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "plugin_interface.h"

namespace plugins {

// ---------------------------------------------------------------------------
//  PluginManager – owns all registered plugin instances.
//
//  For now plugins are statically linked (no dynamic .dll loading).
//  Call registerPlugin() for each built-in plugin at startup,
//  then init / shutdown through the manager.
// ---------------------------------------------------------------------------
class PluginManager {
public:
    PluginManager() = default;
    ~PluginManager() { shutdownAll(); }

    // Register a plugin instance (takes ownership).
    void registerPlugin(std::unique_ptr<IPlugin> plugin);

    // Initialise every registered plugin.
    void initAll(const std::shared_ptr<engine::renderer::Device>& device);

    // Shut down and remove all plugins.
    void shutdownAll();

    // Lookup by name (returns nullptr if not found).
    IPlugin* findPlugin(const std::string& name) const;

    // Iterate.
    const std::vector<std::unique_ptr<IPlugin>>& plugins() const { return plugins_; }

    // Draw ImGui panels for every loaded plugin.
    void drawAllImGui();

private:
    std::vector<std::unique_ptr<IPlugin>> plugins_;
    std::unordered_map<std::string, IPlugin*> name_map_;
};

} // namespace plugins
