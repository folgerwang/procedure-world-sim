#pragma once
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace engine {
namespace renderer {
    class Device;
}
}

namespace plugins {

// Plugin lifecycle states.
enum class PluginState {
    kUnloaded,
    kLoaded,
    kRunning,
    kFinished,
    kError
};

// Progress callback: (current_step, total_steps, description).
using ProgressCallback = std::function<void(int, int, const std::string&)>;

// ---------------------------------------------------------------------------
//  IPlugin – base interface every plugin must implement.
// ---------------------------------------------------------------------------
class IPlugin {
public:
    virtual ~IPlugin() = default;

    // Human-readable name and version.
    virtual const char* getName()    const = 0;
    virtual const char* getVersion() const = 0;

    // Lifecycle.
    virtual bool init(const std::shared_ptr<engine::renderer::Device>& device) = 0;
    virtual void shutdown() = 0;

    // Current state.
    virtual PluginState getState() const = 0;

    // Optional: let the plugin draw its own ImGui panel.
    virtual void drawImGui() {}

    // Optional: show/hide the plugin's ImGui window.
    virtual void setVisible(bool v) { (void)v; }
    virtual bool isVisible() const { return false; }

    // Optional: set a progress callback so the host can display status.
    virtual void setProgressCallback(ProgressCallback cb) { (void)cb; }
};

// ---------------------------------------------------------------------------
//  Plugin registry helpers – each plugin implements a factory function
//  with this signature, exported as `createPlugin`.
// ---------------------------------------------------------------------------
using PluginFactory = std::unique_ptr<IPlugin>(*)();

} // namespace plugins
