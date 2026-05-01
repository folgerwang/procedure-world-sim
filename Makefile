# =============================================================================
# RealWorld — cross-platform Makefile (replaces realworld.sln)
#
# Supported platforms: Linux (x86_64), macOS (arm64/x86_64), Windows (MinGW)
#
# Usage:
#   make                         # Debug build (auto-exports ML model if missing)
#   make BUILD=release           # Release build
#   make model                   # Export ML model only
#   make shaders                 # Compile all GLSL shaders to SPIR-V
#   make clean                   # Remove all build artifacts
#   make submodules              # Initialize / update all git submodules
#   make help                    # Show this message
#
# Key variables (override on command line):
#   BUILD        debug (default) | release
#   CXX          C++ compiler    (default: g++ on Linux, clang++ on macOS)
#   CC           C compiler      (default: gcc on Linux, clang on macOS)
#   VULKAN_SDK   Path to the LunarG Vulkan SDK (auto-detected if set in env)
#   LIBTORCH_DIR Path to LibTorch installation (enables ML-based auto-rig)
# =============================================================================

BUILD ?= debug

# ML model path
MODEL_PT := realworld/assets/models/rig_diffusion.pt

# Derived directories
BUILD_DIR := build/$(BUILD)
OBJ_DIR   := $(BUILD_DIR)/obj
LIB_DIR   := realworld/src/lib

# ── Detect host platform ──────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    PLATFORM := windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        PLATFORM := linux
    else ifeq ($(UNAME_S),Darwin)
        PLATFORM := macos
    else
        $(error Unsupported platform: $(UNAME_S))
    endif
endif

# ── LibTorch auto-detection / download config ────────────────────────────────
LIBTORCH_VERSION ?= 2.7.0
LIBTORCH_LOCAL    = realworld/src/sim_engine/third_parties/libtorch

# If LIBTORCH_DIR is not explicitly set, check third_parties/libtorch
LIBTORCH_DIR ?=
ifeq ($(LIBTORCH_DIR),)
    ifneq ($(wildcard $(LIBTORCH_LOCAL)/share/cmake/Torch/TorchConfig.cmake),)
        LIBTORCH_DIR := $(LIBTORCH_LOCAL)
    endif
endif

# Platform-specific download URL
LIBTORCH_BASE_URL := https://download.pytorch.org/libtorch
ifeq ($(PLATFORM),linux)
    LIBTORCH_ZIP_NAME := libtorch-cxx11-abi-shared-with-deps-$(LIBTORCH_VERSION)%2Bcpu.zip
    LIBTORCH_URL      := $(LIBTORCH_BASE_URL)/cpu/$(LIBTORCH_ZIP_NAME)
else ifeq ($(PLATFORM),macos)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
        LIBTORCH_ZIP_NAME := libtorch-macos-arm64-$(LIBTORCH_VERSION).zip
    else
        LIBTORCH_ZIP_NAME := libtorch-macos-x86_64-$(LIBTORCH_VERSION).zip
    endif
    LIBTORCH_URL := $(LIBTORCH_BASE_URL)/cpu/$(LIBTORCH_ZIP_NAME)
else ifeq ($(PLATFORM),windows)
    LIBTORCH_ZIP_NAME := libtorch-win-shared-with-deps-$(LIBTORCH_VERSION)%2Bcpu.zip
    LIBTORCH_URL      := $(LIBTORCH_BASE_URL)/cpu/$(LIBTORCH_ZIP_NAME)
endif

# ── Toolchain ─────────────────────────────────────────────────────────────────
ifeq ($(PLATFORM),macos)
    CXX ?= clang++
    CC  ?= clang
else
    CXX ?= g++
    CC  ?= gcc
endif

AR      := ar
ARFLAGS := rcs

# ── Compiler flags ────────────────────────────────────────────────────────────
CXXFLAGS := -std=c++20 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers

ifeq ($(BUILD),debug)
    CXXFLAGS += -g -O0 -D_DEBUG
    CFLAGS   := -g -O0 -D_DEBUG
else ifeq ($(BUILD),release)
    CXXFLAGS += -O2 -DNDEBUG
    CFLAGS   := -O2 -DNDEBUG
else
    $(error BUILD must be 'debug' or 'release', got '$(BUILD)')
endif

# Definitions shared across most translation units
COMMON_DEFINES := -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS

ifeq ($(PLATFORM),windows)
    COMMON_DEFINES += -DWIN32 -DSTBI_MSC_SECURE_CRT
endif

CXXFLAGS += $(COMMON_DEFINES)
CFLAGS   += $(COMMON_DEFINES)

# ── LibTorch integration (optional) ──────────────────────────────────────────
ifneq ($(LIBTORCH_DIR),)
    CXXFLAGS += -DHAS_LIBTORCH=1
    LIBTORCH_INCLUDES := -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
    LIBTORCH_LDFLAGS  := -L$(LIBTORCH_DIR)/lib -Wl,-rpath,$(LIBTORCH_DIR)/lib
    LIBTORCH_LDLIBS   := -ltorch -ltorch_cpu -lc10
    # Add CUDA libs if available
    ifneq ($(wildcard $(LIBTORCH_DIR)/lib/libtorch_cuda.so),)
        LIBTORCH_LDLIBS += -ltorch_cuda -lc10_cuda
    endif
    ifneq ($(wildcard $(LIBTORCH_DIR)/lib/libtorch_cuda.dylib),)
        LIBTORCH_LDLIBS += -ltorch_cuda -lc10_cuda
    endif
else
    LIBTORCH_INCLUDES :=
    LIBTORCH_LDFLAGS  :=
    LIBTORCH_LDLIBS   :=
endif

# ── Key directory paths ───────────────────────────────────────────────────────
ENGINE_DIR   := realworld/src/sim_engine
TP_DIR       := $(ENGINE_DIR)/third_parties
SRC_DIR      := realworld/src

GLFW_DIR     := $(TP_DIR)/glfw
GLFW_SRC     := $(GLFW_DIR)/src
IMGUI_DIR    := $(TP_DIR)/imgui
OPENMESH_DIR := $(TP_DIR)/OpenMesh/src
VULKAN_H_DIR := $(TP_DIR)/Vulkan-Headers/Include

# ── Include paths ─────────────────────────────────────────────────────────────
BASE_INCLUDES := \
    -I$(TP_DIR)/OpenMesh/src          \
    -I$(TP_DIR)/imgui                 \
    -I$(TP_DIR)/imgui/backends        \
    -I$(GLFW_DIR)/include             \
    -I$(TP_DIR)/glm                   \
    -I$(VULKAN_H_DIR)                 \
    -I$(TP_DIR)/tinygltf              \
    -I$(ENGINE_DIR)/shaders           \
    -I$(ENGINE_DIR)                   \
    -I$(SRC_DIR)

# Optional Vulkan SDK (system-wide) include path
ifneq ($(VULKAN_SDK),)
    BASE_INCLUDES += -I$(VULKAN_SDK)/include
endif

# On non-Windows we need the compat stub (provides MessageBoxA etc.)
ifneq ($(PLATFORM),windows)
    BASE_INCLUDES += -Irealworld/compat
endif

# ── GLFW: platform-specific sources & defines ─────────────────────────────────
# Sources shared across all platforms
GLFW_COMMON_C := \
    context.c       \
    egl_context.c   \
    init.c          \
    input.c         \
    monitor.c       \
    null_init.c     \
    null_joystick.c \
    null_monitor.c  \
    null_window.c   \
    osmesa_context.c\
    platform.c      \
    vulkan.c        \
    window.c

ifeq ($(PLATFORM),windows)
    GLFW_PLATFORM_C := \
        wgl_context.c   \
        win32_init.c    \
        win32_joystick.c\
        win32_module.c  \
        win32_monitor.c \
        win32_thread.c  \
        win32_time.c    \
        win32_window.c
    GLFW_PLATFORM_DEF  := -D_GLFW_WIN32
    GLFW_LINK_EXTRAS   :=
else ifeq ($(PLATFORM),linux)
    GLFW_PLATFORM_C := \
        glx_context.c   \
        x11_init.c      \
        x11_monitor.c   \
        x11_window.c    \
        posix_module.c  \
        posix_poll.c    \
        posix_thread.c  \
        posix_time.c    \
        linux_joystick.c\
        xkb_unicode.c
    GLFW_PLATFORM_DEF  := -D_GLFW_X11
    GLFW_LINK_EXTRAS   := -lX11 -lXrandr -lXi -lXxf86vm -lXcursor -lXinerama
else ifeq ($(PLATFORM),macos)
    GLFW_PLATFORM_C := \
        cocoa_time.c    \
        posix_module.c  \
        posix_thread.c
    GLFW_PLATFORM_M := \
        cocoa_init.m    \
        cocoa_joystick.m\
        cocoa_monitor.m \
        cocoa_window.m  \
        nsgl_context.m
    GLFW_PLATFORM_DEF  := -D_GLFW_COCOA
    GLFW_LINK_EXTRAS   := \
        -framework Cocoa          \
        -framework IOKit          \
        -framework CoreVideo      \
        -framework CoreFoundation
endif

# Expand to full paths
GLFW_ALL_C := $(addprefix $(GLFW_SRC)/, $(GLFW_COMMON_C) $(GLFW_PLATFORM_C))
GLFW_C_OBJS := $(patsubst $(GLFW_SRC)/%.c, $(OBJ_DIR)/glfw/%.o, $(GLFW_ALL_C))

ifeq ($(PLATFORM),macos)
    GLFW_ALL_M  := $(addprefix $(GLFW_SRC)/, $(GLFW_PLATFORM_M))
    GLFW_M_OBJS := $(patsubst $(GLFW_SRC)/%.m, $(OBJ_DIR)/glfw/%.o, $(GLFW_ALL_M))
else
    GLFW_M_OBJS :=
endif

GLFW_OBJS := $(GLFW_C_OBJS) $(GLFW_M_OBJS)
GLFW3_LIB := $(LIB_DIR)/libglfw3_$(BUILD).a

# ── ImGui sources ─────────────────────────────────────────────────────────────
IMGUI_SRCS := \
    $(IMGUI_DIR)/imgui.cpp                        \
    $(IMGUI_DIR)/imgui_draw.cpp                   \
    $(IMGUI_DIR)/imgui_tables.cpp                 \
    $(IMGUI_DIR)/imgui_widgets.cpp                \
    $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp     \
    $(IMGUI_DIR)/backends/imgui_impl_vulkan.cpp

IMGUI_INCLUDES := \
    -I$(IMGUI_DIR)           \
    -I$(IMGUI_DIR)/backends  \
    -I$(GLFW_DIR)/include    \
    -I$(VULKAN_H_DIR)

IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp, $(OBJ_DIR)/imgui/%.o, $(IMGUI_SRCS))
IMGUI_LIB  := $(LIB_DIR)/libimgui_$(BUILD).a

# ── OpenMesh sources ──────────────────────────────────────────────────────────
OPENMESH_SRCS := \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/BinaryHelper.cc                  \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/IOManager.cc                     \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/OMFormat.cc                      \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/BaseReader.cc             \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/OBJReader.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/OFFReader.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/OMReader.cc               \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/PLYReader.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/reader/STLReader.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/BaseWriter.cc             \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/OBJWriter.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/OFFWriter.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/OMWriter.cc               \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/PLYWriter.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/STLWriter.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/IO/writer/VTKWriter.cc              \
    $(OPENMESH_DIR)/OpenMesh/Core/Mesh/ArrayKernel.cc                 \
    $(OPENMESH_DIR)/OpenMesh/Core/Mesh/BaseKernel.cc                  \
    $(OPENMESH_DIR)/OpenMesh/Core/Mesh/PolyConnectivity.cc            \
    $(OPENMESH_DIR)/OpenMesh/Core/Mesh/TriConnectivity.cc             \
    $(OPENMESH_DIR)/OpenMesh/Core/System/omstream.cc                  \
    $(OPENMESH_DIR)/OpenMesh/Core/Utils/BaseProperty.cc               \
    $(OPENMESH_DIR)/OpenMesh/Core/Utils/Endian.cc                     \
    $(OPENMESH_DIR)/OpenMesh/Core/Utils/PropertyCreator.cc            \
    $(OPENMESH_DIR)/OpenMesh/Core/Utils/RandomNumberGenerator.cc      \
    $(OPENMESH_DIR)/OpenMesh/Tools/Decimater/Observer.cc

OPENMESH_DEFINES  := -DOM_STATIC_BUILD
OPENMESH_INCLUDES := -I$(OPENMESH_DIR)

OPENMESH_OBJS := $(patsubst $(OPENMESH_DIR)/%.cc, $(OBJ_DIR)/openmesh/%.o, $(OPENMESH_SRCS))
OPENMESH_LIB  := $(LIB_DIR)/libopen-mesh_$(BUILD).a

# ── Engine (sim_engine) sources ───────────────────────────────────────────────
ENGINE_CPP_SRCS := \
    $(ENGINE_DIR)/game_object/box.cpp               \
    $(ENGINE_DIR)/game_object/camera.cpp            \
    $(ENGINE_DIR)/game_object/camera_object.cpp     \
    $(ENGINE_DIR)/game_object/conemap_obj.cpp       \
    $(ENGINE_DIR)/game_object/conemap_test.cpp      \
    $(ENGINE_DIR)/game_object/debug_draw.cpp        \
    $(ENGINE_DIR)/game_object/drawable_object.cpp   \
    $(ENGINE_DIR)/game_object/hair_patch.cpp        \
    $(ENGINE_DIR)/game_object/hair_test.cpp         \
    $(ENGINE_DIR)/game_object/lbm_patch.cpp         \
    $(ENGINE_DIR)/game_object/lbm_test.cpp          \
    $(ENGINE_DIR)/game_object/object_file.cpp       \
    $(ENGINE_DIR)/game_object/patch.cpp             \
    $(ENGINE_DIR)/game_object/plane.cpp             \
    $(ENGINE_DIR)/game_object/player_controller.cpp \
    $(ENGINE_DIR)/game_object/shape_base.cpp        \
    $(ENGINE_DIR)/game_object/sphere.cpp            \
    $(ENGINE_DIR)/game_object/terrain.cpp           \
    $(ENGINE_DIR)/game_object/view_object.cpp       \
    $(ENGINE_DIR)/helper/bvh.cpp                    \
    $(ENGINE_DIR)/helper/collision_debug_draw.cpp   \
    $(ENGINE_DIR)/helper/collision_mesh.cpp         \
    $(ENGINE_DIR)/helper/engine_helper.cpp          \
    $(ENGINE_DIR)/helper/mesh_tool.cpp              \
    $(ENGINE_DIR)/ray_tracing/raytracing_callable.cpp       \
    $(ENGINE_DIR)/ray_tracing/raytracing_shadow.cpp         \
    $(ENGINE_DIR)/renderer/renderer.cpp                     \
    $(ENGINE_DIR)/renderer/renderer_helper.cpp              \
    $(ENGINE_DIR)/renderer/vulkan/vk_command_buffer.cpp     \
    $(ENGINE_DIR)/renderer/vulkan/vk_device.cpp             \
    $(ENGINE_DIR)/renderer/vulkan/vk_renderer_helper.cpp    \
    $(ENGINE_DIR)/scene_rendering/conemap.cpp               \
    $(ENGINE_DIR)/scene_rendering/ibl_creator.cpp           \
    $(ENGINE_DIR)/scene_rendering/object_scene_view.cpp     \
    $(ENGINE_DIR)/scene_rendering/prt_shadow.cpp            \
    $(ENGINE_DIR)/scene_rendering/skydome.cpp               \
    $(ENGINE_DIR)/scene_rendering/terrain_scene_view.cpp    \
    $(ENGINE_DIR)/scene_rendering/view_capture.cpp          \
    $(ENGINE_DIR)/scene_rendering/ssao.cpp                   \
    $(ENGINE_DIR)/scene_rendering/cluster_renderer.cpp      \
    $(ENGINE_DIR)/scene_rendering/volume_cloud.cpp          \
    $(ENGINE_DIR)/scene_rendering/volume_noise.cpp          \
    $(ENGINE_DIR)/scene_rendering/weather_system.cpp        \
    $(ENGINE_DIR)/ui/chat_box.cpp                           \
    $(ENGINE_DIR)/ui/menu.cpp                               \
    $(ENGINE_DIR)/helper/gpu_profiler.cpp

# ── Plugin sources ────────────────────────────────────────────────────────────
PLUGIN_CPP_SRCS := \
    $(SRC_DIR)/plugins/plugin_manager.cpp                   \
    $(SRC_DIR)/plugins/auto_rig/auto_rig_plugin.cpp         \
    $(SRC_DIR)/plugins/auto_rig/simple_rasterizer.cpp       \
    $(SRC_DIR)/plugins/auto_rig/rig_diffusion_model.cpp

ENGINE_CPP_SRCS += $(PLUGIN_CPP_SRCS)

ENGINE_C_SRCS := \
    $(ENGINE_DIR)/third_parties/fbx/ufbx.c

# Engine objects: engine sources live under ENGINE_DIR, plugin sources under SRC_DIR
ENGINE_ENGDIR_CPP := $(filter $(ENGINE_DIR)/%, $(ENGINE_CPP_SRCS))
ENGINE_SRCDIR_CPP := $(filter $(SRC_DIR)/%, $(ENGINE_CPP_SRCS))

ENGINE_CPP_OBJS := $(patsubst $(ENGINE_DIR)/%.cpp, $(OBJ_DIR)/engine/%.o, $(ENGINE_ENGDIR_CPP)) \
                   $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/plugins/%.o, $(ENGINE_SRCDIR_CPP))
ENGINE_C_OBJS   := $(patsubst $(ENGINE_DIR)/%.c,   $(OBJ_DIR)/engine/%.o, $(ENGINE_C_SRCS))
ENGINE_OBJS     := $(ENGINE_CPP_OBJS) $(ENGINE_C_OBJS)
ENGINE_LIB      := $(LIB_DIR)/libengine_$(BUILD).a

# ── Main application sources ──────────────────────────────────────────────────
APP_SRCS := \
    $(SRC_DIR)/application.cpp  \
    $(SRC_DIR)/main.cpp

APP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/app/%.o, $(APP_SRCS))

# Executable lives beside assets/ so relative paths like "assets/..." resolve.
ifeq ($(PLATFORM),windows)
    TARGET := realworld/RealWorld.exe
else
    TARGET := realworld/RealWorld
endif

# ── Linker settings ───────────────────────────────────────────────────────────
LDFLAGS := -L$(LIB_DIR) $(LIBTORCH_LDFLAGS)

# Static libraries (order matters: dependents before dependencies)
LDLIBS := \
    -lengine_$(BUILD)     \
    -limgui_$(BUILD)      \
    -lglfw3_$(BUILD)      \
    -lopen-mesh_$(BUILD)  \
    $(LIBTORCH_LDLIBS)

ifeq ($(PLATFORM),windows)
    # MinGW: use bundled Vulkan import library
    LDFLAGS += -L$(TP_DIR)/vulkan_lib
    LDLIBS  += -lvulkan-1 -lgdi32 -luser32 -lshell32 -lole32
else ifeq ($(PLATFORM),linux)
    LDLIBS  += -lvulkan -ldl -lpthread $(GLFW_LINK_EXTRAS)
else ifeq ($(PLATFORM),macos)
    LDLIBS  += -lvulkan -ldl -lpthread $(GLFW_LINK_EXTRAS)
    ifneq ($(VULKAN_SDK),)
        LDFLAGS += -L$(VULKAN_SDK)/lib
    endif
endif

# ── Shader compiler ───────────────────────────────────────────────────────────
SHADER_DIR := $(ENGINE_DIR)/shaders

ifeq ($(PLATFORM),windows)
    GLSLC := $(TP_DIR)/vulkan_lib/glslc.exe
else
    # Prefer SDK glslc if VULKAN_SDK is set, else fall back to PATH
    ifneq ($(VULKAN_SDK),)
        GLSLC := $(VULKAN_SDK)/bin/glslc
    else
        GLSLC := glslc
    endif
endif

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all clean shaders submodules model libtorch help

# ── Default target ────────────────────────────────────────────────────────────
all: libtorch model $(TARGET)
	@echo ""
	@echo "Build complete: $(TARGET)  (run from the realworld/ directory)"
ifneq ($(LIBTORCH_DIR),)
	@echo "  LibTorch: $(LIBTORCH_DIR)  (ML auto-rig enabled)"
else
	@echo "  LibTorch: not found  (auto-rig uses heuristic mode)"
	@echo "  Run 'make libtorch' or set LIBTORCH_DIR to enable ML inference"
endif

# ── LibTorch auto-download (to third_parties/libtorch) ───────────────────────
libtorch:
	@if [ -n "$(LIBTORCH_DIR)" ] && [ -d "$(LIBTORCH_DIR)" ]; then \
	    echo "[libtorch] Using: $(LIBTORCH_DIR)"; \
	elif [ -f "$(LIBTORCH_LOCAL)/share/cmake/Torch/TorchConfig.cmake" ]; then \
	    echo "[libtorch] Found: $(LIBTORCH_LOCAL)"; \
	else \
	    echo "[libtorch] Not found — downloading LibTorch $(LIBTORCH_VERSION) (CPU)..."; \
	    echo "[libtorch] URL: $(LIBTORCH_URL)"; \
	    echo "[libtorch] This is a one-time ~200 MB download."; \
	    echo ""; \
	    _tmpzip="$(BUILD_DIR)/libtorch-download.zip"; \
	    mkdir -p "$(BUILD_DIR)"; \
	    if command -v curl >/dev/null 2>&1; then \
	        curl -L --progress-bar -o "$$_tmpzip" "$(LIBTORCH_URL)"; \
	    elif command -v wget >/dev/null 2>&1; then \
	        wget --show-progress -q -O "$$_tmpzip" "$(LIBTORCH_URL)"; \
	    else \
	        echo "[libtorch] ERROR: Neither curl nor wget found."; \
	        echo "         Install curl or wget, or download manually from:"; \
	        echo "         $(LIBTORCH_URL)"; \
	        echo "         Extract to: $(LIBTORCH_LOCAL)"; \
	        exit 1; \
	    fi; \
	    echo "[libtorch] Extracting to $(LIBTORCH_LOCAL)..."; \
	    mkdir -p "$$(dirname $(LIBTORCH_LOCAL))"; \
	    unzip -q -o "$$_tmpzip" -d "$$(dirname $(LIBTORCH_LOCAL))"; \
	    rm -f "$$_tmpzip"; \
	    if [ -f "$(LIBTORCH_LOCAL)/share/cmake/Torch/TorchConfig.cmake" ]; then \
	        echo "[libtorch] Installed successfully: $(LIBTORCH_LOCAL)"; \
	    else \
	        echo "[libtorch] ERROR: Extraction failed — TorchConfig.cmake not found."; \
	        exit 1; \
	    fi; \
	fi

# ── ML model export (only if .pt is missing) ─────────────────────────────────
model:
	@if [ ! -f "$(MODEL_PT)" ]; then \
	    echo "[model] $(MODEL_PT) not found — exporting..."; \
	    mkdir -p $$(dirname $(MODEL_PT)); \
	    if command -v python3 >/dev/null 2>&1; then PY=python3; \
	    elif command -v python >/dev/null 2>&1; then PY=python; \
	    else \
	        echo "[model] Python not found — skipping ML model export."; \
	        echo "        Auto-rig will use heuristic mode."; \
	        exit 0; \
	    fi; \
	    echo "[model] Trying distillation (downloads pretrained weights)..."; \
	    $$PY ml_training/scripts/download_and_export_model.py \
	        --method distill --output $(MODEL_PT) --resolution 256 2>&1 \
	    || { \
	        echo "[model] Distillation failed — falling back to skeleton export..."; \
	        $$PY ml_training/scripts/download_and_export_model.py \
	            --method skeleton --output $(MODEL_PT) --resolution 256 2>&1 \
	        || echo "[model] Export failed. Install PyTorch: pip install torch torchvision"; \
	    }; \
	else \
	    echo "[model] $(MODEL_PT) found — OK"; \
	fi

# ── Final executable ──────────────────────────────────────────────────────────
$(TARGET): $(APP_OBJS) $(ENGINE_LIB) $(IMGUI_LIB) $(GLFW3_LIB) $(OPENMESH_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(APP_OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

# ── Static libraries ──────────────────────────────────────────────────────────
$(ENGINE_LIB): $(ENGINE_OBJS)
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^
	@echo "  [lib] $@"

$(IMGUI_LIB): $(IMGUI_OBJS)
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^
	@echo "  [lib] $@"

$(GLFW3_LIB): $(GLFW_OBJS)
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^
	@echo "  [lib] $@"

$(OPENMESH_LIB): $(OPENMESH_OBJS)
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^
	@echo "  [lib] $@"

# ── Object rules: application ─────────────────────────────────────────────────
$(OBJ_DIR)/app/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(BASE_INCLUDES) -c $< -o $@

# ── Object rules: engine ──────────────────────────────────────────────────────
$(OBJ_DIR)/engine/%.o: $(ENGINE_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(BASE_INCLUDES) $(LIBTORCH_INCLUDES) -c $< -o $@

$(OBJ_DIR)/engine/%.o: $(ENGINE_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(BASE_INCLUDES) -c $< -o $@

# ── Object rules: plugins (under SRC_DIR) ────────────────────────────────────
$(OBJ_DIR)/plugins/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(BASE_INCLUDES) $(LIBTORCH_INCLUDES) -c $< -o $@

# ── Object rules: ImGui ───────────────────────────────────────────────────────
$(OBJ_DIR)/imgui/%.o: $(IMGUI_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(IMGUI_INCLUDES) -c $< -o $@

# ── Object rules: GLFW (C) ────────────────────────────────────────────────────
$(OBJ_DIR)/glfw/%.o: $(GLFW_SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(GLFW_DIR)/include -I$(GLFW_SRC) $(GLFW_PLATFORM_DEF) -c $< -o $@

# ── Object rules: GLFW (Objective-C, macOS only) ─────────────────────────────
$(OBJ_DIR)/glfw/%.o: $(GLFW_SRC)/%.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(GLFW_DIR)/include -I$(GLFW_SRC) $(GLFW_PLATFORM_DEF) -c $< -o $@

# ── Object rules: OpenMesh ────────────────────────────────────────────────────
$(OBJ_DIR)/openmesh/%.o: $(OPENMESH_DIR)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENMESH_DEFINES) $(OPENMESH_INCLUDES) -c $< -o $@

# ── Shader compilation ────────────────────────────────────────────────────────
# Reads shaders-compile.cfg and compiles each entry with glslc.
# Backslashes in paths (Windows-style) are converted to forward slashes.
shaders:
	@echo "Compiling shaders in $(SHADER_DIR) ..."
	@cd $(SHADER_DIR) && \
	while IFS= read -r line || [ -n "$$line" ]; do \
	    [ -z "$$line" ] && continue; \
	    args=$$(printf '%s' "$$line" | tr '\\' '/'); \
	    echo "  $(GLSLC) $$args"; \
	    $(GLSLC) $$args || exit 1; \
	done < shaders-compile.cfg
	@echo "Shaders compiled."

# ── Submodule helpers ─────────────────────────────────────────────────────────
submodules:
	git submodule update --init --recursive

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf build/ $(LIB_DIR) realworld/RealWorld realworld/RealWorld.exe

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "RealWorld — cross-platform Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build everything: download LibTorch, export model, compile (default)"
	@echo "  libtorch    Download LibTorch to third_parties/ (if not present)"
	@echo "  model       Export ML auto-rig model (if missing)"
	@echo "  shaders     Compile all GLSL shaders → SPIR-V (.spv)"
	@echo "  submodules  Run: git submodule update --init --recursive"
	@echo "  clean       Remove build/ and realworld/src/lib/"
	@echo "  help        Show this help"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  BUILD            debug (default) or release"
	@echo "  CXX              C++ compiler   [$(CXX)]"
	@echo "  CC               C   compiler   [$(CC)]"
	@echo "  VULKAN_SDK       Path to LunarG Vulkan SDK (optional)"
	@echo "  LIBTORCH_DIR     Custom LibTorch path (auto-downloaded if omitted)"
	@echo "  LIBTORCH_VERSION LibTorch version to download [$(LIBTORCH_VERSION)]"
	@echo ""
	@echo "Examples:"
	@echo "  make                           # Debug build (auto-downloads LibTorch)"
	@echo "  make BUILD=release             # Release build"
	@echo "  make libtorch                  # Download LibTorch only"
	@echo "  make model                     # Export ML model only"
	@echo "  make shaders                   # Compile shaders only"
	@echo "  make VULKAN_SDK=~/VulkanSDK/1.3.x/x86_64  # Custom SDK path"
	@echo "  make CXX=clang++ CC=clang      # Use Clang"
	@echo ""
	@echo "Prerequisites (Linux):"
	@echo "  apt install build-essential libvulkan-dev vulkan-tools glslc"
	@echo "  apt install libx11-dev libxrandr-dev libxi-dev libxxf86vm-dev"
	@echo "  apt install libxcursor-dev libxinerama-dev"
	@echo ""
	@echo "Prerequisites (macOS):"
	@echo "  xcode-select --install"
	@echo "  Install LunarG Vulkan SDK from https://vulkan.lunarg.com"
	@echo "  Set: export VULKAN_SDK=~/VulkanSDK/<version>/macOS"
