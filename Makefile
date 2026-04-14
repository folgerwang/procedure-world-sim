# =============================================================================
# RealWorld — cross-platform Makefile (replaces realworld.sln)
#
# Supported platforms: Linux (x86_64), macOS (arm64/x86_64), Windows (MinGW)
#
# Usage:
#   make                         # Debug build
#   make BUILD=release           # Release build
#   make shaders                 # Compile all GLSL shaders to SPIR-V
#   make clean                   # Remove all build artifacts
#   make submodules              # Initialize / update all git submodules
#   make help                    # Show this message
#
# Key variables (override on command line):
#   BUILD       debug (default) | release
#   CXX         C++ compiler    (default: g++ on Linux, clang++ on macOS)
#   CC          C compiler      (default: gcc on Linux, clang on macOS)
#   VULKAN_SDK  Path to the LunarG Vulkan SDK (auto-detected if set in env)
# =============================================================================

BUILD ?= debug

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
    $(ENGINE_DIR)/game_object/shape_base.cpp        \
    $(ENGINE_DIR)/game_object/sphere.cpp            \
    $(ENGINE_DIR)/game_object/terrain.cpp           \
    $(ENGINE_DIR)/game_object/view_object.cpp       \
    $(ENGINE_DIR)/helper/bvh.cpp                    \
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
    $(ENGINE_DIR)/scene_rendering/volume_cloud.cpp          \
    $(ENGINE_DIR)/scene_rendering/volume_noise.cpp          \
    $(ENGINE_DIR)/scene_rendering/weather_system.cpp        \
    $(ENGINE_DIR)/ui/chat_box.cpp                           \
    $(ENGINE_DIR)/ui/menu.cpp

ENGINE_C_SRCS := \
    $(ENGINE_DIR)/third_parties/fbx/ufbx.c

ENGINE_CPP_OBJS := $(patsubst $(ENGINE_DIR)/%.cpp, $(OBJ_DIR)/engine/%.o, $(ENGINE_CPP_SRCS))
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
LDFLAGS := -L$(LIB_DIR)

# Static libraries (order matters: dependents before dependencies)
LDLIBS := \
    -lengine_$(BUILD)     \
    -limgui_$(BUILD)      \
    -lglfw3_$(BUILD)      \
    -lopen-mesh_$(BUILD)

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
.PHONY: all clean shaders submodules help

# ── Default target ────────────────────────────────────────────────────────────
all: $(TARGET)
	@echo ""
	@echo "Build complete: $(TARGET)  (run from the realworld/ directory)"

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
	$(CXX) $(CXXFLAGS) $(BASE_INCLUDES) -c $< -o $@

$(OBJ_DIR)/engine/%.o: $(ENGINE_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(BASE_INCLUDES) -c $< -o $@

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
	@echo "  all         Build the RealWorld executable (default)"
	@echo "  shaders     Compile all GLSL shaders → SPIR-V (.spv)"
	@echo "  submodules  Run: git submodule update --init --recursive"
	@echo "  clean       Remove build/ and realworld/src/lib/"
	@echo "  help        Show this help"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  BUILD       debug (default) or release"
	@echo "  CXX         C++ compiler   [$(CXX)]"
	@echo "  CC          C   compiler   [$(CC)]"
	@echo "  VULKAN_SDK  Path to LunarG Vulkan SDK (optional)"
	@echo ""
	@echo "Examples:"
	@echo "  make                           # Debug build"
	@echo "  make BUILD=release             # Release build"
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
