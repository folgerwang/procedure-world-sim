cmake_minimum_required(VERSION 3.20)

# Objective-C is only needed on macOS (for GLFW's Cocoa backend .m files)
if(APPLE)
    project(RealWorld LANGUAGES C CXX OBJC)
else()
    project(RealWorld LANGUAGES C CXX)
endif()

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Optional LibTorch for ML-based auto-rigging ──────────────────────────────
#  LibTorch is resolved in this order:
#    1) CMAKE_PREFIX_PATH (user-specified via -DCMAKE_PREFIX_PATH=...)
#    2) third_parties/libtorch (auto-downloaded by GenerateProjectFiles.bat / make)
#    3) Not found → auto-rig uses heuristic mode
#
#  To download LibTorch automatically, run GenerateProjectFiles.bat or
#    make libtorch
#  before configuring CMake.

set(LIBTORCH_LOCAL "${CMAKE_SOURCE_DIR}/realworld/src/sim_engine/third_parties/libtorch")

# Check third_parties/libtorch if find_package hasn't been given a path
find_package(Torch QUIET)
if(NOT Torch_FOUND AND EXISTS "${LIBTORCH_LOCAL}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL}")
    find_package(Torch QUIET)
endif()

if(Torch_FOUND)
    message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
    add_compile_definitions(HAS_LIBTORCH=1)
else()
    # Attempt auto-download at configure time (Windows only, via PowerShell)
    set(LIBTORCH_VERSION "2.7.0")
    if(WIN32)
        set(_lt_url "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
    elseif(APPLE)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _arch OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_arch STREQUAL "arm64")
            set(_lt_url "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
        else()
            set(_lt_url "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${LIBTORCH_VERSION}.zip")
        endif()
    else()
        set(_lt_url "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip")
    endif()

    message(STATUS "LibTorch not found — downloading v${LIBTORCH_VERSION} (CPU)...")
    message(STATUS "  URL: ${_lt_url}")
    message(STATUS "  Destination: ${LIBTORCH_LOCAL}")

    set(_lt_zip "${CMAKE_BINARY_DIR}/libtorch-download.zip")
    file(DOWNLOAD "${_lt_url}" "${_lt_zip}"
         STATUS _dl_status
         SHOW_PROGRESS
         TLS_VERIFY ON)

    list(GET _dl_status 0 _dl_code)
    if(_dl_code EQUAL 0)
        message(STATUS "Extracting LibTorch...")
        file(ARCHIVE_EXTRACT INPUT "${_lt_zip}"
             DESTINATION "${LIBTORCH_LOCAL}/..")
        file(REMOVE "${_lt_zip}")

        # Retry find_package
        list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL}")
        find_package(Torch QUIET)
        if(Torch_FOUND)
            message(STATUS "LibTorch downloaded and found: ${TORCH_INSTALL_PREFIX}")
            add_compile_definitions(HAS_LIBTORCH=1)
        else()
            message(WARNING "LibTorch downloaded but TorchConfig.cmake not found.")
        endif()
    else()
        list(GET _dl_status 1 _dl_msg)
        message(STATUS "LibTorch download failed: ${_dl_msg}")
        message(STATUS "Auto-rig will use heuristic mode. To enable ML inference:")
        message(STATUS "  Option A: make libtorch   (then re-run CMake)")
        message(STATUS "  Option B: cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch")
    endif()
endif()

# ── Key source paths ──────────────────────────────────────────────────────────
set(ENGINE_DIR  "${CMAKE_SOURCE_DIR}/realworld/src/sim_engine")
set(TP_DIR      "${ENGINE_DIR}/third_parties")
set(SRC_DIR     "${CMAKE_SOURCE_DIR}/realworld/src")
set(GLFW_SRC    "${TP_DIR}/glfw/src")
set(OM_SRC      "${TP_DIR}/OpenMesh/src")

# ── Global compile definitions ────────────────────────────────────────────────
add_compile_definitions(
    _USE_MATH_DEFINES
    _CRT_SECURE_NO_WARNINGS
)

if(WIN32)
    add_compile_definitions(WIN32 UNICODE _UNICODE STBI_MSC_SECURE_CRT)
endif()

# ── Common include directories (shared by engine + app) ───────────────────────
set(COMMON_INCLUDES
    "${OM_SRC}"
    "${TP_DIR}/imgui"
    "${TP_DIR}/imgui/backends"
    "${TP_DIR}/glfw/include"
    "${TP_DIR}/glm"
    "${TP_DIR}/Vulkan-Headers/Include"
    "${TP_DIR}/tinygltf"
    "${ENGINE_DIR}/shaders"
    "${ENGINE_DIR}"
    "${SRC_DIR}"
)

# Compat stub: provides MessageBoxA etc. for non-Windows builds
if(NOT WIN32)
    list(APPEND COMMON_INCLUDES "${CMAKE_SOURCE_DIR}/realworld/compat")
endif()

# =============================================================================
# glfw3  (static library)
# =============================================================================
set(GLFW_COMMON_SRCS
    ${GLFW_SRC}/context.c
    ${GLFW_SRC}/egl_context.c
    ${GLFW_SRC}/init.c
    ${GLFW_SRC}/input.c
    ${GLFW_SRC}/monitor.c
    ${GLFW_SRC}/null_init.c
    ${GLFW_SRC}/null_joystick.c
    ${GLFW_SRC}/null_monitor.c
    ${GLFW_SRC}/null_window.c
    ${GLFW_SRC}/osmesa_context.c
    ${GLFW_SRC}/platform.c
    ${GLFW_SRC}/vulkan.c
    ${GLFW_SRC}/window.c
)

if(WIN32)
    set(GLFW_PLATFORM_SRCS
        ${GLFW_SRC}/wgl_context.c
        ${GLFW_SRC}/win32_init.c
        ${GLFW_SRC}/win32_joystick.c
        ${GLFW_SRC}/win32_module.c
        ${GLFW_SRC}/win32_monitor.c
        ${GLFW_SRC}/win32_thread.c
        ${GLFW_SRC}/win32_time.c
        ${GLFW_SRC}/win32_window.c
    )
    set(GLFW_PLATFORM_DEF _GLFW_WIN32)
elseif(APPLE)
    set(GLFW_PLATFORM_SRCS
        ${GLFW_SRC}/cocoa_init.m
        ${GLFW_SRC}/cocoa_joystick.m
        ${GLFW_SRC}/cocoa_monitor.m
        ${GLFW_SRC}/cocoa_window.m
        ${GLFW_SRC}/nsgl_context.m
        ${GLFW_SRC}/cocoa_time.c
        ${GLFW_SRC}/posix_module.c
        ${GLFW_SRC}/posix_thread.c
    )
    set(GLFW_PLATFORM_DEF _GLFW_COCOA)
else()  # Linux
    set(GLFW_PLATFORM_SRCS
        ${GLFW_SRC}/glx_context.c
        ${GLFW_SRC}/x11_init.c
        ${GLFW_SRC}/x11_monitor.c
        ${GLFW_SRC}/x11_window.c
        ${GLFW_SRC}/posix_module.c
        ${GLFW_SRC}/posix_poll.c
        ${GLFW_SRC}/posix_thread.c
        ${GLFW_SRC}/posix_time.c
        ${GLFW_SRC}/linux_joystick.c
        ${GLFW_SRC}/xkb_unicode.c
    )
    set(GLFW_PLATFORM_DEF _GLFW_X11)
endif()

add_library(glfw3 STATIC ${GLFW_COMMON_SRCS} ${GLFW_PLATFORM_SRCS})
target_compile_definitions(glfw3 PRIVATE ${GLFW_PLATFORM_DEF})
target_include_directories(glfw3 PRIVATE
    "${TP_DIR}/glfw/include"
    "${GLFW_SRC}"
)

# =============================================================================
# imgui  (static library)
# =============================================================================
add_library(imgui STATIC
    "${TP_DIR}/imgui/imgui.cpp"
    "${TP_DIR}/imgui/imgui_draw.cpp"
    "${TP_DIR}/imgui/imgui_tables.cpp"
    "${TP_DIR}/imgui/imgui_widgets.cpp"
    "${TP_DIR}/imgui/backends/imgui_impl_glfw.cpp"
    "${TP_DIR}/imgui/backends/imgui_impl_vulkan.cpp"
)
target_include_directories(imgui PRIVATE
    "${TP_DIR}/imgui"
    "${TP_DIR}/imgui/backends"
    "${TP_DIR}/glfw/include"
    "${TP_DIR}/Vulkan-Headers/Include"
)

# =============================================================================
# open-mesh  (static library)
# =============================================================================
add_library(open-mesh STATIC
    ${OM_SRC}/OpenMesh/Core/IO/BinaryHelper.cc
    ${OM_SRC}/OpenMesh/Core/IO/IOManager.cc
    ${OM_SRC}/OpenMesh/Core/IO/OMFormat.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/BaseReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/OBJReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/OFFReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/OMReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/PLYReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/reader/STLReader.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/BaseWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/OBJWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/OFFWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/OMWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/PLYWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/STLWriter.cc
    ${OM_SRC}/OpenMesh/Core/IO/writer/VTKWriter.cc
    ${OM_SRC}/OpenMesh/Core/Mesh/ArrayKernel.cc
    ${OM_SRC}/OpenMesh/Core/Mesh/BaseKernel.cc
    ${OM_SRC}/OpenMesh/Core/Mesh/PolyConnectivity.cc
    ${OM_SRC}/OpenMesh/Core/Mesh/TriConnectivity.cc
    ${OM_SRC}/OpenMesh/Core/System/omstream.cc
    ${OM_SRC}/OpenMesh/Core/Utils/BaseProperty.cc
    ${OM_SRC}/OpenMesh/Core/Utils/Endian.cc
    ${OM_SRC}/OpenMesh/Core/Utils/PropertyCreator.cc
    ${OM_SRC}/OpenMesh/Core/Utils/RandomNumberGenerator.cc
    ${OM_SRC}/OpenMesh/Tools/Decimater/Observer.cc
)
target_compile_definitions(open-mesh PRIVATE OM_STATIC_BUILD)
target_include_directories(open-mesh PRIVATE "${OM_SRC}")

# OpenMesh uses heavy template instantiation — /bigobj required on MSVC
if(MSVC)
    target_compile_options(open-mesh PRIVATE /bigobj)
endif()

# =============================================================================
# engine  (static library — sim_engine submodule)
# =============================================================================
add_library(engine STATIC
    "${ENGINE_DIR}/game_object/box.cpp"
    "${ENGINE_DIR}/game_object/camera.cpp"
    "${ENGINE_DIR}/game_object/camera_object.cpp"
    "${ENGINE_DIR}/game_object/cluster_debug_draw.cpp"
    "${ENGINE_DIR}/game_object/conemap_obj.cpp"
    "${ENGINE_DIR}/game_object/conemap_test.cpp"
    "${ENGINE_DIR}/game_object/debug_draw.cpp"
    "${ENGINE_DIR}/game_object/drawable_object.cpp"
    "${ENGINE_DIR}/game_object/hair_patch.cpp"
    "${ENGINE_DIR}/game_object/hair_test.cpp"
    "${ENGINE_DIR}/game_object/lbm_patch.cpp"
    "${ENGINE_DIR}/game_object/lbm_test.cpp"
    "${ENGINE_DIR}/game_object/mesh_load_task_manager.cpp"
    "${ENGINE_DIR}/game_object/object_file.cpp"
    "${ENGINE_DIR}/game_object/patch.cpp"
    "${ENGINE_DIR}/game_object/plane.cpp"
    "${ENGINE_DIR}/game_object/shape_base.cpp"
    "${ENGINE_DIR}/game_object/sphere.cpp"
    "${ENGINE_DIR}/game_object/terrain.cpp"
    "${ENGINE_DIR}/game_object/view_object.cpp"
    "${ENGINE_DIR}/helper/bvh.cpp"
    "${ENGINE_DIR}/helper/cluster_mesh.cpp"
    "${ENGINE_DIR}/helper/engine_helper.cpp"
    "${ENGINE_DIR}/helper/gpu_profiler.cpp"
    "${ENGINE_DIR}/helper/mesh_tool.cpp"
    "${ENGINE_DIR}/ray_tracing/raytracing_callable.cpp"
    "${ENGINE_DIR}/ray_tracing/raytracing_shadow.cpp"
    "${ENGINE_DIR}/renderer/renderer.cpp"
    "${ENGINE_DIR}/renderer/renderer_helper.cpp"
    "${ENGINE_DIR}/renderer/vulkan/vk_command_buffer.cpp"
    "${ENGINE_DIR}/renderer/vulkan/vk_device.cpp"
    "${ENGINE_DIR}/renderer/vulkan/vk_renderer_helper.cpp"
    "${ENGINE_DIR}/scene_rendering/conemap.cpp"
    "${ENGINE_DIR}/scene_rendering/ibl_creator.cpp"
    "${ENGINE_DIR}/scene_rendering/object_scene_view.cpp"
    "${ENGINE_DIR}/scene_rendering/prt_shadow.cpp"
    "${ENGINE_DIR}/scene_rendering/skydome.cpp"
    "${ENGINE_DIR}/scene_rendering/terrain_scene_view.cpp"
    "${ENGINE_DIR}/scene_rendering/view_capture.cpp"
    "${ENGINE_DIR}/scene_rendering/ssao.cpp"
    "${ENGINE_DIR}/scene_rendering/cluster_renderer.cpp"
    "${ENGINE_DIR}/scene_rendering/volume_cloud.cpp"
    "${ENGINE_DIR}/scene_rendering/volume_noise.cpp"
    "${ENGINE_DIR}/scene_rendering/weather_system.cpp"
    "${ENGINE_DIR}/third_parties/fbx/ufbx.c"
    "${ENGINE_DIR}/ui/chat_box.cpp"
    "${ENGINE_DIR}/ui/menu.cpp"
    "${SRC_DIR}/plugins/plugin_manager.cpp"
    "${SRC_DIR}/plugins/auto_rig/auto_rig_plugin.cpp"
    "${SRC_DIR}/plugins/auto_rig/simple_rasterizer.cpp"
    "${SRC_DIR}/plugins/auto_rig/rig_diffusion_model.cpp"
)
target_include_directories(engine PUBLIC ${COMMON_INCLUDES})

# Link LibTorch to engine if found.
if(Torch_FOUND)
    target_link_libraries(engine PUBLIC ${TORCH_LIBRARIES})
    target_include_directories(engine PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

# =============================================================================
# RealWorld  (executable)
# =============================================================================
add_executable(RealWorld
    "${SRC_DIR}/application.cpp"
    "${SRC_DIR}/main.cpp"
)
target_include_directories(RealWorld PRIVATE ${COMMON_INCLUDES})

# ── Platform-specific linking ─────────────────────────────────────────────────
if(WIN32)
    # Use bundled Vulkan import library
    target_link_directories(RealWorld PRIVATE "${TP_DIR}/vulkan_lib")
    target_link_libraries(RealWorld PRIVATE
        engine imgui glfw3 open-mesh
        vulkan-1
        gdi32 user32 shell32 ole32
    )
elseif(APPLE)
    # Prefer VULKAN_SDK env var; fall back to system libvulkan
    find_library(VULKAN_LIB
        NAMES vulkan
        HINTS "$ENV{VULKAN_SDK}/lib" "/usr/local/lib" "/opt/homebrew/lib"
        REQUIRED
    )
    target_link_libraries(RealWorld PRIVATE
        engine imgui glfw3 open-mesh
        ${VULKAN_LIB}
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
        "-framework CoreFoundation"
        dl pthread
    )
else()  # Linux
    find_package(Vulkan REQUIRED)
    target_link_libraries(RealWorld PRIVATE
        engine imgui glfw3 open-mesh
        Vulkan::Vulkan
        X11 Xrandr Xi Xxf86vm Xcursor Xinerama
        dl pthread
    )
endif()

# ── Output executable next to assets/ so relative paths resolve ───────────────
# The game loads assets via paths like "assets/..." relative to the exe.
# Place the binary directly in the realworld/ folder for all configs.
set_target_properties(RealWorld PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${CMAKE_SOURCE_DIR}/realworld"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_SOURCE_DIR}/realworld"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_SOURCE_DIR}/realworld"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}/realworld"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_SOURCE_DIR}/realworld"
    # Debugger working directory = realworld/ so assets/ is found at runtime
    VS_DEBUGGER_WORKING_DIRECTORY           "${CMAKE_SOURCE_DIR}/realworld"
)

# ── Copy LibTorch DLLs next to executable (Windows) ─────────────────────────
# The exe lives in realworld/ but the LibTorch .dlls are in libtorch/lib/.
# Without this step, Windows can't find torch_cpu.dll etc. at runtime.
if(Torch_FOUND AND WIN32)
    # TORCH_INSTALL_PREFIX points to the libtorch root; DLLs are in lib/
    set(_torch_dll_dir "${TORCH_INSTALL_PREFIX}/lib")
    file(GLOB _torch_dlls "${_torch_dll_dir}/*.dll")
    if(_torch_dlls)
        add_custom_command(TARGET RealWorld POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_torch_dlls}
                "$<TARGET_FILE_DIR:RealWorld>"
            COMMENT "Copying LibTorch DLLs to output directory..."
        )
        message(STATUS "LibTorch DLLs will be copied from: ${_torch_dll_dir}")
    endif()
elseif(Torch_FOUND AND NOT WIN32)
    # On Linux/macOS the rpath is set via the linker; no copy needed.
    # Just ensure the runtime path includes the LibTorch lib directory.
    set_target_properties(RealWorld PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# ── Auto-export ML model if missing ──────────────────────────────────────────
# If the TorchScript model file doesn't exist, try to generate it via Python.
# This runs once at build time; subsequent builds skip it.
set(MODEL_PT "${CMAKE_SOURCE_DIR}/realworld/assets/models/rig_diffusion.pt")
set(MODEL_SCRIPT "${CMAKE_SOURCE_DIR}/ml_training/scripts/download_and_export_model.py")

if(NOT EXISTS "${MODEL_PT}" AND EXISTS "${MODEL_SCRIPT}")
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        message(STATUS "ML model not found — will export at build time")
        # Try distillation first; if it fails, fall back to skeleton export.
        # We use a small wrapper script via cmake -P to handle the fallback.
        set(_export_script "${CMAKE_BINARY_DIR}/export_model.cmake")
        file(WRITE "${_export_script}" "
execute_process(
    COMMAND \"${Python3_EXECUTABLE}\" \"${MODEL_SCRIPT}\"
        --method distill --output \"${MODEL_PT}\" --resolution 256
    WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}/ml_training\"
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(STATUS \"Distillation failed — trying skeleton export...\")
    execute_process(
        COMMAND \"${Python3_EXECUTABLE}\" \"${MODEL_SCRIPT}\"
            --method skeleton --output \"${MODEL_PT}\" --resolution 256
        WORKING_DIRECTORY \"${CMAKE_SOURCE_DIR}/ml_training\"
        RESULT_VARIABLE _rc2
    )
    if(NOT _rc2 EQUAL 0)
        message(WARNING \"ML model export failed. Auto-rig will use heuristic mode.\")
    endif()
endif()
")
        add_custom_command(
            OUTPUT "${MODEL_PT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_SOURCE_DIR}/realworld/assets/models"
            COMMAND ${CMAKE_COMMAND} -P "${_export_script}"
            COMMENT "Exporting auto-rig ML model..."
        )
        add_custom_target(export_ml_model DEPENDS "${MODEL_PT}")
        add_dependencies(RealWorld export_ml_model)
    else()
        message(STATUS "Python3 not found — skipping ML model export")
    endif()
elseif(EXISTS "${MODEL_PT}")
    message(STATUS "ML model found: ${MODEL_PT}")
endif()

# ── Set RealWorld as the startup project in Visual Studio ─────────────────────
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT RealWorld)

# ── MSVC-specific tweaks ──────────────────────────────────────────────────────
if(MSVC)
    # Match original vcxproj: Warning Level 3, permissive-off (conformance mode)
    foreach(_tgt RealWorld engine imgui glfw3 open-mesh)
        target_compile_options(${_tgt} PRIVATE /W3 /permissive-)
    endforeach()

    # Subsystem: Console (matches SubSystem=Console in vcxproj)
    set_target_properties(RealWorld PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )
endif()
