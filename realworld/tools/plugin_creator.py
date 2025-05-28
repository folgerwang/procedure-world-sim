import os
import json
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

def sanitize_name_for_path(name, default="my_plugin_dir"):
    """
    Sanitizes a string to be suitable for use as a directory or file name.
    Removes special characters, converts to lowercase, and replaces spaces/hyphens with underscores.
    """
    if not name:
        return default
    sanitized = re.sub(r'[^\w\s-]', '', str(name)).strip().lower()
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    if not sanitized: # If sanitization results in an empty string
        return default
    return sanitized

def to_pascal_case(text):
    """
    Converts a string to PascalCase.
    """
    if not text:
        return "MyPlugin" # Default if input is empty
    return "".join(word.capitalize() for word in re.split(r'[\s_-]+', str(text)) if word)


class PluginGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("C++ Plugin Generator") # Updated title
        master.geometry("650x780")

        self.style = ttk.Style()
        self.style.theme_use('clam')

        input_frame = ttk.LabelFrame(master, text="C++ Plugin Details", padding=(10, 5))
        input_frame.pack(padx=10, pady=10, fill="x")

        self.fields = {}
        field_labels = [
            ("Plugin Name:", "My Awesome Cpp Tool"), # Updated placeholder
            ("Author Name:", "Your Name"),
            ("Plugin Version:", "0.1.0"),
            ("Plugin Description:", "A C++ plugin for awesome things."), # Updated
            ("Main Plugin Class Name:", "") # Default will be generated
        ]

        for i, (label_text, placeholder) in enumerate(field_labels):
            label = ttk.Label(input_frame, text=label_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            
            entry_var = tk.StringVar()
            if label_text == "Plugin Name:":
                entry_var.trace_add("write", self.update_default_class_name)
            
            entry = ttk.Entry(input_frame, width=65, textvariable=entry_var)
            if placeholder and label_text != "Main Plugin Class Name:":
                entry.insert(0, placeholder)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.fields[label_text.replace(":", "").replace(" ", "_").lower()] = entry_var
        
        dir_label = ttk.Label(input_frame, text="Output Directory:")
        dir_label.grid(row=len(field_labels), column=0, padx=5, pady=5, sticky="w")
        
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        dir_entry = ttk.Entry(input_frame, textvariable=self.output_dir_var, width=55)
        dir_entry.grid(row=len(field_labels), column=1, padx=5, pady=5, sticky="ew")
        
        browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_directory)
        browse_button.grid(row=len(field_labels), column=2, padx=5, pady=5)

        input_frame.columnconfigure(1, weight=1)

        generate_button = ttk.Button(master, text="Generate C++ Plugin Container", command=self.process_generation) # Updated text
        generate_button.pack(pady=10)

        status_frame = ttk.LabelFrame(master, text="Output Log", padding=(10,5))
        status_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, height=20, width=75)
        self.status_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.status_text.configure(state='disabled')

        self.log_message("Welcome to the C++ Plugin Generator!")
        self.log_message("Fill in the details above and click 'Generate'.")
        self.update_default_class_name()

    def update_default_class_name(self, *args):
        plugin_name_var = self.fields.get("plugin_name")
        class_name_var = self.fields.get("main_plugin_class_name")
        
        if plugin_name_var and class_name_var:
            plugin_name = plugin_name_var.get()
            if plugin_name:
                default_class_name_base = to_pascal_case(plugin_name)
                class_name_default = f"{default_class_name_base}Plugin"
                
                current_class_name = class_name_var.get()
                if not current_class_name or \
                   (hasattr(self, '_previous_default_class_name') and \
                    current_class_name == self._previous_default_class_name):
                    class_name_var.set(class_name_default)
                self._previous_default_class_name = class_name_default
            elif not class_name_var.get():
                 class_name_var.set("")
                 self._previous_default_class_name = ""

    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)

    def log_message(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END) 
        self.status_text.configure(state='disabled')
        self.master.update_idletasks() 

    def clear_log(self):
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', tk.END)
        self.status_text.configure(state='disabled')

    def process_generation(self):
        self.clear_log()
        self.log_message("Starting C++ plugin generation process...")

        plugin_name_str = self.fields["plugin_name"].get().strip()
        author_name = self.fields["author_name"].get().strip()
        version = self.fields["plugin_version"].get().strip() or "0.1.0"
        description = self.fields["plugin_description"].get().strip()
        class_name_str = self.fields["main_plugin_class_name"].get().strip() # User input class name
        base_output_dir = self.output_dir_var.get().strip()

        if not plugin_name_str:
            messagebox.showerror("Error", "Plugin Name cannot be empty.")
            self.log_message("Error: Plugin Name cannot be empty.")
            return
        if not class_name_str:
            messagebox.showerror("Error", "Main Plugin Class Name cannot be empty.")
            self.log_message("Error: Main Plugin Class Name cannot be empty.")
            return
        if not base_output_dir:
            messagebox.showerror("Error", "Output Directory cannot be empty.")
            self.log_message("Error: Output Directory cannot be empty.")
            return

        plugin_dir_name_sanitized = sanitize_name_for_path(plugin_name_str, "my_cpp_plugin")
        # For C++, class names are often PascalCase directly from user input or derived
        plugin_class_name_pascal = to_pascal_case(class_name_str) # Ensure it's PascalCase
        if not plugin_class_name_pascal: plugin_class_name_pascal = "DefaultPlugin" # Fallback

        # Define project name for CMake (often PascalCase version of directory)
        cmake_project_name = to_pascal_case(plugin_dir_name_sanitized)
        if not cmake_project_name: cmake_project_name = "MyCppPlugin"


        full_plugin_path = os.path.join(base_output_dir, plugin_dir_name_sanitized)
        
        include_dir_name = "include"
        plugin_specific_include_dir_name = plugin_dir_name_sanitized # for include/{plugin_name}/...
        src_dir_name = "src"

        full_include_path = os.path.join(full_plugin_path, include_dir_name, plugin_specific_include_dir_name)
        full_src_path = os.path.join(full_plugin_path, src_dir_name)

        self.log_message(f"Sanitized Plugin Directory Name: {plugin_dir_name_sanitized}")
        self.log_message(f"Plugin Class Name: {plugin_class_name_pascal}")
        self.log_message(f"Full path for plugin: {full_plugin_path}")

        if os.path.exists(full_plugin_path):
            self.log_message(f"Warning: Directory '{full_plugin_path}' already exists.")
            if not messagebox.askyesno("Overwrite?", f"Directory '{full_plugin_path}' already exists.\nDo you want to overwrite it?"):
                self.log_message("Operation cancelled by user (overwrite denied).")
                return
            else:
                import shutil
                try:
                    shutil.rmtree(full_plugin_path)
                    self.log_message(f"Removed existing directory: {full_plugin_path}")
                except OSError as e:
                    messagebox.showerror("Error", f"Error removing existing directory: {e}")
                    self.log_message(f"Error removing existing directory: {e}")
                    return
        
        try:
            self.log_message(f"Generating C++ plugin container in {full_plugin_path}/")
            os.makedirs(full_plugin_path)
            os.makedirs(full_include_path)
            os.makedirs(full_src_path)
            self.log_message(f"  Created directory structure.")

            # --- Create include/{plugin_name}/plugin_export.h ---
            # Used for DLL export/import macros
            plugin_export_upper = plugin_dir_name_sanitized.upper() + "_API" # e.g., MY_CPP_PLUGIN_API
            plugin_export_h_content = f"""\
#ifndef {plugin_dir_name_sanitized.upper()}_PLUGIN_EXPORT_H
#define {plugin_dir_name_sanitized.upper()}_PLUGIN_EXPORT_H

#if defined(_WIN32) || defined(_WIN64)
    #ifdef {plugin_dir_name_sanitized.upper()}_EXPORTS // Defined by CMake when building the DLL
        #define {plugin_export_upper} __declspec(dllexport)
    #else
        #define {plugin_export_upper} __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define {plugin_export_upper} __attribute__((visibility("default")))
#else
    #define {plugin_export_upper}
#endif

#endif // {plugin_dir_name_sanitized.upper()}_PLUGIN_EXPORT_H
"""
            with open(os.path.join(full_include_path, "plugin_export.h"), "w", encoding="utf-8") as f:
                f.write(plugin_export_h_content)
            self.log_message(f"  Created file: {include_dir_name}/{plugin_specific_include_dir_name}/plugin_export.h")

            # --- Create include/{plugin_name}/iplugin.h ---
            # A generic plugin interface
            iplugin_h_content = f"""\
#ifndef IPLUGIN_H
#define IPLUGIN_H

#include "{plugin_specific_include_dir_name}/plugin_export.h" // For API export macro
#include <string> // For std::string (optional, can use const char*)

// Forward declaration for HostAPI if used
// class HostAPI; 

class {plugin_export_upper} IPlugin {{
public:
    virtual ~IPlugin() {{}}

    // Called by the host to initialize the plugin.
    // 'hostApi' can be a pointer to an interface provided by the host for callbacks.
    virtual bool initialize(/* HostAPI* hostApi */) = 0;

    // Called by the host to perform the plugin's main task(s).
    // 'inputData' can be any structure or type the host passes.
    // The return value can be any result the host expects.
    virtual void execute(const char* inputData) = 0; 

    // Called by the host before unloading the plugin for cleanup.
    virtual void shutdown() = 0;

    // Returns the name of the plugin.
    virtual const char* getName() const = 0;

    // Returns the version of the plugin.
    virtual const char* getVersion() const = 0;
}};

// Define function pointer types for the factory functions
// These are what the host application will look for using dlsym/GetProcAddress.
typedef IPlugin* (*CreatePluginFunc)();
typedef void (*DestroyPluginFunc)(IPlugin*);

#endif // IPLUGIN_H
"""
            with open(os.path.join(full_include_path, "iplugin.h"), "w", encoding="utf-8") as f:
                f.write(iplugin_h_content)
            self.log_message(f"  Created file: {include_dir_name}/{plugin_specific_include_dir_name}/iplugin.h")

            # --- Create include/{plugin_name}/{PluginClassName}.h ---
            # Header for the specific plugin class
            plugin_class_h_content = f"""\
#ifndef {plugin_class_name_pascal.upper()}_H
#define {plugin_class_name_pascal.upper()}_H

#include "{plugin_specific_include_dir_name}/iplugin.h"
#include "{plugin_specific_include_dir_name}/plugin_export.h" // For API export macro
#include <string> // For std::string

class {plugin_export_upper} {plugin_class_name_pascal} : public IPlugin {{
public:
    {plugin_class_name_pascal}();
    ~{plugin_class_name_pascal}() override;

    // IPlugin interface implementation
    bool initialize(/* HostAPI* hostApi */) override;
    void execute(const char* inputData) override;
    void shutdown() override;
    const char* getName() const override;
    const char* getVersion() const override;

private:
    std::string m_pluginName;
    std.string m_pluginVersion;
    // Add other member variables here
    // HostAPI* m_hostApi; // Example if host API is used
}};

// Factory functions to create and destroy an instance of the plugin.
// These are the functions the host application will call.
extern "C" {plugin_export_upper} IPlugin* CreatePlugin();
extern "C" {plugin_export_upper} void DestroyPlugin(IPlugin* plugin);

#endif // {plugin_class_name_pascal.upper()}_H
"""
            with open(os.path.join(full_include_path, f"{plugin_class_name_pascal.lower()}.h"), "w", encoding="utf-8") as f:
                f.write(plugin_class_h_content)
            self.log_message(f"  Created file: {include_dir_name}/{plugin_specific_include_dir_name}/{plugin_class_name_pascal.lower()}.h")
            
            # --- Create src/{PluginClassName}.cpp ---
            # Implementation of the specific plugin class
            plugin_class_cpp_content = f"""\
#include "{plugin_specific_include_dir_name}/{plugin_class_name_pascal.lower()}.h"
#include <iostream> // For example output

{plugin_class_name_pascal}::{plugin_class_name_pascal}() 
    : m_pluginName("{plugin_name_str}"), m_pluginVersion("{version}") {{
    std::cout << "[{plugin_class_name_pascal}] Constructor called." << std::endl;
}}

{plugin_class_name_pascal}::~{plugin_class_name_pascal}() {{
    std::cout << "[{plugin_class_name_pascal}] Destructor called." << std::endl;
}}

bool {plugin_class_name_pascal}::initialize(/* HostAPI* hostApi */) {{
    // m_hostApi = hostApi; // Store host API if provided
    std::cout << "[" << m_pluginName << "] Initialized. Version: " << m_pluginVersion << std::endl;
    // if (m_hostApi) {{ m_hostApi->logMessage("Plugin initialized by host."); }} // Escaped braces
    return true;
}}

void {plugin_class_name_pascal}::execute(const char* inputData) {{
    std::cout << "[" << m_pluginName << "] Executing with data: " << (inputData ? inputData : "null") << std::endl;
    // Add your plugin's core logic here
    // Example: if (m_hostApi) {{ m_hostApi->performAction(); }} // Escaped braces
}}

void {plugin_class_name_pascal}::shutdown() {{
    std::cout << "[" << m_pluginName << "] Shutting down." << std::endl;
    // Perform cleanup tasks here
}}

const char* {plugin_class_name_pascal}::getName() const {{
    return m_pluginName.c_str();
}}

const char* {plugin_class_name_pascal}::getVersion() const {{
    return m_pluginVersion.c_str();
}}

// --- Factory Function Implementations ---
// These are the C-style functions exported by the DLL/SO.
IPlugin* CreatePlugin() {{
    return new {plugin_class_name_pascal}();
}}

void DestroyPlugin(IPlugin* plugin) {{
    delete plugin;
}}
"""
            with open(os.path.join(full_src_path, f"{plugin_class_name_pascal.lower()}.cpp"), "w", encoding="utf-8") as f:
                f.write(plugin_class_cpp_content)
            self.log_message(f"  Created file: {src_dir_name}/{plugin_class_name_pascal.lower()}.cpp")

            # --- Create plugin_manifest.json ---
            # Updated for C++: 'library_name' might be how host finds the .dll/.so
            # The exact details depend on the host application's loading mechanism.
            library_file_name = f"{plugin_dir_name_sanitized}" # Base name, host adds .dll/.so/.dylib
            if os.name == 'nt': # Windows
                library_file_name = f"{plugin_dir_name_sanitized}.dll"
            elif sys.platform == 'darwin': # macOS
                library_file_name = f"lib{plugin_dir_name_sanitized}.dylib"
            else: # Linux/other Unix
                library_file_name = f"lib{plugin_dir_name_sanitized}.so"


            manifest_data = {
                "plugin_name": plugin_name_str,
                "version": version,
                "author": author_name,
                "description": description,
                "language": "C++",
                "entry_point": {
                    "library_name": library_file_name, # e.g., my_cpp_plugin.dll or libmy_cpp_plugin.so
                    "create_function": "CreatePlugin", # Name of the exported factory function
                    "destroy_function": "DestroyPlugin" # Name of the exported cleanup function
                },
                "dependencies": [], 
                "host_compatibility": {"min_version": "1.0.0", "max_version": None}
            }
            with open(os.path.join(full_plugin_path, "plugin_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            self.log_message(f"  Created file: plugin_manifest.json")

            # --- Create CMakeLists.txt ---
            cmake_content = f"""\
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project({cmake_project_name} VERSION {version} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the plugin name (used for library target and export definitions)
set(PLUGIN_TARGET_NAME "{plugin_dir_name_sanitized}")

# --- Include Directories ---
include_directories(
    "${{CMAKE_CURRENT_SOURCE_DIR}}/{include_dir_name}"
)

# --- Source Files ---
set(PLUGIN_SOURCES
    "{src_dir_name}/{plugin_class_name_pascal.lower()}.cpp"
)

# --- Build the Shared Library ---
add_library(\${{PLUGIN_TARGET_NAME}} SHARED \${{PLUGIN_SOURCES}})

# Define a preprocessor macro to indicate that we are building the DLL/SO
# This is used by plugin_export.h to switch between dllexport/dllimport or visibility.
target_compile_definitions(\${{PLUGIN_TARGET_NAME}} PRIVATE "{plugin_dir_name_sanitized.upper()}_EXPORTS")

# --- Set Properties for Windows DLLs (Optional but good practice) ---
if(WIN32)
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF # We explicitly export with {plugin_export_upper}
        DEFINE_SYMBOL "{plugin_dir_name_sanitized.upper()}_EXPORTS" # Ensure this is defined for the build
    )
    # Set output directory for DLLs to be more predictable (e.g., in build/bin or build/lib)
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/lib")
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/lib")
else()
    # For non-Windows, ensure correct RPATH handling for finding dependencies if any
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(\${{PLUGIN_TARGET_NAME}} PROPERTIES MACOSX_RPATH TRUE) # For macOS
endif()


# --- Installation ---
# Install the compiled shared library
install(TARGETS \${{PLUGIN_TARGET_NAME}}
    LIBRARY DESTINATION "lib/my_application/plugins/\${{PLUGIN_TARGET_NAME}}" # Adjust as needed
    RUNTIME DESTINATION "bin/my_application/plugins/\${{PLUGIN_TARGET_NAME}}" # For Windows DLLs
    ARCHIVE DESTINATION "lib/my_application/plugins/\${{PLUGIN_TARGET_NAME}}" # For import libs on Windows
    COMPONENT Runtime
)

# Install the manifest file
install(
    FILES plugin_manifest.json
    DESTINATION "share/my_application/plugins/\${{PLUGIN_TARGET_NAME}}" # Adjust as needed
    COMPONENT Runtime
)

# Install public headers (optional, if other C++ code needs to link against this plugin directly)
# install(
#     DIRECTORY "{include_dir_name}/" # Installs the entire include directory
#     DESTINATION "include/my_application/plugins/\${{PLUGIN_TARGET_NAME}}"
#     COMPONENT Development
# )

# --- Testing (Optional, placeholder) ---
# enable_testing()
# add_executable(\${{PLUGIN_TARGET_NAME}}_tester tests/main_test.cpp) # Example test executable
# target_link_libraries(\${{PLUGIN_TARGET_NAME}}_tester PRIVATE \${{PLUGIN_TARGET_NAME}})
# add_test(NAME \${{PLUGIN_TARGET_NAME}}_basic_test COMMAND \${{PLUGIN_TARGET_NAME}}_tester)

message(STATUS "CMake configured for C++ plugin: \${{PROJECT_NAME}} (\${{PLUGIN_TARGET_NAME}})")
message(STATUS "  Version: \${{PROJECT_VERSION}}")
message(STATUS "  Sources: \${{PLUGIN_SOURCES}}")
message(STATUS "  To build, navigate to your build directory and run: cmake --build .")
message(STATUS "  To install (after build): cmake --install . --prefix <your_install_dir>")
"""
            with open(os.path.join(full_plugin_path, "CMakeLists.txt"), "w", encoding="utf-8") as f:
                f.write(cmake_content)
            self.log_message(f"  Created file: CMakeLists.txt")

            self.log_message("\n--- C++ Plugin container generated successfully! ---")
            self.log_message(f"Plugin Name: {plugin_name_str}")
            self.log_message(f"Directory:   {full_plugin_path}/")
            self.log_message("Files created:")
            self.log_message(f"  ├── {include_dir_name}/{plugin_specific_include_dir_name}/plugin_export.h")
            self.log_message(f"  ├── {include_dir_name}/{plugin_specific_include_dir_name}/iplugin.h")
            self.log_message(f"  ├── {include_dir_name}/{plugin_specific_include_dir_name}/{plugin_class_name_pascal.lower()}.h")
            self.log_message(f"  ├── {src_dir_name}/{plugin_class_name_pascal.lower()}.cpp")
            self.log_message(f"  ├── plugin_manifest.json")
            self.log_message(f"  └── CMakeLists.txt")
            messagebox.showinfo("Success", f"C++ Plugin '{plugin_name_str}' generated successfully in:\n{full_plugin_path}")

        except OSError as e:
            messagebox.showerror("Error", f"Error creating C++ plugin container: {e}")
            self.log_message(f"Error creating C++ plugin container: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.log_message(f"An unexpected error occurred: {e}")
            # import traceback # For debugging
            # self.log_message(traceback.format_exc())

def main():
    root = tk.Tk()
    app = PluginGeneratorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
