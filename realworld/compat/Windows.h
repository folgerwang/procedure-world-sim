// =============================================================================
// compat/Windows.h  —  minimal Windows API stub for Linux / macOS builds
//
// Provides just enough for application.cpp to compile.
// Full cross-platform runtime support still requires source changes in
// engine_helper.cpp (glslc.exe path, '\\' path separators).
// =============================================================================
#pragma once

#ifndef _WIN32   // only active on non-Windows

#include <cstdio>
#include <cstdlib>

// MessageBox flags
#define MB_OK               0x00000000L
#define MB_OKCANCEL         0x00000001L
#define MB_ICONERROR        0x00000010L
#define MB_ICONWARNING      0x00000030L

// Stub MessageBoxA: prints to stderr instead
inline int MessageBoxA(void* /*hwnd*/, const char* text,
                       const char* caption, unsigned int /*type*/)
{
    fprintf(stderr, "[%s] %s\n", caption ? caption : "", text ? text : "");
    return 1; // IDOK
}

#endif // !_WIN32
