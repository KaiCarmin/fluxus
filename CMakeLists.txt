cmake_minimum_required(VERSION 3.15)
project(fluxus_core LANGUAGES CXX)

# 1. Setup Python & Pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 2. Define source files (Globbing is okay for small projects, but explicit list is safer)
file(GLOB_RECURSE SOURCES "src/core/lib/*.cpp")
file(GLOB_RECURSE BINDINGS "src/core/bindings.cpp")

# 3. Create the Python Extension Module
# output name must match what you import in Python (e.g., _core)
python_add_library(_core MODULE ${SOURCES} ${BINDINGS})

# This tells CMake: "Put the compiled .pyd file inside src/fluxus/"
set_target_properties(_core PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/fluxus"
)

# Link Pybind11 headers so the compiler can find "pybind11.h"
target_link_libraries(_core PRIVATE pybind11::headers)

# 4. Include Directories (Where to look for .hpp files)
target_include_directories(_core PRIVATE "src/core/include")

# 5. Compiler Flags (Optimization is crucial for CFD!)
if(MSVC)
    target_compile_options(_core PRIVATE /O2 /fp:fast)
else()
    target_compile_options(_core PRIVATE -O3 -march=native -ffast-math)
endif()

# 6. Install location (handled by scikit-build-core, but good to define)
install(TARGETS _core DESTINATION src/fluxus)