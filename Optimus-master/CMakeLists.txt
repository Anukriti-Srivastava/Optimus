cmake_minimum_required(VERSION 3.13)
project(CodeOptimizer)

set(CMAKE_CXX_STANDARD 17)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Include LLVM headers and definitions
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Select only necessary LLVM components
llvm_map_components_to_libnames(llvm_libs
  Core
  IRReader
  Support
  Passes
  Analysis
  TransformUtils
  Target
 X86       # ✅ Required for InitializeNativeTarget
  MC        # ✅ Required for TargetMachine
  CodeGen  
)

# Define the executable
add_executable(CodeOptimizer main.cpp)

# Apply include directories and definitions to this target
target_include_directories(CodeOptimizer PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(CodeOptimizer PRIVATE ${LLVM_DEFINITIONS})

# Link the appropriate LLVM libraries
target_link_libraries(CodeOptimizer PRIVATE ${llvm_libs})
