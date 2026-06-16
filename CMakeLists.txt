cmake_minimum_required(VERSION 3.20)
project(aes_gcm_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to enable CUDA if available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found — GPU targets will be built")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found — building CPU-only targets")
endif()

# ── CPU library ───────────────────────────────────────────────────────────────
add_library(aes_gcm_cpu STATIC
    cpu/aes_core.cpp
    cpu/aes_ctr.cpp
    cpu/ghash.cpp
    cpu/aes_gcm_cpu.cpp
)
target_include_directories(aes_gcm_cpu PUBLIC include)
target_compile_options(aes_gcm_cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# ── NIST test runner ──────────────────────────────────────────────────────────
add_executable(run_tests tests/nist_vectors.cpp)
target_link_libraries(run_tests PRIVATE aes_gcm_cpu)
target_include_directories(run_tests PRIVATE include)

# ── Demo binary (CPU encrypt/decrypt) ────────────────────────────────────────
add_executable(aes_gcm_demo tests/demo.cpp)
target_link_libraries(aes_gcm_demo PRIVATE aes_gcm_cpu)
target_include_directories(aes_gcm_demo PRIVATE include)

# ── GPU targets (only if CUDA available) ─────────────────────────────────────
if(CUDA_AVAILABLE)
    add_library(aes_gcm_gpu_lib STATIC
        gpu/aes_ctr.cu
        gpu/ghash_kernel.cu
        gpu/mapreduce.cu
        gpu/aes_gcm_gpu.cu
    )
    target_include_directories(aes_gcm_gpu_lib PUBLIC include)
    target_compile_options(aes_gcm_gpu_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    )

    add_executable(bench tests/bench_gpu.cu)
    target_link_libraries(bench PRIVATE aes_gcm_cpu aes_gcm_gpu_lib)
    target_include_directories(bench PRIVATE include)
endif()
