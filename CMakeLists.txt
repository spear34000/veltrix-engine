cmake_minimum_required(VERSION 3.15)
project(veltrix C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" VX_PROCESSOR)
set(VX_X86_PROCESSOR OFF)
if(VX_PROCESSOR MATCHES "^(x86_64|amd64|i.86|x86)$")
    set(VX_X86_PROCESSOR ON)
endif()

if(MSVC)
    if(VX_X86_PROCESSOR)
        add_compile_options(/W3 /O2 /fp:fast /arch:AVX2 /openmp)
    else()
        add_compile_options(/W3 /O2 /fp:fast /openmp)
    endif()
else()
    add_compile_options(-Wall -Wextra -Ofast -funroll-loops -fopenmp -flto)
    add_link_options(-flto)
    if(VX_X86_PROCESSOR)
        add_compile_options(-mavx2 -mfma -march=native)
    endif()
endif()

set(SOURCES
    src/main.c
    src/gguf.c
    src/safetensors.c
    src/tokenizer.c
    src/onnx.c
    src/tensor.c
    src/model.c
    src/format.c
    src/metalearn.c
    src/scheduler.c
    src/quantize.c
    src/lora.c
    src/speculative.c
    src/runtime_profile.c
    src/simd/avx2.c
)

add_executable(veltrix ${SOURCES})
target_include_directories(veltrix PRIVATE include)

set(TEST_SOURCES ${SOURCES})
list(REMOVE_ITEM TEST_SOURCES src/main.c)
add_executable(veltrix_test tests/test_gguf.c tests/test_runtime_profile.c ${TEST_SOURCES})
target_include_directories(veltrix_test PRIVATE include)

add_executable(veltrix_bench tests/benchmark.c ${TEST_SOURCES})
target_include_directories(veltrix_bench PRIVATE include)

add_executable(veltrix_e2e tests/test_e2e.c ${TEST_SOURCES})
target_include_directories(veltrix_e2e PRIVATE include)

add_executable(veltrix_onnx tests/test_onnx.c ${TEST_SOURCES})
target_include_directories(veltrix_onnx PRIVATE include)

add_executable(veltrix_tokjson tests/test_tokenizer_json.c ${TEST_SOURCES})
target_include_directories(veltrix_tokjson PRIVATE include)

add_executable(veltrix_tiktok tests/test_tiktoken.c ${TEST_SOURCES})
target_include_directories(veltrix_tiktok PRIVATE include)

add_executable(veltrix_lora tests/test_lora.c ${TEST_SOURCES})
target_include_directories(veltrix_lora PRIVATE include)

if(WIN32)
    target_link_libraries(veltrix PRIVATE gomp)
    target_link_libraries(veltrix_test PRIVATE gomp)
    target_link_libraries(veltrix_bench PRIVATE gomp)
    target_link_libraries(veltrix_e2e PRIVATE gomp)
    target_link_libraries(veltrix_onnx PRIVATE gomp)
    target_link_libraries(veltrix_tokjson PRIVATE gomp)
    target_link_libraries(veltrix_tiktok PRIVATE gomp)
    target_link_libraries(veltrix_lora PRIVATE gomp)
else()
    target_link_libraries(veltrix PRIVATE gomp m pthread)
    target_link_libraries(veltrix_test PRIVATE gomp m pthread)
    target_link_libraries(veltrix_bench PRIVATE gomp m pthread)
    target_link_libraries(veltrix_e2e PRIVATE gomp m pthread)
    target_link_libraries(veltrix_onnx PRIVATE gomp m pthread)
    target_link_libraries(veltrix_tokjson PRIVATE gomp m pthread)
    target_link_libraries(veltrix_tiktok PRIVATE gomp m pthread)
    target_link_libraries(veltrix_lora PRIVATE gomp m pthread)
endif()
