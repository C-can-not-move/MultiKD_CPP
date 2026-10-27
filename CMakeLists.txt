cmake_minimum_required(VERSION 4.1)
project(MultiKD_C++ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- libtorch ----------------
if(WIN32)
    set(LIBTORCH_ROOT "D:/DATA/C++_lib/libtorch_cu126")
else()
    set(LIBTORCH_ROOT "/home/chenzhijiang/userFile/C++_lib/libtorch")
    set(CUDA_TOOLKIT_ROOT_DIR "/home/chenzhijiang/userFile/miniConda3/envs/cuda126/targets/x86_64-linux")
    set(CMAKE_CUDA_FLAGS "-I${CUDA_TOOLKIT_ROOT_DIR}/include" CACHE STRING "CUDA flags")
endif()

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED)
message(STATUS "libtorch: ${TORCH_INSTALL_PREFIX}   (${Torch_VERSION})")

# ---------------- 源文件 ----------------
# 加新方法：kd/ 里加一对 .h/.cpp，然后在这里加两行。CMake 不会自动扫目录。
set(APP_SOURCES
    "src/MultiKD_C++.h"
    "src/MultiKD_C++.cpp"
    "src/config.h"
    "src/config.cpp"
    "src/data/cifar100.h"
    "src/data/cifar100.cpp"
    "src/data/transforms.h"
    "src/data/transforms.cpp"
    # kd 通用契约
    "src/kd/loss.h"
    "src/kd/detail.h"
    # 方法：CE
    "src/kd/ce.h"
    "src/kd/ce.cpp"
    # 方法：KD
    "src/kd/kd.h"
    "src/kd/kd.cpp"
    # 方法：DKD
    "src/kd/dkd.h"
    "src/kd/dkd.cpp"
    "src/model/resnet.h"
    "src/model/resnet.cpp"
    "src/model/checkpoint.h"
    "src/model/checkpoint.cpp"
    # train
    "src/train/batch.h"
    "src/train/trainer.h"
    "src/train/trainer.cpp"
    # 自检
    "src/test_tool.h"
    "src/test_tool.cpp")
add_executable(MultiKD_C++ ${APP_SOURCES})

# ---------------- 头文件路径与链接 ----------------
target_include_directories(MultiKD_C++ PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(MultiKD_C++ PRIVATE torch)

# ---------------- 编译器选项（仅 MSVC）----------------
if(MSVC)
  target_compile_options(MultiKD_C++ PRIVATE /bigobj /EHsc /wd4267 /wd4244 /utf-8)
  target_compile_definitions(MultiKD_C++ PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------- 运行期依赖 ----------------
if(WIN32)
  # ⚠ 这一步会给每个 build 目录塞一整套 libtorch DLL（~1.5-2 GB）。
  #   所以只切 Release，别 Debug 和 Release 各编一次。
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MultiKD_C++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} "$<TARGET_FILE_DIR:MultiKD_C++>")
else()
  set_target_properties(MultiKD_C++ PROPERTIES
    BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()
