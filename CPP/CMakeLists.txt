# CMake 最低版本要求 (Minimum CMake version required)
cmake_minimum_required(VERSION 3.10)

# 项目名称 (Project name)
project(NBodySimulation_CPP)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 设置 C++ 标准为 C++17 (Set C++ standard to C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenMP, 用于多线程 (Find OpenMP for multithreading)
find_package(OpenMP REQUIRED)

# 查找 FFTW3 库, 要求是线程安全的版本 (fftw3_threads)
# Find FFTW3 library, requiring the thread-safe version
find_package(FFTW3 REQUIRED COMPONENTS threads)

# 查找 Eigen3 库
# Eigen 是一个纯头文件库，我们只需要找到它的包含目录
# Find Eigen3 library. It's header-only, so we just need its include directory.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 如果找到了 OpenMP, 添加编译和链接标志
# If OpenMP is found, add the necessary compile and link flags
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel compilation.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found. The simulation will run in single-threaded mode.")
endif()

# 添加可执行文件 (Add the executable)
add_executable(nbody_simulation ./src/main.cpp)

# --- 链接库和头文件 (Link libraries and include headers) ---

# 添加 Eigen3 的头文件目录 (Add Eigen3's include directory)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3, adding include directory.")
    target_include_directories(nbody_simulation PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 library not found. Please install it (e.g., 'sudo apt-get install libeigen3-dev').")
endif()

# 链接 FFTW3 库 (Link FFTW3 libraries)
if(FFTW3_FOUND)
    message(STATUS "Found FFTW3 library, linking...")
    # 包含 FFTW3 的头文件目录 (Include FFTW3's header directory)
    target_include_directories(nbody_simulation PRIVATE ${FFTW3_INCLUDE_DIRS})
    
    # -- 修复: 使用 ${FFTW3_LIBRARIES} 变量进行链接 --
    # FIX: Link using the ${FFTW3_LIBRARIES} variable for broader compatibility
    # This is more robust than using imported targets like FFTW3::fftw3_threads,
    # which may not be available on all systems.
    target_link_libraries(nbody_simulation PRIVATE ${FFTW3_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "FFTW3 library not found. Please install it (e.g., 'sudo apt-get install libfftw3-dev').")
endif()

# 为了在某些系统上（如macOS或Linux）正确链接数学库，显式添加 m
# Explicitly link the math library (m) for some systems like macOS or Linux
if(NOT WIN32)
    target_link_libraries(nbody_simulation PRIVATE m)
endif()

# 输出提示信息
message(STATUS "Configuration complete. To build, run 'make' in your build directory.")
