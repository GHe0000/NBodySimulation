cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME NBodyCalc)

project(${PROJECT_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE 
    src/main.cpp
)

set(INCLUDE_DIRS 
    include
)

add_executable(${PROJECT_NAME} ${SOURCE})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

# 导入库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
    # target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 not found..")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found.")
endif()

find_package(FFTW3 REQUIRED COMPONENTS threads)
if(FFTW3_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FFTW3_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "FFTW3 library not found.")
endif()

if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE m)
endif()

