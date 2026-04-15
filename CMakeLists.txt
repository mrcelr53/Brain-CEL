cmake_minimum_required(VERSION 3.18)
project(BrainCEL)

set(VERSION_SHORT 1.0.0)
project(BrainCEL VERSION ${VERSION_SHORT} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Virtual environment
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    message(STATUS "Creating virtual environment...")
    execute_process(
            COMMAND python3 -m venv "${CMAKE_SOURCE_DIR}/.venv"
            RESULT_VARIABLE VENV_RESULT
    )
    if(NOT VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create virtual environment")
    endif()

    # Install required packages
    execute_process(
            COMMAND "${CMAKE_SOURCE_DIR}/.venv/bin/pip" install -r "${CMAKE_SOURCE_DIR}/requirements.txt"
            RESULT_VARIABLE PIP_RESULT
    )
    if(NOT PIP_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install Python packages")
    endif()
endif()

# Python
set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv")
set(Python3_FIND_VIRTUALENV FIRST)          # prefer the venv over system Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3_EXECUTABLE:    ${Python3_EXECUTABLE}")
message(STATUS "Python3_LIBRARIES:     ${Python3_LIBRARIES}")
message(STATUS "Python3_INCLUDE_DIRS:  ${Python3_INCLUDE_DIRS}")

# JSON
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(json)
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)

# Subdirectories
add_subdirectory(src)

# Examples
add_subdirectory(examples/nestcompare)
add_subdirectory(examples/ftcompare)
add_subdirectory(examples/timingwindow)

