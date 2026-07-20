add_executable(quickstart main.cpp)

target_link_libraries(quickstart PRIVATE braincel)

target_include_directories(quickstart PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(quickstart PRIVATE VENV_PYTHON="${Python3_EXECUTABLE}")

configure_file(visualize.py visualize.py COPYONLY)
