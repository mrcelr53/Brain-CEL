add_executable(homeostasis main.cpp)

target_link_libraries(homeostasis PRIVATE braincel)

target_include_directories(homeostasis PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(homeostasis PRIVATE VENV_PYTHON="${Python3_EXECUTABLE}")

configure_file(buildparams.json buildparams.json COPYONLY)
configure_file(simparams.json   simparams.json   COPYONLY)
configure_file(visualize.py     visualize.py     COPYONLY)
