add_executable(timingwindow main.cpp)
target_link_libraries(timingwindow PRIVATE braincel)
set_target_properties(timingwindow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/timingwindow"
)
configure_file(buildparams.json ${CMAKE_BINARY_DIR}/examples/timingwindow/buildparams.json COPYONLY)
configure_file(simparams.json   ${CMAKE_BINARY_DIR}/examples/timingwindow/simparams.json   COPYONLY)
configure_file(visualize.py     ${CMAKE_BINARY_DIR}/examples/timingwindow/visualize.py     COPYONLY)
