add_executable(ftcompare main.cpp)
target_link_libraries(ftcompare PRIVATE braincel)
set_target_properties(ftcompare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/ftcompare"
)
configure_file(buildparams.json ${CMAKE_BINARY_DIR}/examples/ftcompare/buildparams.json COPYONLY)
configure_file(simparams.json   ${CMAKE_BINARY_DIR}/examples/ftcompare/simparams.json   COPYONLY)
configure_file(visualize.py     ${CMAKE_BINARY_DIR}/examples/ftcompare/visualize.py     COPYONLY)
