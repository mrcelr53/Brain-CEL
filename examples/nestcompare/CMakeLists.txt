add_executable(nestcompare main.cpp)
target_link_libraries(nestcompare PRIVATE braincel)
set_target_properties(nestcompare PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/nestcompare"
)
configure_file(buildparams.json   ${CMAKE_BINARY_DIR}/examples/nestcompare/buildparams.json   COPYONLY)
configure_file(simparams.json     ${CMAKE_BINARY_DIR}/examples/nestcompare/simparams.json     COPYONLY)
configure_file(nest_bench.py      ${CMAKE_BINARY_DIR}/examples/nestcompare/nest_bench.py      COPYONLY)
configure_file(visualize.py       ${CMAKE_BINARY_DIR}/examples/nestcompare/visualize.py       COPYONLY)
