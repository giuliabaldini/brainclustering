PYTHON_ADD_MODULE(mapping mapping.cpp)
add_dependencies(main mapping)

PYTHON_ADD_MODULE(mapping_simple mapping_simple.cpp)
add_dependencies(main mapping_simple)