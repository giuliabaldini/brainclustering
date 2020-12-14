PYTHON_ADD_MODULE(kmeans kmeans.cpp)
add_dependencies(main kmeans)

PYTHON_ADD_MODULE(leveled_kmeans leveled_kmeans.cpp)
add_dependencies(main leveled_kmeans)

PYTHON_ADD_MODULE(nesting nesting.cpp)
add_dependencies(main nesting)