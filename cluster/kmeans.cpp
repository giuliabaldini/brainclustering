#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <limits>
#include "util.hpp"
#include "clustering_util.hpp"

namespace brainclustering {
    template<unsigned int dimensions, bool weighted>
    python::tuple cluster(ndarray const &data_vector,
                          ndarray const &previous_centers, size_t const n_clusters,
                          ndarray const &weights_vector,
                          size_t cluster_addition = 1, size_t const init_method = 1, size_t const n_iterations = 100,
                          size_t const n_runs = 10, double const tolerance = 1e-4,
                          size_t const seed = 0, size_t const verbose = 0) {

        using value_type = double;
        using point_type = point_nd<value_type, dimensions>;
        check_dimensions<value_type>(data_vector, dimensions);
        check_dimensions<value_type>(previous_centers, dimensions);

        double *weights = nullptr;
        if constexpr (weighted){
            weights = get_c_array<double>(weights_vector);
        }
        auto *data = get_c_ndarray<double, dimensions>(data_vector);
        auto *prev_centers = get_c_ndarray<double, dimensions>(previous_centers);
        size_t n_points = data_vector.shape(0);

        // Set best inertia: If there is more than one run, it should be the highest number.
        // Otherwise, we don't want to check it, so we put it to a very low number
        double best_inertia = (n_runs > 1) ? std::numeric_limits<double>::max()
                                           : std::numeric_limits<double>::min();


        auto *labels = new double[n_points];
        auto *centers = new point_type[n_points];

        // TODO: Can this be done without instantiating a new array?
        // Copy the current centers to another array
        if (init_method == 0) {
            for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
                centers[cluster] = prev_centers[cluster];
            }
        }

        // Set a pointer to the actual centers and to the actual labels
        point_type *current_centers = centers;
        value_type *current_labels = labels;

        // Set the seed, if the seed is 0, select a random one from random device
        size_t seed_ = seed ? seed : dev();

        for (size_t run = 0; run < n_runs; ++run) {
            // For keeping track of the current inertia
            double current_inertia = kmeans<point_type, true, weighted>(data, n_points, current_centers, n_clusters,
                                                                        current_labels, cluster_addition, init_method,
                                                                        n_iterations, tolerance, seed_, weights,
                                                                        verbose);
            if (verbose) {
                std::cout << "Inertia at run " << run + 1 << ": " << current_inertia << std::endl;
            }
            // If the current inertia is smaller than the best one
            if (current_inertia < best_inertia) {
                // Set the current inertia
                best_inertia = current_inertia;

                // Swap pointers for labels and centers
                std::swap(current_centers, centers);
                std::swap(current_labels, labels);
                if (verbose) {
                    print_ndarray(centers, n_clusters, "Modified centers: ");
                }
            }
            if (verbose) {
                std::cout << std::endl;
            }

            if (run == 0) {
                current_centers = new point_type[n_clusters];
                current_labels = new double[n_points];
            }
        }

        delete[] current_centers;
        delete[] current_labels;

        if (verbose) {
            print_ndarray(centers, n_clusters, "Final centers: ");
        }
        return python::make_tuple(get_numpy_1darray(labels, n_points),
                                  get_numpy_2darray<point_type, value_type>(centers, n_clusters, dimensions));
    }


    BOOST_PYTHON_MODULE (kmeans) {
        numpy::initialize();
        def("cluster1d", cluster<1, false>);
        def("cluster2d", cluster<2, false>);
        def("cluster3d", cluster<3, false>);
        def("cluster4d", cluster<4, false>);
        def("cluster2d_weighted", cluster<2, true>);
        def("cluster3d_weighted", cluster<3, true>);
        def("cluster4d_weighted", cluster<4, true>);
    }

}