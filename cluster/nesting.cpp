#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <limits>
#include "util.hpp"
#include "clustering_util.hpp"
#include "kmeans1d.hpp"

// TODO: Is this needed?
//#define BOOST_PYTHON_STATIC_LIB

namespace brainclustering {


    void dendogram_mapping_diff(std::vector<std::vector<size_t>> &dendogram1,
                                std::vector<std::vector<size_t>> &dendogram2,
                                ndarray const &ignore_indices_vec1,
                                ndarray const &ignore_indices_vec2,
                                const object &clustering_vec1,
                                const object &clustering_vec2,
                                size_t n_points) {
        std::vector<size_t> clustering_steps1 = to_std_vector<size_t>(clustering_vec1);
        std::vector<size_t> clustering_steps2 = to_std_vector<size_t>(clustering_vec2);
        if (clustering_steps1[clustering_steps1.size() - 1] != clustering_steps2[clustering_steps2.size() - 1]) {
            std::cerr << "The main number of clusters should be the same for both dendograms." << std::endl;
            std::abort();
        }
        auto *ignore1 = get_c_array<size_t>(ignore_indices_vec1);
        auto *ignore2 = get_c_array<size_t>(ignore_indices_vec2);
        size_t size_ignore1 = ignore_indices_vec1.shape(0);
        size_t size_ignore2 = ignore_indices_vec2.shape(0);
        std::vector<int> prev_labels1(n_points, -1);
        std::vector<int> curr_labels1(n_points, -1);
        std::vector<int> prev_labels2(n_points, -1);
        std::vector<int> curr_labels2(n_points, -1);
        size_t main_clusters = clustering_steps1[clustering_steps1.size() - 1];
        std::vector<std::vector<int>> cost_matrix(main_clusters, std::vector<int>(main_clusters, 0));

        for (size_t i = 0; i < main_clusters; ++i) {
            prev_labels1[i] = i;
            prev_labels2[i] = i;
        }

        // TODO: Remove these two vectors (they are for sanity checks)
        //std::vector<size_t> in1;
        //std::vector<size_t> in2;
        for (size_t i = clustering_steps1.size(); i > 0;) {
            --i;
            size_t step_init = 0;
            for (size_t j = 0; j < i; ++j) {
                step_init += clustering_steps1[j];
            }
            size_t step_end = step_init + clustering_steps1[i];
            size_t lab_iter = 0;

            for (size_t j = step_init; j < step_end; ++j) {
                if (i == 0) {
                    for (size_t k = 1; k < dendogram1[j].size(); ++k) {
                        if (std::binary_search(ignore1, ignore1 + size_ignore1, dendogram1[j][k])) {
                            //in1.push_back(dendogram1[j][k]);
                            continue;
                        }
                        curr_labels1[dendogram1[j][k]] = prev_labels1[lab_iter];
                    }
                } else {
                    for (size_t k = 1; k < dendogram1[j].size(); ++k) {
                        curr_labels1[dendogram1[j][k]] = prev_labels1[lab_iter];
                    }
                }
                lab_iter++;
            }
            std::swap(prev_labels1, curr_labels1);
        }

        size_t init_top = 0;
        for (size_t i = clustering_steps2.size(); i > 0;) {
            --i;
            size_t step_init = 0;
            for (size_t j = 0; j < i; ++j) {
                step_init += clustering_steps2[j];
            }

            if (i == clustering_steps2.size() - 1) {
                init_top = step_init;
            }

            size_t step_end = step_init + clustering_steps2[i];
            size_t lab_iter = 0;
            for (size_t j = step_init; j < step_end; ++j) {
                if (i == 0) {
                    for (size_t k = 1; k < dendogram2[j].size(); ++k) {
                        if (std::binary_search(ignore2, ignore2 + size_ignore2, dendogram2[j][k])) {
                            //in2.push_back(dendogram2[j][k]);
                            continue;
                        }
                        curr_labels2[dendogram2[j][k]] = prev_labels2[lab_iter];
                    }
                } else {
                    for (size_t k = 1; k < dendogram2[j].size(); ++k) {
                        curr_labels2[dendogram2[j][k]] = prev_labels2[lab_iter];
                    }
                }
                lab_iter++;
            }

            std::swap(prev_labels2, curr_labels2);
        }
        /*// Sanity check
        if (in1.size() != size_ignore1) {
            std::cerr
                    << "The number of ignored indices found for the first dendogram does not correspond to the expected value."
                    << std::endl;
            std::abort();
        }
        // Sanity check
        if (in2.size() != size_ignore2) {
            std::cerr
                    << "The number of ignored indices found for the second dendogram does not correspond to the expected value."
                    << std::endl;
            std::abort();
        }*/

        for (size_t i = 0; i < n_points; ++i) {
            if (prev_labels1[i] == -1 || prev_labels2[i] == -1) {
                continue;
            }
            cost_matrix[prev_labels2[i]][prev_labels1[i]]++;
        }

        auto mapping_tuples = Hungarian::Solve(cost_matrix);
        if (mapping_tuples.size() != main_clusters) {
            std::cerr << "The length of the tuples is not correct." << std::endl;
            std::abort();
        }

        /*for (size_t i = 0; i < mapping_tuples.size(); ++i) {
            std::cout << "(" << i << ", " << mapping_tuples[i] << ") ";
        }
        std::cout << std::endl;*/

        for (size_t i = 0; i < mapping_tuples.size(); ++i) {
            while (i != mapping_tuples[i]) {
                size_t swap_val = mapping_tuples[i];
                std::swap(dendogram2[init_top + i], dendogram2[init_top + swap_val]);
                std::swap(mapping_tuples[i], mapping_tuples[swap_val]);
            }
        }

        /*std::cout << std::endl;
        for (size_t i = init_top; i < dendogram2.size(); ++i) {
            std::cout << i << ": ";
            print_container(dendogram2[i]);
        }*/

    }

    uint32_t upper_power_of_two(uint32_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;

    }

    template<unsigned int dimensions>
    object return_cluster_sizes(ndarray const &data_vector, size_t n_clusters_min, size_t n_clusters_max,
                                int const verbose = 0) {
        using point_type = point_nd<double, dimensions>;
        check_dimensions<double>(data_vector, dimensions);
        auto *data = get_c_ndarray<double, dimensions>(data_vector);
        size_t n_points = data_vector.shape(0);

        if (n_clusters_max > n_points) {
            std::cerr << "The number of clusters is greater than the number of points." << std::endl;
            std::abort();
        }


        // Build a vector that contains the cluster costs and their positions
        size_t total_clusterings = (n_clusters_max - n_clusters_min) + 1;
        if (verbose) {
            std::cout << "We are interested in " << total_clusterings << " clusterings." << std::endl;
        }
        std::vector<pair<double>>
                cluster_costs(total_clusterings);

        std::vector<double> sorted_array(n_points);
        Matrix<double> D;
        if constexpr (dimensions == 1) {
            for (ulong i = 0; i < n_points; ++i) {
                sorted_array[i] = data[i][0];
            }
            sort(sorted_array.begin(), sorted_array.end());
            D.resize(n_clusters_max, n_points);
            Matrix<ulong> T(n_clusters_max, n_points);
            fill_matrices(sorted_array, D, T);
            D.set(n_clusters_max - 1, n_points - 1, 0);
        }

        // We start from n_points - 1, we don't really need the clustering for k = n
        for (size_t cluster_size = n_clusters_max; cluster_size > n_clusters_min - 1; --cluster_size) {
            if (verbose) {
                std::cout << "Building cluster cost for cluster size " << cluster_size << std::endl;
            }
            // In case we are in 1D, run the optimal clustering
            cluster_costs[n_clusters_max - cluster_size].idx_ = cluster_size;
            cluster_costs[n_clusters_max - cluster_size].value_ = 0;
            if constexpr (dimensions == 1) {
                cluster_costs[n_clusters_max - cluster_size].value_ = D.get(cluster_size - 1, n_points - 1);
            } else { // Otherwise run the kmeans++ approximation
                std::vector<double> temp_labels(n_points);
                // Build a dummy array that contains the centers
                // We don't save them because they might be just too many
                std::vector<point_type> cluster_temp(cluster_size);
                kmeans(data, n_points, cluster_temp.data(), cluster_size, temp_labels.data(), 0);
                // Assign the cluster cost of the current cluster
                for (size_t point = 0; point < n_points; ++point) {
                    double distance = squared_euclidean_distance(data[point], cluster_temp[temp_labels[point]]);
                    // Compute the cost of this clustering
                    cluster_costs[n_clusters_max - cluster_size].value_ += distance;
                }
            }
        }
        // We sort the elements by increasing value only if we do not have an initial optimal clustering
        if constexpr (dimensions != 1) {
            std::sort(cluster_costs.begin(), cluster_costs.end(),
                      [](pair<double> const &a, pair<double> const &b) {
                          return a.value_ < b.value_;
                      });
        }

        // Deterministic
        std::vector<size_t> highlight_elements;
        size_t current_element = 0, min_k;
        // because we don't actually need to sort them, they are already sorted in descending order
        // but this is better to understand what is going on
        do {
            // If the cost of the current clustering is 0
            if (cluster_costs[current_element].value_ == 0) {
                ++current_element;
                continue;
            }
            // Find the first cluster that has an actual cost
            auto rounded = (uint32_t) ceil(cluster_costs[current_element].value_);

            // The interval is made up by cost values between rounded and the next power of two
            uint32_t next_power = upper_power_of_two(rounded);

            //std::cout << "The cost at the beginning of the interval is " << cluster_costs[current_element].value_
            //          << " with k=" << cluster_costs[current_element].idx_ << "." << std::endl;
            //std::cout << "The next power of two is " << next_power << "." << std::endl;

            // Set the minimum to the first element of the interval
            min_k = cluster_costs[current_element].idx_;
            // Start scanning the interval from the next element
            size_t element_in_interval = current_element + 1;
            // Loop until the interval is over
            while (cluster_costs[element_in_interval].value_ < next_power &&
                   element_in_interval < cluster_costs.size()) {
                // Save the element with the smallest k
                //std::cout << "Considering id " << element_in_interval << ", with k="
                //          << cluster_costs[element_in_interval].idx_ <<
                //          " and value " << cluster_costs[element_in_interval].value_ << "." << std::endl;
                min_k = std::min(cluster_costs[element_in_interval].idx_, min_k);
                ++element_in_interval;
            }

            //std::cout << "The interval ends with k = " << cluster_costs[element_in_interval - 1].idx_ << "." << std::endl;

            //std::cout << "The smallest k in this interval is " << min_k << "." << std::endl;

            // The next interval starts from the next element
            highlight_elements.push_back(min_k);
            current_element = element_in_interval;
            //std::cout << std::endl;
            // Stop if we have reached the end of the points
        } while (current_element < cluster_costs.size());

        if (verbose) {
            print_container(highlight_elements, "The main elements are:");
        }

        return to_python_list(highlight_elements);
    }


    template<typename point_type>
    size_t nesting_routine(std::vector<std::vector<size_t>> &dendogram,
                           point_type *centers1, size_t size1,
                           point_type *centers2, size_t size2,
                           size_t depth) {
        // Always assume that the size of centers1 is larger

        std::vector<point_type> output_centers(size2);
        // Keep track of how many clusters of C1 are assigned to each center of C2
        std::vector<size_t> count(size1, 0);
        // Keep track of the merges that have to be performed on the dendogram
        std::vector<std::vector<size_t>> merges(size2);
        for (auto &merge : merges) {
            merge.push_back(depth);
        }

        for (size_t i = 0; i < size1; ++i) {
            // Find the closest center in C2 to the current center of C1
            best_pair closest_center = {0, 0};
            if constexpr (point_type::dimensions == 1) {
                closest_center = nearest_center_binsearch(centers1[i], centers2, size2);
            } else {
                closest_center = nearest_center(centers1[i], centers2, size2);
            }
            merges[closest_center.cluster].push_back(i);
            // Add the centers to the output
            output_centers[closest_center.cluster] += centers1[i];
            // Which will then be divided by how many elements are there
            ++count[closest_center.cluster];
        }
        for (auto &merge : merges) {
            if (merge.size() > 1) {
                dendogram.push_back(merge);
            }
        }

        size_t new_element = 0;
        for (size_t i = 0; i < size2; ++i) {
            if (count[i] == 0) {
                continue;
            }
            // Finally, compute the new centers
            centers1[new_element] = output_centers[i] / std::max<size_t>(1, count[i]);
            ++new_element;
        }
        return new_element;
    }

    template<unsigned int dimensions>
    std::vector<std::vector<size_t>> cluster(ndarray const &data_vector, list &ks_vector, int const verbose = 0) {
        check_dimensions<double>(data_vector, dimensions);
        using value_type = double;
        using point_type = point_nd<value_type, dimensions>;
        auto *data = get_c_ndarray<double, dimensions>(data_vector);
        size_t n_points = data_vector.shape(0);
        std::vector<size_t> highlight_elements = to_std_vector<size_t>(ks_vector);

        for (unsigned long highlight_element : highlight_elements) {
            if (highlight_element > n_points) {
                std::cerr << "The division in " << highlight_element
                          << " clusters cannot be performed with data of size " << n_points << "." << std::endl;
                std::abort();
            }
        }

        // Initialize dendogram
        std::vector<std::vector<size_t>> dendogram;

        // The elements are theoretically already sorted
        //std::sort(highlight_elements.begin(), highlight_elements.end(), std::greater<>());
        // It is important to add the first centers to the dendogram too,
        // such that we can always have the original points
        std::vector<point_type> centers1_vec(n_points);
        point_type *const centers1 = centers1_vec.data();
        // We keep centers for the entire run, because we can just smaller and smaller portions of it
        // The first centers are the data points
        for (size_t i = 0; i < n_points; ++i) {
            centers1[i] = data[i];
        }
        std::vector<double> sorted_array;
        Matrix<ulong> T;
        if (point_type::dimensions == 1) {
            sorted_array.resize(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                sorted_array[i] = data[i][0];
            }
            std::sort(sorted_array.begin(), sorted_array.end());
            T.resize(highlight_elements[0], n_points);
            Matrix<double> D(highlight_elements[0], n_points);
            fill_matrices(sorted_array, D, T);
        }
        // Since at each level multiple merges might happen, to have all possible numbers of clusters we could
        // select a random element out of the children, but this is for later
        size_t cluster1_size = n_points;
        for (size_t j = 0; j < highlight_elements.size(); ++j) {
            if (verbose) {
                std::cout << "Looking at element " << highlight_elements[j] << std::endl;
            }
            // Start the actual nesting routine
            // If the second clustering (C2) has a larger size than our current clustering
            if (highlight_elements[j] >= cluster1_size) {
                // Then we can move to the next step
                ks_vector.pop(j);
                continue;
            }
            size_t size2 = highlight_elements[j];
            std::vector<point_type> centers2(size2);
            // Compute the clustering for the smaller size
            if constexpr (point_type::dimensions == 1) {
                return_labels<false>(T, sorted_array.data(), (double *) centers2.data(), size2);
            } else {
                kmeans<point_type, false>(data, n_points, centers2.data(), size2);
            }

            cluster1_size = nesting_routine(dendogram, centers1, cluster1_size, centers2.data(), size2,
                                            (highlight_elements.size() - j - 1));
            ks_vector[j] = cluster1_size;
            if (verbose) {
                std::cout << "The size of the resulting clustering is " << cluster1_size << "." << std::endl;
            }
        }

        /*for (auto &i : dendogram) {
            print_container(i, "");
        }

        std::cout << dendogram.size() << std::endl;*/
        return dendogram;
    }

    BOOST_PYTHON_MODULE (nesting) {
        python::type_info infoVectorValue = python::type_id<std::vector<size_t>>();
        const python::converter::registration *regVectorValue = python::converter::registry::query(infoVectorValue);
        if (regVectorValue == NULL || (*regVectorValue).m_to_python == NULL) {
            python::class_<std::vector<size_t >>("vector").def(
                    python::vector_indexing_suite<std::vector<size_t >>());
            python::class_<std::vector<std::vector<size_t>>>("vector_vector").def(
                    python::vector_indexing_suite<std::vector<std::vector<size_t>>>());
        }

        numpy::initialize();
        def("cluster1d", cluster<1>);
        def("cluster2d", cluster<2>);
        def("cluster3d", cluster<3>);
        def("cluster4d", cluster<4>);
        def("return_cluster_sizes1d", return_cluster_sizes<1>);
        def("return_cluster_sizes2d", return_cluster_sizes<2>);
        def("return_cluster_sizes3d", return_cluster_sizes<3>);
        def("return_cluster_sizes4d", return_cluster_sizes<4>);
        def("build_labels", build_labels);
        def("build_labels_cluster_color", build_labels_cluster_color);
        def("dendogram_mapping", dendogram_mapping);
        def("dendogram_mapping_diff", dendogram_mapping_diff);
    }
}