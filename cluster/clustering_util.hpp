#pragma once

#include <cstddef>
#include <random>
#include <iostream>
#include "munkres.cpp"

namespace brainclustering {

    struct best_pair {
        size_t cluster;
        double distance;
    };

    struct label_pair {
        double init{0.0};
        double end{0.0};
    };

    void build_labels_divisive(ndarray const &labels_vector, std::vector<std::vector<size_t>> &dendogram,
                               std::vector<size_t> &clustering_steps, double const shading_factor = 1.0) {
        auto *labels = get_c_array<double>(labels_vector);
        size_t n_points = labels_vector.shape(0);
        // Counter to keep track of how many of each level we have seen already
        std::vector<size_t> counter(clustering_steps.size(), 0);
        std::vector<label_pair> labels_interval(n_points, {0.0, 1.0 * shading_factor});

        /*for (int m = 0; m < dendogram.size(); ++m) {
            std::cout << m << " ";
            print_container(dendogram[m]);
        }*/

        double portion, init, end;

        //Iterate through the lines of the dendogram
        for (auto &node : dendogram) {
            //print_container(node);
            // The first element of the row represents the iteration
            size_t k_index = node[0];
            //std::cout << "Considering " << clustering_steps[k_index] << std::endl;
            // If the counter reaches the same number as "k" then we need to reset it
            if (counter[k_index] == clustering_steps[k_index]) {
                //std::cout << "Resetting the counter" << std::endl;
                // And we also reset all the counters after it
                for (size_t i = k_index; i < clustering_steps.size(); ++i) {
                    counter[i] = 0;
                }
            }
            //std::cout << "Setting label init to " << init << std::endl;
            //std::cout << "Setting label end to " << end << std::endl;
            // Go through the other elements of the row
            for (size_t j = 1; j < node.size(); ++j) {
                // The current portion to add depends on the current element
                // and the size of the elements
                portion = (labels_interval[node[j]].end - labels_interval[node[j]].init) /
                          (double) (clustering_steps[k_index]);
                // The initial part is the initial plus a portion muliplied
                // how many times the element has been seen already
                init = labels_interval[node[j]].init + portion * counter[k_index];
                // And the end is the same plus one
                end = labels_interval[node[j]].init + portion * (double) (counter[k_index] + 1);
                //std::cout << "New init " << init << ", new end " << end << std::endl;
                labels_interval[node[j]].init = init;
                labels_interval[node[j]].end = end;
            }
            counter[k_index]++;
        }

        for (size_t i = 0; i < labels_interval.size(); ++i) {
            labels[i] = (labels_interval[i].end - labels_interval[i].init) / 2 + labels_interval[i].init;
        }

    }

    void build_labels_agglomerative(ndarray const &labels_vector, std::vector<std::vector<size_t>> &dendogram,
                                    std::vector<size_t> &clustering_steps, double const shading_factor = 1.0) {
        auto *labels = get_c_array<double>(labels_vector);
        size_t n_points = labels_vector.shape(0);
        std::vector<label_pair> prev_labels_interval(n_points);
        std::vector<label_pair> curr_labels_interval(n_points);

        double lab_init = 0.0;
        double lab_end = 1.0 * shading_factor;
        double portion = (lab_end - lab_init) / (double) (clustering_steps[clustering_steps.size() - 1]);
        for (size_t i = 0; i < clustering_steps[clustering_steps.size() - 1]; ++i) {
            prev_labels_interval[i].init = lab_init;
            prev_labels_interval[i].end = lab_init + portion;
            lab_init += portion;
        }

        /*for (size_t i = 0; i < prev_labels_interval.size(); ++i) {
            std::cout << "(" << prev_labels_interval[i].init << ", " << prev_labels_interval[i].end << ") ";
        }
        std::cout << std::endl;*/


        /*for (size_t i = 0; i < dendogram.size(); ++i) {
            std::cout << i << ": ";
            print_container(dendogram[i]);
        }*/

        // Start with the smallest cluster (top)
        for (size_t i = clustering_steps.size(); i > 0;) {
            --i;
            //std::cout << "Considering clustering step: " << clustering_steps[i] << std::endl;
            size_t step_init = 0;
            // Compute where the rows of this clustering step start
            for (size_t j = 0; j < i; ++j) {
                step_init += clustering_steps[j];
            }

            size_t step_end = step_init + clustering_steps[i];
            //std::cout << "Start at " << step_init << ", end at " << step_end << std::endl;
            // This determines in which portion of the cluster we are
            size_t lab_iter = 0;

            /*for (size_t j = 0; j < prev_labels_interval.size(); ++j) {
                std::cout << "(" << prev_labels_interval[i].init << ", " << prev_labels_interval[i].end << ") ";
            }
            std::cout << std::endl;
             */
            for (size_t j = step_init; j < step_end; ++j) {
                /*std::cout << "Current step " << j << std::endl;
                if (!std::is_sorted(dendogram[j].begin() + 1, dendogram[j].end())) {
                    print_container(dendogram[j], "Not Sorted");
                }*/
                lab_init = prev_labels_interval[lab_iter].init;
                lab_end = prev_labels_interval[lab_iter].end;
                //std::cout << "init: " << lab_init << ", end: " << lab_end << std::endl;
                if (i == 0) {
                    // If we are at biggest cluster (the one containing the indices of the labels)
                    // then we can set the labels values already
                    for (size_t k = 1; k < dendogram[j].size(); ++k) {
                        labels[dendogram[j][k]] = (lab_end - lab_init) / 2 + lab_init;
                    }
                } else {
                    // Otherwise we need to compute the parents for the next level
                    portion = (lab_end - lab_init) / (double) (dendogram[j].size() - 1);
                    for (size_t k = 1; k < dendogram[j].size(); ++k) {
                        curr_labels_interval[dendogram[j][k]].init = lab_init;
                        curr_labels_interval[dendogram[j][k]].end = lab_init + portion;
                        lab_init += portion;
                    }
                }
                lab_iter++;
            }

            std::swap(prev_labels_interval, curr_labels_interval);
        }

        /*for (size_t i = 0; i < prev_labels_interval.size(); ++i) {
            labels[i] = (prev_labels_interval[i].end - prev_labels_interval[i].init) / 2 + prev_labels_interval[i].init;
        }*/
    }

    void dendogram_mapping(std::vector<std::vector<size_t>> &dendogram1,
                           std::vector<std::vector<size_t>> &dendogram2,
                           ndarray const &ignore_indices_vec1,
                           ndarray const &ignore_indices_vec2,
                           const object &clustering_vec,
                           size_t const n_points) {
        std::vector<size_t> clustering_steps = to_std_vector<size_t>(clustering_vec);
        auto *ignore1 = get_c_array<size_t>(ignore_indices_vec1);
        auto *ignore2 = get_c_array<size_t>(ignore_indices_vec2);
        size_t size_ignore1 = ignore_indices_vec1.shape(0);
        size_t size_ignore2 = ignore_indices_vec2.shape(0);
        std::vector<std::pair<int, int>> prev_labels(n_points, std::pair<int, int>(-1, -1));
        std::vector<std::pair<int, int>> curr_labels(n_points, std::pair<int, int>(-1, -1));
        size_t main_clusters = clustering_steps[clustering_steps.size() - 1];
        std::vector<std::vector<int>> cost_matrix(main_clusters, std::vector<int>(main_clusters, 0));

        for (size_t i = 0; i < main_clusters; ++i) {
            prev_labels[i].first = i;
            prev_labels[i].second = i;
        }

        /*for (size_t i = 0; i < prev_labels_interval.size(); ++i) {
            std::cout << "(" << prev_labels_interval[i].first << ", " << prev_labels_interval[i].second << ") ";
        }
        std::cout << std::endl;*/


        /*for (size_t i = 0; i < dendogram1.size(); ++i) {
            std::cout << i << ": ";
            print_container(dendogram1[i]);
        }
        std::cout << std::endl;
        for (size_t i = 0; i < dendogram2.size(); ++i) {
            std::cout << i << ": ";
            print_container(dendogram2[i]);
        }
        */
        size_t init_top = 0;

        // TODO: Remove these two vectors (they are for sanity checks)
        std::vector<size_t> in1;
        std::vector<size_t> in2;

        // Start with the smallest cluster (top)
        for (size_t i = clustering_steps.size(); i > 0;) {
            --i;
            //std::cout << "Considering clustering step: " << clustering_steps[i] << std::endl;
            size_t step_init = 0;
            // Compute where the rows of this clustering step start
            for (size_t j = 0; j < i; ++j) {
                step_init += clustering_steps[j];
            }

            if (i == clustering_steps.size() - 1) {
                init_top = step_init;
            }

            size_t step_end = step_init + clustering_steps[i];
            //std::cout << "Start at " << step_init << ", end at " << step_end << std::endl;
            // This determines in which portion of the cluster we are
            size_t lab_iter = 0;

            for (size_t j = step_init; j < step_end; ++j) {
                //std::cout << "init: " << lab_init << ", end: " << lab_end << std::endl;
                // If we are at biggest cluster (the one containing the indices of the labels)
                // then we can set the labels values already
                if (i == 0) {
                    for (size_t k = 1; k < dendogram1[j].size(); ++k) {
                        if (std::binary_search(ignore1, ignore1 + size_ignore1, dendogram1[j][k])) {
                            in1.push_back(dendogram1[j][k]);
                            continue;
                        }
                        curr_labels[dendogram1[j][k]].first = prev_labels[lab_iter].first;
                    }
                    for (size_t k = 1; k < dendogram2[j].size(); ++k) {
                        if (std::binary_search(ignore2, ignore2 + size_ignore2, dendogram2[j][k])) {
                            in2.push_back(dendogram2[j][k]);
                            continue;
                        }
                        curr_labels[dendogram2[j][k]].second = prev_labels[lab_iter].second;
                    }
                } else {
                    for (size_t k = 1; k < dendogram1[j].size(); ++k) {
                        curr_labels[dendogram1[j][k]].first = prev_labels[lab_iter].first;
                    }
                    for (size_t k = 1; k < dendogram2[j].size(); ++k) {
                        curr_labels[dendogram2[j][k]].second = prev_labels[lab_iter].second;
                    }
                }
                lab_iter++;
            }

            std::swap(prev_labels, curr_labels);
        }
        // Sanity check
        /*if (in1.size() != size_ignore1) {
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
        /*for (auto &labels : prev_labels_interval) {
            std::cout << "(" << labels.first << ", " << labels.second << ") ";
        }
        std::cout << std::endl;
        */
        for (auto &labels : prev_labels) {
            if (labels.first == -1 || labels.second == -1) {
                continue;
            }
            cost_matrix[labels.second][labels.first]++;
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

    void build_labels(ndarray const &labels_vector, std::vector<std::vector<size_t>> &dendogram,
                      const object &clustering_vec, const bool agglomerative, double const shading_factor = 1.0) {
        std::vector<size_t> clustering_steps = to_std_vector<size_t>(clustering_vec);
        if (agglomerative) {
            build_labels_agglomerative(labels_vector, dendogram, clustering_steps, shading_factor);
        } else {
            build_labels_divisive(labels_vector, dendogram, clustering_steps, shading_factor);
        }
    }

    // Only works for 1dim
    void build_labels_cluster_color(ndarray const &data_vector, ndarray const &labels_vector,
                                    std::vector<std::vector<size_t>> &dendogram,
                                    size_t last_level_size) {
        check_dimensions<double>(data_vector, 1);
        auto *labels = get_c_array<double>(labels_vector);
        auto *data = get_c_ndarray<double, 1>(data_vector);
        std::vector<double> center_colour(last_level_size);
        for (size_t i = 0; i < last_level_size; ++i) {
            for (size_t j = 0; j < dendogram[i].size(); ++j) {
                center_colour[i] += data[dendogram[i][j]][0];
            }
            //std::cout << "center color: " << center_colour[i] << std::endl;
            center_colour[i] /= dendogram[i].size();
            for (size_t j = 0; j < dendogram[i].size(); ++j) {
                labels[dendogram[i][j]] = center_colour[i];
            }
        }

    }


    template<typename point_type>
    size_t binary_search(point_type *arr, int l, int r, point_type &val) {
        int best_index = l;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            //std::cout << l << ", " << r << ", " << mid << std::endl;
            //std::cout << arr[mid][0] << ", " << val[0] << std::endl;
            if (arr[mid][0] < val[0]) {
                l = mid + 1;
            } else if (arr[mid][0] > val[0]) {
                r = mid - 1;
            } else {
                best_index = mid;
                break;
            }
            //std::cout << squared_euclidean_distance(arr[mid], val) << ", "
            //          << squared_euclidean_distance(arr[best_index], val) << std::endl;
            if (squared_euclidean_distance(arr[mid], val) < squared_euclidean_distance(arr[best_index], val)) {
                best_index = mid;
            }
        }
        return best_index;
    }

    template<typename point_type>
    inline static best_pair nearest_center_binsearch(point_type const &point, point_type const *const centers,
                                                     size_t const initialized_limit) {
        //print_ndarray(centers, initialized_limit, "centers: ");
        size_t cluster = binary_search(centers, 0, initialized_limit - 1, point);
        double distance = squared_euclidean_distance(centers[cluster], point);
        best_pair best = {cluster, distance};
        return best;
    }

    template<typename point_type, bool weighted = false>
    inline static best_pair nearest_center(point_type const &point, point_type const *const centers,
                                           size_t const initialized_limit,
                                           [[maybe_unused]]double const *const weights = nullptr) {
        best_pair current_best = {0, 0};
        double distance;
        if constexpr (weighted) {
            current_best = {0, squared_euclidean_distance(point, centers[0], weights)};
        } else {
            current_best = {0, squared_euclidean_distance(point, centers[0])};
        }
        for (size_t cluster = 1; cluster < initialized_limit; ++cluster) {
            if constexpr (weighted) {
                distance = squared_euclidean_distance(point, centers[cluster], weights);
            } else {
                distance = squared_euclidean_distance(point, centers[cluster]);
            }
            if (distance < current_best.distance) {
                current_best.distance = distance;
                current_best.cluster = cluster;
            }
        }
        return current_best;
    }

    template<typename point_type, bool weighted = false>
    void kmeans_plus_plus(point_type const *const data, size_t const n_points,
                          point_type *centers, size_t n_clusters,
                          size_t const seed, double const *const weights = nullptr) {
        std::mt19937 random_number_generator(seed);
        std::uniform_int_distribution<size_t> distribution(0, n_points - 1);
        // Choose the first center at random from the data points
        centers[0] = data[distribution(random_number_generator)];
        std::vector<double> D(n_points);

        for (size_t cluster = 1; cluster < n_clusters; ++cluster) {
            double sum_distances = 0;
            // Since we are using a custom distribution, we use the CDF instead of the PDF
            // The CDF is always distributed between 0 and 1
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            // We draw a value from the CDF
            double prob = distribution(random_number_generator);
            for (size_t point = 0; point < n_points; ++point) {
                // For each point, compute the distance between the points and the closest centers
                // Considering only the centers that have already been chosen
                best_pair best_cluster = nearest_center<point_type, weighted>(data[point], centers, cluster + 1,
                                                                              weights);
                D[point] = best_cluster.distance;
                // We sum all distances, such that we know what is the highest value
                sum_distances += best_cluster.distance;
            }

            // We find the value of the point that has been chosen
            sum_distances = sum_distances * prob;

            for (size_t point = 0; point < n_points; ++point) {
                // We subtract from sum distances
                sum_distances = sum_distances - D[point];
                // If we get a value less than 0, then it means that we reached the chosen point
                if (sum_distances <= 0) {
                    centers[cluster] = data[point];
                    break;
                }
            }
        }
    }


    template<typename point_type, bool ret_labels = true, bool weighted = false>
    double kmeans(point_type const *const data, size_t const n_points,
                  point_type *centers, size_t n_clusters,
                  double *labels = nullptr, size_t cluster_addition = 1,
                  size_t const init_method = 1, size_t const n_iterations = 100,
                  double const tolerance = 1e-4, size_t const seed = 0, double const *const weights = nullptr,
                  size_t const verbose = 0) {
        if (init_method == 1) { // k-means++
            if (verbose) {
                std::cout << "Setting up the centers with kmeans++." << std::endl;
            }
            kmeans_plus_plus(data, n_points, centers, n_clusters, seed, weights);
        } else if (init_method == 2) { // Random
            if (verbose) {
                std::cout << "Setting up the centers randomly." << std::endl;
            }
            static std::mt19937 random_number_generator(seed);
            std::uniform_int_distribution<size_t> distribution(0, n_points - 1);
            // Set each cluster to a random data points
            for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
                centers[cluster] = data[distribution(random_number_generator)];
            }
        } else { // The centers are already initialized
            if (verbose) {
                std::cout << "Considering previously initialized centers." << std::endl;
            }
        }

        if (verbose) {
            print_ndarray(centers, n_clusters, "Initialized centers: ");
        }

        // Initialize Frobenius norm
        double difference_norm = std::numeric_limits<double>::max();
        double current_inertia = 0;

        // Start the iterations
        // Stop if the number of iterations has been reached or if the centers have converged
        for (size_t iteration = 0; iteration < n_iterations; ++iteration) {
            if (difference_norm < tolerance) {
                if (verbose) {
                    std::cout << "Convergence was reached at iteration " << iteration + 1 << "." << std::endl;
                }
                break;
            }

            // Create an array to store the new centers
            std::vector<point_type> new_centers(n_clusters);
            // And a vector to count how many points are in each cluster
            std::vector<size_t> counts(n_clusters, 0);
            // Set the current inertia to zero
            current_inertia = 0;
            // Assign each point to their nearest center
            for (size_t point = 0; point < n_points; ++point) {
                // Find the closest cluster to this point
                best_pair best_cluster = nearest_center<point_type, weighted>(data[point], centers, n_clusters,
                                                                              weights);
                // Assign point to the closest cluster + 1, because we want cluster 0 to be empty
                if constexpr (ret_labels) {
                    labels[point] = best_cluster.cluster * 1.0 + cluster_addition;
                }
                // Add the inertia to the current value
                current_inertia += best_cluster.distance;

                // Add this point to the new centers of that cluster
                new_centers[best_cluster.cluster] += data[point];
                // And increase the counter
                counts[best_cluster.cluster] += 1; // Count how many points are in the cluster
            }

            // Compute the Frobenius norm
            difference_norm = 0;
            // Divide the means by how many elements are in the cluster
            for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
                point_type new_centroid = new_centers[cluster] / std::max<size_t>(1, counts[cluster]);
                // The norm is the sum of squared distances
                difference_norm += squared_euclidean_distance(centers[cluster], new_centroid);
                centers[cluster] = new_centroid;
            }

            /*if (verbose) {
                print_ndarray(centers, n_clusters, "The new centers are: ");
            }*/

        }
        return current_inertia;
    }

    template<bool weighted, typename... weights>
    int get_weight(int a, weights... params) {
        static_assert(!weighted || (sizeof...(weights) >= 1));
        if constexpr (weighted) {
            // Theoretically fix it to get weight and index
            return a * std::get<0>(std::tuple<weights...>{params...});
        } else {
            return a;
        }
    }

    template<typename T>
    inline static T square(T value) {
        return value * value;
    }

    template<bool weighted, typename point_type>
    inline static typename point_type::value_type sedtest(point_type const &point_a,
                                                          point_type const &point_b,
                                                          double const *const weights) {
        typename point_type::value_type distance = 0;
        for (size_t i = 0; i < point_type::dimensions; ++i) {
            distance += get_weight<weighted>(square(point_a[i] - point_b[i]), weights, i);
        }
        return distance;
    }


    template<typename point_type>
    inline static typename point_type::value_type squared_euclidean_distance(point_type const &point_a,
                                                                             point_type const &point_b) {
        typename point_type::value_type distance = 0;
        for (size_t i = 0; i < point_type::dimensions; ++i) {
            distance += square(point_a[i] - point_b[i]);
        }
        return distance;
    }

    template<typename point_type>
    inline static typename point_type::value_type squared_euclidean_distance(point_type const &point_a,
                                                                             point_type const &point_b,
                                                                             double const *const weights) {
        typename point_type::value_type distance = 0;
        for (size_t i = 0; i < point_type::dimensions; ++i) {
            distance += weights[i] * square(point_a[i] - point_b[i]);
        }
        return distance;
    }

}