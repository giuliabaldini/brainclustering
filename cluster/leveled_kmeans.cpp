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
#include "kmeans1d.hpp"

namespace brainclustering {

    // TODO: Tested up to 3 levels, it might not work flawlessy for more levels
    std::vector<std::vector<size_t>> build_agglomerative_dendogram(std::vector<std::vector<size_t>> div_dendogram,
                                                                   const object &ks_vector) {
        /*for (auto & d : div_dendogram) {
            print_container(d, "");
        }*/
        std::vector<std::vector<size_t>> a_dendogram;
        std::vector<size_t> ks = to_std_vector<size_t>(ks_vector);
        size_t last_element = (ks.size() - 1);
        for (auto &d : div_dendogram) {
            // If the row is belongs to the bottom of the dendogram
            if (d[0] == last_element) {
                a_dendogram.push_back(d);
            }
        }
        // For the other nodes, build "fake" indices
        for (size_t i = last_element; i > 0;) {
            --i;
            size_t start = 0;
            size_t count_fake_level = ks[i];
            for (size_t j = i; j > 0;) {
                --j;
                count_fake_level *= ks[j];
            }
            for (size_t j = 0; j < count_fake_level; ++j) {
                std::vector<size_t> higher_level(ks[i + 1] + 1, 0);
                higher_level[0] = i;
                for (size_t k = 1; k < higher_level.size(); ++k) {
                    higher_level[k] = start;
                    start++;
                }
                a_dendogram.push_back(higher_level);
            }
        }
        /*for (size_t i = 0; i < a_dendogram.size(); ++i) {
            print_container(a_dendogram[i], "");
        }*/
        return a_dendogram;
    }

    template<typename point_type>
    void recursive_clustering(point_type const *const data,
                              std::vector<size_t>::iterator data_ids_start, std::vector<size_t>::iterator data_ids_end,
                              std::vector<size_t> &divisive_k_vector, size_t divisive_k_iter,
                              std::vector<std::vector<size_t>> &dendogram) {
        //std::cout << std::endl;
        size_t current_cluster_division = divisive_k_vector[divisive_k_iter];

        //std::cout << "We want to divide " << data_ids.size() << " points into " << current_cluster_division
        //          << " clusters." << std::endl;

        /*// If the number of clusters is 1, or there is only one point, we can use the previous division of the parent
        if (current_cluster_division == 1 || data_ids.size() == 1) {
            return;
        }*/
        size_t data_length = data_ids_end - data_ids_start;
        // Create a local variable to contain the resulting labels
        std::vector<double> new_labels(data_length);
        // If we have any points
        if (data_length > 0) {
            if (current_cluster_division == 1 || data_length == 1) {
                // If there is only one cluster, or there is only one point
                for (size_t i = 0; i < data_length; ++i) {
                    new_labels[i] = 0;
                }
            } else if (current_cluster_division < data_length) {
                // If the number of clusters is less than the number of points
                //std::cout << "There are enough points to cluster." << std::endl;
                // Then we run the clustering
                std::vector<point_type> new_data(data_length);
                //std::cout << "Creating new data vector." << std::endl;
                int data_i = 0;
                for (auto it = data_ids_start; it != data_ids_end; ++it, ++data_i) {
                    new_data[data_i] = data[*it];
                }
                //print_ndcontainer(new_data, "Considering these data points");
                //std::cout << "Creating centers of size " << current_cluster_division << "." << std::endl;
                std::vector<point_type> centers(current_cluster_division);
                if constexpr (point_type::dimensions == 1) {
                    cluster_1d<true, false>((double *) new_data.data(), data_length, (double *) centers.data(),
                                            current_cluster_division, new_labels.data());
                } else {
                    kmeans(new_data.data(), data_length, centers.data(), current_cluster_division,
                           new_labels.data(), 0);
                }
            } else {
                // Otherwise, if the number of points is equal to the number of clusters, or there are too many clusters
                // Then each point becomes a cluster
                //std::cout << "There are not enough points to cluster." << std::endl;
                for (size_t i = 0; i < data_length; ++i) {
                    new_labels[i] = i;
                }
            }
        }

        //print_array(new_labels, data_ids.size(), "The new labels are");
        // If there are still clusters to compute
        if (divisive_k_iter + 1 < divisive_k_vector.size()) {
            for (size_t i = 0; i < current_cluster_division; ++i) {
                // Create a new data vector, that contains the indices of the new children
                std::vector<size_t> current_ids;
                current_ids.push_back((divisive_k_iter - 1));
                int data_i = 0;
                for (auto it = data_ids_start; it != data_ids_end; ++it, ++data_i) {
                    // If the label is the same as the cluster we are considering
                    if (new_labels[data_i] == i) {
                        current_ids.push_back(*it);
                    }
                }
                // Push into the dendogram
                dendogram.push_back(current_ids);

                // Run recursively on the new IDs
                recursive_clustering(data, current_ids.begin() + 1, current_ids.end(),
                                     divisive_k_vector, divisive_k_iter + 1, dendogram);
            }
        } else {
            // We are in the last iteration
            size_t new_id = dendogram.size();

            for (size_t i = 0; i < current_cluster_division; ++i) {
                std::vector<size_t> current_ids;
                // Add level
                current_ids.push_back((divisive_k_iter - 1));
                dendogram.push_back(current_ids);
            }
            // For each point in new_data
            int data_i = 0;
            for (auto it = data_ids_start; it != data_ids_end; ++it, ++data_i) {
                // Add data to this cluster
                dendogram[new_id + new_labels[data_i]].push_back(*it);
            }
        }
    }

    template<unsigned int dimensions>
    std::vector<std::vector<size_t>> cluster(ndarray const &data_vector, const object &ks_vector) {
        using value_type = double;
        using point_type = point_nd<value_type, dimensions>;
        check_dimensions<value_type>(data_vector, dimensions);
        auto *data = get_c_ndarray<value_type, dimensions>(data_vector);
        size_t n_points = data_vector.shape(0);
        std::vector<size_t> ks = to_std_vector<size_t>(ks_vector);
        size_t needed_points = 1;
        for (size_t i = 0; i < ks.size(); ++i) {
            needed_points *= ks[i];
        }

        if (needed_points > n_points) {
            std::cerr << "The division with the given cluster list cannot be performed with data of size "
                      << n_points << "." << std::endl;
            std::abort();
        }

        ks.insert(ks.begin(), 1);

        std::vector<std::vector<size_t>> dendogram;
        std::vector<point_type> sorted_array;
        std::vector<size_t> sort_idxs;
        if constexpr (dimensions == 1) {
            sort_idxs.resize(n_points);
            iota(sort_idxs.begin(), sort_idxs.end(), 0);
            sort(sort_idxs.begin(),
                 sort_idxs.end(),
                 [&data](size_t a, size_t b) { return data[a][0] < data[b][0]; });
            sorted_array.resize(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                sorted_array[i] = data[sort_idxs[i]];
            }
        }

        std::vector<size_t> parent_data_ids(n_points, 0);
        for (size_t i = 0; i < n_points; ++i) {
            parent_data_ids[i] = i;
        }
        if constexpr (dimensions == 1) {
            recursive_clustering(sorted_array.data(), parent_data_ids.begin(), parent_data_ids.end(),
                                 ks, 1, dendogram);
        } else {
            recursive_clustering(data, parent_data_ids.begin(), parent_data_ids.end(), ks, 1, dendogram);
        }
        if constexpr (dimensions == 1) {
            for (auto &row : dendogram) {
                for (size_t j = 1; j < row.size(); ++j) {
                    row[j] = sort_idxs[row[j]];
                }
            }
        }

        /*for (size_t i = 0; i < dendogram.size(); ++i) {
            print_container(dendogram[i], "");
        }*/

        return dendogram;
    }

    BOOST_PYTHON_MODULE (leveled_kmeans) {
        python::type_info infoVectorValue = python::type_id<std::vector<size_t >>();
        const python::converter::registration *regVectorValue = python::converter::registry::query(infoVectorValue);
        if (regVectorValue == NULL || (*regVectorValue).m_to_python == NULL) {
            python::class_<std::vector<size_t >>("vector").def(
                    python::vector_indexing_suite<std::vector<size_t >>());
            python::class_<std::vector<std::vector<size_t >>>("vector_vector").def(
                    python::vector_indexing_suite<std::vector<std::vector<size_t >>>());
        }

        numpy::initialize();
        def("cluster1d", cluster<1>);
        def("cluster2d", cluster<2>);
        def("cluster3d", cluster<3>);
        def("cluster4d", cluster<4>);
        def("build_agglomerative_dendogram", build_agglomerative_dendogram);
        def("build_labels", build_labels);
        def("build_labels_cluster_color", build_labels_cluster_color);
        def("dendogram_mapping", dendogram_mapping);
    }

}