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

    void
    generate_labels(ndarray const &merge_vector, double min_dist, ndarray const &labels_vector,
                    double const shading_factor = 1.0, int const verbose = 0) {
        auto *merge = get_c_array<double>(merge_vector);
        auto *labels = get_c_array<double>(labels_vector);
        size_t n_points = labels_vector.shape(0);

        /*
        The output of linkage is stepwise dendrogram, which is represented as an (N − 1) × 4 NumPy array
        with floating point entries (dtype=numpy.double). The first two columns contain the node indices
        which are joined in each step. The input nodes are labeled 0,..,N − 1, and the newly generated
        nodes have the labels N,.., 2N − 2.
        The third column contains the distance between the two nodes at each step, ie. the current minimal
        distance at the time of the merge. The fourth column counts the number of points which comprise
        each new node.
        */
        if (!verbose) {
            std::cout.setstate(std::ios_base::failbit);
        }
        std::vector<label_pair> nodes_labels((n_points - 1) * 2 + 1);
        nodes_labels[(n_points - 1) * 2].init = 0.0;
        nodes_labels[(n_points - 1) * 2].end = 1.0 * shading_factor;
        double next_increase;
        for (size_t last_merge = n_points - 1; last_merge > 0;) {
            --last_merge;
            double merge_id1 = merge[last_merge * 4 + 0];
            double merge_id2 = merge[last_merge * 4 + 1];
            //std::cout << "Currently investigating merge step " << last_merge << " with merge ids " << merge_id1
            //          << " and " << merge_id2 << "." << std::endl;
            double parent = n_points + last_merge;
            next_increase = (nodes_labels[parent].end - nodes_labels[parent].init) / 2;
            if (merge[last_merge * 4 + 2] > min_dist) {
                nodes_labels[merge_id1].init = nodes_labels[parent].init;
                nodes_labels[merge_id1].end = nodes_labels[parent].end - next_increase;
                nodes_labels[merge_id2].init = nodes_labels[merge_id1].end;
                nodes_labels[merge_id2].end = nodes_labels[parent].end;
            } else {
                nodes_labels[merge_id1].init = nodes_labels[merge_id2].init = nodes_labels[parent].init;
                nodes_labels[merge_id1].end = nodes_labels[merge_id2].end = nodes_labels[parent].end;
            }
            /*std::cout << "The current parent is " << parent << std::endl;
            std::cout << "Node " << merge_id1 << " with label (" << nodes_labels[merge_id1].init << ", "
                      << nodes_labels[merge_id1].end << ")." << std::endl;
            std::cout << "Node " << merge_id2 << " with label (" << nodes_labels[merge_id2].init << ", "
                      << nodes_labels[merge_id2].end << ")." << std::endl;

            std::cout << "The next increase/decrease is " << next_increase << std::endl;

            std::cout << std::endl;*/

        }
        for (size_t i = 0; i < n_points; ++i) {
            labels[i] = (nodes_labels[i].end - nodes_labels[i].init) / 2 + nodes_labels[i].init;
        }

        std::cout.clear();
    }


    std::vector<std::vector<size_t>>
    build_agglomerative_dendogram(ndarray const &merge_vector, const object &ks_vector, int const verbose = 0) {
        auto *merge = get_c_array<double>(merge_vector);
        size_t n_points = merge_vector.shape(0) + 1;
        std::vector<size_t> ks = to_std_vector<size_t>(ks_vector);

        /*
        The output of linkage is stepwise dendrogram, which is represented as an (N − 1) × 4 NumPy array
        with floating point entries (dtype=numpy.double). The first two columns contain the node indices
        which are joined in each step. The input nodes are labeled 0,..,N − 1, and the newly generated
        nodes have the labels N,.., 2N − 2.
        The third column contains the distance between the two nodes at each step, ie. the current minimal
        distance at the time of the merge. The fourth column counts the number of points which comprise
        each new node.
        */
        if (!verbose) {
            std::cout.setstate(std::ios_base::failbit);
        }

        std::vector<std::vector<size_t>> dendogram;
        std::vector<std::vector<size_t>> d(2 * n_points - 1);
        for (size_t i = 0; i < n_points; ++i) {
            d[i].push_back(i);
        }
        size_t k_iter = 0;
        size_t current_int = n_points;
        for (size_t curr_merge = 0; curr_merge < n_points - 1; ++curr_merge) {
            if (k_iter == ks.size()) {
                break;
            }
            double merge_id1 = merge[curr_merge * 4 + 0];
            double merge_id2 = merge[curr_merge * 4 + 1];

            //std::cout << "Currently investigating merge step " << curr_merge << " with merge ids " << merge_id1
            //          << " and " << merge_id2 << "." << std::endl;

            for (size_t j = 0; j < d[merge_id1].size(); ++j) {
                d[current_int].push_back(d[merge_id1][j]);
            }
            for (size_t j = 0; j < d[merge_id2].size(); ++j) {
                d[current_int].push_back(d[merge_id2][j]);
            }
            d[merge_id1].erase(d[merge_id1].begin(), d[merge_id1].end());
            d[merge_id2].erase(d[merge_id2].begin(), d[merge_id2].end());
            current_int++;
            if ((n_points - curr_merge - 1) == ks[k_iter]) {
                /*for (size_t i = 0; i < d.size(); ++i) {
                    std::cout << i << ": ";
                    print_container(d[i]);
                }*/
                //std::cout << "Reached desired size of " << ks[k_iter] << std::endl;
                size_t insert_id = 0;
                for (size_t i = 0; i < d.size(); ++i) {
                    if (!d[i].empty()) {
                        std::vector<size_t> ids = d[i];
                        ids.insert(ids.begin(), ks.size() - k_iter - 1);
                        dendogram.push_back(ids);
                        d[i].erase(d[i].begin(), d[i].end());
                        d[i].push_back(insert_id);
                        insert_id++;
                    }
                }
                k_iter++;
            }
            /*for (size_t i = 0; i < d.size(); ++i) {
                std::cout << i << ": ";
                print_container(d[i]);
            }*/
        }

        /*for (size_t i = 0; i < dendogram.size(); ++i) {
            print_container(dendogram[i]);
        }*/

        std::cout.clear();
        return dendogram;
    }

    std::vector<std::vector<size_t>>
    build_agglomerative_dendogram_sorted(ndarray const &merge_vector, const object &ks_vector, int const verbose = 0) {
        auto *merge = get_c_array<double>(merge_vector);
        size_t n_points = merge_vector.shape(0) + 1;
        std::vector<size_t> ks = to_std_vector<size_t>(ks_vector);

        /*
        The output of linkage is stepwise dendrogram, which is represented as an (N − 1) × 4 NumPy array
        with floating point entries (dtype=numpy.double). The first two columns contain the node indices
        which are joined in each step. The input nodes are labeled 0,..,N − 1, and the newly generated
        nodes have the labels N,.., 2N − 2.
        The third column contains the distance between the two nodes at each step, ie. the current minimal
        distance at the time of the merge. The fourth column counts the number of points which comprise
        each new node.
        */
        if (!verbose) {
            std::cout.setstate(std::ios_base::failbit);
        }

        std::vector<std::vector<size_t>> dendogram;
        std::vector<std::vector<size_t>> d(2 * n_points - 1);
        for (size_t i = 0; i < n_points; ++i) {
            d[i].push_back(i);
        }
        size_t k_iter = 0;
        size_t current_int = n_points;
        for (size_t curr_merge = 0; curr_merge < n_points - 1; ++curr_merge) {
            if (k_iter == ks.size()) {
                break;
            }
            double merge_id1 = merge[curr_merge * 4 + 0];
            double merge_id2 = merge[curr_merge * 4 + 1];

            //std::cout << "Currently investigating merge step " << curr_merge << " with merge ids " << merge_id1
            //          << " and " << merge_id2 << "." << std::endl;

            for (size_t j = 0; j < d[merge_id1].size(); ++j) {
                d[current_int].push_back(d[merge_id1][j]);
            }
            for (size_t j = 0; j < d[merge_id2].size(); ++j) {
                d[current_int].push_back(d[merge_id2][j]);
            }
            d[merge_id1].erase(d[merge_id1].begin(), d[merge_id1].end());
            d[merge_id2].erase(d[merge_id2].begin(), d[merge_id2].end());
            current_int++;
            if ((n_points - curr_merge - 1) == ks[k_iter]) {
                /*for (int i = 0; i < d.size(); ++i) {
                    std::cout << i << ": ";
                    print_container(d[i]);
                }*/
                //std::cout << "Reached desired size of " << ks[k_iter] << std::endl;
                size_t insert_id = 0;
                for (size_t i = 0; i < d.size(); ++i) {
                    if (!d[i].empty()) {
                        std::vector<size_t> ids = d[i];
                        std::sort(ids.begin(), ids.end());
                        ids.insert(ids.begin(), ks.size() - k_iter - 1);
                        dendogram.push_back(ids);
                        d[i].erase(d[i].begin(), d[i].end());
                        d[i].push_back(insert_id);
                        insert_id++;
                    }
                }
                k_iter++;
            }
            /*for (int i = 0; i < d.size(); ++i) {
                std::cout << i << ": ";
                print_container(d[i]);
            }*/
        }

        /*for (int i = 0; i < dendogram.size(); ++i) {
            print_container(dendogram[i]);
        }*/

        std::cout.clear();
        return dendogram;
    }


    BOOST_PYTHON_MODULE (agglomerative) {
        python::type_info infoVectorValue = python::type_id<std::vector<size_t >>();
        const python::converter::registration *regVectorValue = python::converter::registry::query(infoVectorValue);
        if (regVectorValue == NULL || (*regVectorValue).m_to_python == NULL) {
            python::class_<std::vector<size_t >>("vector").def(
                    python::vector_indexing_suite<std::vector<size_t >>());
            python::class_<std::vector<std::vector<size_t >>>("vector_vector").def(
                    python::vector_indexing_suite<std::vector<std::vector<size_t >>>());
        }

        numpy::initialize();
        def("generate_labels", generate_labels);
        def("build_agglomerative_dendogram", build_agglomerative_dendogram);
        def("build_agglomerative_dendogram_sorted", build_agglomerative_dendogram_sorted);
        def("build_labels", build_labels);
        def("build_labels_cluster_color", build_labels_cluster_color);
        def("dendogram_mapping", dendogram_mapping);
    }
}