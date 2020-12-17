#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <limits>
#include <map>
#include <set>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "util.hpp"
#include "munkres.cpp"

namespace brainclustering {

    class Table {
    private:
        friend class boost::serialization::access;

        using value_type = double;
        using duipair = std::pair<value_type, size_t>;
        size_t n_cluster_main_;
        size_t n_cluster_secondary_;
        double background_value_;
        std::vector <std::map<value_type, duipair>> vec_map_;

        template<class Archive>
        void serialize(Archive &ar, const unsigned int /* file_version */) {
            ar & n_cluster_main_ & n_cluster_secondary_ & background_value_ & vec_map_;
        }

    public:
        Table(size_t const n_main, size_t const n_secondary, double background = 0.0) :
                n_cluster_main_(n_main),
                n_cluster_secondary_(n_secondary),
                background_value_(background) {
            vec_map_.resize(n_cluster_main_);
        };

        void add_entry(ndarray const &data_vector_seq1, std::vector <std::vector<size_t>> const &dendogram_seq1,
                       ndarray const &data_vector_seq2, std::vector <std::vector<size_t>> const &dendogram_seq2) {

            // Data from the first sequencing
            auto *data_seq1 = get_c_array<value_type>(data_vector_seq1);
            // Data from the second sequencing
            auto *data_seq2 = get_c_array<value_type>(data_vector_seq2);

            // Sanity check
            if (data_vector_seq1.shape(0) != data_vector_seq2.shape(0)) {
                std::cerr << "The two sequencing files should have the same number of elements. ";
                std::cerr << "Please convert them such that they do. " << std::endl;
                std::abort();
            }
            // Sanity check
            if (dendogram_seq1.size() != (n_cluster_main_ + n_cluster_secondary_) ||
                dendogram_seq2.size() != (n_cluster_main_ + n_cluster_secondary_)) {
                std::cerr
                        << "The size of the dendograms has to be equal to the number of main clusters + number of sub clusters ";
                std::abort();
            }

            size_t n_points = data_vector_seq1.shape(0);

            std::vector <std::vector<size_t>> cluster_indices_seq1(n_cluster_main_);
            std::vector <std::vector<size_t>> cluster_indices_seq2(n_cluster_main_);
            std::vector <size_t> cluster_val_small_seq1(n_points);
            std::vector <size_t> cluster_val_small_seq2(n_points);
            std::vector <std::vector<int>> cost_matrix(n_cluster_main_, std::vector<int>(n_cluster_main_, 0));
            std::vector <std::pair<int, int>> cost_pairs(n_points);

            // TODO: This won't work for more than 2 levels, but do we really need more than 2 levels?
            // Go through the main clusters, which always start at index n_cluster_secondary_
            for (size_t i = n_cluster_secondary_; i < n_cluster_secondary_ + n_cluster_main_; ++i) {
                // Sanity check
                if (dendogram_seq1[i][0] != 0 || dendogram_seq2[i][0] != 0) {
                    std::cout << "The mapping should be considering the first level of the dendogram, but it is not."
                              << std::endl;
                }
                // Go through the nodes of the bigger clusters to reach the leaves
                for (size_t j = 1; j < dendogram_seq1[i].size(); ++j) {
                    // Node that contains the secondary level of the cluster
                    size_t current_smaller_cluster = dendogram_seq1[i][j];
                    // That node can be found exactly at row current_smaller_cluster of the dendogram
                    for (size_t k = 1; k < dendogram_seq1[current_smaller_cluster].size(); ++k) {
                        // Go through the elements inside this cluster
                        size_t current_index = dendogram_seq1[current_smaller_cluster][k];
                        // We give this cluster a label,
                        // which is the current iteration starting from n_cluster_secondary_
                        cost_pairs[current_index].first = i - n_cluster_secondary_;
                        // Add this element to the list of indices for this cluster
                        cluster_indices_seq1[i - n_cluster_secondary_].push_back(current_index);
                        // We also save the index of the position in which the smaller cluster is (0 - n_cluster_secondary)
                        cluster_val_small_seq1[current_index] = j - 1;
                    }
                }

                // The same happens here, but for the second image
                for (size_t j = 1; j < dendogram_seq2[i].size(); ++j) {
                    size_t current_smaller_cluster = dendogram_seq2[i][j];
                    for (size_t k = 1; k < dendogram_seq2[current_smaller_cluster].size(); ++k) {
                        size_t current_index = dendogram_seq2[current_smaller_cluster][k];
                        cost_pairs[current_index].second = i - n_cluster_secondary_;
                        cluster_indices_seq2[i - n_cluster_secondary_].push_back(current_index);
                        cluster_val_small_seq2[current_index] = j - 1;
                    }
                }
            }
            for (auto &cost_pair : cost_pairs) {
                // Build contingengy matrix
                cost_matrix[cost_pair.first][cost_pair.second]++;
            }

            // Run Hungarian algorithm
            auto mapping_tuples = Hungarian::Solve(cost_matrix);
            if (mapping_tuples.size() != n_cluster_main_) {
                std::cerr << "The length of the tuples is not correct." << std::endl;
                std::abort();
            }

            for (size_t t = 0; t < mapping_tuples.size(); ++t) {
                size_t cluster_seq1 = t;
                size_t cluster_seq2 = mapping_tuples[t];
                //std::cout << "The tuple is (" << cluster_seq1 << ", " << cluster_seq2 << ")" << std::endl;
                size_t size_cluster = std::max(dendogram_seq1[n_cluster_secondary_ + cluster_seq1].size(),
                                               dendogram_seq2[n_cluster_secondary_ + cluster_seq2].size()) - 1;

                std::vector <std::vector<int>> smaller_cost_matrix(size_cluster,
                                                                   std::vector<int>(size_cluster, 0));

                std::sort(cluster_indices_seq1[cluster_seq1].begin(),
                          cluster_indices_seq1[cluster_seq1].end());
                std::sort(cluster_indices_seq2[cluster_seq2].begin(),
                          cluster_indices_seq2[cluster_seq2].end());
                std::vector<int> intersection(std::max(cluster_indices_seq1[cluster_seq1].size(),
                                                       cluster_indices_seq2[cluster_seq2].size()));

                // Find the intersection between the values of one vector and the values of the other vector
                // so we know which elements are in the same clusters, and we can use those for munkres
                // Using the intersection values, find in which cluster they are stored
                auto it = std::set_intersection(cluster_indices_seq1[cluster_seq1].begin(),
                                                cluster_indices_seq1[cluster_seq1].end(),
                                                cluster_indices_seq2[cluster_seq2].begin(),
                                                cluster_indices_seq2[cluster_seq2].end(),
                                                intersection.begin());
                intersection.resize(it - intersection.begin());

                // We check for the points that are in both of the large clusters
                for (auto id : intersection) {
                    //std::cout << id << std::endl;
                    // We saved the values of the index in which the smaller cluster is saved
                    size_t cluster_id1 = cluster_val_small_seq1[id];
                    size_t cluster_id2 = cluster_val_small_seq2[id];
                    //std::cout << cluster_id1 << ", " << cluster_id2 << std::endl;
                    // Build contingengy matrix
                    smaller_cost_matrix[cluster_id1][cluster_id2]++;
                }

                auto smaller_mapping_tuples = Hungarian::Solve(smaller_cost_matrix);

                // Iterate through all points that correspond to each small particular fraction and
                // compute the average of these points
                for (size_t tt = 0; tt < smaller_mapping_tuples.size(); ++tt) {
                    // Since we considered the maximum of the two sizes, we need to take into account the fact that
                    // some of the mapping might not be valid.
                    // It might be that cluster 7 (in seq1) is mapped to cluster 5 (in seq2), which doesn't exist.
                    // So we need to ignore these mappings
                    if (tt >= dendogram_seq1[n_cluster_secondary_ + cluster_seq1].size() - 1 ||
                        smaller_mapping_tuples[tt] >= dendogram_seq2[n_cluster_secondary_ + cluster_seq2].size() - 1) {
                        continue;
                    }

                    size_t smaller_cluster_seq1 =
                            dendogram_seq1[n_cluster_secondary_ + cluster_seq1][tt + 1];
                    size_t smaller_cluster_seq2 =
                            dendogram_seq2[n_cluster_secondary_ + cluster_seq2][smaller_mapping_tuples[tt] + 1];

                    // Sanity check
                    if (dendogram_seq1[smaller_cluster_seq1][0] != 1 || dendogram_seq2[smaller_cluster_seq2][0] != 1) {
                        std::cout
                                << "The mapping should be considering the second level of the dendogram, but it is not."
                                << std::endl;
                    }

                    if (dendogram_seq1[smaller_cluster_seq1].size() == 1 ||
                        dendogram_seq2[smaller_cluster_seq2].size() == 1) {
                        continue;
                    }
                    double x = 0.0;
                    size_t n_x = 0;
                    for (size_t j = 1; j < dendogram_seq1[smaller_cluster_seq1].size(); ++j) {
                        // Sum up the values only if they are not zero
                        if (data_seq1[dendogram_seq1[smaller_cluster_seq1][j]] > background_value_) {
                            n_x++;
                            x += data_seq1[dendogram_seq1[smaller_cluster_seq1][j]];
                        }
                    }
                    // If n_x is zero and would produce a /0, then set it to zero (it will not be considered)
                    x = (n_x > 0) ? x / (n_x * 1.0) : 0.0;
                    double y = 0.0;
                    size_t n_y = 0;
                    for (size_t j = 1; j < dendogram_seq2[smaller_cluster_seq2].size(); ++j) {
                        if (data_seq2[dendogram_seq2[smaller_cluster_seq2][j]] > background_value_) {
                            n_y++;
                            y += data_seq2[dendogram_seq2[smaller_cluster_seq2][j]];
                        }
                    }
                    // If n_y is zero and would produce a /0, then set it to zero (it will not be considered)
                    y = (n_y > 0) ? y / (n_y * 1.0) : 0.0;

                    auto found = vec_map_[t].find(x);
                    if (found == vec_map_[t].end()) {
                        vec_map_[t].insert({x, duipair(y, 1)});
                    } else {
                        found->second.second++;
                        found->second.first += std::abs(y - found->second.first) / found->second.second;
                    }
                }
            }

            /*for (size_t k = 0; k < vec_map_.size(); ++k) {
                std::cout << "Looking at the " << k << "th map" << std::endl;
                for (auto p : vec_map_[k]) {
                    std::cout << "x: " << p.first << ", y: " << p.second.y_ << ", " <<
                              p.second.count_ << " times." << std::endl;
                }
            }*/
        }

        void return_sequencing(ndarray const &data_vector_seq1, ndarray const &labels_vector_seq1,
                               ndarray const &new_data_vector_seq2) {
            // Data of source sequencing
            auto *data_seq1 = get_c_array<value_type>(data_vector_seq1);
            // Labels of source sequencing
            auto *labels_seq1 = get_c_array<value_type>(labels_vector_seq1);
            // Empty array for new sequencing
            auto *new_data_seq2 = get_c_array<value_type>(new_data_vector_seq2);


            /*for (size_t cluster = 0; cluster < n_cluster_main_; ++cluster) {
                std::cout << "Looking up the " << cluster + 1 << "th map" << std::endl;
                for (auto p : vec_map_[cluster]) {
                    std::cout << "x: " << p.first << ", y: " << p.second.first << ", " <<
                              p.second.second << " times." << std::endl;
                }
            }*/
            std::vector <std::vector<int>> cost_matrix(n_cluster_main_, std::vector<int>(n_cluster_main_, 0));

            for (int i = 0; i < labels_vector_seq1.shape(0); ++i) {
                // Skip background points
                if (data_seq1[i] == background_value_) {
                    continue;
                }
                size_t mapped_cluster = labels_seq1[i];

                // Try to find the point
                auto found = vec_map_[mapped_cluster].find(data_seq1[i]);
                // If the point does not exist
                if (found == vec_map_[mapped_cluster].end()) {
                    // Find the upper bound
                    auto upper = vec_map_[mapped_cluster].upper_bound(data_seq1[i]);
                    // Set the lower bound to the upperbound
                    auto lower = upper;
                    // If we are at the beginning of the table
                    if (upper == vec_map_[mapped_cluster].begin()) {
                        new_data_seq2[i] = (upper->second.first * data_seq1[i]) / upper->first;
                        // If we are at the end of the table
                    } else if (upper == vec_map_[mapped_cluster].end()) {
                        // Decrease upper
                        lower = --upper;
                        new_data_seq2[i] = (upper->second.first * data_seq1[i]) / upper->first;
                    } else {
                        // Otherwise do interpolation
                        --lower;
                        new_data_seq2[i] =
                                (lower->second.first * (upper->first - data_seq1[i]) +
                                 upper->second.first * (data_seq1[i] - lower->first)) /
                                (upper->first - lower->first);
                    }
                    /*std::cout << "Point " << data_seq1[i] << ", cluster " << labels_seq1[i] <<
                              "; Lowerbound: " << lower->first <<
                              ", upperbound " << upper->first << std::endl;
                    std::cout << "New point " << new_data_seq2[i] << "; " <<
                              "Lowerbound: " << lower->second.first <<
                              ", upperbound " << upper->second.first << "" << std::endl;*/
                } else {
                    // If found, give the same value
                    new_data_seq2[i] = found->second.first;
                }
            }
        }

    };

    void save_mapping(const Table &t, const char *filename) {
        // Make an archive
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << t;
    }

    void restore_mapping(Table &t, const char *filename) {
        // Open the archive
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);

        // Restore the table from the archive
        ia >> t;
    }


    BOOST_PYTHON_MODULE (mapping) {
            numpy::initialize();

            python::class_<Table, boost::noncopyable>("Table", python::init<size_t, size_t, double>())
            .def("add_entry", &Table::add_entry)
            .def("return_sequencing", &Table::return_sequencing);
            python::def("save_mapping", save_mapping);
            python::def("restore_mapping", restore_mapping);

    }
}
