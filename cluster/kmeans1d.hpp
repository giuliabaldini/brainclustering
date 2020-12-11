#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

/********************************************************************
 ********************************************************************
 ** https://github.com/dstein64/kmeans1d
 ** MIT License
 **
 ** Copyright (c) 2019 Daniel Steinberg
 **
 ** Permission is hereby granted, free of charge, to any person obtaining a copy
 ** of this software and associated documentation files (the "Software"), to deal
 ** in the Software without restriction, including without limitation the rights
 ** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 ** copies of the Software, and to permit persons to whom the Software is
 ** furnished to do so, subject to the following conditions:
 **
 ** The above copyright notice and this permission notice shall be included in all
 ** copies or substantial portions of the Software.
 **
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 ** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 ** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 ** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 ** SOFTWARE.
 **
 ********************************************************************
 ********************************************************************/

typedef unsigned long ulong;

/*
 *  Internal implementation of the SMAWK algorithm.
 */
template<typename T>
void _smawk(
        const std::vector<ulong> &rows,
        const std::vector<ulong> &cols,
        const std::function<T(ulong, ulong)> &lookup,
        std::vector<ulong> *result) {
    // Recursion base case
    if (rows.size() == 0) return;

    // ********************************
    // * REDUCE
    // ********************************

    std::vector<ulong> _cols;  // Stack of surviving columns
    for (ulong col : cols) {
        while (true) {
            if (_cols.size() == 0) break;
            ulong row = rows[_cols.size() - 1];
            if (lookup(row, col) >= lookup(row, _cols.back()))
                break;
            _cols.pop_back();
        }
        if (_cols.size() < rows.size())
            _cols.push_back(col);
    }

    // Call recursively on odd-indexed rows
    std::vector<ulong> odd_rows;
    for (ulong i = 1; i < rows.size(); i += 2) {
        odd_rows.push_back(rows[i]);
    }
    _smawk(odd_rows, _cols, lookup, result);

    std::unordered_map<ulong, ulong> col_idx_lookup;
    for (ulong idx = 0; idx < _cols.size(); ++idx) {
        col_idx_lookup[_cols[idx]] = idx;
    }

    // ********************************
    // * INTERPOLATE
    // ********************************

    // Fill-in even-indexed rows
    ulong start = 0;
    for (ulong r = 0; r < rows.size(); r += 2) {
        ulong row = rows[r];
        ulong stop = _cols.size() - 1;
        if (r < rows.size() - 1)
            stop = col_idx_lookup[(*result)[rows[r + 1]]];
        ulong argmin = _cols[start];
        T min = lookup(row, argmin);
        for (ulong c = start + 1; c <= stop; ++c) {
            T value = lookup(row, _cols[c]);
            if (c == start || value < min) {
                argmin = _cols[c];
                min = value;
            }
        }
        (*result)[row] = argmin;
        start = stop;
    }
}

/*
 *  Interface for the SMAWK algorithm, for finding the minimum value in each row
 *  of an implicitly-defined totally monotone matrix.
 */
template<typename T>
std::vector<ulong> smawk(
        const ulong num_rows,
        const ulong num_cols,
        const std::function<T(ulong, ulong)> &lookup) {
    std::vector<ulong> result;
    result.resize(num_rows);
    std::vector<ulong> rows(num_rows);
    iota(begin(rows), end(rows), 0);
    std::vector<ulong> cols(num_cols);
    iota(begin(cols), end(cols), 0);
    _smawk<T>(rows, cols, lookup, &result);
    return result;
}

/*
 *  Calculates cluster costs in O(1) using prefix sum arrays.
 */
class CostCalculator {
    std::vector<double> cumsum;
    std::vector<double> cumsum2;

public:
    CostCalculator(const std::vector<double> &vec, ulong n) {
        cumsum.push_back(0.0);
        cumsum2.push_back(0.0);
        for (ulong i = 0; i < n; ++i) {
            double x = vec[i];
            cumsum.push_back(x + cumsum[i]);
            cumsum2.push_back(x * x + cumsum2[i]);
        }
    }

    double calc(ulong i, ulong j) {
        if (j < i) return 0.0;
        double mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1);
        double result = cumsum2[j + 1] - cumsum2[i];
        result += (j - i + 1) * (mu * mu);
        result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);
        return result;
    }
};

template<typename T>
class Matrix {
    std::vector<T> data;
public:
    ulong num_rows;
    ulong num_cols;

public:
    Matrix() : num_rows(), num_cols() {} // zero-initialize

    Matrix(ulong num_rows, ulong num_cols) : num_rows(num_rows), num_cols(num_cols) {
        data.resize(num_rows * num_cols);
    }

    inline void resize(ulong n_rows, ulong n_cols) {
        this->num_rows = n_rows;
        this->num_cols = n_cols;
        data.resize(n_rows * n_cols);
    }

    inline T get(ulong i, ulong j) {
        return data[i * num_cols + j];
    }

    inline void set(ulong i, ulong j, T value) {
        data[i * num_cols + j] = value;
    }
};

extern "C++" {
// "__declspec(dllexport)" causes the function to be exported when compiling with
// Visual Studio on Windows. Otherwise, the function is not exported and the code
// raises "AttributeError: function 'cluster' not found".
#if defined (_MSC_VER)
__declspec(dllexport)
#endif

void fill_matrices(
        std::vector<double> &sorted_array,
        Matrix<double> &D,
        Matrix<ulong> &T) {
    // ***************************************************
    // * Set D and T using dynamic programming algorithm
    // ***************************************************

    // Algorithm as presented in section 2.2 of (Gronlund et al., 2017).
    ulong k = D.num_rows;
    ulong n = D.num_cols;
    CostCalculator cost_calculator(sorted_array, n);

    for (ulong i = 0; i < n; ++i) {
        D.set(0, i, cost_calculator.calc(0, i));
        T.set(0, i, 0);
    }

    for (ulong k_ = 1; k_ < k; ++k_) {
        // C_i matrix where i = k_
        auto C = [&D, &k_, &cost_calculator](ulong i, ulong j) -> double {
            // Choose the minimum min(j - 1, m)
            ulong col = i < j - 1 ? i : j - 1;
            // C_i[m][j] = D[i-1][min(j - 1, m)] = CC(j, m)
            return D.get(k_ - 1, col) + cost_calculator.calc(j, i);
        };
        std::vector<ulong> row_argmins = smawk<double>(n, n, C);
        for (ulong i = 0; i < row_argmins.size(); ++i) {
            ulong argmin = row_argmins[i];
            double min = C(i, argmin);
            D.set(k_, i, min);
            T.set(k_, i, argmin);
        }
    }
}

template<bool ret_labels = true, bool sort_array = true>
void return_labels(
        Matrix<ulong> &T,
        const double *sorted_array,
        double *centroids,
        ulong k,
        ulong *undo_sort_lookup = nullptr,
        double *clusters = nullptr) {
    ulong n = T.num_cols;
    std::vector<double> sorted_clusters;
    if constexpr (sort_array){
        sorted_clusters.resize(n);
    }
    ulong t = n;
    ulong k_ = k - 1;
    ulong n_ = n - 1;
    // The do/while loop was used in place of:
    //   for (k_ = k - 1; k_ >= 0; --k_)
    // to avoid wraparound of an unsigned type.
    do {
        ulong t_ = t;
        t = T.get(k_, n_);
        double centroid = 0.0;
        for (ulong i = t; i < t_; ++i) {
            if constexpr (ret_labels) {
                if constexpr (sort_array){
                    sorted_clusters[i] = k_;
                } else {
                    clusters[i] = k_;
                }
            }
            centroid += (sorted_array[i] - centroid) / (i - t + 1);
        }
        centroids[k_] = centroid;
        k_ -= 1;
        n_ = t - 1;
    } while (t > 0);

    // ***************************************************
    // * Order cluster assignments to match de-sorted
    // * ordering
    // ***************************************************
    if constexpr (ret_labels && sort_array) {
        for (ulong i = 0; i < n; ++i) {
            clusters[i] = sorted_clusters[undo_sort_lookup[i]];
        }
    }
}

template<bool ret_labels = true, bool sort_array = true>
void cluster_1d(
        double *array,
        ulong n,
        double *centroids,
        ulong k,
        double *clusters = nullptr) {
    // ***************************************************
    // * Sort input array and save info for de-sorting
    // ***************************************************
    std::vector<double> sorted_array(n);
    std::vector<ulong> undo_sort_lookup;
    if constexpr (sort_array) {
        std::vector<ulong> sort_idxs(n);
        iota(sort_idxs.begin(), sort_idxs.end(), 0);
        std::sort(sort_idxs.begin(), sort_idxs.end(),
                  [&array](ulong a, ulong b) { return array[a] < array[b]; });
        undo_sort_lookup.resize(n);
        for (ulong i = 0; i < n; ++i) {
            sorted_array[i] = array[sort_idxs[i]];
            undo_sort_lookup[sort_idxs[i]] = i;
        }
    } else {
        for (ulong i = 0; i < n; ++i) {
            sorted_array[i] = array[i];
        }
    }


    // ***************************************************
    // * Set D and T using dynamic programming algorithm
    // ***************************************************

    // Algorithm as presented in section 2.2 of (Gronlund et al., 2017).
    Matrix<double> D(k, n);
    Matrix<ulong> T(k, n);

    fill_matrices(sorted_array, D, T);

    // ***************************************************
    // * Extract cluster assignments by backtracking
    // ***************************************************

    // TODO: This step requires O(kn) memory usage due to saving the entire
    //       T matrix. However, it can be modified so that the memory usage is O(n).
    //       D and T would not need to be retained in full (D already doesn't need
    //       to be fully retained, although it currently is).
    //       Details are in section 3 of (Grønlund et al., 2017).

    return_labels<ret_labels, sort_array>(T, sorted_array.data(), centroids, k, undo_sort_lookup.data(), clusters);
}
} // extern "C"