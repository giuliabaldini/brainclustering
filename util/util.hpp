#pragma once

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/python/list.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace brainclustering {

    namespace python = boost::python;
    namespace numpy = python::numpy;
    using ndarray = numpy::ndarray;
    using object = python::object;
    using list = python::list;
    static std::random_device dev;

    template<typename T>
    inline
    std::vector<T> to_std_vector(const python::object &iterable) {
        return std::vector<T>(python::stl_input_iterator<T>(iterable),
                              python::stl_input_iterator<T>());
    }

    template<typename Container>
    python::list to_python_list(const Container &vec) {
        typedef typename Container::value_type T;
        python::list lst;
        std::for_each(vec.begin(), vec.end(), [&](const T &t) { lst.append(t); });
        return lst;
    }

    template<typename value_type>
    inline void check_dimensions(ndarray const &arr, unsigned int dimensions) {
        size_t check2d = arr.get_nd();
        if ((check2d == 1 && dimensions != 1) || (check2d == 2 && dimensions != arr.shape(1)) || (check2d > 2) ||
            (check2d == 1 and arr.strides(0) != sizeof(value_type)) ||
            (check2d == 2 and arr.strides(1) != sizeof(value_type))) {
            std::cerr << "Wrong array type supplied." << std::endl;
            std::abort();
        }
    }


    template<typename pvalue_type, unsigned int pdimensions>
    struct point_nd {
        using value_type = pvalue_type;
        constexpr static unsigned int dimensions = pdimensions;

        value_type data[dimensions] = {0};

        inline value_type &operator[](size_t const idx) {
            return data[idx];
        }

        inline value_type const &operator[](size_t const idx) const {
            return data[idx];
        }

        inline point_nd &operator=(point_nd const &other) = default;

        inline point_nd &operator+=(point_nd const &other) {
            for (size_t k = 0; k < dimensions; ++k) {
                data[k] += other[k];
            }
            return *this;
        };

        inline point_nd &operator/(value_type const v) {
            for (size_t k = 0; k < dimensions; ++k) {
                data[k] /= v;
            }
            return *this;
        };

        inline bool operator==(value_type const v) {
            for (size_t k = 0; k < dimensions; ++k) {
                if (data[k] == v) {
                    return true;
                }
            }
            return false;
        };

        inline void print() {
            for (int i = 0; i < dimensions; ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }
    };

    template<typename T>
    struct pair {
        T value_;
        size_t idx_;

        pair(size_t id, T value) : value_(value), idx_(id) {};

        pair() = default;

        // The following one have to be default to make the sorting work
        pair(const pair &) = default;

        pair &operator=(const pair &) = default;

        ~pair() = default;
    };

    template<typename point_type>
    inline static void
    print_ndarray(point_type const *const c_arr, size_t const rows, size_t const cols,
                  std::string const &message = "") {
        if (message.length() > 1) {
            std::cout << message << std::endl;
        }
        for (size_t i = 0; i < rows; ++i) {
            std::cout << i << ": ";
            for (size_t j = 0; j < cols; ++j) {
                std::cout << c_arr[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    template<typename point_type>
    inline static void print_ndarray(point_type const *const c_arr, size_t const rows, std::string const &message) {
        print_ndarray(c_arr, rows, point_type::dimensions, message);
    }

    template<typename ContainerType>
    inline static void print_ndcontainer(ContainerType &vec, const std::string &message = "") {
        print_ndarray(&vec[0], vec.size(), message);
    }

    template<typename value_type>
    inline static void
    print_array(value_type *c_arr, size_t const n, const std::string &message = "", bool indices = 0) {
        if (message.length() > 1) {
            std::cout << message << std::endl;
        }
        if (indices) {
            for (size_t i = 0; i < n; ++i) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        for (size_t i = 0; i < n; ++i) {
            std::cout << c_arr[i] << " ";
        }
        std::cout << std::endl;
    }

    template<typename value_type>
    inline static value_type *get_c_array(ndarray const &ndarray) {
        return (value_type *) ndarray.get_data();
    }


    template<typename value_type, unsigned int dimensions>
    inline static auto get_c_ndarray(ndarray const &ndarray) {
        return (point_nd<value_type, dimensions> *) ndarray.get_data();
    }


    template<typename value_type>
    inline static ndarray get_numpy_1darray(value_type *c_arr, size_t const n) {
        return numpy::from_data(
                c_arr,
                numpy::dtype::get_builtin<value_type>(),
                python::make_tuple(n),
                python::make_tuple(sizeof(value_type)),
                python::object()
        );
    }

    template<typename point_type, typename value_type>
    inline static ndarray get_numpy_2darray(point_type const *const c_arr, size_t const n, size_t const dimensions) {
        return numpy::from_data(
                c_arr,
                numpy::dtype::get_builtin<value_type>(),
                python::make_tuple(n, dimensions), // n x d representing the shape
                python::make_tuple(dimensions * sizeof(value_type),
                                   sizeof(value_type)), // d * sizeof(double) x sizeof(double) representing the strides
                python::object()
        );
    }

    template<typename ContainerType>
    inline static ndarray get_ndarray(ContainerType &vec) {
        return get_ndarray(&vec[0], vec.size());
    }


    template<typename ContainerType>
    inline static void print_container(ContainerType &vec, const std::string &message = "") {
        print_array(&vec[0], vec.size(), message);
    }
}