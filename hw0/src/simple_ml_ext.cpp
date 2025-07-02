#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


namespace Matrix{
    template<class T1, class T2>
    void multi(
        const T1* m1, const T2* m2,
        size_t m, size_t r, size_t n,
        const float* ret) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                ret[i*n+j] = 0.;
                for (size_t k = 0; k < r; k++) ret[i*n+j] += m1[i*r+k] * m2[k*n+j];
            }
        }
    }

    template<class T1, class T2>
    void sub(
        const T1* m1, const T2* m2,
        size_t m, size_t n,
        const float* ret) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                ret[i*n+j] = m1[i*n+j] - m2[i*n+j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t b_ = 0; b_ < m; b_ += batch) {
        //std::cout << b_ << "----->" << std::endl;

        const float* X_ = X + n * b_;
        const unsigned char* y_ = y + b_;

        float* h_hat = new float[batch*k]();
        //std::cout << "y_hat" << std::endl;
        for (size_t i_ = 0; i_ < batch; i_++) {
            for (size_t j_ = 0; j_ < k; j_++) {
                for (size_t k_ = 0; k_ < n; k_++) h_hat[i_*k+j_] += X_[i_*n+k_] * theta[k_*k+j_];
                //std::cout << h_hat[i_*k+j_] << " ";
            }
            //std::cout << std::endl;
        }

        float* Z = new float[batch*k]();
        //std::cout << "Z" << std::endl;
        for (size_t i_ = 0; i_ < batch; i_++) {
            float sum = 0;
            for (size_t j_ = 0; j_ < k; j_++) {
                Z[i_*k+j_] = std::exp(h_hat[i_*k+j_]);
                sum += Z[i_*k+j_];
            }
            for (size_t j_ = 0; j_ < k; j_++) {
                Z[i_*k+j_] /= sum;
                //std::cout << Z[i_*k+j_] << " ";
            }
            //std::cout << std::endl;
        }

        unsigned char* Iy = new unsigned char[batch*k]();
        for (size_t i_ = 0; i_ < batch; i_++) Iy[i_*k+y_[i_]] = 1;

        float* Z_Iy = new float[batch*k]();
        //std::cout << "Z_Iy" << std::endl;
        for (size_t i_ = 0; i_ < batch; i_++) {
            for (size_t j_ = 0; j_ < k; j_++) {
                Z_Iy[i_*k+j_] = Z[i_*k+j_] - Iy[i_*k+j_];
                //std::cout << Z_Iy[i_*k+j_] << " ";
            }
            //std::cout << std::endl;
        }

        float* grad = new float[n*k]();
        //std::cout << "grad" << std::endl;
        for (size_t i_ = 0; i_ < n; i_++) {
            for (size_t j_ = 0; j_ < k; j_++) {
                for (size_t k_ = 0; k_ < batch; k_++) grad[i_*k+j_] += X_[k_*n+i_] * Z_Iy[k_*k+j_];
                grad[i_*k+j_] /= batch;
                //std::cout << grad[i_*k+j_] << " ";
            }
            //std::cout << std::endl;
        }

        //std::cout << "theta" << std::endl;
        for (size_t i_ = 0; i_ < n; i_++) {
            for (size_t j_ = 0; j_ < k; j_++) {
                theta[i_*k+j_] -= lr * grad[i_*k+j_];
                //std::cout << theta[i_*k+j_] << " ";
            }
            //std::cout << std::endl;
        }

        delete[] h_hat;
        delete[] Iy;
        delete[] Z;
        delete[] Z_Iy;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
