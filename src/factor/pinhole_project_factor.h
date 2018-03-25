
#ifndef DEPTH_PROJECT_FACTOR_H
#define DEPTH_PROJECT_FACTOR_H
#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "../utility/utility.h"

/**
 *
 */
class PinholeProjectFactor:public ceres::SizedCostFunction<2, /* num of residual */
        7, /* parameter of pose */
        7, /* parameter of pose */
        7,
        1>{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /// \brief The base in ceres we derive from
    typedef ::ceres::SizedCostFunction<2, 7, 7> base_t;

    /// \brief The number of residuals
    static const int kNumResiduals = 2;

    /// \brief The type of the covariance.
    typedef Eigen::Matrix<double, 2, 2> covariance_t;

    /// \brief The type of the information (same matrix dimension as covariance).
    typedef covariance_t information_t;

    /// \brief The type of hte overall Jacobian.
    typedef Eigen::Matrix<double, 2, 7> jacobian_t;

    /// \brief The type of the Jacobian w.r.t. poses --
    /// \warning This is w.r.t. minimal tangential space coordinates...
    typedef Eigen::Matrix<double, 2, 6> jacobian0_t;

    PinholeProjectFactor() = delete;
    PinholeProjectFactor(const Eigen::Vector3d& uv_C0,
                       const Eigen::Vector3d& uv_C1);

    /// \brief Trivial destructor.
    virtual ~PinholeProjectFactor() {}

    virtual bool Evaluate(double const *const *parameters, double *residuals,
                          double **jacobians) const;

    bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians,
                                      double **jacobiansMinimal) const;

private:

    Eigen::Vector3d C0uv;
    Eigen::Vector3d C1uv;

    // information matrix and its square root
    mutable information_t information_; ///< The information matrix for this error term.
    mutable information_t squareRootInformation_; ///< The square root information matrix for this error term.
};

#endif


/*
 *
 */