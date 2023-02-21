#include "pose_local_parameterization.h"
#include "utility/utility.h"

//Instances of LocalParameterization implement the ⊞ operation.
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

//And its derivative with respect to Δx at Δx=0.
// This is correct! This jacobian should be 4*3 matrix and be the transpose matrix of lift jacobian matrix(3*4)
// In projectotion_factor we can deduce the lift jacobian is [I3 0].
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
    j.setZero();
    j.block<3,3>(0,0).setIdentity();
    j.block<4,3>(3,3) = 0.5 * Utility::quatPlus(_q).leftCols(3);

    return true;
}

// bool PoseLocalParameterization::plusJacobian(const double *x, double *jacobian) const
// {
//     Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
//     Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
//     j.setZero();
//     j.block<3,3>(0,0).setIdentity();
//     j.block<4,3>(3,3) = 0.5 * Utility::quatPlus(_q).leftCols(3);

//     return true;
// }



