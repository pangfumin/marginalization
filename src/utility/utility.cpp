#include "utility.h"



Eigen::Matrix3d rotX(double angle){
    Eigen::Matrix3d R;
    R<<1, 0, 0,
            0, cos(angle), -sin(angle),
            0, sin(angle), cos(angle);
    return R;
}

Eigen::Matrix3d rotY(double angle){
    Eigen::Matrix3d R;
    R<<cos(angle), 0,sin(angle),
            0, 1, 0,
            -sin(angle),0,cos(angle) ;
    return R;
}


Eigen::Matrix3d rotZ(double angle){
    Eigen::Matrix3d R;
    R<<cos(angle),-sin(angle),0,
            sin(angle),cos(angle),0,
            0,0,1;
    return R;
}

Eigen::Vector3d Utility::unskew3d(const Eigen::Matrix3d & Omega) {
    return 0.5 * Eigen::Vector3d(Omega(2,1) - Omega(1,2), Omega(0,2) - Omega(2,0), Omega(1,0) - Omega(0,1));
}

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0}; // gravity in ground
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}

Eigen::Matrix4d Utility::quatPlus(Eigen::Quaterniond& quat){
    Eigen::Matrix4d L;

    double x = quat.x();
    double y = quat.y();
    double z = quat.z();
    double w = quat.w();
    L<< w, -z, y, x,
        z, w, -x, y,
        -y, x, w, z,
        -x, -y, -z, w;
    return L;

}
Eigen::Matrix4d Utility::quatOplus(Eigen::Quaterniond& quat){
    Eigen::Matrix4d R;

    double x = quat.x();
    double y = quat.y();
    double z = quat.z();
    double w = quat.w();
    R << w, z, -y, x,
        -z, w, x, y,
        y, -x, w, z,
        -x, -y, -z, w;

    return R;
}
