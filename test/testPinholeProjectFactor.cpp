#include "../src/factor/pinhole_project_factor.h"
#include <iostream>
#include "../src/utility/num-diff.hpp"
#include "../src/factor/pose_local_parameterization.h"
void T2double(Eigen::Isometry3d& T,double* ptr){

    Eigen::Vector3d trans = T.matrix().topRightCorner(3,1);
    Eigen::Matrix3d R = T.matrix().topLeftCorner(3,3);
    Eigen::Quaterniond q(R);

    ptr[0] = trans(0);
    ptr[1] = trans(1);
    ptr[2] = trans(2);
    ptr[3] = q.x();
    ptr[4] = q.y();
    ptr[5] = q.z();
    ptr[6] = q.w();
}

void applyNoise(const Eigen::Isometry3d& Tin,Eigen::Isometry3d& Tout){


    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.5*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;

    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3) = Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}

int main(){

    // simulate

    Eigen::Isometry3d T_WI0, T_WI1, T_IC;
    Eigen::Vector3d C0p(4, 3, 10);
    T_WI0 = T_WI1 = T_IC = Eigen::Isometry3d::Identity();

    T_WI1.matrix().topRightCorner(3,1) = Eigen::Vector3d(1,0,0);
    T_IC.matrix().topRightCorner(3,1) = Eigen::Vector3d(0.1,0.10,0);

    Eigen::Isometry3d T_WC0, T_WC1;
    T_WC0 = T_WI0 * T_IC;
    T_WC1 = T_WI1 * T_IC;

    Eigen::Isometry3d T_C0C1 = T_WC0.inverse()*T_WC1;
    Eigen::Isometry3d T_C1C0 = T_C0C1.inverse();

    Eigen::Vector3d C1p = T_C1C0.matrix().topLeftCorner(3,3)*C0p+T_C1C0.matrix().topRightCorner(3,1);

    Eigen::Vector3d p0(C0p(0)/C0p(2), C0p(1)/C0p(2), 1);
    double z = C0p(2);
    double rho = 1.0/z;

    Eigen::Vector3d p1(C1p(0)/C1p(2), C1p(1)/C1p(2), 1);



    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;

    PinholeProjectFactor* pinholeProjectFactor = new PinholeProjectFactor(p0,p1);

    double* param_T_WI0 = new double[7];
    double* param_T_WI1 = new double[7];
    double* param_T_IC = new double[7];
    double* param_rho;


    T2double(T_WI0,param_T_WI0);
    T2double(T_WI1,param_T_WI1);
    T2double(T_IC,param_T_IC);
    param_rho = &rho;



    double* paramters[4] = {param_T_WI0, param_T_WI1, param_T_IC, param_rho};

    Eigen::Matrix<double, 2,1> residual;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
    Eigen::Matrix<double,2,1> jacobian3_min;
    double* jacobians_min[4] = {jacobian0_min.data(), jacobian1_min.data(),
                                jacobian2_min.data(), jacobian3_min.data()};


    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian1;
    Eigen::Matrix<double,2,7,Eigen::RowMajor> jacobian2;
    Eigen::Matrix<double,2,1> jacobian3;
    double* jacobians[4] = {jacobian0.data(), jacobian1.data(), jacobian2.data(), jacobian3.data()};

    pinholeProjectFactor->EvaluateWithMinimalJacobians(paramters,residual.data(),jacobians,jacobians_min);

    std::cout<<"residual: "<<residual.transpose()<<std::endl;
    CHECK_EQ(residual.norm()< 0.001,true)<<"Residual is Not zero, zero check not passed!";
//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */


    std::cout<<"------------  Jacobian Check -----------------"<<std::endl;

    Eigen::Isometry3d T_WI0_noised,  T_WI1_noised, T_IC_noised;
    double rho_noised;

    applyNoise(T_WI0,T_WI0_noised);
    applyNoise(T_WI1,T_WI1_noised);
    applyNoise(T_IC,T_IC_noised);
    rho_noised = rho + 0.02;

    double* param_T_WI0_noised = new double[7];
    double* param_T_WI1_noised = new double[7];
    double* param_T_IC_noised = new double[7];
    double* param_rho_noised;



    T2double(T_WI0_noised,param_T_WI0_noised);
    T2double(T_WI1_noised,param_T_WI1_noised);
    T2double(T_IC_noised,param_T_IC_noised);

    param_rho_noised = &rho_noised;

    double* parameters_noised[4] = {param_T_WI0_noised ,param_T_WI1_noised,
                                    param_T_IC_noised, param_rho_noised};

    pinholeProjectFactor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(),jacobians,jacobians_min);


    std::cout<<"residual: "<<residual.transpose()<<std::endl;

    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian0_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian1_min;
    Eigen::Matrix<double,2,6,Eigen::RowMajor> num_jacobian2_min;
    Eigen::Matrix<double,2,1> num_jacobian3_min;

    NumDiff<PinholeProjectFactor,4> localizer_num_differ(pinholeProjectFactor);

    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameterization>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;

    std::cout<<"Check jacobian0: "<<std::endl;
    localizer_num_differ.isJacobianEqual<2,6>(jacobian0_min.data(),num_jacobian0_min.data(),1e-2);

    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameterization>(parameters_noised,1,num_jacobian1_min.data());

    std::cout<<"jacobian1_min: "<<std::endl<<jacobian1_min<<std::endl;
    std::cout<<"num_jacobian1_min: "<<std::endl<<num_jacobian1_min<<std::endl;

    std::cout<<"Check jacobian1: "<<std::endl;
    localizer_num_differ.isJacobianEqual<2,6>(jacobian1_min.data(),num_jacobian1_min.data(),1e-2);

    localizer_num_differ.df_r_xi<2,7,6,PoseLocalParameterization>(parameters_noised,2,num_jacobian2_min.data());

    std::cout<<"jacobian2_min: "<<std::endl<<jacobian2_min<<std::endl;
    std::cout<<"num_jacobian2_min: "<<std::endl<<num_jacobian2_min<<std::endl;

    std::cout<<"Check jacobian2: "<<std::endl;
    localizer_num_differ.isJacobianEqual<2,6>(jacobian2_min.data(),num_jacobian2_min.data(),1e-2);


    localizer_num_differ.df_r_xi<2,1>(parameters_noised,3,num_jacobian3_min.data());
    std::cout<<"jacobian3_min: "<<std::endl<<jacobian3_min<<std::endl;
    std::cout<<"num_jacobian3_min: "<<std::endl<<num_jacobian3_min<<std::endl;

    return 0;
}
