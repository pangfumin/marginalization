#include "../src/factor/pinhole_project_factor.h"
#include "../src/factor/binary_pose_error_factor.hpp"

#include "../src/factor/marginalization_factor.h"
#include <iostream>
#include "../src/utility/NumbDifferentiator.hpp"
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


void double2T(double* ptr, Eigen::Isometry3d& T){
    Eigen::Vector3d trans( ptr[0], ptr[1], ptr[2]) ;
    Eigen::Quaterniond q(ptr[6], ptr[3], ptr[4], ptr[5]);

    T.setIdentity();
    T.matrix().topLeftCorner(3,3) = q.toRotationMatrix();
    T.matrix().topRightCorner(3,1) = trans;
}

void applyNoise(const Eigen::Isometry3d Tin,Eigen::Isometry3d& Tout) {
    Tout.setIdentity();

    Eigen::Vector3d delat_trans = 0.14*Eigen::Matrix<double,3,1>::Random();
    Eigen::Vector3d delat_rot = 0.16*Eigen::Matrix<double,3,1>::Random();

    Eigen::Quaterniond delat_quat(1.0,delat_rot(0),delat_rot(1),delat_rot(2)) ;
    delat_quat.normalize();
    Tout.matrix().topRightCorner(3,1) = Tin.matrix().topRightCorner(3,1) + delat_trans;
    Tout.matrix().topLeftCorner(3,3)
            = Tin.matrix().topLeftCorner(3,3)*delat_quat.toRotationMatrix();
}


void printParameter(double* param, int dim) {
    std::cout<<"Parameter: "<<std::endl;
    for (int i = 0 ; i < dim; i++) {
        std::cout<<i <<" : " << param[i]<<std::endl;
    }
}
int main(){

    // simulate

    Eigen::Isometry3d T_WI0, T_WI1, T_IC;

    T_WI0 = T_WI1 = T_IC = Eigen::Isometry3d::Identity();

    T_WI1.matrix().topRightCorner(3,1) = Eigen::Vector3d(1,0,0);
    T_IC.matrix().topRightCorner(3,1) = Eigen::Vector3d(0.1,0.10,0);

    Eigen::Isometry3d T_WC0, T_WC1;
    T_WC0 = T_WI0 * T_IC;
    T_WC1 = T_WI1 * T_IC;

    Eigen::Isometry3d T_C0C1 = T_WC0.inverse()*T_WC1;
    Eigen::Isometry3d T_C1C0 = T_C0C1.inverse();

    std::vector<Eigen::Vector3d> C0p_vec, C1p_vec, p0_vec, p1_vec;
    std::vector<double> rho_vec;
    for (int i = 0; i < 2; i ++) {
        for (int j=0; j < 2 ; j++) {
            Eigen::Vector3d C0p = Eigen::Vector3d(-3 + i*6.0/10, -4 + j*8.0/10, 8+ i*10.0/10);
            Eigen::Vector3d C1p = T_C1C0.matrix().topLeftCorner(3,3)*C0p+T_C1C0.matrix().topRightCorner(3,1);
            Eigen::Vector3d p0(C0p(0)/C0p(2), C0p(1)/C0p(2), 1);

            double z = C0p(2);
            double rho = 1.0/z;
            Eigen::Vector3d p1(C1p(0)/C1p(2), C1p(1)/C1p(2), 1);

            p0_vec.push_back(p0);
            p1_vec.push_back(p1);
            rho_vec.push_back(rho);
        }
    }



    /*
     * Zero Test
     * Passed!
     */

    std::cout<<"------------ Zero Test -----------------"<<std::endl;
    double* param_T_WI0 = new double[7];
    double* param_T_WI1 = new double[7];
    double* param_T_IC = new double[7];
    std::vector<double*> param_rho;

    T2double(T_WI0,param_T_WI0);
    T2double(T_WI1,param_T_WI1);
    T2double(T_IC,param_T_IC);
    for(int i = 0; i < rho_vec.size(); i++ ) {
        param_rho.push_back(&rho_vec[i]);
    }


    MarginalizationInfo *marginalization_info = new MarginalizationInfo();

    for (int i = 0; i < rho_vec.size(); i++) {

        PinholeProjectFactor *pinholeProjectFactor = new PinholeProjectFactor(p0_vec.at(i), p1_vec.at(i));

        {
            ResidualBlockInfo *residual_block_info =
                    new ResidualBlockInfo(pinholeProjectFactor,
                                          NULL,
                                          std::vector<double *>{param_T_WI0,
                                                                param_T_WI1,
                                                                param_T_IC,
                                                                param_rho.at(i)},
                                          std::vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
    }




    Binary_pose_error_facotr* binary_pose_error_facotr
            = new Binary_pose_error_facotr(T_WC0, T_WC1, 0,0);
    {
        ResidualBlockInfo *residual_block_info =
                new ResidualBlockInfo(binary_pose_error_facotr,
                                      NULL,
                                      std::vector<double *>{param_T_WI0,
                                                            param_T_WI1},
                                      std::vector<int>{0});
        marginalization_info->addResidualBlockInfo(residual_block_info);
    }


    marginalization_info->preMarginalize();
    marginalization_info->marginalize();

    std::unordered_map<long, double *> addr_shift;
    addr_shift[reinterpret_cast<long>(param_T_WI1)] = param_T_WI1;
    addr_shift[reinterpret_cast<long>(param_T_IC)] = param_T_IC;
    std::vector<double*> residual_parameter = marginalization_info->getParameterBlocks(addr_shift);



    MarginalizationFactor* marginalization_factor = new MarginalizationFactor(marginalization_info);

    double** paramters = &residual_parameter[0];   // vector to array


    Eigen::VectorXd residual(marginalization_factor->num_residuals());
    std::cout<<"num_residuals: "<<marginalization_factor->num_residuals()<<std::endl;
    std::cout<<"parameter_block_sizes: "<<marginalization_factor->parameter_block_sizes().size()<<std::endl;
    for (auto i:marginalization_factor->parameter_block_sizes()) {
        std::cout<<"size: "<< i <<std::endl;
    }

    marginalization_factor->EvaluateWithMinimalJacobians(paramters,residual.data(), NULL, NULL);
    std::cout<<"residual: "<< residual.transpose()<<std::endl;



//
    /*
     * Jacobian Check: compare the analytical jacobian to num-diff jacobian
     */
    // disturbance
    Eigen::Isometry3d param_T0, param_T1;
    Eigen::Isometry3d param_T0_noised, param_T1_noised;
    double2T(paramters[0], param_T0);
    double2T(paramters[1], param_T1);

    applyNoise(param_T0, param_T0_noised);
    applyNoise(param_T1, param_T1_noised);

    double* param0_noised = new double[7];
    double* param1_noised = new double[7];

    T2double(param_T0_noised, param0_noised);
    T2double(param_T1_noised, param1_noised);


    Eigen::Matrix<double,12,6,Eigen::RowMajor> jacobian0_min;
    Eigen::Matrix<double,12,6,Eigen::RowMajor> jacobian1_min;

    double* jacobians_min[2] = {jacobian0_min.data(), jacobian1_min.data()};


    Eigen::Matrix<double,12,7,Eigen::RowMajor> jacobian0;
    Eigen::Matrix<double,12,7,Eigen::RowMajor> jacobian1;

    double* jacobians[2] = {jacobian0.data(), jacobian1.data()};

    double* parameters_noised[2] = {param0_noised, param1_noised};
    marginalization_factor->EvaluateWithMinimalJacobians(parameters_noised,residual.data(), jacobians, jacobians_min);
    std::cout<<"residual: "<< residual.transpose()<<std::endl;


    Eigen::Matrix<double,12,6,Eigen::RowMajor> num_jacobian0_min;
    Eigen::Matrix<double,12,6,Eigen::RowMajor> num_jacobian1_min;

    NumbDifferentiator<MarginalizationFactor,2> localizer_num_differ(marginalization_factor);

    localizer_num_differ.df_r_xi<12,7,6,PoseLocalParameterization>(parameters_noised,0,num_jacobian0_min.data());

    std::cout<<"jacobian0_min: "<<std::endl<<jacobian0_min<<std::endl;
    std::cout<<"num_jacobian0_min: "<<std::endl<<num_jacobian0_min<<std::endl;


    localizer_num_differ.df_r_xi<12,7,6,PoseLocalParameterization>(parameters_noised,1,num_jacobian1_min.data());

    std::cout<<"jacobian1_min: "<<std::endl<<jacobian1_min<<std::endl;
    std::cout<<"num_jacobian1_min: "<<std::endl<<num_jacobian1_min<<std::endl;


    return 0;
}