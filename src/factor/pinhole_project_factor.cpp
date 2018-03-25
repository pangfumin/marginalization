#include "pinhole_project_factor.h"
PinholeProjectFactor::PinholeProjectFactor(const Eigen::Vector3d& uv_C0,
                                       const Eigen::Vector3d& uv_C1):
        C0uv(uv_C0),
        C1uv(uv_C1)  {}


bool PinholeProjectFactor::Evaluate(double const *const *parameters,
                                  double *residuals,
                                  double **jacobians) const {
    return EvaluateWithMinimalJacobians(parameters,
                                        residuals,
                                        jacobians, NULL);
}


bool PinholeProjectFactor::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                      double *residuals,
                                                      double **jacobians,
                                                      double **jacobiansMinimal) const {

    // T_WI0
    Eigen::Vector3d t_WI0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Q_WI0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // T_WI1
    Eigen::Vector3d t_WI1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Q_WI1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // T_IC
    Eigen::Vector3d t_IC(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond Q_IC(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // rho
    double inv_dep = parameters[3][0];


    Eigen::Vector3d C0p = C0uv / inv_dep;
    Eigen::Vector3d I0p = Q_IC * C0p + t_IC;
    Eigen::Vector3d Wp = Q_WI0 * I0p + t_WI0;
    Eigen::Vector3d I1p = Q_WI1.inverse() * (Wp - t_WI1);
    Eigen::Vector3d C1p = Q_IC.inverse() * (I1p - t_IC);

    Eigen::Matrix<double, 2, 1> error;

    double inv_z = 1/C1p(2);
    Eigen::Vector2d hat_C1uv(C1p(0)*inv_z, C1p(1)*inv_z);

    Eigen::Matrix<double,2,3> H;
    H << 1, 0, -C1p(0)*inv_z,
        0, 1, -C1p(1)*inv_z;
    H *= inv_z;

    error = hat_C1uv - C1uv.head<2>();
    squareRootInformation_.setIdentity();
    //squareRootInformation_ = weightScalar_* squareRootInformation_; //Weighted

    // weight it
    Eigen::Map<Eigen::Matrix<double, 2, 1> > weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

    Eigen::Matrix3d R_WI0 = Q_WI0.toRotationMatrix();
    Eigen::Matrix3d R_WI1 = Q_WI1.toRotationMatrix();
    Eigen::Matrix3d R_IC = Q_IC.toRotationMatrix();

    // calculate jacobians
    if(jacobians != NULL){

        if(jacobians[0] != NULL){

            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian0_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian0(jacobians[0]);

            Eigen::Matrix<double, 3, 6> tmp;
            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_IC.transpose()*R_WI1.transpose();
            tmp.topRightCorner(3,3) =  - R_IC.transpose()*R_WI1.transpose()*R_WI0*Utility::skewSymmetric(I0p);


            jacobian0_min  =  H*tmp;

            jacobian0_min = squareRootInformation_*jacobian0_min;

            jacobian0 << jacobian0_min, Eigen::Matrix<double,2,1>::Zero(); // lift

            if(jacobiansMinimal != NULL && jacobiansMinimal[0] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian0_min(jacobiansMinimal[0]);
                map_jacobian0_min = jacobian0_min;
            }
        }

        if(jacobians[1] != NULL){

            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian1_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian1(jacobians[1]);

            Eigen::Matrix<double, 3, 6> tmp;

            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = -R_IC.transpose()*R_WI1.transpose();
            tmp.bottomRightCorner(3,3) =  R_IC.transpose() * Utility::skewSymmetric(I1p);

            jacobian1_min = H*tmp;
            jacobian1_min = squareRootInformation_*jacobian1_min;

            jacobian1 << jacobian1_min, Eigen::Matrix<double,2,1>::Zero(); // lift

            if(jacobiansMinimal != NULL && jacobiansMinimal[1] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian1_min(jacobiansMinimal[1]);
                map_jacobian1_min = jacobian1_min;
            }
        }
        if(jacobians[2] != NULL){

            Eigen::Matrix<double,2,6,Eigen::RowMajor> jacobian2_min;
            Eigen::Map<Eigen::Matrix<double,2,7,Eigen::RowMajor>> jacobian2(jacobians[2]);

            Eigen::Matrix<double, 3, 6> tmp;

            tmp.setIdentity();
            tmp.topLeftCorner(3,3) = R_IC.transpose() * (R_WI1.transpose() * R_WI0 - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = R_IC.transpose() * R_WI1.transpose() * R_WI0 * R_IC;
            tmp.bottomRightCorner(3,3) =  -tmp_r * Utility::skewSymmetric(C0p) + Utility::skewSymmetric(tmp_r * C0p) +
                      Utility::skewSymmetric(R_IC.transpose() * (R_WI1.transpose() * (R_WI0 * t_IC + t_WI0 - t_WI1) - t_IC));

            jacobian2_min = H*tmp;
            jacobian2_min = squareRootInformation_*jacobian2_min;

            jacobian2 << jacobian2_min, Eigen::Matrix<double,2,1>::Zero(); // lift

            if(jacobiansMinimal != NULL && jacobiansMinimal[2] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,6,Eigen::RowMajor>> map_jacobian2_min(jacobiansMinimal[2]);
                map_jacobian2_min = jacobian2_min;
            }
        }

        if(jacobians[3] != NULL){
            Eigen::Map<Eigen::Matrix<double,2,1>> jacobian3(jacobians[3]);
            jacobian3 = - H*R_IC.transpose()*R_WI1.transpose()*R_WI0*R_IC*C0uv/(inv_dep*inv_dep);
            jacobian3 = squareRootInformation_*jacobian3;


            if(jacobiansMinimal != NULL && jacobiansMinimal[3] != NULL){
                Eigen::Map<Eigen::Matrix<double,2,1>> map_jacobian3_min(jacobiansMinimal[3]);
                map_jacobian3_min = jacobian3;
            }
        }


    }

    return true;
}
