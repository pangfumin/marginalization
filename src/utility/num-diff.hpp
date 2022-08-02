/*! A simple tools to do finity num-diff
 *
 * Filename: num-diff.hpp
 * Version: 0.10
 * Author: Pang Fumin 
 */

#ifndef NUMBDIFFERENTIATOR_H
#define NUMBDIFFERENTIATOR_H
#include <ceres/ceres.h>

#include <memory>

#define Eps  (1e-6)


/**
  *  A simple class to do quickly finite differencing of an error functor to get
  *  a numeric-diff jacobian for minimal parameterization.
  *
  *  You can also use ceres::GradientChecker.
  *
  *  @Author: Pang Fumin
  */

template<typename Functor,int ParamBlockSize /* num of parameter blocks */>
class NumDiff{

public:


    NumDiff(Functor* ptrErrorFunctor):
            ptrErrorFunctor_(ptrErrorFunctor){

        if(ptrErrorFunctor_ == NULL){
            LOG(FATAL)<<"Error functor pointor is NULL!";
        }

    };

    /**
     * Diff function for the parameters with minimal-parameterization
     * @tparam ResidualDim
     * @tparam ParamDim
     * @tparam MinimalParamDim
     * @tparam LoaclPrameter
     * @param parameters
     * @param paramId
     * @param jacobiansMinimal
     * @return
     */
    template <int ResidualDim,/*  dim of residual*/
              int ParamDim,
              int MinimalParamDim,
              typename LoaclPrameter>
    bool df_r_xi(double** parameters, /* full parameters*/
                 unsigned int paramId,
                 double* jacobiansMinimal){

        
        std::shared_ptr<LoaclPrameter> ptrlocalParemeter(new LoaclPrameter);
        Eigen::Map<Eigen::Matrix<double,ResidualDim,MinimalParamDim,Eigen::RowMajor>> miniJacobian(jacobiansMinimal);
        Eigen::Map<Eigen::Matrix<double,ParamDim,1>> xi(parameters[paramId]);
        Eigen::Matrix<double,ResidualDim,1> residual_plus;
        Eigen::Matrix<double,ResidualDim,1> residual_minus;

        Eigen::Matrix<double,ParamDim,1>  xi_plus_delta, xi_minus_delta;
        Eigen::Matrix<double,MinimalParamDim,1> delta;

        for(unsigned int i = 0; i < MinimalParamDim; i++){

            // plus
            delta.setZero();
            delta(i) = Eps;
            ptrlocalParemeter->Plus(xi.data(),delta.data(),xi_plus_delta.data());
            double* parameter_plus[ParamBlockSize];
            applyDistribance(parameters,xi_plus_delta.data(),parameter_plus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_plus,residual_plus.data(),NULL);

            // minus
            delta.setZero();
            delta(i) = -Eps;
            ptrlocalParemeter->Plus(xi.data(),delta.data(),xi_minus_delta.data());
            double* parameter_minus[ParamBlockSize];
            applyDistribance(parameters,xi_minus_delta.data(),parameter_minus,paramId);
            ptrErrorFunctor_->Evaluate(parameter_minus,residual_minus.data(),NULL);

            // diff
            miniJacobian.col(i) = (residual_plus - residual_minus)/(2.0*Eps);


        }

        return true;
    };

    template <int ResidualDim,
            int ParamDim>
    bool df_r_xi(double** parameters,
                 unsigned int paramId,
                 double* jacobian){
        typedef Eigen::Matrix<double, ResidualDim, ParamDim, (ParamDim > 1? Eigen::RowMajor : Eigen::ColMajor)> JocabinType;

        Eigen::Map<JocabinType> Jacobian(jacobian);
        Eigen::Map<Eigen::Matrix<double, ParamDim, 1>> xi(parameters[paramId]);
        Eigen::Matrix<double, ResidualDim, 1> residual_plus;
        Eigen::Matrix<double, ResidualDim, 1> residual_minus;

        Eigen::Matrix<double, ParamDim, 1> xi_plus_delta, xi_minus_delta;
        Eigen::Matrix<double, ParamDim, 1> delta;

        for (unsigned int i = 0; i < ParamDim; i++) {

            delta.setZero();
            delta(i) = Eps;
            xi_plus_delta = xi + delta;
            double *parameter_plus[ParamBlockSize];
            applyDistribance(parameters, xi_plus_delta.data(), parameter_plus, paramId);
            ptrErrorFunctor_->Evaluate(parameter_plus, residual_plus.data(), NULL);


            xi_minus_delta = xi - delta;
            double *parameter_minus[ParamBlockSize];
            applyDistribance(parameters, xi_minus_delta.data(), parameter_minus, paramId);
            ptrErrorFunctor_->Evaluate(parameter_minus, residual_minus.data(), NULL);

            Jacobian.col(i) = (residual_plus - residual_minus) / (2.0 * Eps);

        }

        return true;
    };


    /**
     *
     * @tparam ResidualDim
     * @tparam ParamDim
     * @param Jacobian_a
     * @param Jacobian_b
     * @param relTol
     * @return
     */
    template <int ResidualDim,
              int ParamDim>
    static bool isJacobianEqual(double* Jacobian_a,
                                  double* Jacobian_b,
                                  double relTol = 1e-4) {

        Eigen::Map<Eigen::Matrix<double,ResidualDim,ParamDim,Eigen::RowMajor>> jacobian_a(Jacobian_a);
        Eigen::Map<Eigen::Matrix<double,ResidualDim,ParamDim,Eigen::RowMajor>> jacobian_b(Jacobian_b);


        bool isCorrect = true;
        // check
        double norm = jacobian_a.norm();
        Eigen::MatrixXd J_diff = jacobian_a - jacobian_b;
        double maxDiff = std::max(-J_diff.minCoeff(), J_diff.maxCoeff());
        if (maxDiff / norm > relTol) {
            std::cout << "Jacobian inconsistent: " << std::endl;
            std::cout << " Jacobian a: ";
            std::cout << std::endl<< jacobian_a << std::endl;
            std::cout << "provided Jacobian b: ";
            std::cout << std::endl<< jacobian_b;
            std::cout << std::endl << "relative error: " << maxDiff / norm
                      << ", relative tolerance: " << relTol << std::endl;
            isCorrect = false;
        }

        return isCorrect;

    }

private:
    /**
     *
     * @param parameters
     * @param parameter_i
     * @param parameters_plus
     * @param ith
     */
    void applyDistribance(double** parameters,
                          double* parameter_i,
                          double** parameters_plus,
                          unsigned int ith){
        for(unsigned int i = 0; i < ith; i++){
            parameters_plus[i] = parameters[i];
        }
        parameters_plus[ith] = parameter_i;
        for(unsigned int i = ith + 1; i < ParamBlockSize; i++){
            parameters_plus[i] = parameters[i];
        }
    }
    Functor* ptrErrorFunctor_;
};


#endif
