#define _USE_MATH_DEFINES
#include "math.h"
#include "kalman_filter.h"
#include "tools.h"
#include <stdlib.h>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //h(x') vector
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  float h1 = sqrt(pow(px, 2) + pow(py, 2));
  float h2 = atan2(py,px);
  float h3 = (px*vx + py*vy) / h1;
  VectorXd Hx = VectorXd(3);
  Hx << h1, h2, h3;

  //y vector
  VectorXd y = z - Hx;
  //std::cout << "phi:" << y(1) << std::endl;

  if (fabs(y(1)) > M_PI){
    //std::cout << "phi before normalization:" << y(1) << std::endl;
    y(1) -= round(y(1) / (2.0 * M_PI)) * (2.0 * M_PI);
    //std::cout << "phi angle normalized:" << y(1) << std::endl;
  }

  /*
  while(!(y(1)<M_PI && y(1)>-M_PI)){
    std::cout << "phi before normalization:" << y(1) << std::endl;
    if(y(1)>M_PI){
      y(1)=y(1)-2*M_PI;
    }
    else{
      y(1)=y(1)+2*M_PI;
    }
    std::cout << "phi angle normalized:" << y(1) << std::endl;
  }*/

  //S and K matrices
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate and P matrix
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
