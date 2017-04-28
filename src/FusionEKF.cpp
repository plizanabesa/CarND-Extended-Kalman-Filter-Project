#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  //measurement matrix - radar
  Hj_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  //state covariance matrix P
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  //state covariance matrix P
  MatrixXd Q_ = MatrixXd(4, 4);
  Q_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;
    
  //measurement matrix
  MatrixXd H_ = MatrixXd(2, 4);
  H_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  //the initial transition matrix F_
  MatrixXd F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
    
  //state vector
  VectorXd x_ = VectorXd(4);
    
  //initialize EKF
  ekf_.Init(x_,P_,F_,H_,R_laser_,Q_);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float px = ro * cos(phi);
      float py = ro * sin(phi);
      float v = measurement_pack.raw_measurements_(2);
      float vx = v * cos(phi);
      float vy = v * sin(phi);
      ekf_.x_ << px, py, vx, vy;

      //update Hj
      Hj_ = tools.CalculateJacobian(ekf_.x_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);
      float vx = 0;
      float vy = 0;
      ekf_.x_ << px, py, vx, vy;
    }

    //update previous time
    previous_timestamp_ = measurement_pack.timestamp_;

    //done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the acceleration noise components
  float noise_ax = 9;
  float noise_ay = 9;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
         0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
         dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
         0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  VectorXd z;
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    z = VectorXd(3);
    float z0 = measurement_pack.raw_measurements_[0];
    float z1 = measurement_pack.raw_measurements_[1];
    float z2 = measurement_pack.raw_measurements_[2];
    z << z0,z1,z2;
    ekf_.UpdateEKF(z);
  } else {      
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    z = VectorXd(2);
    float z0 = measurement_pack.raw_measurements_[0];
    float z1 = measurement_pack.raw_measurements_[1];
    z << z0, z1;
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
