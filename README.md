# UTIAS-SLAM-
A summary of different Estimation algorithms and SLAM techniques on the UTIAS SLAM Dataset produced by Keith Leung


How to use the repository?
--------------------------

Clone the repo and download the MRCLAM Dataset1 from the official page below
http://asrl.utias.utoronto.ca/datasets/mrclam/

The repo was tested on Ubuntu 16.04 and should work with the other OS provided necessary python packages are installed.
Make sure the data and other files are referenced with the correct path.


Python Packages needed
----------------------
1. numpy
2. matplotlib
3. scipy (for sqrtm function for drawing covariance ellipse)


Scope of Repository
-------------------

1. Baseline EKF (inspired by Andrew Kramer's work)
2. Effect of models on filter performace
3. Comparing UKF and CKF
4. Workflow for KF (inspired by Sensor Fusion and Non linear filtering course on edX)



1.Baseline EKF (EKF_known_correlation_baseline.ipynb)
---------------------------------------
Primarily Python adaptation of Andrew Kramer's EKF_known_corr.m with added visualisation functions;
RMSE found is used as baseline for improvement and also tuning of filters.

Params set for the file
sample_time = 0.02 seconds, start index = 600, refresh rate = 5.0 seconds;



Reference
----------

1. Andrew Kramer's work
https://github.com/1988kramer/UTIAS-practice
Output of matlab script (localization/EKF_known_corr) is stored as test.dat which is used as baseline to improve upon.

2. Roger Labbe's work on KF is great place to learn 
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


3. Sensor Fusion and Non-Linear filtering course on edX
https://www.edx.org/course/sensor-fusion-and-non-linear-filtering-for-automot

