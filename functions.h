#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "Face.h"

using namespace cv;

/*Shift the quadrants of a fourier transform to have the low frequencies in the middle*/
void shift(Mat magI);

/*Compute the dft of the image*/
Mat computeDFT(Mat image);

/* Return the value of the buttereworth filter at*/
double Butterworth(double kx, double ky, double K, int n);

/*Apply homomorphic filtering to input*/
void HomomorphicFiltering(const Mat& input, Mat& output, double K = 0.5);

/*Find the closest face to "face" in "previousFaces", 
it compares the distance between face centers and if a face is close enough it returns true and update "closest" */
bool identify(const Face& face, std::vector<Face>& previousFaces, Face** closest, double maxGap = 100.0); 