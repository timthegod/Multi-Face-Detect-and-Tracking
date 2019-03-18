#include "functions.h"

void shift(Mat magI) {

	// crop if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left 
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat computeDFT(Mat image) {
	Mat padded;
	int m = getOptimalDFTSize(image.rows);
	int n = getOptimalDFTSize(image.cols);
	// create output image of optimal size
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
	// copy the source image, on the border add zero values
	Mat planes[] = { Mat_< float>(padded), Mat::zeros(padded.size(), CV_32F) };
	// create a complex matrix
	Mat complex;
	merge(planes, 2, complex);
	dft(complex, complex, DFT_COMPLEX_OUTPUT);  // fourier transform
	return complex;
}

double Butterworth(double kx, double ky, double K, int n) {

	return 1.0 / (1.0 + pow(K / sqrt(kx*kx + ky * ky), 2 * n));
}

void HomomorphicFiltering(const Mat& input, Mat& output, double K) {

	Mat lT, FlT;
	input.convertTo(lT, CV_32F);

	lT = lT + 1;
	log(lT, lT);//lT = ln(1+input)

	FlT = computeDFT(lT);
	shift(FlT);

	double kx, ky;
	double MaxDist = sqrt(1.0 / 4.0*(FlT.rows*FlT.rows + FlT.cols*FlT.cols));
	double centerX = (double)FlT.cols / 2.0;
	double centerY = (double)FlT.rows / 2.0;

	//Applying filtering in fourier space 
	for (int i = 0; i < FlT.rows; ++i)
	{
		for (int j = 0; j < FlT.cols; ++j) {
			kx = abs((double)j - centerX) / MaxDist;
			ky = abs((double)i - centerY) / MaxDist;

			FlT.at<Vec2f>(i, j) *= (1.0 - Butterworth(kx, ky, K, 10));
		}
	}

	Mat lT_filtered;
	idft(FlT, lT_filtered, DFT_SCALE);
	Mat planes[] = {
		Mat::zeros(FlT.size(), CV_32F),
		Mat::zeros(FlT.size(), CV_32F)
	};
	split(lT_filtered, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], lT_filtered); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)

	exp(lT_filtered, lT_filtered); 
	lT_filtered -= 1;//exponential to go back to original image but filtered

	normalize(lT_filtered, lT_filtered, 0, 255, NORM_MINMAX);
	convertScaleAbs(lT_filtered, lT_filtered);

	output = lT_filtered.clone();
}

bool identify(const Face& face, std::vector<Face>& previousFaces, Face** closest, double maxGap) {

	bool found = false;
	for (Face& f : previousFaces) {
		if (face.distance(f) < maxGap) {
			maxGap = face.distance(f);
			*closest = &f;
			found = true;
		}
	}

	return found;
}