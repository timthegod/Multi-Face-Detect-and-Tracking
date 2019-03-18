#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/detection_based_tracker.hpp"
#include <opencv2/core/ocl.hpp>
#include "core.hpp"

#include <iostream>
#include <vector>
#include <stdio.h>

#include "Face.h"
#include "functions.h"

using namespace std;
using namespace cv;

vector<Face> previousFaces;

void detectDisplay(Mat frame, Mat& result, CascadeClassifier& face_cascade)
{
	
	//Image processing of the frame
	Mat frameGray;
	cvtColor(frame, frameGray, CV_BGR2GRAY);
	HomomorphicFiltering(frameGray, frameGray);
	equalizeHist(frameGray, frameGray);

	imshow("gray", frameGray);
	namedWindow("result", CV_WINDOW_AUTOSIZE);
	int minNeighbors = 3;
	cvCreateTrackbar("minNeighbors", "gray", &minNeighbors, 50);

	//Face detection
	vector<Rect> faces;
	face_cascade.detectMultiScale(frameGray, faces, 1.1, minNeighbors, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++) {
		//Mat faceROI = frame(faces[i]);
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);
		String locationText = "x: " + to_string((int)(faces[i].x + faces[i].width*0.5)) + " y: " + to_string((int)(faces[i].y + faces[i].height*0.5));
		putText(frame, locationText, Point(faces[i].x + faces[i].width*0.5, faces[i].y), 1, 1, Scalar(0, 255, 0), 2, 8);
		//imshow("face" + to_string(i), )
	}

	bool exist = false;
	vector<Face> detectedFaces;
	Face* closest = nullptr;
	for (Rect fR : faces) {
		detectedFaces.push_back(Face(fR));
	}
	//initialise 
	for (Face& f: previousFaces) {
		f.check = false;
	}

	for (const Face& f : detectedFaces) {
		exist = identify(f, previousFaces, &closest);
		if (exist) {
			//old face
			closest->previousPositions.push_back(f.center);
			closest->check = true;
		}
		else {
			//new face
			previousFaces.push_back(f);
		}
	}
	//face disappear
	vector<Face> tmp;
	for (auto& f : previousFaces) {
		if (f.check) {
			tmp.push_back(f);
			std::cout << previousFaces.size() << std::endl;
		}
	}
	previousFaces = tmp;

	for (const Face& f : previousFaces) {
		f.draw(frame);
	}
	imshow("result", frame);
	result = frame.clone();
	
}


int main()
{
	String face_cascade_name = "haarcascade_frontalface_alt2.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	VideoCapture cap(0); // open the default camera
	VideoWriter video("outcpp.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(640, 480));
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	Mat frame, result;

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n"); return -1;
	};
	//if (!eyes_cascade.load(eyes_cascade_name))
	//{
	//	printf("--(!)Error loading\n"); return -1;
	//};
	while (1) {
		cap >> frame;
		flip(frame, frame, 1);
		if (!frame.empty()) {
			detectDisplay(frame, result,face_cascade);
			//video.write(result);
		}
		else {
			printf(" --(!) No captured frame -- Break!"); break;
		}
		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}

	return 0;
}