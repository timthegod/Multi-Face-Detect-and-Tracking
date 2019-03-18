#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <vector>

using namespace cv;

class Face {
private:
    Rect contour;
    
public:
    Point center;
    std::vector<Point> previousPositions;
    bool check;
    
    Face(Rect _contour) :contour(_contour), check(true) {
        center = Point(contour.x + contour.width*0.5, contour.y + contour.height*0.5);
        previousPositions.push_back(center);
    }
    
    Face() : check(true) {};
    
    double distance(const Face& f) const {
        double dX = center.x - f.center.x;
        double dY = center.y - f.center.y;
        return std::sqrt(dX*dX + dY * dY);
    };
    
    void draw(Mat& img) const {
        for (int i = 0; i < previousPositions.size() - 1; ++i) {
            line(img, previousPositions[i], previousPositions[i + 1], Scalar(255 * i / previousPositions.size(), 0, 0), 2);
        }
    }
};




