#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;



Mat rotateImage1(Mat img, int degree, Point2f g_center);
Mat rotateImage2(Mat img, int degree, Point2f g_center);
void Getcontours(InputArray src, OutputArray dst);