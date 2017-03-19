#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "head.h"
using namespace cv;
using namespace std;

Mat rotateImage1(Mat img, int degree, Point2f g_center)
{
	degree = -degree;
	Mat img_rotate;
	Mat rot = getRotationMatrix2D(g_center, degree, 1.0);
	Rect bbox = RotatedRect(g_center, img.size(), degree).boundingRect();
	rot.at<double>(0, 2) += bbox.width / 2.0 - g_center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - g_center.y;

	cv::Mat dst;
	cv::warpAffine(img, img_rotate, rot, bbox.size());
	return img_rotate;
}
Mat rotateImage2(Mat img, int degree, Point2f g_center)
{
	degree = -degree;
	double angle = degree  * CV_PI / 180.; // ����  
	double a = sin(angle), b = cos(angle);
	int width = img.cols;
	int height = img.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6];
	Mat map_matrix = Mat(2, 3, CV_32F, map);
	// ��ת����
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 = map_matrix;
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	Mat img_rotate;
	//��ͼ��������任
	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�
	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.
	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
	warpAffine(img, img_rotate, map_matrix, Size(width_rotate, height_rotate), 1, 0, 0);
	return img_rotate;
}