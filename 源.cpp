#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


float max_radius,first_max_radius; Point2f max_center;//ROI����
float sec_radius; Point2f sec_center;//������Ϊ��ת����


//��ȡ����Ե����,���ROI����/��ȡ��������Բ�ĺͰ뾶
void Getcontours(InputArray src, OutputArray dst)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;

	src.copyTo(src_gray);
	//��������
	findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	vector<Point2f>  center(contours.size());
	vector<float> radius(contours.size());

	vector <vector<Point>> contours_poly(contours.size());
	
	vector<float> temp(contours.size());
	//������
	Mat drawing = Mat::zeros(src_gray.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);//�ƽ����ߣ�Ӧ��Ҫ����
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		
	}

	//��ʼ�������ǵõ���С���������Բ��ȷ��roi
	max_radius = radius[0];
	max_center = center[0];

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);   //���Բ
		temp[i] = radius[i];
		//������������ԲROI����
		if (max_radius < radius[i])
		{
	     	max_radius = radius[i];
			max_center = center[i];
		}
	}
	circle(drawing,max_center, (int)max_radius, Scalar(0, 255, 255), 2, 8, 0);

	sort(temp.begin(), temp.end());
	for (int i = 0; i < contours.size(); i++)
	{
		if (radius[i] == temp[contours.size() - 2])
		{
			sec_center = center[i];
			sec_radius = radius[i];
		}
	}
	circle(drawing, sec_center, (int)sec_radius, Scalar(0, 255, 255), 2, 8, 0);
	circle(drawing, sec_center, 4, Scalar(0, 255, 255), -1, 8, 0);

	drawing.copyTo(dst);
	/*...........................*/
}

//��ת����
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
int main()
{
	Mat img = imread("222.jpg", 1);
	Mat rotate_maps[15]; Mat ImageROIS[15];//��תͼ�κ���תROI
	Mat temp_rotate;//��ת֮���ROI�����ݴ�
	Mat gray_img, dst_img, dst_contours,temp_contours,real_no_use;//ԭͼ�����������Ӧ�ñ���
	//����ֱ�����Զ�����ֵ����������
	cvtColor(img, gray_img, COLOR_RGB2GRAY);//ԭʼ�Ҷ�ͼ
	threshold(gray_img,dst_img,150,255,3);//dst_img��ֵ֮��Ķ�ֵ��ͼ
	
	Getcontours(dst_img,dst_contours);//������õ������ǻ�ȡ����center������radius��ͼ��û���κ�Ӱ��
	first_max_radius = max_radius;

	Mat temp = dst_img;
	//��������ȡ����Ȥ����
	Mat imageROI; 
	imageROI = temp(Rect(max_center.x - max_radius, max_center.y - max_radius, max_radius * 2, max_radius * 2));//��С����������
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//���������������
	findContours(imageROI, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	drawContours(imageROI, contours,-1,(255,255,255),FILLED,8,hierarchy);

	

	circle(temp, sec_center, (int)sec_radius, Scalar(0, 255, 255), 2, 8, 0);
	circle(temp, sec_center, 4, Scalar(0, 255, 255), -1, 8, 0);
		

	namedWindow("result", WINDOW_NORMAL);
	imshow("result", temp);
	
	
	//ת��һ��֮��,�����������Դ�ȥ��СROI�Ϳ��Ա�����ִ�С��һ�������
    

	rotate_maps[0] = imageROI.clone();
	


	//��������sec_centerΪ������������ת24��*15��=360��
	for (int i = 1; i < 15; i++)
	{
		temp_contours = rotateImage2(imageROI, 24*i, sec_center);
		temp_rotate = temp_contours.clone();
		Getcontours(temp_rotate, real_no_use);
		ImageROIS[i] = temp_rotate(Rect(max_center.x - first_max_radius, max_center.y - first_max_radius, first_max_radius * 2, first_max_radius * 2)).clone();//��С����������
		rotate_maps[i] = ImageROIS[i].clone();
	}

	Mat average = Mat::zeros(imageROI.size(), CV_32F); //�ۼӺʹ洢
	
	accumulate(rotate_maps[0], average);

	for (int i = 1; i < 15; i++)
	{
		accumulate(rotate_maps[i], average);
	}
	average = average / 15;

	threshold(average, average, 150, 255, 3);
	imwrite("average.jpg", average);
	namedWindow("average", WINDOW_NORMAL);
	imshow("average", average);

	waitKey(0);


}