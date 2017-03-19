#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "head.h"
using namespace cv;
using namespace std;


float max_radius, first_max_radius; Point2f max_center;//ROI参数
float sec_radius; Point2f sec_center;//内轮廓为旋转中心
//获取最大边缘轮廓,获得ROI区域/获取内轮廓的圆心和半径

int getGeaer(Mat src)
{
	
	Mat gray_img,  threshold_output;
	int g_nThresh = 150;
	int g_maxThresh = 255;
	int areamax = 0; int imax = 0;
	vector<vector<Point>> g_contours;
	vector<Vec4i> g_hierarchy;

	gray_img = src.clone();
	blur(gray_img, gray_img, Size(3, 3));

	threshold(gray_img, threshold_output, g_nThresh, 255, THRESH_BINARY);
	findContours(threshold_output, g_contours, g_hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int idx = 0; idx >= 0; idx = g_hierarchy[idx][0])
	{
		Rect rect = boundingRect(g_contours[idx]);
		int area = rect.width * rect.height;
		if (area > areamax)
		{
			areamax = area;
			imax = idx;
		}
	}
	vector<Point> hull;
	vector<Point> contours_poly;

	approxPolyDP(Mat(g_contours[imax]), contours_poly, 3, true);
	convexHull(contours_poly, hull, false);
	return hull.size();
	
}

int getConstnum(Mat src)
{
	Mat src_next; 
	Mat src_first=src.clone();
	for (int i = 1; i < 50; i++)
	{
		resize(src_first, src_next, Size(), 0.5, 0.5);

		if ((getGeaer(src_first) - getGeaer(src_next)) / getGeaer(src_next)>=0.5)
		{
			return getGeaer(src_first);
			break;
		}
		else
		{
			src_first = src_next.clone();
			resize(src_first, src_next, Size(), 0.5, 0.5);
		}


	}



}

void Getcontours(InputArray src, OutputArray dst)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;

	src.copyTo(src_gray);
	//查找轮廓
	findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	vector<Point2f>  center(contours.size());
	vector<float> radius(contours.size());

	vector <vector<Point>> contours_poly(contours.size());

	vector<float> temp(contours.size());
	//画轮廓
	Mat drawing = Mat::zeros(src_gray.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//逼近曲线，应该要调整
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);

	}

	//初始化，我们得到最小外轮廓外接圆来确定roi
	max_radius = radius[0];
	max_center = center[0];

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);   //外接圆
		temp[i] = radius[i];
		//最大轮廓的外接圆ROI参数
		if (max_radius < radius[i])
		{
			max_radius = radius[i];
			max_center = center[i];
		}
	}
	circle(drawing, max_center, (int)max_radius, Scalar(0, 255, 255), 2, 8, 0);

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


int main()
{
	Mat img = imread("222.jpg", 1);
	Mat rotate_maps[15]; Mat ImageROIS[15];//旋转图形和旋转ROI
	Mat diffcv;
	Mat temp_rotate;//旋转之后的ROI区域暂存
	Mat gray_img, dst_img, dst_contours,temp_contours,real_no_use;//原图像轮廓的相关应用变量
	//下面直接用自动的阈值操作来除噪
	cvtColor(img, gray_img, COLOR_RGB2GRAY);//原始灰度图
	threshold(gray_img,dst_img,140,255,3);//dst_img阈值之后的二值化图
	
	Getcontours(dst_img,dst_contours);//这个调用的作用是获取两个center和两个radius对图像没有任何影响
	first_max_radius = max_radius;

	Mat temp = dst_img;
	//下面来获取赶兴趣区域
	Mat imageROI; 
	//circle(temp, max_center, (int)max_radius, Scalar(255, 255, 255), 2, 8, 0);
	imageROI = temp(Rect(max_center.x - max_radius, max_center.y - max_radius, max_radius * 2, max_radius * 2));//最小正方形区域
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//查找轮廓，并填充
	findContours(imageROI, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	drawContours(imageROI, contours,-1,(255,255,255),FILLED,8,hierarchy);
	
	imwrite("000.jpg", imageROI);
	//int num = getGeaer(imageROI);

	//circle(temp, sec_center, (int)sec_radius, Scalar(0, 255, 255), 2, 8, 0);
	//circle(temp, sec_center, 4, Scalar(0, 255, 255), -1, 8, 0);
		
	//转过一次之后,那我们再来对此去最小ROI就可以避免出现大小不一的情况了


	int num = getConstnum(imageROI);

	rotate_maps[0] = imageROI.clone();


	//接下来用sec_center为中心来进行旋转24度*15次=360度
	for (int i = 1; i < 15; i++)
	{
		if (i < 8)
		{
			temp_contours = rotateImage2(imageROI, 24 * i, sec_center);
		}
		else
		{
			temp_contours = rotateImage2(imageROI, -24 * i, sec_center);
		}
		temp_rotate = temp_contours.clone();
		Getcontours(temp_rotate, real_no_use);
		ImageROIS[i] = temp_rotate(Rect(max_center.x - first_max_radius, max_center.y - first_max_radius, first_max_radius * 2, first_max_radius * 2)).clone();//最小正方形区域
		rotate_maps[i] = ImageROIS[i].clone();
	}

	Mat average = Mat::zeros(imageROI.size(), CV_32F); //累加和存储
	
	accumulate(rotate_maps[0], average);

	for (int i = 1; i < 15; i++)
	{
		accumulate(rotate_maps[i], average);
	}
	average = average / 15;

	//threshold(average, average,150, 255, 3);


	diffcv = Mat::zeros(imageROI.size(), CV_32F);
	imageROI.convertTo(imageROI, CV_32F);
	average.convertTo(average, CV_32F);
	
	diffcv = imageROI - average;

	imwrite("diffv.jpg", diffcv);
	
	threshold(diffcv, diffcv, 200, 255, 3);


	namedWindow("diff", WINDOW_NORMAL);
	imshow("diff", diffcv);


	waitKey(0);


}