#include <iostream>
#include <stdio.h>
#include <ctime>
#include "IPM.h"
//opencv includes
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace std;
using namespace cv;

//class Camera {
//public:
//	struct Parameters {
//		float fx, fy;
//		float cx, cy;
//		float k1, k2, k3;
//		float p1, p2;
//		Parameters(float _fx, float _fy,
//			float _cx, float _cy,
//			float _k1 = 0.0, float _k2 = 0.0, float _k3 = 0.0,
//			float _p1 = 0.0, float _p2 = 0.0) : fx(_fx), fy(_fy), cx(_cx), cy(_cy), k1(_k1), k2(_k2), k3(_k3), p1(_p1), p2(_p2) {}
//		Parameters() {}
//	};
//	Camera() {}
//	Camera(float fx, float fy,
//		float cx, float cy,
//		float k1 = 0.0, float k2 = 0.0, float k3 = 0.0,
//		float p1 = 0.0, float p2 = 0.0) :params(fx, fy, cx, cy, k1, k2, k3, p1, p2), useDistortion(false) {}
//	Parameters params;
//	bool useDistortion;
//	void distortPoint(float x, float y, float *u, float *v);
//	void undistortPoint(float u, float v, float *x, float *y);
//};

#pragma region CrossDetection
//const double M_PI = 3.14;
//const double DEG_TO_RAD = M_PI / 180;
//const double RAD_TO_DEG = 180 / M_PI;
//Camera cam;
//Mat map_x;
//Mat map_y;
//
//Mat inv_map_x;
//Mat inv_map_y;
//bool calibrateZ = false;
//int lastId = 0;
//unsigned short h_cutout;
//Mat lastR;
//Mat lastT;
//float distanceToCrossing;
//
//
//void logZForCalibration(cv::Mat &image);
//
//void ComputeRemappingLUT(cv::Mat image, cv::Mat *map_x, cv::Mat *map_y);
//
//void ComputeInvRemappingLUT(cv::Mat image, cv::Mat *map_x, cv::Mat *map_y);
//
//int main(int _argc, char** _argv)
//{
//
//	lastT = cv::Mat::zeros(3, 1, CV_32FC1);
//	lastR = cv::Mat::zeros(3, 1, CV_32FC1);
//
//	string imgPath = "G:\\Study stuff\\Semester 3\\AADC\\Work\\images\\Crossroads\\crossroad Detection\\550.png";
//	Mat image = imread(imgPath, 1);
//	imshow("Original Image", image);
//	Mat gray = image(Rect(0, 0, 640, 480));
//	Mat transformedImage = image.clone();
//	Mat sobeledImage = image.clone();
//	Mat groundPlane = image.clone();
//
//	Mat remapped = image(Rect(0, 0, 640, 480));
//
//	if (map_x.size() != gray.size() || map_y.size() != gray.size()) 
//	{
//		map_x.create(gray.size(), CV_32FC1); map_y.create(gray.size(), CV_32FC1);
//		inv_map_x.create(gray.size(), CV_32FC1); inv_map_y.create(gray.size(), CV_32FC1);
//		ComputeRemappingLUT(gray, &map_x, &map_y);
//		ComputeInvRemappingLUT(gray, &inv_map_x, &inv_map_y);
//	}
//	
//	Mat croppedEqualized;
//	Mat cannyImage;
//	Mat croppedRemaped = image(Rect(0, 0, 640, 480));
//	Mat croppedColor;
//	croppedColor = gray;
//	imshow("Gray", gray);
//	if ((calibrateZ || lastId) && false)
//	{ //Slow
//		//            cv::Ptr<cv::CLAHE> clahe = createCLAHE();
//		//            clahe->setClipLimit(4);
//		//            clahe->apply(croppedImage,croppedEqualized);
//		//croppedEqualized = croppedImage;
//		//cv::Canny(croppedEqualized, cannyImage, 350,600,3, true); //1FPS
//		//cv::remap(croppedEqualized, croppedRemaped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar(0,0,0)); //5FPS
//		//croppedRemaped = image(Rect(0,480,200,250));
//		cvtColor(croppedRemaped, croppedColor, COLOR_GRAY2RGB);
//		imshow("Cropped Color", croppedColor);
//	}
//	if (calibrateZ) {
//		logZForCalibration(croppedColor);
//	}
//	else if (lastId) { //Sign detected
//		Mat rMat;
//		Rodrigues(lastR, rMat);
//		Mat corners[4];
//		corners[0] = (Mat_<float>(3, 1) << 0.1, 0.1, 0.35); //  x:down, y:right, z:towards camera
//		corners[1] = corners[0] + (Mat_<float>(3, 1) << 0., 1., 0.);
//		corners[2] = corners[0] + (Mat_<float>(3, 1) << 0., 0., 1.);
//		corners[3] = corners[2] + (Mat_<float>(3, 1) << 0., 1., 0.);
//		//            Mat nonRot = (Mat_<float>(3,3) << 0, 1, 0,
//		//                                              1, 0, 0,
//		//                                              0, 0,-1);
//		float yaw = M_PI / 2, pitch = 0.0, roll = atan2(rMat.at<float>(7), rMat.at<float>(8));//M_PI;
//		//Mat rot = (Mat_<float>(3,3) << cos(yaw)*cos(pitch),cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll),cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll),sin(yaw)*cos(pitch),sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll),sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll),-sin(pitch),cos(pitch)*sin(roll),cos(pitch)*cos(roll));
//		Mat rot = (Mat_<float>(3, 3) << 0, -cos(roll), sin(roll),
//			1, 0, 0,
//			0, sin(roll), cos(roll));
//		//Compute distance
//		Mat x = /*nonRot*/rot*(Mat_<float>(3, 1) << 0.1, 0.15, 0.35) + lastT;
//		distanceToCrossing = x.at<float>(2);
//
//		//evaluateCrossing(croppedRemaped,&croppedColor,corners,/*rMat*/rot,lastT,true);
//		timeAtLastSign = time(0);
//	}
//
//
//	waitKey(0);
//	return 0;
//
//
//}
//
//void logZForCalibration(cv::Mat &image)
//{
//	Mat zer = (Mat_<float>(3, 1) << 0.1, 0.0, 0.0);
//	Mat nonRot = (Mat_<float>(3, 3) << 0, 1, 0,
//		1, 0, 0,
//		0, 0, -1);
//	Mat xx = nonRot*zer + lastT;
//	float xp = xx.at<float>(0) / xx.at<float>(2);
//	float yp = xx.at<float>(1) / xx.at<float>(2);
//	float u, v;
//	cam.distortPoint(xp, yp, &u, &v);
//	if ((u >= 0 && u<inv_map_x.cols) && ((v - h_cutout) >= 0 && (v - h_cutout)<inv_map_x.rows)) {
//		float remaped_u = inv_map_x.at<float>(v - h_cutout, u);
//		float remaped_v = inv_map_y.at<float>(v - h_cutout, u);
//		cv::circle(image, Point(remaped_u, remaped_v), 1, Scalar(0, 255, 0), 5);
//		LOG_WARNING(adtf_util::cString::Format("%.03f\",\"%.03f", xx.at<float>(2), remaped_v));
//	}
//}
//
//void ComputeRemappingLUT(cv::Mat image, cv::Mat *map_x, cv::Mat *map_y)
//{
//	//Focal Length (mm)
//	double f_u = cam.params.fx, f_v = cam.params.fy;
//	//Principal Point
//	double c_u = cam.params.cx, c_v = cam.params.cy;
//	double pitch_angle = 1 * DEG_TO_RAD;
//	//cos(pitch_angle), cos(yaw_angle)
//	double c_1 = cos(pitch_angle);
//	double c_2 = 1.00;
//	//sin(pitch_angle), sin(yaw_angle)
//	double s_1 = sin(pitch_angle);
//	double s_2 = 0.0;
//	//height of camera (cm)
//	double h = 22.5;
//
//	//Transformation matrix T > Image points to ground plane
//	cv::Mat T = (cv::Mat_<double>(4, 4) << -c_2 / f_u, s_1*s_2 / f_v, c_u*c_2 / f_u - c_v*s_1*s_2 / f_v - c_1*s_2, 0,
//		s_2 / f_u, s_1*c_1 / f_v, -c_u*s_2 / f_u - c_v*s_1*c_2 / f_v - c_1*c_2, 0,
//		0, c_1 / f_v, -c_v*c_1 / f_v + s_1, 0,
//		0, -c_1 / (f_v*h), c_v*c_1 / (h*f_v) - s_1 / h, 0);
//	T = h * T;
//
//	cv::Mat T_inv = (cv::Mat_<double>(4, 4) << f_u*c_2 + c_u*c_1*s_2, c_u*c_1*c_2 - s_2*f_u, -c_u*s_1, 0,
//		s_2*(c_v*c_1 - f_v*s_1), c_2*(c_v*c_1 - f_v*s_1), -f_v*c_1 - c_v*s_1, 0,
//		c_1*s_2, c_1*c_2, -s_1, 0,
//		c_1*s_2, c_1*c_2, -s_1, 0);
//
//	for (int p_y = 0; p_y<480; p_y++) {
//		for (int p_x = 0; p_x<640; p_x++) {
//			cv::Mat P_G = (cv::Mat_<double>(4, 1) << p_x - 210, p_y + 40, -h, 1);
//			cv::Mat P_I = T_inv*P_G;
//			P_I /= P_I.at<double>(3, 0);
//			float x = (float)P_I.at<double>(0, 0);
//			float y = (float)P_I.at<double>(1, 0);
//			if (x >= 0 && x<image.cols && y >= h_cutout && y<image.rows) {
//				(*map_y).at<float>(p_y, p_x) = y - h_cutout;  // y displacement, this LUT if for the cropped image.
//				(*map_x).at<float>(p_y, p_x) = x;
//			}
//		}
//	}
//}
//
//void ComputeInvRemappingLUT(cv::Mat image, cv::Mat *map_x, cv::Mat *map_y)
//{
//	//Focal Length (mm)
//	double f_u = cam.params.fx, f_v = cam.params.fy;
//	//Principal Point
//	double c_u = cam.params.cx, c_v = cam.params.cy;
//	double pitch_angle = 2 * DEG_TO_RAD;
//	//cos(pitch_angle), cos(yaw_angle)
//	double c_1 = cos(pitch_angle);
//	double c_2 = 1.00;
//	//sin(pitch_angle), sin(yaw_angle)
//	double s_1 = sin(pitch_angle);
//	double s_2 = 0.0;
//	//height of camera (cm)
//	double h = 22.5;
//
//	//Transformation matrix T > Image points to ground plane
//	cv::Mat T = (cv::Mat_<double>(4, 4) << -c_2 / f_u, s_1*s_2 / f_v, c_u*c_2 / f_u - c_v*s_1*s_2 / f_v - c_1*s_2, 0,
//		s_2 / f_u, s_1*c_1 / f_v, -c_u*s_2 / f_u - c_v*s_1*c_2 / f_v - c_1*c_2, 0,
//		0, c_1 / f_v, -c_v*c_1 / f_v + s_1, 0,
//		0, -c_1 / (f_v*h), c_v*c_1 / (h*f_v) - s_1 / h, 0);
//	T = h * T;
//
//	for (int y = 0; y<image.rows; y++) {
//		for (int x = 0; x<image.cols; x++) {
//			cv::Mat P_I = (cv::Mat_<double>(4, 1) << x, y, 1, 1);
//			cv::Mat P_G = T*P_I;
//			P_G /= P_G.at<double>(3, 0);
//			float x_value = (float)P_G.at<double>(0, 0);
//			float y_value = (float)P_G.at<double>(1, 0);
//			x_value += 210; //To get the frame in the image
//			y_value -= 40;
//			if (x_value >= 0 && y_value >= 0 && x_value < image.cols && y_value < image.rows && y >= h_cutout) {
//				(*map_y).at<float>(y - h_cutout, x) = y_value;  //y-line_height: For the cropped image
//				(*map_x).at<float>(y - h_cutout, x) = x_value;
//			}
//		}
//	}
//}

#pragma endregion


#pragma region Old Code
///// perform the Simplest Color Balancing algorithm
//void SimplestCB(Mat& in, Mat& out, float percent) {
//	assert(in.channels() == 3);
//	assert(percent > 0 && percent < 100);
//
//	float half_percent = percent / 200.0f;
//
//	vector<Mat> tmpsplit; split(in, tmpsplit);
//	for (int i = 0; i<3; i++) {
//		//find the low and high precentile values (based on the input percentile)
//		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
//		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
//		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
//		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
//		//cout << lowval << " " << highval << endl;
//
//		//saturate below the low percentile and above the high percentile
//		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
//		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);
//
//		//scale the channel
//		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
//	}
//	merge(tmpsplit, out);
//}
//
//void showImage(String imgPath, String windowName)
//{
//	Mat image;
//	//String imgPath = "C:\\Users\\Amanullah Tariq\\Documents\\Visual Studio 2013\\Projects\\LaneDetection\\LaneDetection\\Data\\image1.jpg";
//	image = imread(imgPath, IMREAD_COLOR); // Read the file
//
//	if (!image.data) // Check for invalid input
//	{
//		cout << "Could not open or find the image" << std::endl;
//		//return -1;
//	}
//
//	namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window for display.
//	imshow(windowName, image); // Show our image inside it.
//}
//
//void showImage(Mat img, String windowName)
//{
//	namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window for display.
//	imshow(windowName, img); // Show our image inside it.
//}
//
//void Search(Point *psPoints, unsigned  int *pui8PointsCount, unsigned  int ui8Limit, cv::Mat matCannyImg)
//{
//	//LOG_INFO("cLaneTracking::Search Called");
//	/*A_UTIL_ASSERT(NULL != psPoints);
//	A_UTIL_ASSERT(NULL != pui8PointsCount);*/
//
//	*pui8PointsCount = 0;
//	int nColumnLast = -1;
//	Size szCannySize = matCannyImg.size();
//
//	for (int nColumn = 0; nColumn < szCannySize.width; nColumn++)
//	{
//		if (matCannyImg.at<uchar>(szCannySize.height / 2, nColumn) != 0)
//		{
//			if (abs(nColumnLast - nColumn) >= 2 && abs(nColumnLast - nColumn) < ui8Limit && nColumnLast != -1)
//			{
//				psPoints[*pui8PointsCount].x = static_cast<int>(nColumn - (abs(nColumnLast - nColumn) / 2));
//				(*pui8PointsCount)++;
//				nColumnLast = -1;
//			}
//			else
//			{
//				nColumnLast = nColumn;
//			}
//		}
//	}
//
//	
//}
//
//#define MAX_DEVIATION 150
//static int m_nBlindCounter = 0;
//static Point m_sPlaceToBe;
//static int m_i16Error;
//static int m_i16ErrorSum = 0;
//static float m_f32SteeringAngle = 0.0f;
//static float m_f64Kp = 0.08;
//
//static float m_f64Ki = 0.1445;
//static float m_f64Kd = 0.01238;
//static int m_i16ErrorOld = 0;
//static float m_f64PT1ScaledError;
//static float m_f64PT1InputFactor = 0.1;
//static float m_f32PT1SteeringOut = 0.0f;
//static float m_f64PT1Gain;
//static float m_f32PT1LastSteeringOut = 0.0f;
//static float m_f32PT1NextSteeringOut;
//static float m_f64PT1Tau = 0.1;
//static float m_f64PT1Sample = 0.9;
//static float m_f32Ts = 0.033f;
//static int   m_i16LaneWidth = 390;
//static int m_i16LaneWidthMinNear = 320;
//static int m_i16LaneWidthMaxNear = 430;
//
//static int        m_nCenterFromLeft = 0;
//static int        m_nCenterFromRight = 0;
//
//static Point      m_sLaneCenterNear;
//
//void LateralControl(Point *psPoints, unsigned int *pui8PointsCount)
//{
//	
//	//m_f32PT1LastSteeringOut = 0.0f;
//	//m_f32PT1SteeringOut = 0.0f;
//
//	if (*pui8PointsCount == 2)
//	{
//		int i16LaneWidthCalc = psPoints[1].x - psPoints[0].x;
//		//LOG_INFO(cString::Format("Point0: %i, Point1: %i, LaneWidth calc: %i", points[0].x, points[1].x, LaneWidth_calc));
//		int nLaneCenterCalc = psPoints[0].x + (static_cast<int>(i16LaneWidthCalc / 2));
//
//		if ((i16LaneWidthCalc > m_i16LaneWidthMinNear) && (i16LaneWidthCalc < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16LaneWidthCalc;
//			m_sLaneCenterNear.x = nLaneCenterCalc;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//
//	else if (*pui8PointsCount == 1)
//	{
//		// If just one point is found the lane center shall be calculated based on this single point and the half of the previously calculated lane width
//		// If the right point of the lane was found LaneWidth/2 must be subtracted from the position of the point, if the left point was found LaneWidth/2
//		// must be added. Whether to add or to subtract LaneWidth/2 is determined by comparing the calculated new LaneCenter with the old LaneCenter.
//
//		m_nCenterFromLeft = psPoints[0].x + (static_cast<int>(m_i16LaneWidth / 2));
//
//		m_nCenterFromRight = psPoints[0].x - (static_cast<int>(m_i16LaneWidth / 2));
//
//		if (abs(m_nCenterFromLeft - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_sLaneCenterNear.x = m_nCenterFromLeft;
//			m_nBlindCounter = 0;
//		}
//		else if (abs(m_nCenterFromRight - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_sLaneCenterNear.x = m_nCenterFromRight;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//	else if (*pui8PointsCount == 3)
//	{
//		// If three points are found the correct lane is chosen according to the calculated lanewidth
//		int i16Width1 = psPoints[2].x - psPoints[1].x;
//		int i16Width2 = psPoints[1].x - psPoints[0].x;
//		int nLaneCenterCalc1 = psPoints[1].x + (static_cast<int>(i16Width1 / 2));
//		int nLaneCenterCalc2 = psPoints[0].x + (static_cast<int>(i16Width2 / 2));
//
//		if ((i16Width1 > m_i16LaneWidthMinNear) && (i16Width1 < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc1 - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16Width1;
//			m_sLaneCenterNear.x = nLaneCenterCalc1;
//			m_nBlindCounter = 0;
//		}
//		else if ((i16Width2 > m_i16LaneWidthMinNear) && (i16Width2 < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc2 - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16Width2;
//			m_sLaneCenterNear.x = nLaneCenterCalc2;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//	else
//	{
//		// If more than three or no point (=lane) is detected, continue with the previously calculated values
//		// Due to safety reasons a counter is implemented, that will cause the car to stop, if the car did not find any lane for more than X times consecutively (see parameter in logitudinal control)
//
//		m_nBlindCounter++;
//	}
//	//PID - Controller for Steering Angle
//	m_i16Error = m_sLaneCenterNear.x - m_sPlaceToBe.x;
//	m_i16ErrorSum = m_i16ErrorSum + m_i16Error;
//	m_f32SteeringAngle = static_cast<float>(m_f64Kp*m_i16Error + m_f64Ki*m_f32Ts*m_i16ErrorSum + (m_f64Kd*(m_i16Error - m_i16ErrorOld)) / m_f32Ts);
//	m_i16ErrorOld = m_i16Error;
//
//	// Limitation of Steering Angle
//	if (m_f32SteeringAngle > 30)
//	{
//		//      LOG_INFO(cString::Format("Error greater: LaneCenter %d Place2Be %d Steering %f error_val: %d error_sum: %d", m_LaneCenter.x, m_PlaceToBe.x, m_SteeringAngle, error, e_sum).GetPtr());
//		m_f32SteeringAngle = 30.0;
//	}
//	else if (m_f32SteeringAngle < -30)
//	{
//		//      LOG_INFO(cString::Format("Error smaller: LaneCenter %d Place2Be %d Steering %f error_val: %d error_sum: %d", m_LaneCenter.x, m_PlaceToBe.x, m_SteeringAngle, error, e_sum).GetPtr());
//		m_f32SteeringAngle = -30.0;
//	}
//
//
//	m_f64PT1ScaledError = m_i16Error * m_f64PT1InputFactor/*/ 5*/;
//	m_f32PT1SteeringOut = static_cast<float> ((m_f64PT1ScaledError + m_f64PT1Gain * m_f32PT1LastSteeringOut) / (1 + m_f64PT1Gain));
//	m_f32PT1NextSteeringOut = m_f32PT1SteeringOut;
//	m_f32PT1SteeringOut = m_f32PT1LastSteeringOut;
//	m_f32PT1LastSteeringOut = m_f32PT1NextSteeringOut;
//
//	if (m_f32PT1SteeringOut > 30.0f)
//	{
//		m_f32PT1SteeringOut = 30.0f;
//	}
//	else if (m_f32PT1SteeringOut < -30.0f)
//	{
//		m_f32PT1SteeringOut = -30.0f;
//	}
//
//	printf("Steering Angle %f \n", m_f32PT1SteeringOut);
//
//}
//
//static int m_nNearLine = 350;
//static int m_i16FarLaneWidthCalc = 0;
//static int m_i16LaneWidthMinFar = 120;
//static int m_i16LaneWidthMaxFar = 380;
//static int m_i16FarLaneWidth = 200;
//static int m_i16FarLaneCenter = 320;
//static int m_nBlindCounterFar = 0;
//static float m_f32AccelerateOut = 0;
//static float m_f32AccelerationMax = 1.5;
//static float m_f32AccelerationMin = 0.6;
//static int m_nAccelerationFarNearDiff = 40;
//static int m_nCurrentNearLine = m_nNearLine;
//static int m_nNearLineMaxOffset = -25;
//static bool m_bActive;
//
//void LongitudinalControl(Point *psPointsFar, unsigned int *pui8PointsCountFar)
//{
//	
//
//	if (*pui8PointsCountFar == 2)
//	{
//		m_i16FarLaneWidthCalc = psPointsFar[1].x - psPointsFar[0].x;
//
//		//LOG_INFO(cString::Format("FarLaneWidth calc: %i", FarLaneWidth_calc));
//
//		if ((m_i16FarLaneWidthCalc > m_i16LaneWidthMinFar) && (m_i16FarLaneWidthCalc < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneWidth = m_i16FarLaneWidthCalc;
//			m_i16FarLaneCenter = psPointsFar[0].x + (static_cast<int>(m_i16FarLaneWidth / 2));
//			m_nBlindCounterFar = 0;
//		}
//
//		//LOG_INFO(cString::Format("FarLaneCenter: %i", FarLaneCenter));
//		//LOG_INFO(cString::Format("FarLaneWidth: %i", FarLaneWidth));
//	}
//	else if (*pui8PointsCountFar == 1)
//	{
//		int nFarCenterLeft = psPointsFar[0].x + (static_cast<int>(m_i16FarLaneWidth / 2));
//
//		int nFarCenterRight = psPointsFar[0].x - (static_cast<int>(m_i16FarLaneWidth / 2));
//
//		if (abs(nFarCenterRight - m_i16FarLaneCenter) < MAX_DEVIATION)
//		{
//			m_i16FarLaneCenter = nFarCenterRight;
//			m_nBlindCounterFar = 0;
//		}
//		else if (abs(nFarCenterLeft - m_i16FarLaneCenter) < MAX_DEVIATION)
//		{
//			m_i16FarLaneCenter = nFarCenterLeft;
//			m_nBlindCounterFar = 0;
//		}
//
//	}
//	else if (*pui8PointsCountFar == 3)
//	{
//		int i16FarWidth1 = psPointsFar[1].x - psPointsFar[0].x;
//		int i16FarWidth2 = psPointsFar[2].x - psPointsFar[1].x;
//
//		if ((i16FarWidth2 > m_i16LaneWidthMinFar) && (i16FarWidth2 < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneCenter = psPointsFar[1].x + (static_cast<int>(i16FarWidth2 / 2));
//			m_nBlindCounterFar = 0;
//		}
//		else if ((i16FarWidth1 > m_i16LaneWidthMinFar) && (i16FarWidth1 < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneCenter = psPointsFar[0].x + (static_cast<int>(i16FarWidth1 / 2));
//			m_nBlindCounterFar = 0;
//		}
//	}
//	else
//	{    // If no or more than 3 points are found, it is supposed that the car is located in a curve, looking anywhere. 
//		// increase the blind counter
//		m_nBlindCounterFar++;
//	}
//
//	if (m_nBlindCounterFar > 30)
//	{
//		// For this reason the variable Lanecenter is set to 0, since this will 
//		// cause the car to continue driving at m_AccelerationMin.
//		m_i16FarLaneCenter = 0;
//	}
//
//
//	if (m_nBlindCounter > 50)
//	{
//		m_f32AccelerateOut = 0.0;        // If the car can't find lanes anymore --> stop
//	}
//	else
//	{
//		//        tInt16 i16Difference = abs(m_i16FarLaneCenter - m_sPlaceToBe.x);
//		int i16Difference = abs(m_i16FarLaneCenter - m_sLaneCenterNear.x);
//		float f32AccelerationDifference = m_f32AccelerationMax - m_f32AccelerationMin;
//		float f32Factor = (m_nAccelerationFarNearDiff - i16Difference) > 0 ? (m_nAccelerationFarNearDiff - i16Difference) / (float)m_nAccelerationFarNearDiff : 0.0f;
//		m_f32AccelerateOut = m_f32AccelerationMin + (f32Factor * f32AccelerationDifference);
//		// Adjust current near line according to absolute output acceleration (speed) 
//		// higher speed results in farther distance => look ahead!
//		f32Factor = m_f32AccelerateOut / 100;
//		m_nCurrentNearLine = (int)(f32Factor * m_nNearLineMaxOffset) + m_nNearLine;
//	}
//	
//	if (m_bActive)
//	{
//		// transmit acceleration out only if the "global" active flag is set to true
//		//TransmitAcceleration(m_f32AccelerateOut, tsInputTime);        // Send data to output pin
//	}
//	//else if (m_hEmergencyStopTimerNegative == NULL &&
//	//	m_hEmergencyStopTimerZero == NULL &&
//	//	//m_hEmergencyStopTimerResume == NULL && 
//	//	m_hStopTimerNegative == NULL &&
//	//	m_hStopTimerZero == NULL)
//	//{
//	//	TransmitAcceleration(0.0f, tsInputTime);
//	//}
//	printf("Acceleration %f \n", m_f32AccelerateOut);
//
//}
//
//Mat ApplyClahe(Mat gray_img, int limit, String name)
//{
//	Ptr<CLAHE> clahe = createCLAHE();
//	clahe->setClipLimit(limit);
//
//	Mat dst;
//	clahe->apply(gray_img, dst);
//
//	imshow(name, dst);
//
//	return dst;
//}
//
//
//void ApplyLaneDetection()
//{
//	m_f64PT1Gain = m_f64PT1Tau / m_f64PT1Sample;
//	m_sPlaceToBe.x = 320 + 15;
//	m_sLaneCenterNear.x = 320 + 15;
//
//	int m_nNearLine = 350;
//	int m_nCurrentNearLine;
//	int nWidth = 640;
//	int nHeight = 480;
//	cv::Mat m_matGreyNear;
//	cv::Mat m_matGreyThreshNear;
//	cv::Mat m_matLineCannyNear;
//	int m_nThresholdValue = 120;
//
//	m_nCurrentNearLine = m_nNearLine;
//	Mat image;
//	for (int x = 0; x < 1; x++)
//	{
//	
//		for (int count = 0; count < 200; count++)
//		{
//
//			// String imgPath = "C:\\Users\\Amanullah Tariq\\Documents\\Visual Studio 2013\\Projects\\LaneDetection\\LaneDetection\\Data\\image" + to_string(count) + ".png";
//			String imgPath = "G:\\Study stuff\\Semester 3\\AADC\\Work\\images\\good\\" + to_string(count) + ".png";
//			image = imread(imgPath, IMREAD_COLOR); // Read the file
//			showImage(image, "Original Image");
//
//			Mat imgH;
//			//image.convertTo(imgH, -1, 0, 0);
//			image = image - Scalar(100, 100, 100);
//			showImage(image, "Increase Brightness near imagee");
//
//			
//#pragma region Search Near Line
//
//			// Transform nearfield (Nearfield is used for lateral control)
//			Mat matImageCutNear = image(cv::Range(m_nCurrentNearLine - 20, m_nCurrentNearLine + 20), cv::Range(0, nWidth)).clone(); //Cut Image    
//			//showImage(matImageCutNear, "Near Cut Image");
//			
//
//
//			//SimplestCB(image, matImageCutNear, 1);
//			//showImage(matImageCutNear, "Balanced Image");
//			//GaussianBlur(matImageCutNear, matImageCutNear, Size(11, 11), 0, 0, BORDER_ISOLATED); // Filter
//			//showImage(matImageCutNear, "Guassian Blur Image");
//
//
//			cvtColor(matImageCutNear, m_matGreyNear, CV_RGB2GRAY);// Grey Image
//			//showImage(m_matGreyNear, "Grey Image");
//			threshold(m_matGreyNear, m_matGreyThreshNear, m_nThresholdValue, 500, THRESH_BINARY);// Generate Binary Image
//			//showImage(m_matGreyThreshNear, "Generate Binary Image");
//
//			Canny(m_matGreyThreshNear, m_matLineCannyNear, 0, 2, 3, false);// Detect Edges
//			showImage(m_matLineCannyNear, "NEAR Canny Image");
//			Point m_asAllpointsNear[640];
//			unsigned int m_ui8NearPointsCount;
//			Search(m_asAllpointsNear, &m_ui8NearPointsCount, 40, m_matLineCannyNear);
//
//#pragma endregion Search Near Line
//
//#pragma region Search Far
//			// search points within the canny near
//			//	Point      m_asAllpointsNear[640];
//			//unsigned int      m_ui8NearPointsCount;
//			Point      m_asAllpointsFar[640];
//			unsigned int      m_ui8FarPointsCount;
//			int m_nFarLine = 300;
//			cv::Mat m_matGreyFar;
//			cv::Mat m_matGreyThreshFar;
//			cv::Mat m_matLineCannyFar;
//
//			// Transform farfield (farfield is used for longitudinal control)
//			Mat matImageCutFar = image(cv::Range(m_nFarLine - 20, m_nFarLine + 20), cv::Range(0, nWidth)).clone(); //Cut Image   
//			//showImage(matImageCutFar, "Far Cut Image");
//			//SimplestCB(matImageCutFar, matImageCutFar, 0.1);
//			//showImage(matImageCutFar, "Far Balanced Image");
//			//GaussianBlur(matImageCutFar, matImageCutFar, Size(11, 11), 0, 0, BORDER_TRANSPARENT); // Filter
//			//showImage(matImageCutFar, "FAR Guassian Blur Image");
//
//			cvtColor(matImageCutFar, m_matGreyFar, CV_RGB2GRAY);// Grey Image
//			//showImage(m_matGreyFar, "FAR Grey Image");
//			threshold(m_matGreyFar, m_matGreyThreshFar, m_nThresholdValue, 500, THRESH_BINARY);// Generate Binary Image
//			//showImage(m_matGreyThreshFar, "Far Generate Binary Image");
//
//			Canny(m_matGreyThreshFar, m_matLineCannyFar, 0, 2, 3, false);// Detect Edges
//			showImage(m_matLineCannyFar, "FAR Canny Image");
//
//
//			// search points within the canny far
//			Search(m_asAllpointsFar, &m_ui8FarPointsCount, 40, m_matLineCannyFar);
//
//			LateralControl(m_asAllpointsNear, &m_ui8NearPointsCount);
//			LongitudinalControl(m_asAllpointsFar, &m_ui8FarPointsCount);
//#pragma endregion Search Far
//			 // Wait for a keystroke in the window
//
//			waitKey(100);
//		}
//	}
//}
//
///**
//* @function on_trackbar
//* @brief Callback for trackbar
//*/
//
//const int alpha_slider_max = 100;
//int alpha_slider;
//double alpha;
//double beta;
//
//const int limit_max = 4;
//int limit_slider;
///// Matrices to store images
//Mat src1;
//Mat src2;
//Mat dst;
//int limit;
//
//void on_trackbar(int, void*)
//{
//	alpha = (double)alpha_slider / alpha_slider_max;
//	beta = (1.0 - alpha);
//
//	addWeighted(src1, alpha, src2, beta, 0.0, dst);
//
//	imshow("Linear Blend", dst);
//}
//
//void ApplyClahe(int, void*)
//{
//	limit  = limit_slider;
//	ApplyClahe(src1, limit, "Limit Check");
//}
//
//
////
////void ApplySimplestCB(int, void*)
////{
////	 = limit_slider;
////	
////	SimplestCB(src1, src1, 5.0);
////}
////
////void ApplyIPM()
////{
////	// Images
////
////	
////
////	Mat inputImgGray;
////	Mat outputImg;
////
////	/*if (_argc != 2)
////	{
////		cout << "Usage: ipm.exe <videofile>" << endl;
////		return 1;
////	}*/
////
////	// Video
////	/*string videoFileName = _argv[1];
////	cv::VideoCapture video;*/
////
////	// Show video information
////	int width = 0, height = 0, fps = 0, fourcc = 0;
////	width = 640;
////	height = 480;
////
////
////	// The 4-points at the input image	
////	vector<Point2f> origPoints;
////	origPoints.push_back(Point2f(0, height));
////	origPoints.push_back(Point2f(width, height));
////	origPoints.push_back(Point2f(width / 2 + 30, 140));
////	origPoints.push_back(Point2f(width / 2 - 50, 140));
////
////	// The 4-points correspondences in the destination image
////	vector<Point2f> dstPoints;
////	dstPoints.push_back(Point2f(0, height));
////	dstPoints.push_back(Point2f(width, height));
////	dstPoints.push_back(Point2f(width, 0));
////	dstPoints.push_back(Point2f(0, 0));
////
////	// IPM object
////	IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);
////
////	// Main loop
////	for (int i = 0;i < 1000; i++)
////	{
////		String imgPath = "G:\\Study stuff\\Semester 3\\AADC\\images\\dark0\\" + to_string(i) +".png";
////		Mat  inputImg = imread(imgPath, IMREAD_COLOR); // Read the file
////
////		if (inputImg.empty())
////			break;
////
////		// Color Conversion
////		if (inputImg.channels() == 3)
////			cvtColor(inputImg, inputImgGray, CV_BGR2GRAY);
////		else
////			inputImg.copyTo(inputImgGray);
////
////		// Process
////		clock_t begin = clock();
////		ipm.applyHomography(inputImg, outputImg);
////		ipm.drawPoints(origPoints, inputImg);
////
////		// View		
////		imshow("Input", inputImg);
////		imshow("Output", outputImg);
////		waitKey(1);
////	}
////}
//
//#pragma endregion#pragma region Old Code
///// perform the Simplest Color Balancing algorithm
//void SimplestCB(Mat& in, Mat& out, float percent) {
//	assert(in.channels() == 3);
//	assert(percent > 0 && percent < 100);
//
//	float half_percent = percent / 200.0f;
//
//	vector<Mat> tmpsplit; split(in, tmpsplit);
//	for (int i = 0; i<3; i++) {
//		//find the low and high precentile values (based on the input percentile)
//		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
//		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
//		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
//		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
//		//cout << lowval << " " << highval << endl;
//
//		//saturate below the low percentile and above the high percentile
//		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
//		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);
//
//		//scale the channel
//		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
//	}
//	merge(tmpsplit, out);
//}
//
//void showImage(String imgPath, String windowName)
//{
//	Mat image;
//	//String imgPath = "C:\\Users\\Amanullah Tariq\\Documents\\Visual Studio 2013\\Projects\\LaneDetection\\LaneDetection\\Data\\image1.jpg";
//	image = imread(imgPath, IMREAD_COLOR); // Read the file
//
//	if (!image.data) // Check for invalid input
//	{
//		cout << "Could not open or find the image" << std::endl;
//		//return -1;
//	}
//
//	namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window for display.
//	imshow(windowName, image); // Show our image inside it.
//}
//
//void showImage(Mat img, String windowName)
//{
//	namedWindow(windowName, WINDOW_AUTOSIZE); // Create a window for display.
//	imshow(windowName, img); // Show our image inside it.
//}
//
//void Search(Point *psPoints, unsigned  int *pui8PointsCount, unsigned  int ui8Limit, cv::Mat matCannyImg)
//{
//	//LOG_INFO("cLaneTracking::Search Called");
//	/*A_UTIL_ASSERT(NULL != psPoints);
//	A_UTIL_ASSERT(NULL != pui8PointsCount);*/
//
//	*pui8PointsCount = 0;
//	int nColumnLast = -1;
//	Size szCannySize = matCannyImg.size();
//
//	for (int nColumn = 0; nColumn < szCannySize.width; nColumn++)
//	{
//		if (matCannyImg.at<uchar>(szCannySize.height / 2, nColumn) != 0)
//		{
//			if (abs(nColumnLast - nColumn) >= 2 && abs(nColumnLast - nColumn) < ui8Limit && nColumnLast != -1)
//			{
//				psPoints[*pui8PointsCount].x = static_cast<int>(nColumn - (abs(nColumnLast - nColumn) / 2));
//				(*pui8PointsCount)++;
//				nColumnLast = -1;
//			}
//			else
//			{
//				nColumnLast = nColumn;
//			}
//		}
//	}
//
//	
//}
//
//#define MAX_DEVIATION 150
//static int m_nBlindCounter = 0;
//static Point m_sPlaceToBe;
//static int m_i16Error;
//static int m_i16ErrorSum = 0;
//static float m_f32SteeringAngle = 0.0f;
//static float m_f64Kp = 0.08;
//
//static float m_f64Ki = 0.1445;
//static float m_f64Kd = 0.01238;
//static int m_i16ErrorOld = 0;
//static float m_f64PT1ScaledError;
//static float m_f64PT1InputFactor = 0.1;
//static float m_f32PT1SteeringOut = 0.0f;
//static float m_f64PT1Gain;
//static float m_f32PT1LastSteeringOut = 0.0f;
//static float m_f32PT1NextSteeringOut;
//static float m_f64PT1Tau = 0.1;
//static float m_f64PT1Sample = 0.9;
//static float m_f32Ts = 0.033f;
//static int   m_i16LaneWidth = 390;
//static int m_i16LaneWidthMinNear = 320;
//static int m_i16LaneWidthMaxNear = 430;
//
//static int        m_nCenterFromLeft = 0;
//static int        m_nCenterFromRight = 0;
//
//static Point      m_sLaneCenterNear;
//
//void LateralControl(Point *psPoints, unsigned int *pui8PointsCount)
//{
//	
//	//m_f32PT1LastSteeringOut = 0.0f;
//	//m_f32PT1SteeringOut = 0.0f;
//
//	if (*pui8PointsCount == 2)
//	{
//		int i16LaneWidthCalc = psPoints[1].x - psPoints[0].x;
//		//LOG_INFO(cString::Format("Point0: %i, Point1: %i, LaneWidth calc: %i", points[0].x, points[1].x, LaneWidth_calc));
//		int nLaneCenterCalc = psPoints[0].x + (static_cast<int>(i16LaneWidthCalc / 2));
//
//		if ((i16LaneWidthCalc > m_i16LaneWidthMinNear) && (i16LaneWidthCalc < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16LaneWidthCalc;
//			m_sLaneCenterNear.x = nLaneCenterCalc;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//
//	else if (*pui8PointsCount == 1)
//	{
//		// If just one point is found the lane center shall be calculated based on this single point and the half of the previously calculated lane width
//		// If the right point of the lane was found LaneWidth/2 must be subtracted from the position of the point, if the left point was found LaneWidth/2
//		// must be added. Whether to add or to subtract LaneWidth/2 is determined by comparing the calculated new LaneCenter with the old LaneCenter.
//
//		m_nCenterFromLeft = psPoints[0].x + (static_cast<int>(m_i16LaneWidth / 2));
//
//		m_nCenterFromRight = psPoints[0].x - (static_cast<int>(m_i16LaneWidth / 2));
//
//		if (abs(m_nCenterFromLeft - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_sLaneCenterNear.x = m_nCenterFromLeft;
//			m_nBlindCounter = 0;
//		}
//		else if (abs(m_nCenterFromRight - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_sLaneCenterNear.x = m_nCenterFromRight;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//	else if (*pui8PointsCount == 3)
//	{
//		// If three points are found the correct lane is chosen according to the calculated lanewidth
//		int i16Width1 = psPoints[2].x - psPoints[1].x;
//		int i16Width2 = psPoints[1].x - psPoints[0].x;
//		int nLaneCenterCalc1 = psPoints[1].x + (static_cast<int>(i16Width1 / 2));
//		int nLaneCenterCalc2 = psPoints[0].x + (static_cast<int>(i16Width2 / 2));
//
//		if ((i16Width1 > m_i16LaneWidthMinNear) && (i16Width1 < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc1 - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16Width1;
//			m_sLaneCenterNear.x = nLaneCenterCalc1;
//			m_nBlindCounter = 0;
//		}
//		else if ((i16Width2 > m_i16LaneWidthMinNear) && (i16Width2 < m_i16LaneWidthMaxNear) && abs(nLaneCenterCalc2 - m_sLaneCenterNear.x) < MAX_DEVIATION)
//		{
//			m_i16LaneWidth = i16Width2;
//			m_sLaneCenterNear.x = nLaneCenterCalc2;
//			m_nBlindCounter = 0;
//		}
//		else
//		{
//			m_nBlindCounter++;
//		}
//	}
//	else
//	{
//		// If more than three or no point (=lane) is detected, continue with the previously calculated values
//		// Due to safety reasons a counter is implemented, that will cause the car to stop, if the car did not find any lane for more than X times consecutively (see parameter in logitudinal control)
//
//		m_nBlindCounter++;
//	}
//	//PID - Controller for Steering Angle
//	m_i16Error = m_sLaneCenterNear.x - m_sPlaceToBe.x;
//	m_i16ErrorSum = m_i16ErrorSum + m_i16Error;
//	m_f32SteeringAngle = static_cast<float>(m_f64Kp*m_i16Error + m_f64Ki*m_f32Ts*m_i16ErrorSum + (m_f64Kd*(m_i16Error - m_i16ErrorOld)) / m_f32Ts);
//	m_i16ErrorOld = m_i16Error;
//
//	// Limitation of Steering Angle
//	if (m_f32SteeringAngle > 30)
//	{
//		//      LOG_INFO(cString::Format("Error greater: LaneCenter %d Place2Be %d Steering %f error_val: %d error_sum: %d", m_LaneCenter.x, m_PlaceToBe.x, m_SteeringAngle, error, e_sum).GetPtr());
//		m_f32SteeringAngle = 30.0;
//	}
//	else if (m_f32SteeringAngle < -30)
//	{
//		//      LOG_INFO(cString::Format("Error smaller: LaneCenter %d Place2Be %d Steering %f error_val: %d error_sum: %d", m_LaneCenter.x, m_PlaceToBe.x, m_SteeringAngle, error, e_sum).GetPtr());
//		m_f32SteeringAngle = -30.0;
//	}
//
//
//	m_f64PT1ScaledError = m_i16Error * m_f64PT1InputFactor/*/ 5*/;
//	m_f32PT1SteeringOut = static_cast<float> ((m_f64PT1ScaledError + m_f64PT1Gain * m_f32PT1LastSteeringOut) / (1 + m_f64PT1Gain));
//	m_f32PT1NextSteeringOut = m_f32PT1SteeringOut;
//	m_f32PT1SteeringOut = m_f32PT1LastSteeringOut;
//	m_f32PT1LastSteeringOut = m_f32PT1NextSteeringOut;
//
//	if (m_f32PT1SteeringOut > 30.0f)
//	{
//		m_f32PT1SteeringOut = 30.0f;
//	}
//	else if (m_f32PT1SteeringOut < -30.0f)
//	{
//		m_f32PT1SteeringOut = -30.0f;
//	}
//
//	printf("Steering Angle %f \n", m_f32PT1SteeringOut);
//
//}
//
//static int m_nNearLine = 350;
//static int m_i16FarLaneWidthCalc = 0;
//static int m_i16LaneWidthMinFar = 120;
//static int m_i16LaneWidthMaxFar = 380;
//static int m_i16FarLaneWidth = 200;
//static int m_i16FarLaneCenter = 320;
//static int m_nBlindCounterFar = 0;
//static float m_f32AccelerateOut = 0;
//static float m_f32AccelerationMax = 1.5;
//static float m_f32AccelerationMin = 0.6;
//static int m_nAccelerationFarNearDiff = 40;
//static int m_nCurrentNearLine = m_nNearLine;
//static int m_nNearLineMaxOffset = -25;
//static bool m_bActive;
//
//void LongitudinalControl(Point *psPointsFar, unsigned int *pui8PointsCountFar)
//{
//	
//
//	if (*pui8PointsCountFar == 2)
//	{
//		m_i16FarLaneWidthCalc = psPointsFar[1].x - psPointsFar[0].x;
//
//		//LOG_INFO(cString::Format("FarLaneWidth calc: %i", FarLaneWidth_calc));
//
//		if ((m_i16FarLaneWidthCalc > m_i16LaneWidthMinFar) && (m_i16FarLaneWidthCalc < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneWidth = m_i16FarLaneWidthCalc;
//			m_i16FarLaneCenter = psPointsFar[0].x + (static_cast<int>(m_i16FarLaneWidth / 2));
//			m_nBlindCounterFar = 0;
//		}
//
//		//LOG_INFO(cString::Format("FarLaneCenter: %i", FarLaneCenter));
//		//LOG_INFO(cString::Format("FarLaneWidth: %i", FarLaneWidth));
//	}
//	else if (*pui8PointsCountFar == 1)
//	{
//		int nFarCenterLeft = psPointsFar[0].x + (static_cast<int>(m_i16FarLaneWidth / 2));
//
//		int nFarCenterRight = psPointsFar[0].x - (static_cast<int>(m_i16FarLaneWidth / 2));
//
//		if (abs(nFarCenterRight - m_i16FarLaneCenter) < MAX_DEVIATION)
//		{
//			m_i16FarLaneCenter = nFarCenterRight;
//			m_nBlindCounterFar = 0;
//		}
//		else if (abs(nFarCenterLeft - m_i16FarLaneCenter) < MAX_DEVIATION)
//		{
//			m_i16FarLaneCenter = nFarCenterLeft;
//			m_nBlindCounterFar = 0;
//		}
//
//	}
//	else if (*pui8PointsCountFar == 3)
//	{
//		int i16FarWidth1 = psPointsFar[1].x - psPointsFar[0].x;
//		int i16FarWidth2 = psPointsFar[2].x - psPointsFar[1].x;
//
//		if ((i16FarWidth2 > m_i16LaneWidthMinFar) && (i16FarWidth2 < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneCenter = psPointsFar[1].x + (static_cast<int>(i16FarWidth2 / 2));
//			m_nBlindCounterFar = 0;
//		}
//		else if ((i16FarWidth1 > m_i16LaneWidthMinFar) && (i16FarWidth1 < m_i16LaneWidthMaxFar))
//		{
//			m_i16FarLaneCenter = psPointsFar[0].x + (static_cast<int>(i16FarWidth1 / 2));
//			m_nBlindCounterFar = 0;
//		}
//	}
//	else
//	{    // If no or more than 3 points are found, it is supposed that the car is located in a curve, looking anywhere. 
//		// increase the blind counter
//		m_nBlindCounterFar++;
//	}
//
//	if (m_nBlindCounterFar > 30)
//	{
//		// For this reason the variable Lanecenter is set to 0, since this will 
//		// cause the car to continue driving at m_AccelerationMin.
//		m_i16FarLaneCenter = 0;
//	}
//
//
//	if (m_nBlindCounter > 50)
//	{
//		m_f32AccelerateOut = 0.0;        // If the car can't find lanes anymore --> stop
//	}
//	else
//	{
//		//        tInt16 i16Difference = abs(m_i16FarLaneCenter - m_sPlaceToBe.x);
//		int i16Difference = abs(m_i16FarLaneCenter - m_sLaneCenterNear.x);
//		float f32AccelerationDifference = m_f32AccelerationMax - m_f32AccelerationMin;
//		float f32Factor = (m_nAccelerationFarNearDiff - i16Difference) > 0 ? (m_nAccelerationFarNearDiff - i16Difference) / (float)m_nAccelerationFarNearDiff : 0.0f;
//		m_f32AccelerateOut = m_f32AccelerationMin + (f32Factor * f32AccelerationDifference);
//		// Adjust current near line according to absolute output acceleration (speed) 
//		// higher speed results in farther distance => look ahead!
//		f32Factor = m_f32AccelerateOut / 100;
//		m_nCurrentNearLine = (int)(f32Factor * m_nNearLineMaxOffset) + m_nNearLine;
//	}
//	
//	if (m_bActive)
//	{
//		// transmit acceleration out only if the "global" active flag is set to true
//		//TransmitAcceleration(m_f32AccelerateOut, tsInputTime);        // Send data to output pin
//	}
//	//else if (m_hEmergencyStopTimerNegative == NULL &&
//	//	m_hEmergencyStopTimerZero == NULL &&
//	//	//m_hEmergencyStopTimerResume == NULL && 
//	//	m_hStopTimerNegative == NULL &&
//	//	m_hStopTimerZero == NULL)
//	//{
//	//	TransmitAcceleration(0.0f, tsInputTime);
//	//}
//	printf("Acceleration %f \n", m_f32AccelerateOut);
//
//}
//
//Mat ApplyClahe(Mat gray_img, int limit, String name)
//{
//	Ptr<CLAHE> clahe = createCLAHE();
//	clahe->setClipLimit(limit);
//
//	Mat dst;
//	clahe->apply(gray_img, dst);
//
//	imshow(name, dst);
//
//	return dst;
//}
//
//
//void ApplyLaneDetection()
//{
//	m_f64PT1Gain = m_f64PT1Tau / m_f64PT1Sample;
//	m_sPlaceToBe.x = 320 + 15;
//	m_sLaneCenterNear.x = 320 + 15;
//
//	int m_nNearLine = 350;
//	int m_nCurrentNearLine;
//	int nWidth = 640;
//	int nHeight = 480;
//	cv::Mat m_matGreyNear;
//	cv::Mat m_matGreyThreshNear;
//	cv::Mat m_matLineCannyNear;
//	int m_nThresholdValue = 120;
//
//	m_nCurrentNearLine = m_nNearLine;
//	Mat image;
//	for (int x = 0; x < 1; x++)
//	{
//	
//		for (int count = 0; count < 200; count++)
//		{
//
//			// String imgPath = "C:\\Users\\Amanullah Tariq\\Documents\\Visual Studio 2013\\Projects\\LaneDetection\\LaneDetection\\Data\\image" + to_string(count) + ".png";
//			String imgPath = "G:\\Study stuff\\Semester 3\\AADC\\Work\\images\\good\\" + to_string(count) + ".png";
//			image = imread(imgPath, IMREAD_COLOR); // Read the file
//			showImage(image, "Original Image");
//
//			Mat imgH;
//			//image.convertTo(imgH, -1, 0, 0);
//			image = image - Scalar(100, 100, 100);
//			showImage(image, "Increase Brightness near imagee");
//
//			
//#pragma region Search Near Line
//
//			// Transform nearfield (Nearfield is used for lateral control)
//			Mat matImageCutNear = image(cv::Range(m_nCurrentNearLine - 20, m_nCurrentNearLine + 20), cv::Range(0, nWidth)).clone(); //Cut Image    
//			//showImage(matImageCutNear, "Near Cut Image");
//			
//
//
//			//SimplestCB(image, matImageCutNear, 1);
//			//showImage(matImageCutNear, "Balanced Image");
//			//GaussianBlur(matImageCutNear, matImageCutNear, Size(11, 11), 0, 0, BORDER_ISOLATED); // Filter
//			//showImage(matImageCutNear, "Guassian Blur Image");
//
//
//			cvtColor(matImageCutNear, m_matGreyNear, CV_RGB2GRAY);// Grey Image
//			//showImage(m_matGreyNear, "Grey Image");
//			threshold(m_matGreyNear, m_matGreyThreshNear, m_nThresholdValue, 500, THRESH_BINARY);// Generate Binary Image
//			//showImage(m_matGreyThreshNear, "Generate Binary Image");
//
//			Canny(m_matGreyThreshNear, m_matLineCannyNear, 0, 2, 3, false);// Detect Edges
//			showImage(m_matLineCannyNear, "NEAR Canny Image");
//			Point m_asAllpointsNear[640];
//			unsigned int m_ui8NearPointsCount;
//			Search(m_asAllpointsNear, &m_ui8NearPointsCount, 40, m_matLineCannyNear);
//
//#pragma endregion Search Near Line
//
//#pragma region Search Far
//			// search points within the canny near
//			//	Point      m_asAllpointsNear[640];
//			//unsigned int      m_ui8NearPointsCount;
//			Point      m_asAllpointsFar[640];
//			unsigned int      m_ui8FarPointsCount;
//			int m_nFarLine = 300;
//			cv::Mat m_matGreyFar;
//			cv::Mat m_matGreyThreshFar;
//			cv::Mat m_matLineCannyFar;
//
//			// Transform farfield (farfield is used for longitudinal control)
//			Mat matImageCutFar = image(cv::Range(m_nFarLine - 20, m_nFarLine + 20), cv::Range(0, nWidth)).clone(); //Cut Image   
//			//showImage(matImageCutFar, "Far Cut Image");
//			//SimplestCB(matImageCutFar, matImageCutFar, 0.1);
//			//showImage(matImageCutFar, "Far Balanced Image");
//			//GaussianBlur(matImageCutFar, matImageCutFar, Size(11, 11), 0, 0, BORDER_TRANSPARENT); // Filter
//			//showImage(matImageCutFar, "FAR Guassian Blur Image");
//
//			cvtColor(matImageCutFar, m_matGreyFar, CV_RGB2GRAY);// Grey Image
//			//showImage(m_matGreyFar, "FAR Grey Image");
//			threshold(m_matGreyFar, m_matGreyThreshFar, m_nThresholdValue, 500, THRESH_BINARY);// Generate Binary Image
//			//showImage(m_matGreyThreshFar, "Far Generate Binary Image");
//
//			Canny(m_matGreyThreshFar, m_matLineCannyFar, 0, 2, 3, false);// Detect Edges
//			showImage(m_matLineCannyFar, "FAR Canny Image");
//
//
//			// search points within the canny far
//			Search(m_asAllpointsFar, &m_ui8FarPointsCount, 40, m_matLineCannyFar);
//
//			LateralControl(m_asAllpointsNear, &m_ui8NearPointsCount);
//			LongitudinalControl(m_asAllpointsFar, &m_ui8FarPointsCount);
//#pragma endregion Search Far
//			 // Wait for a keystroke in the window
//
//			waitKey(100);
//		}
//	}
//}
//
///**
//* @function on_trackbar
//* @brief Callback for trackbar
//*/
//
//const int alpha_slider_max = 100;
//int alpha_slider;
//double alpha;
//double beta;
//
//const int limit_max = 4;
//int limit_slider;
///// Matrices to store images
//Mat src1;
//Mat src2;
//Mat dst;
//int limit;
//
//void on_trackbar(int, void*)
//{
//	alpha = (double)alpha_slider / alpha_slider_max;
//	beta = (1.0 - alpha);
//
//	addWeighted(src1, alpha, src2, beta, 0.0, dst);
//
//	imshow("Linear Blend", dst);
//}
//
//void ApplyClahe(int, void*)
//{
//	limit  = limit_slider;
//	ApplyClahe(src1, limit, "Limit Check");
//}
//
//
////
////void ApplySimplestCB(int, void*)
////{
////	 = limit_slider;
////	
////	SimplestCB(src1, src1, 5.0);
////}
////
////void ApplyIPM()
////{
////	// Images
////
////	
////
////	Mat inputImgGray;
////	Mat outputImg;
////
////	/*if (_argc != 2)
////	{
////		cout << "Usage: ipm.exe <videofile>" << endl;
////		return 1;
////	}*/
////
////	// Video
////	/*string videoFileName = _argv[1];
////	cv::VideoCapture video;*/
////
////	// Show video information
////	int width = 0, height = 0, fps = 0, fourcc = 0;
////	width = 640;
////	height = 480;
////
////
////	// The 4-points at the input image	
////	vector<Point2f> origPoints;
////	origPoints.push_back(Point2f(0, height));
////	origPoints.push_back(Point2f(width, height));
////	origPoints.push_back(Point2f(width / 2 + 30, 140));
////	origPoints.push_back(Point2f(width / 2 - 50, 140));
////
////	// The 4-points correspondences in the destination image
////	vector<Point2f> dstPoints;
////	dstPoints.push_back(Point2f(0, height));
////	dstPoints.push_back(Point2f(width, height));
////	dstPoints.push_back(Point2f(width, 0));
////	dstPoints.push_back(Point2f(0, 0));
////
////	// IPM object
////	IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);
////
////	// Main loop
////	for (int i = 0;i < 1000; i++)
////	{
////		String imgPath = "G:\\Study stuff\\Semester 3\\AADC\\images\\dark0\\" + to_string(i) +".png";
////		Mat  inputImg = imread(imgPath, IMREAD_COLOR); // Read the file
////
////		if (inputImg.empty())
////			break;
////
////		// Color Conversion
////		if (inputImg.channels() == 3)
////			cvtColor(inputImg, inputImgGray, CV_BGR2GRAY);
////		else
////			inputImg.copyTo(inputImgGray);
////
////		// Process
////		clock_t begin = clock();
////		ipm.applyHomography(inputImg, outputImg);
////		ipm.drawPoints(origPoints, inputImg);
////
////		// View		
////		imshow("Input", inputImg);
////		imshow("Output", outputImg);
////		waitKey(1);
////	}
////}

//int main(int argc, const char * argv[])
//{
//
//	ApplyLaneDetection();
//
//	return 0;
//}
#pragma endregion

#pragma region IPM

//int main(int _argc, char** _argv)
//{
//	// Images
//	Mat inputImg, inputImgGray;
//	Mat outputImg;
//
//
//	// Show video information
//	int width = 640, height = 480, fps = 0, fourcc = 0;
//
//
//	// The 4-points at the input image	
//	vector<Point2f> origPoints;
//	origPoints.push_back(Point2f(0, height));
//	origPoints.push_back(Point2f(width + 150, height));
//	origPoints.push_back(Point2f(width / 2 + 180, 230));
//	origPoints.push_back(Point2f(width / 2 - 220, 230));
//
//	// The 4-points correspondences in the destination image
//	vector<Point2f> dstPoints;
//	dstPoints.push_back(Point2f(0, 240));
//	dstPoints.push_back(Point2f(320, 240));
//	dstPoints.push_back(Point2f(320, 0));
//	dstPoints.push_back(Point2f(0, 0));
//
//	// IPM object
//	IPM ipm(Size(width, height), Size(320, 240), origPoints, dstPoints);
//	string path;
//	string saveImg; 
//	for (int i = 0; i <= 3500; i++)
//	{
//		path = "G:\\Study stuff\\Semester 3\\AADC\\IMAGES\\straight1\\" + to_string(i) +".png";
//		
//		inputImg = imread(path, 1);
//		if (!inputImg.empty())
//		{
//			// Color Conversion
//			if (inputImg.channels() == 3)
//				cvtColor(inputImg, inputImgGray, CV_BGR2GRAY);
//			else
//				inputImg.copyTo(inputImgGray);
//
//			ipm.applyHomography(inputImgGray, outputImg);
//			ipm.drawPoints(origPoints, inputImgGray);
//
//			// View		
//			//imshow("Input", inputImgGray);
//			imshow("Output", outputImg);
//
//			//imwrite("inputImg.jpg", inputImgGray);
//			saveImg = "G:\\Study stuff\\Semester 3\\AADC\\IMAGES\\ipm_straight1\\" + to_string(i) + ".png";
//			imwrite(saveImg, outputImg);
//			waitKey(1);
//		}
//	}
//	return 0;
//}

#pragma endregion



//our sensitivity value to be used in the threshold() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the image to remove possible noise and
//increase the size of the object we are trying to track. (Much like dilate and erode)
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = { 0, 0 };
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0, 0, 0, 0);


//int to string helper function
string intToString(int number){

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed){
	//notice how we use the '&' operator for the cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed in this function which is then displayed in the main() function.
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);// retrieves external contours

	//if contours vector is not empty, we have found some objects
	if (contours.size()>0)objectDetected = true;
	else objectDetected = false;

	if (objectDetected){
		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		vector< vector<Point> > largestContourVec;
		largestContourVec.push_back(contours.at(contours.size() - 1));
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
		int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

		//update the objects positions by changing the 'theObject' array values
		theObject[0] = xpos, theObject[1] = ypos;
	}
	//make some temp x and y variables so we dont have to type out so much
	int x = theObject[0];
	int y = theObject[1];
	//draw some crosshairs on the object
	circle(cameraFeed, Point(x, y), 20, Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	line(cameraFeed, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	putText(cameraFeed, "Tracking object at (" + intToString(x) + "," + intToString(y) + ")", Point(x, y), 1, 1, Scalar(255, 0, 0), 2);



}
int main(){

	//some boolean variables for added functionality
	bool objectDetected = true;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = true;
	bool trackingEnabled = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat frame1, frame2;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1, grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.
	VideoCapture capture;

	while (1){

		//we can loop the video by re-opening the capture every time the video reaches its last frame

		capture.open("C:\\Users\\Amanullah Tariq\\Documents\\Visual Studio 2013\\Projects\\LaneDetection\\LaneDetection\\bouncingBall.avi");

		if (!capture.isOpened()){
			cout << "ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}

		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!
		while (capture.get(CV_CAP_PROP_POS_FRAMES)<capture.get(CV_CAP_PROP_FRAME_COUNT) - 1){

			//read first frame
			capture.read(frame1);
			//convert frame1 to gray scale for frame differencing

			//copy second frame

			//convert frame2 to gray scale for frame differencing

			//perform frame differencing with the sequential images. This will output an "intensity image"
			//do not confuse this with a threshold image, we will need to perform thresholding afterwards.

			//threshold intensity image at a given sensitivity value

			if (debugMode == true){
				//show the difference image and threshold image

			}
			else{
				//if not in debug mode, destroy the windows so we don't see them anymore

			}
			//use blur() to smooth the image, remove possible noise and
			//increase the size of the object we are trying to track. (Much like dilate and erode)

			//threshold again to obtain binary image from blur output

			if (debugMode == true){
				//show the threshold image after it's been "blurred"

			}
			else {
				//if not in debug mode, destroy the windows so we don't see them anymore

			}

			//if tracking enabled, search for contours in our thresholded image

			//show our captured frame
			imshow("Frame1", frame1);
			//check to see if a button has been pressed.
			//this 10ms delay is necessary for proper operation of this program
			//if removed, frames will not have enough time to referesh and a blank 
			//image will appear.
			switch (waitKey(10)){

			case 27: //'esc' key has been pressed, exit program.
				return 0;
			case 116: //'t' has been pressed. this will toggle tracking
				trackingEnabled = !trackingEnabled;
				if (trackingEnabled == false) cout << "Tracking disabled." << endl;
				else cout << "Tracking enabled." << endl;
				break;
			case 100: //'d' has been pressed. this will debug mode
				debugMode = !debugMode;
				if (debugMode == false) cout << "Debug mode disabled." << endl;
				else cout << "Debug mode enabled." << endl;
				break;
			case 112: //'p' has been pressed. this will pause/resume the code.
				pause = !pause;
				if (pause == true){
					cout << "Code paused, press 'p' again to resume" << endl;
					while (pause == true){
						//stay in this loop until 
						switch (waitKey()){
							//a switch statement inside a switch statement? Mind blown.
						case 112:
							//change pause back to false
							pause = false;
							cout << "Code resumed." << endl;
							break;
						}
					}
				}


			}


		}
		//release the capture before re-opening and looping again.
		capture.release();
	}

	return 0;

}