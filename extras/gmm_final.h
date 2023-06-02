#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#define num 2


class GMM
{
private:
	int max_gmm = num;
	float alpha = 0.0005f;
	float t = 0.3f;
	int width;
	int height;
	cv::Mat fore_grnd;
	cv::Mat mask;
	cv::Mat weights[num];
	cv::Mat avg[num];
	cv::Mat std[num];
	cv::Mat differ[num];
	
public:
	GMM(cv::Mat first_frame);
	~GMM();
	void init(cv::Mat img);
	void Calc(cv::Mat frame, std::string no);
};

