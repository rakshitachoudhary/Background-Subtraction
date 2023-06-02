#include "gmm_final.h"
using namespace cv;
using namespace std;

#define pi 3.1415926535

static bool compare_fun(pair<int, float> p1, pair<int, float> p2){
	return p1.second > p2.second;
}

GMM::GMM(Mat img){
	height = img.rows;
	width = img.cols;
}

GMM::~GMM(){}

void GMM::init(cv::Mat img){
	for (int z=0; z<max_gmm; z++){
		weights[z].create(height,width, cv::DataType<float>::type);
		avg[z].create(height,width, cv::DataType<float>::type);
		std[z].create(height,width, cv::DataType<float>::type);
		differ[z].create(height,width, cv::DataType<float>::type);
	}
	for(int y = 0; y<height; y++){
		for(int x = 0; x<width; x++){
			for(int z = 0; z<max_gmm; z++){
				weights[z].at<float>(y,x) = 1/(float)max_gmm;
				avg[z].at<float>(y,x) = (float)(rand()%256);
				std[z].at<float>(y,x) = 6;
				differ[z].at<float>(y,x) = 0.0f;
			}
		}
	}
	fore_grnd.create(height,width,cv::DataType<uchar>::type);
	mask.create(height,width,cv::DataType<uchar>::type);
}

void GMM::Calc(cv::Mat img, string path)
{
	for(int y = 0; y<height; y++){
		for(int x = 0; x<width; x++){
			for(int z = 0; z<max_gmm; z++){
				differ[z].at<float>(y,x) = abs((float)img.at<uchar>(y,x) - avg[z].at<float>(y,x));
			}
		}
	}
	for(int y = 0; y<height; y++){
		for(int x = 0; x<width; x++){
			int idx = -1;
			float wt = FLT_MAX;
			bool found = false;
			float sum_wt = 0;
			uchar val = img.at<uchar>(y,x);

			for(int k = 0; k < max_gmm; k++){
				if(abs(differ[k].at<float>(y,x)) <= 2.5*std[k].at<float>(y,x)){
					found = true;
					float diff_sq = differ[k].at<float>(y,x)*differ[k].at<float>(y,x);
					double expr = (1/sqrt(2*pi*std[k].at<float>(y,x)))*exp(-0.5*diff_sq/std[k].at<float>(y,x));
					weights[k].at<float>(y,x) = (1-alpha)*weights[k].at<float>(y,x) + alpha;
					double p = alpha*expr;
					avg[k].at<float>(y,x) = (1-p)*avg[k].at<float>(y,x) + p*val;
					std[k].at<float>(y,x) = sqrtf((1-p)*(std[k].at<float>(y,x)*std[k].at<float>(y,x)) + p*((val - avg[k].at<float>(y,x))*(val - avg[k].at<float>(y,x))));
				}
				else{
					weights[k].at<float>(y,x) = (1-alpha)*weights[k].at<float>(y,x);
				}
				sum_wt+=weights[k].at<float>(y,x);
			}

			for(int k = 0; k < max_gmm; ++k){
				weights[k].at<float>(y,x)/=sum_wt;
				if(!found and weights[k].at<float>(y,x) < wt){
					idx = k;
					wt = weights[k].at<float>(y,x);
				}
			}
			
			if(!found){
				std[idx].at<float>(y,x) = 6;
				avg[idx].at<float>(y,x) = val;
			}
			
			pair<int, float> order[max_gmm];
			for(int k=0; k<max_gmm; k++){
				order[k].second = weights[k].at<float>(y,x)/std[k].at<float>(y,x);
				order[k].first = k;
			}
			std::sort(order,order+max_gmm,compare_fun);

			found = false;
			fore_grnd.at<uchar>(y,x) = 0;
			
			for (int k = 0; k<max_gmm; k++){
				if(!found and weights[order[k].first].at<float>(y,x)>=t){
					if(abs(differ[order[k].first].at<float>(y,x))<=2.5*std[order[k].first].at<float>(y,x)){
						fore_grnd.at<uchar>(y,x) = 0;
						found = true;
					}
					else fore_grnd.at<uchar>(y,x) = val;
				}
				else break;
			}
			if (!found) mask.at<uchar>(y,x) = 255;
			else mask.at<uchar>(y,x) = 0;
		}
	}
	imshow("fg", mask);
	waitKey(10);
	bool ch = imwrite(path, mask);
	if (!ch) std::cout<<"write failed\n";
}

