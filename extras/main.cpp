#include "gmm_final.h"
#include <string>

int main()
{
	using namespace std;
	string imageFolderPath = "../test/test-baseline/input/";
	ostringstream oss;
	oss<<imageFolderPath+"in000001.jpg";
	cv::Mat im = cv::imread(oss.str());
    cv::Mat frame;
	cv::cvtColor(im, frame, cv::COLOR_BGR2GRAY);
	GMM GMM(frame);
	GMM.init(frame);
	for(int i = 1; i <= 1700; i++)
	{
		oss.str("");
		string sr = to_string(i);
		int le = (6-sr.length());
		if (le == 1) sr = "0";
		else if (le == 2) sr = "00";
		else if (le == 3) sr = "000";
		else if (le == 4) sr = "0000";
		else if (le == 5) sr = "00000";
		oss<<imageFolderPath<<"in"<<sr<<i<<".jpg";
		cv::Mat img = cv::imread(oss.str());
        cv::cvtColor(img,frame,cv::COLOR_BGR2GRAY);
		// blur(frame, frame, cv::Size(3,3));
		// cv::imshow("img", frame);
		oss.str("");
		oss<<"../test/test-baseline/results/gt"<<sr<<i<<".png";
		// cout << oss.str()<<endl;
		GMM.Calc(frame,oss.str());
	}
	return 0;
}
