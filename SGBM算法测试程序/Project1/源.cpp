/******************************/
/*        立体匹配和测距        */
/******************************/

#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <stdio.h>

using namespace std;
using namespace cv;

const int imageWidth = 1920;                             //摄像头的分辨率  
const int imageHeight = 1024;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象


Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 4334.09568, 0, 959.50000,
	0, 4334.09568, 511.50000,
	0, 0, 1.0);
Mat distCoeffL = (Mat_<double>(5, 1) << 0.0, 0.0,0.0, 0.0, 0.0);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 4489.55770, 0, 801.86552,
	0, 4507.13412, 530.72579,
	0, 0, 1.0);
Mat distCoeffR = (Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

Mat T = (Mat_<double>(3, 1) << -518.97666, 01.20629,9.14632);//T平移向量
Mat rec = (Mat_<double>(3, 1) <<0.04345, -0.05236, -0.01810);//rec旋转向量
Mat R;//R 旋转矩阵


static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 16.0e4;
	FILE* fp = fopen(filename, "wt");
	printf("%d %d \n", mat.rows, mat.cols);
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			//printf( "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}


void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
	// color map  
	float max_val = 255.0f;
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int i = 0; i<8; i++)
		sum += map[i][3];

	float weights[8]; // relative   weights  
	float cumsum[8];  // cumulative weights  
	cumsum[0] = 0;
	for (int i = 0; i<7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;
	// for all pixels do  
	for (int v = 0; v<height_; v++) {
		for (int u = 0; u<width_; u++) {

			// get normalized value  
			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

			// find bin  
			int i;
			for (i = 0; i<7; i++)
				if (val<cumsum[i + 1])
					break;

			// compute red/green/blue values  
			float   w = 1.0 - (val - cumsum[i])*weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
			//rgb内存连续存放  
			disp.data[v*width_ * 3 + 3 * u + 0] = b;
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}

	  /*****立体匹配*****/
void stereo_match(int, void*)
{
	sgbm->setPreFilterCap(63);
	int sgbmWinSize = 2 * 0 + 5;//sgbmWinSize以及sgbm->setNumDisparities(416)里的参数还有sgbm->setUniquenessRatio(6);参数最重要，别的用opencv默认即可
	sgbm->setBlockSize(sgbmWinSize);
	int cn = rectifyImageL.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(416);
	sgbm->setUniquenessRatio(6);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(10);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);
    Mat disp, dispf, disp8;
	sgbm->compute(rectifyImageL, rectifyImageR, disp);
	//去黑边
	Mat img1p, img2p;
	copyMakeBorder(rectifyImageL, img1p, 0, 0, 416, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(rectifyImageR, img2p, 0, 0, 416, 0, IPL_BORDER_REPLICATE);
	dispf = disp.colRange(416, img2p.cols-416);

	dispf.convertTo(disp8, CV_8U, 255 / (416*16.));
	reprojectImageTo3D(dispf, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
	Mat color(dispf.size(), CV_8UC3);
	GenerateFalseMap(disp8, color);//转成彩图
	imshow("disparity", color);
	saveXYZ("xyz.xls", xyz);
}



/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}


/*****主函数*****/
int main()
{
	/*
	立体校正
	*/
	Rodrigues(rec, R); //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);

	/*
	读取图片
	*/
	rgbImageL = imread("left_cor.bmp", -1);//CV_LOAD_IMAGE_COLOR
	//cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
	rgbImageR = imread("right_cor.bmp", -1);
	//cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
	/*
	imshow("ImageL Before Rectify", rgbImageL);
	imshow("ImageR Before Rectify", rgbImageR);
	printf("YYY%d %d \n", rgbImageL.rows, rgbImageL.cols);*/
	/*
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(rgbImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);//INTER_LINEAR
	remap(rgbImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	/*
	把校正结果显示出来
	
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);
*/

	//显示在同一张图上
	Mat canvas;
	double sf;
	int w, h;
	sf = 700. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);   //注意通道

										//左图像画到画布上
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	/*
	立体匹配
	*/
	namedWindow("disparity", CV_WINDOW_NORMAL);
	/*
	// 创建SAD窗口 Trackbar
	createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
	// 创建视差唯一性百分比窗口 Trackbar
	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
	// 创建视差窗口 Trackbar
	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
	//鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)*/
	setMouseCallback("disparity", onMouse, 0);//disparity
	stereo_match(0, 0);

	waitKey(0);
	return 0;
}