#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2\opencv.hpp>

#include <stdio.h>
#include <ctime>
#include <math.h>

//using namespace cv;
using namespace std;




__global__ void FindGradKernel(char *mat_grey, char *mat_conture, int width, int height)
{
	int y = blockIdx.x;
	int x = threadIdx.x;

	char tmp1, tmp2;

	tmp1 = (mat_grey[y*width + x]) - (mat_grey[(y + 1)*width + x + 1]);
	tmp2 = (mat_grey[y *width + x]) - (mat_grey[(y + 1)*width + x + 1]);

	mat_conture[y*width + x] = sqrtf((tmp1*tmp1) + (tmp2*tmp2));
	


	
}




int FindGrad(char *mat_grey, char *mat_grad, int width, int height)
{
	int  size = width*height,
		

	char *d_mat_grey, *d_mat_grad;

	


	cudaMalloc((void**)&d_mat_grey, size * sizeof(char));
	cudaMalloc((void**)&d_mat_grad, 3*size * sizeof(char));

	cudaMemcpy(d_mat_grey, mat_grey, size * sizeof(char), cudaMemcpyHostToDevice);



	FindGradKernel << < height, width >> >(d_mat_grey, d_mat_grad, width, height);
	

	cudaMemcpy(mat_grad, d_mat_grad, 3*size * sizeof(char), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();



	cudaFree(d_mat_grey);
	cudaFree(d_mat_grad);


	return 0;
}


int main()
{
	cv::VideoCapture video(0);
	cv::Mat image,im_grey,im_grad;

	int width, height, rad;
		
	width = video.get(CV_CAP_PROP_FRAME_WIDTH);
	height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	rad = 2;

	unsigned int start_time = clock();

	while (true)
	{
		video >> image;

		//cv::GaussianBlur(image,image,cv::Size(3,3),0,0);
		cv::cvtColor(image, im_grey, CV_BGR2GRAY);
		
		im_grad = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
		
		unsigned int start_time = clock();

		FindGrad((char*)im_grey.data, (char*)im_grad.data, width, height);
		
				
		cv::imshow("GraytImage", im_grad);

		unsigned int end_time = clock(); // �������� �����
		unsigned int search_time = end_time - start_time; // ������� �����

		
		std::cout << search_time << "\n";
		

		if (cv::waitKey(1) == 27)
		{
			return 1;
		}
	}

	

	return 0;
}
