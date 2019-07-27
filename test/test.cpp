// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <vector>
#include <iostream>
#include <algorithm>
#include <omp.h>


//#define STRANGE_BEHAVIOR_SWITCH_INT64_TO_INT32_IN_64BIT
//#define STRANGE_BEHAVIOR_DIFFERENT_RADIUS_SIZE_IN_CODE


#ifndef STRANGE_BEHAVIOR_SWITCH_INT64_TO_INT32_IN_64BIT
typedef intptr_t int_t;
#else
typedef int int_t;
#endif	



struct TSize
{
	int_t width;
	int_t height;
};


inline double getTime()
{
	return omp_get_wtime()*1000.0;
}

#define ZPP_TIMER_INIT()       double timer = getTime();
#define ZPP_TIMER_START()      timer = getTime();
#define ZPP_TIMER_POINT(name)  printf("%12.3fms    %s\n", getTime() - timer, name); timer = getTime();
#define ZPP_TIMER_LINE()       printf("\n");


void generateRandomImageData(
	unsigned char * dst, int_t dstStep, int_t width, int_t height);

void copyFirstPixelOnRow(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, int_t width, int_t height);
void copyFirstPixelOnRow_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, int_t width, int_t height, bool parallel);

void copyFirstPixelOnRowUsingTSize(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size);
void copyFirstPixelOnRowUsingTSize_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, bool parallel);

void copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, bool parallel);

void divideImageDataWithParam(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius);
void divideImageDataWithParam(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius, bool parallel);

void boxFilterRow(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius);
void boxFilterRow_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius, bool parallel);


int main()
{
	ZPP_TIMER_INIT();

	const int_t width  = 3840;
	const int_t height = 2160;

	std::vector<unsigned char> src(width*height);
	std::vector<unsigned char> dst(width*height);

	generateRandomImageData(src.data(), width, width, height);
	printf("generateRandomImageData :: %dx%d\n", width, height);

	const int ITER = 100;
	printf("numberOfIterations = %d\n\n", ITER);

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRow(src.data(), width, dst.data(), width, width, height);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRow");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRow_OpenMP(src.data(), width, dst.data(), width, width, height, false);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRow_OpenMP single-threaded");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRow_OpenMP(src.data(), width, dst.data(), width, width, height, true);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRow_OpenMP multi-threaded");
	ZPP_TIMER_LINE();


	TSize size = { width, height };

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRowUsingTSize(src.data(), width, dst.data(), width, size);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRowUsingTSize");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRowUsingTSize_OpenMP(src.data(), width, dst.data(), width, size, false);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRowUsingTSize_OpenMP single-threaded");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRowUsingTSize_OpenMP(src.data(), width, dst.data(), width, size, true);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRowUsingTSize_OpenMP multi-threaded");
	ZPP_TIMER_LINE();
	

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP(src.data(), width, dst.data(), width, size, false);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP single-threaded");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP(src.data(), width, dst.data(), width, size, true);
	}
	ZPP_TIMER_POINT("copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP multi-threaded");
	ZPP_TIMER_LINE();


	const int_t PARAM = 1;

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		divideImageDataWithParam(src.data(), width, dst.data(), width, size, PARAM);
	}
	ZPP_TIMER_POINT("divideImageDataWithParam");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		divideImageDataWithParam(src.data(), width, dst.data(), width, size, PARAM, false);
	}
	ZPP_TIMER_POINT("divideImageDataWithParam single-threaded");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		divideImageDataWithParam(src.data(), width, dst.data(), width, size, PARAM, true);
	}
	ZPP_TIMER_POINT("divideImageDataWithParam multi-threaded");
	ZPP_TIMER_LINE();


	const int_t RADIUS = 1;
	printf("RADIUS = %d\n", RADIUS);

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		boxFilterRow(src.data(), width, dst.data(), width, size, RADIUS);
	}
	ZPP_TIMER_POINT("boxFilterRow");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		boxFilterRow_OpenMP(src.data(), width, dst.data(), width, size, RADIUS, false);
	}
	ZPP_TIMER_POINT("boxFilterRow_OpenMP single-threaded");

	ZPP_TIMER_START();
	for (int i = 0; i < ITER; i++)
	{
		boxFilterRow_OpenMP(src.data(), width, dst.data(), width, size, RADIUS, true);
	}
	ZPP_TIMER_POINT("boxFilterRow_OpenMP multi-threaded");
	ZPP_TIMER_LINE();

	{
		#ifndef STRANGE_BEHAVIOR_DIFFERENT_RADIUS_SIZE_IN_CODE
		const int_t RADIUS = 1;
		#else
		const int_t RADIUS = 2;
		#endif
		printf("RADIUS = %d\n", RADIUS);

		ZPP_TIMER_START();
		for (int i = 0; i < ITER; i++)
		{
			boxFilterRow(src.data(), width, dst.data(), width, size, RADIUS);
		}
		ZPP_TIMER_POINT("boxFilterRow");

		ZPP_TIMER_START();
		for (int i = 0; i < ITER; i++)
		{
			boxFilterRow_OpenMP(src.data(), width, dst.data(), width, size, RADIUS, false);
		}
		ZPP_TIMER_POINT("boxFilterRow_OpenMP single-threaded");

		ZPP_TIMER_START();
		for (int i = 0; i < ITER; i++)
		{
			boxFilterRow_OpenMP(src.data(), width, dst.data(), width, size, RADIUS, true);
		}
		ZPP_TIMER_POINT("boxFilterRow_OpenMP multi-threaded");
		ZPP_TIMER_LINE();
	}

	return EXIT_SUCCESS;
}


void generateRandomImageData(
	unsigned char * dst, int_t dstStep, int_t width, int_t height)
{
	srand(1);

	for (int_t y = 0; y < height; y++)
	{
		for (int_t x = 0; x < width; x++)
		{
			dst[y*dstStep + x] = rand() % 256;
		}
	}
}


void copyFirstPixelOnRow(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, int_t width, int_t height)
{
	for (int_t y = 0; y < height; y++)
	{
		unsigned char value = src[y*srcStep];

		for (int_t x = 0; x < width; x++)
		{
			dst[y*dstStep + x] = value;
		}
	}
}


void copyFirstPixelOnRow_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, int_t width, int_t height, bool parallel)
{
	#pragma omp parallel for if(parallel)
	for (int_t y = 0; y < height; y++)
	{
		unsigned char value = src[y*srcStep];

		for (int_t x = 0; x < width; x++)
		{
			dst[y*dstStep + x] = value;
		}
	}
}


void copyFirstPixelOnRowUsingTSize(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size)
{
	for (int_t y = 0; y < size.height; y++)
	{
		unsigned char value = src[y*srcStep];

		for (int_t x = 0; x < size.width; x++)
		{
			dst[y*dstStep + x] = value;
		}
	}
}


void copyFirstPixelOnRowUsingTSize_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, bool parallel)
{
	#pragma omp parallel for if(parallel)
	for (int_t y = 0; y < size.height; y++)
	{
		unsigned char value = src[y*srcStep];

		for (int_t x = 0; x < size.width; x++)
		{
			dst[y*dstStep + x] = value;
		}
	}
}


void copyFirstPixelOnRowUsingTSizeAsFirstprivate_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, bool parallel)
{
	#pragma omp parallel for if(parallel) firstprivate(size)
	for (int_t y = 0; y < size.height; y++)
	{
		unsigned char value = src[y*srcStep];

		for (int_t x = 0; x < size.width; x++)
		{
			dst[y*dstStep + x] = value;
		}
	}
}


void divideImageDataWithParam(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t param)
{
	for (int_t y = 0; y < size.height; y++)
	{
		for (int_t x = 0; x < size.width; x++)
		{
			dst[y*dstStep + x] = src[y*srcStep + x]/param;
		}
	}
}


void divideImageDataWithParam(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t param, bool parallel)
{
	#pragma omp parallel for if(parallel)
	for (int_t y = 0; y < size.height; y++)
	{
		for (int_t x = 0; x < size.width; x++)
		{
			dst[y*dstStep + x] = src[y*srcStep + x]/param;
		}
	}
}


void boxFilterRow(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius)
{
	const int density = 2*radius + 1;
	const int_t ri = radius + 1;

	for (int_t y = 0; y < size.height; y++)
	{
		int sum = ri*src[y*srcStep];

		for (int_t x = 1; x < ri; x++)
		{ // init
			int_t xx = std::min(x, size.width-1);

			sum += src[y*srcStep + xx];
		}
		dst[y*dstStep] = sum/density;

		for (int_t x = 0; x < size.width; x++)
		{
			int_t xl = std::max(x - ri, (int_t)0);
			int_t xr = std::min(x + ri - 1, size.width-1);

			sum -= src[y*srcStep + xl];
			sum += src[y*srcStep + xr];

			dst[y*dstStep + x] = sum/density;
		}
	}
}


void boxFilterRow_OpenMP(
	const unsigned char * src, int_t srcStep, unsigned char * dst, int_t dstStep, TSize size, int_t radius, bool parallel)
{
	const int density = 2*radius + 1;
	const int_t ri = radius + 1;

	#pragma omp parallel for if(parallel)
	for (int_t y = 0; y < size.height; y++)
	{
		int sum = ri*src[y*srcStep];

		for (int_t x = 1; x < ri; x++)
		{ // init
			int_t xx = std::min(x, size.width-1);

			sum += src[y*srcStep + xx];
		}
		dst[y*dstStep] = sum/density;

		for (int_t x = 0; x < size.width; x++)
		{
			int_t xl = std::max(x - ri, (int_t)0);
			int_t xr = std::min(x + ri - 1, size.width-1);

			sum -= src[y*srcStep + xl];
			sum += src[y*srcStep + xr];

			dst[y*dstStep + x] = sum/density;
		}
	}
}


