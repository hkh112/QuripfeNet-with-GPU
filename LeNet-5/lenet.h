/*
@author : 范文捷
@data    : 2016-04-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api	:

批量训练
void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);

训练
void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT],uint8 label);

预测
uint8 Predict(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 count);

初始化
void Initial(LeNet5 *lenet);
*/

#pragma once
#include "../src/params.h"
#include "../src/rlwe_sife.h"

#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];


typedef struct LeNet5
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

}LeNet5;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];
}Feature;

typedef struct sec_Feature
{
	//double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
#ifdef PLAIN 
	//double input[INPUT][LENGTH_FEATURE1][LENGTH_FEATURE1][LENGTH_KERNEL][LENGTH_KERNEL];
	int input[INPUT][LENGTH_FEATURE1][LENGTH_FEATURE1][LENGTH_KERNEL][LENGTH_KERNEL][TERMS];
#endif   
#if defined(CPU) || defined(GPU)
	uint32_t input[INPUT][LENGTH_FEATURE1][LENGTH_FEATURE1][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N];	//2 means +, -
#endif   
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];
}sec_Feature;

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);

void Train(LeNet5 *lenet, image input, uint8 label);

uint8 Predict(LeNet5 *lenet, image input, uint8 count);

void Initial(LeNet5 *lenet);


#ifdef PLAIN 
	void sec_load_input(sec_Feature *features, image input);
	uint8 sec_Predict(LeNet5 *lenet, sec_Feature *features, uint8 count);
#endif   
#if defined(CPU) || defined(GPU)
	void sec_load_input(sec_Feature *features, image input, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N]);
	uint8 sec_Predict(LeNet5 *lenet, sec_Feature *features, uint8 count, uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]);
#endif   
  