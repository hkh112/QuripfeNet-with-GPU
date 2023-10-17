#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GETLENGTH(array) (int)(sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < (int)count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	uint8 ret = 0;
#ifdef TIME2
    uint64_t CLOCK1, CLOCK2;
    double load_time=0, predict_time=0;
    CLOCK1=cpucycles();
#endif   
	load_input(&features, input);
#ifdef TIME2
    CLOCK2=cpucycles();
    load_time += (double)CLOCK2 - CLOCK1;
    CLOCK1=cpucycles();
#endif   
	forward(lenet, &features, relu);
    ret = get_result(&features, count);
#ifdef TIME2
    CLOCK2=cpucycles();
    predict_time += (double)CLOCK2 - CLOCK1;
    printf("load time: \t \t %.6f ms\n", load_time/CLOCKS_PER_MS);
    printf("predict time: \t %.6f ms\n", predict_time/CLOCKS_PER_MS);
#endif   

	return ret;
}


void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}


#if defined(PLAIN) || defined(CPU) || defined(GPU)
static uint8 sec_get_result(sec_Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}
#endif

#ifdef PLAIN 
void sec_load_input(sec_Feature *features, image input)
{
	int (*layer0)[LENGTH_FEATURE1][LENGTH_FEATURE1][LENGTH_KERNEL][LENGTH_KERNEL][TERMS] = features->input;
	double layer[LENGTH_FEATURE0][LENGTH_FEATURE0] = {0};
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	double temp = 0;

	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		layer[j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}

	for (int o0 = 0; o0 < LENGTH_FEATURE1; ++o0)
		for (int o1 = 0; o1 < LENGTH_FEATURE1; ++o1)
			for (int w0 = 0; w0 < LENGTH_KERNEL; ++w0)
				for (int w1 = 0; w1 < LENGTH_KERNEL; ++w1)
				{
					if(TERMS==1)
					{
						temp = layer[o0 + w0][o1 +w1] * UNKNOWN;
						layer0[0][o0][o1][w0][w1][0] = (int)temp;
					}
					if(TERMS==2)
					{
						temp = layer[o0 + w0][o1 +w1] * UNKNOWN;
						layer0[0][o0][o1][w0][w1][0] = (int)temp;
						temp = (temp - (double)layer0[0][o0][o1][w0][w1][0]) * UNKNOWN;
						layer0[0][o0][o1][w0][w1][1] = (int)temp;
					}
					if(TERMS==3)
					{
						temp = layer[o0 + w0][o1 +w1] * UNKNOWN;
						layer0[0][o0][o1][w0][w1][0] = (int)temp;
						temp = (temp - (double)layer0[0][o0][o1][w0][w1][0]) * UNKNOWN;
						layer0[0][o0][o1][w0][w1][1] = (int)temp;
						temp = (temp - (double)layer0[0][o0][o1][w0][w1][1]) * UNKNOWN;
						layer0[0][o0][o1][w0][w1][2] = (int)temp;
					}
				}
}

static void sec_CONVOLUTION_FORWARD(int input[INPUT][LENGTH_FEATURE1][LENGTH_FEATURE1][LENGTH_KERNEL][LENGTH_KERNEL][TERMS], double output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER1], double(*action)(double))
{
	int weights[TERMS] = {0};
	int term[TERMS*TERMS] = {0};
	double temp = 0;

	for (int y = 0; y < GETLENGTH(*weight); ++y)
		for (int o0 = 0; o0 < GETLENGTH(output[y]); ++o0)
			for (int o1 = 0; o1 < GETLENGTH(*(output[y])); ++o1)
			{
				for (int w0 = 0; w0 < GETLENGTH(weight[0][y]); ++w0)
				{
					for (int w1 = 0; w1 < GETLENGTH(*(weight[0][y])); ++w1)
					{
						if(TERMS==1)
						{
							temp = weight[0][y][w0][w1] * UNKNOWN;
							weights[0] = (int)temp; 							
							term[0] = input[0][o0][o1][w0][w1][0] * weights[0];							
							temp = (double)term[0]/UNKNOWN/UNKNOWN;
						}
						if(TERMS==2)
						{
							temp = weight[0][y][w0][w1] * UNKNOWN;
							weights[0] = (int)temp;
							temp = (temp - (double)weights[0]) * UNKNOWN;
							weights[1] = (int)temp; 
							
							term[0] = input[0][o0][o1][w0][w1][0] * weights[0];
							term[1] = input[0][o0][o1][w0][w1][0] * weights[1];
							term[2] = input[0][o0][o1][w0][w1][1] * weights[0];
							term[3] = input[0][o0][o1][w0][w1][1] * weights[1];
							
							temp = ((double)term[0] + (((double)term[1] + (double)term[2]) + (double)term[3]/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
						}
						if(TERMS==3)
						{
							temp = weight[0][y][w0][w1] * UNKNOWN;
							weights[0] = (int)temp;
							temp = (temp - (double)weights[0]) * UNKNOWN;
							weights[1] = (int)temp; 
							temp = (temp - (double)weights[1]) * UNKNOWN;
							weights[2] = (int)temp; 
							
							term[0] = input[0][o0][o1][w0][w1][0] * weights[0];
							term[1] = input[0][o0][o1][w0][w1][0] * weights[1];
							term[2] = input[0][o0][o1][w0][w1][0] * weights[2];
							term[3] = input[0][o0][o1][w0][w1][1] * weights[0];
							term[4] = input[0][o0][o1][w0][w1][1] * weights[1];
							term[5] = input[0][o0][o1][w0][w1][1] * weights[2];
							term[6] = input[0][o0][o1][w0][w1][2] * weights[0];
							term[7] = input[0][o0][o1][w0][w1][2] * weights[1];
							term[8] = input[0][o0][o1][w0][w1][2] * weights[2];
							
							temp = ((double)term[0] + (((double)term[1] + (double)term[3]) + (((double)term[2] + (double)term[4] + (double)term[6]) + (((double)term[5] + (double)term[7]) + (double)term[8]/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
						}
						
						output[y][o0][o1] += temp;
					}
				}
			}

	FOREACH(j, GETLENGTH(output))
		FOREACH(i, GETCOUNT(output[j]))
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);
}

static void sec_forward(LeNet5 *lenet, sec_Feature *features, double(*action)(double))
{
	sec_CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

uint8 sec_Predict(LeNet5 *lenet, sec_Feature *features, uint8 count)
{
	sec_forward(lenet, features, relu);
	return sec_get_result(features, count);
}
#endif

#if defined(CPU) || defined(GPU)
void sec_load_input(sec_Feature *features, image input, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N])
{
	uint32_t (*layer0)[LENGTH_FEATURE1][LENGTH_FEATURE1][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = features->input;
	double layer[LENGTH_FEATURE0][LENGTH_FEATURE0] = {0};
	uint32_t m[TERMS][2][SIFE_L] = {0};
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	double temp = 0;
	int temp2 = 0;

	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, (int)(sizeof(image) / sizeof(*input)))
		FOREACH(k, (int)(sizeof(*input) / sizeof(**input)))
	{
		layer[j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}

	// input preprocessing
	for (int o0 = 0; o0 < LENGTH_FEATURE1; ++o0)
	{
		for (int o1 = 0; o1 < LENGTH_FEATURE1; ++o1)
		{
			for (int w0 = 0; w0 < LENGTH_KERNEL; ++w0)
			{
				for (int w1 = 0; w1 < LENGTH_KERNEL; ++w1)
				{
					// positive / negative
					temp = layer[o0 + w0][o1 +w1] * UNKNOWN;
					if (temp >= 0)
					{
						m[0][0][w0*LENGTH_KERNEL + w1] = (int)temp;
						m[0][1][w0*LENGTH_KERNEL + w1] = 0;
					}
					else
					{
						m[0][0][w0*LENGTH_KERNEL + w1] = 0;
						m[0][1][w0*LENGTH_KERNEL + w1] = (int)temp * (-1);
					}
					for(int i=1; i<TERMS; i++) {
						temp = (temp - (double)temp2) * UNKNOWN;
						temp2 = (int)temp;
						if (temp >= 0)
						{
							m[i][0][w0*LENGTH_KERNEL + w1] = temp2;
							m[i][1][w0*LENGTH_KERNEL + w1] = 0;
						}
						else
						{
							m[i][0][w0*LENGTH_KERNEL + w1] = 0;
							m[i][1][w0*LENGTH_KERNEL + w1] = temp2 * (-1);
						}
					}
				}
			}
#ifdef CPU
			for (int i = 0; i < TERMS; i++){
				rlwe_sife_encrypt(m[i][0], mpk, layer0[0][o0][o1][i][0]);
				rlwe_sife_encrypt(m[i][1], mpk, layer0[0][o0][o1][i][1]);
			}
#endif
#ifdef GPU
			rlwe_sife_encrypt_gui((uint32_t*)m, mpk, (uint32_t*)layer0[0][o0][o1], 2*TERMS);
#endif
		}
	}
}

void sec_CONVOLUTION_FORWARD(uint32_t input[INPUT][LENGTH_FEATURE1][LENGTH_FEATURE1][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N], double output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], 
	double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER1], double (*action)(double), uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
{
	unsigned int y[TERMS][2][SIFE_L] = {0};     
    uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N];
#ifdef CPU
    mpz_t dy[SIFE_N];
	double term[TERMS*TERMS][4] = {0};
#endif
#ifdef GPU
    uint32_t dy2[SIFE_NMODULI][SIFE_N];
    uint32_t* d_y = (uint32_t*)malloc(TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
    double term[TERMS*TERMS][4] = {0};
#endif
	double temp = 0;
	int temp2 = 0;

	// for channel
	for (int j = 0; j < GETLENGTH(*weight); ++j)		
	{				
		// filter preprocessing
		for (int w0 = 0; w0 < GETLENGTH(weight[0][j]); ++w0)
		{							
			for (int w1 = 0; w1 < GETLENGTH(*(weight[0][j])); ++w1)
			{
				// positive / negative
				temp = weight[0][j][w0][w1] * UNKNOWN;
				if (temp >= 0)
				{
					y[0][0][w0*LENGTH_KERNEL + w1] = (int)temp;
					y[0][1][w0*LENGTH_KERNEL + w1] = 0;
				}
				else
				{
					y[0][0][w0*LENGTH_KERNEL + w1] = 0;
					y[0][1][w0*LENGTH_KERNEL + w1] = (int)temp * (-1);
				}
				for(int i=1; i<TERMS; i++) {
					temp = (temp - temp2) * UNKNOWN;
					temp2 = (int)temp;
					if (temp >= 0)
					{
						y[i][0][w0*LENGTH_KERNEL + w1] = temp2;
						y[i][1][w0*LENGTH_KERNEL + w1] = 0;
					}
					else
					{
						y[i][0][w0*LENGTH_KERNEL + w1] = 0;
						y[i][1][w0*LENGTH_KERNEL + w1] = temp2 * (-1);
					}
				}
			}
		}		

		// filter Key Generation
#ifdef CPU
		for (int i = 0; i < TERMS; i++){
			rlwe_sife_keygen(y[i][0], msk, sk_y[i][0]);
			rlwe_sife_keygen(y[i][1], msk, sk_y[i][1]);
		}
#endif
#ifdef GPU
		rlwe_sife_keygen_gui((uint32_t*)y, msk, (uint32_t*)sk_y, 2*TERMS);
#endif

		// Convolution (Decryption)
#ifdef CPU
        for(int k=0;k<SIFE_N;k++){
            mpz_init(dy[k]);
        }
		for (int o0 = 0; o0 < GETLENGTH(output[j]); ++o0)
		{								
			for (int o1 = 0; o1 < GETLENGTH(*(output[j])); ++o1)
			{
				for(int l=0; l<TERMS; l++)
					for(int m=0; m<TERMS; m++)
						for(int n=0; n<2; n++)
							for(int o=0; o<2; o++)
							{
								rlwe_sife_decrypt_gmp(input[0][o0][o1][l][n], y[m][o], sk_y[m][o], dy);
								round_extract_gmp(dy);
								term[TERMS*l+m][2*n+o] = mpz_get_d(dy[0]);	
								if ((int)term[TERMS*l+m][2*n+o] >= SIFE_P)
									term[TERMS*l+m][2*n+o] = 0;
							}

				for(int l=0; l<TERMS; l++)
					for(int m=0; m<TERMS; m++)
					{
						term[TERMS*l+m][0] = term[TERMS*l+m][0] - term[TERMS*l+m][1] - term[TERMS*l+m][2] + term[TERMS*l+m][3];
					}

				if(TERMS==1)
					output[j][o0][o1] = term[0][0]/UNKNOWN/UNKNOWN;
				else if(TERMS==2)
					output[j][o0][o1] = (term[0][0] + ((term[1][0] + term[2][0]) + term[3][0]/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
				else if(TERMS==3)
					output[j][o0][o1] = (term[0][0] + ((term[1][0] + term[3][0]) + ((term[2][0] + term[4][0] + term[6][0]) + ((term[5][0] + term[7][0]) + term[8][0]/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
			}
		}
#endif
#ifdef GPU
		//Optimized code
		for (int o0 = 0; o0 < GETLENGTH(output[j]); ++o0)
		{								
			for (int o1 = 0; o1 < GETLENGTH(*(output[j])); ++o1)
			{
				for(int m=0; m<TERMS; m++)
					for(int o=0; o<2; o++)
					{
						rlwe_sife_decrypt_gmp_gui3((uint32_t*)input[0][o0][o1], (uint32_t*)y[m][o], (uint32_t*)sk_y[m][o], (uint32_t*)d_y, TERMS*2);
						for(int l=0; l<TERMS; l++)
		 					for(int n=0; n<2; n++)
		 					{
		 						memcpy(dy2, d_y + (l * 2 + n) * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
		 						term[TERMS*l+m][2*n+o] = round_extract_gmp2(dy2);
		 						if ((int)term[TERMS*l+m][2*n+o] >= SIFE_P)
		 							term[TERMS*l+m][2*n+o] = 0;
		 					}
		 			}

		 		for(int l=0; l<TERMS; l++)
		 			for(int m=0; m<TERMS; m++)
		 			{
		 				term[TERMS*l+m][0] = term[TERMS*l+m][0] - term[TERMS*l+m][1] - term[TERMS*l+m][2] + term[TERMS*l+m][3];
		 			}

		 		if(TERMS==1)
         			output[j][o0][o1] = term[0][0]/UNKNOWN/UNKNOWN;
		 		else if(TERMS==2)
		 			output[j][o0][o1] = (term[0][0] + ((term[1][0] + term[2][0]) + term[3][0]/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
		 		else if(TERMS==3)
		 			output[j][o0][o1] = (term[0][0] + ((term[1][0] + term[3][0]) + ((term[2][0] + term[4][0] + term[6][0]) + ((term[5][0] + term[7][0]) + term[8][0]/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN)/UNKNOWN/UNKNOWN;
		 	}
		}
#endif
	}

	for (int j = 0; j < GETLENGTH(*weight); ++j)							
		for (int i = 0; i < (int)GETCOUNT(output[j]); ++i)							
			((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);
#ifdef CPU
	for(int i=0;i<SIFE_N;i++){
		mpz_clear(dy[i]);
	}
#endif
#ifdef GPU
	free(d_y);
#endif
}

static void sec_forward(LeNet5 *lenet, sec_Feature *features, double(*action)(double), uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
{
	sec_CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action, msk);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

uint8 sec_Predict(LeNet5 *lenet, sec_Feature *features, uint8 count, uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
{
	sec_forward(lenet, features, relu, msk);
	return sec_get_result(features, count);
}
#endif
