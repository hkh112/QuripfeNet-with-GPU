#ifndef PARAMS_H
#define PARAMS_H
#include <stdint.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define max_data 100

//#define PERF
#define TIME
#define TIME2
#define CLOCKS_PER_MS 3000000
#define TERMS			1
#define UNKNOWN		32

//#define AVX2
//#define ORI			//Original Lenet+PlainText
//#define PLAIN 		//provided Lenet+PlainText
//#define CPU			//provided Lenet+ChiperText(by IPFE)
#define GPU			//provided Lenet+ChiperText(by cuFE)

#define SEC_LEVEL 1

#elif SEC_LEVEL==1
static const char SIFE_Q_str[] = "76687145727357674227351553";
#define SIFE_LOGQ_BYTE 88
#define SIFE_NMODULI 3
static const uint64_t SIFE_CRT_CONSTS[SIFE_NMODULI]={0, 206923011, 2935204199007202395};	//*
static const uint32_t SIFE_MOD_Q_I[SIFE_NMODULI] = {16760833, 2147352577, 2130706433};//*
#define SIFE_B_x 32		// 32 or 64
#define SIFE_B_y 32		// 32 or 64
#define SIFE_L 25		// 25/49 or 9
#define SIFE_N 4096
#define SIFE_SIGMA 1
#define SIFE_P (SIFE_B_x*SIFE_B_y*SIFE_L + 1)
static const char SIFE_P_str[] = "50241";
static const char SIFE_SCALE_M_str[]="1526385735302993058007";// floor(q/p)
static const uint64_t SIFE_SCALE_M_MOD_Q_I[SIFE_NMODULI]={13798054, 441557681, 1912932552};	//*
#endif

#endif
