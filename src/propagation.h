#ifndef PROPAGATION
#define PROPAGATION

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

// 2D Wave Model used for wave propagation
typedef struct wave_model_2d
{
    int nx;
    int nz;
    int dx;
    float dt;
    int pad_num;
    float total_time;
    int sx;
    int sz;
    int receiver_depth;
    float water_den;
    float water_vel;
    float cutoff;
 } WAVE_MODLE_2D;


// perform propgation for given wave model
// receiver values will be stored at given result buffer
int propagate(WAVE_MODLE_2D* model, float* vel,  float* abs, float* src, float* result);

#endif