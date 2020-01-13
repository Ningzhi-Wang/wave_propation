#ifndef PROPAGATION
#define PROPAGATION

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>


typedef struct wave_model_2d
{
    int nx;
    int nz;
    int dx;
    float dt;
    int pad_num;
    float frequency;
    float total_time;
    float source_amplitude;
    int sx;
    int sz;
    int receiver_depth;
    float* velocity;
    float* abs_model;
} WAVE_MODLE_2D;

float get_time_step_size(WAVE_MODLE_2D* model);

int propagate(WAVE_MODLE_2D* model, float* result);


#endif