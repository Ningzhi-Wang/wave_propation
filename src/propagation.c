#include "propogation.h"

#define COURANT_NUMBER 0.7
#define EPSILON 0.00001
#define MAGIC_SOURCE 2.1

#define isZero(f) fabs(f) < EPSILON
#define notZero(f) fabs(f) > EPSILON

// helper function to simulate wave pressure at source location over time
// maybe changed to other kind of wavelet if needed
float* ricker_source(float frequency, int step_num, float dt, float amplitude);

int propagate(WAVE_MODLE_2D* model, float* result)
{
    int nx = model->nx;
    float dt = model->dt;
    // calucated souce index
    int src_loc = model->sz*nx+model->sx;
    // Create three arrays for time t-1, t and t
    float* u_curr = calloc(model->nx*model->nz, 4);
    float* u_prev = calloc(model->nx*model->nz, 4);
    float* u_next = calloc(model->nx*model->nz, 4);
    // Value used muliple times in propagation simulation
    float dtdx2 = pow(dt, 2)/pow(model->dx, 2);
    int total_steps = model->total_time/dt;
    // Get inital pressures at souce location
    float* src = ricker_source(model->frequency, total_steps, dt, model->source_amplitude);
    // iteration over time steps
    for (int step = 0; step < total_steps; ++step)
    {
        // swith model after each time step progress
        float* temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;
        // inject source value at source location
        u_curr[src_loc] = isZero(src[step])?u_curr[src_loc]:src[step];

        //iteration over the model
        for (int i = 3; i < model->nz-3; ++i)
        {
            for (int j = 3; j < nx-3; ++j)
            {
                // calculate d2u/dx2 using 6th order accuracy
                float d2x = 1/90*(u_curr[i*nx+j-3] + u_curr[i*nx+j+3]) -
                            3/20*(u_curr[i*nx+j-2] + u_curr[i*nx+j+2]) +
                            3/2*(u_curr[i*nx+j-1] + u_curr[i*nx+j+1]) -
                            49/18 * u_curr[i*nx+j];
                float d2z = 1/90*(u_curr[(i-3)*nx+j] + u_curr[(i+3)*nx+j]) -
                            3/20*(u_curr[(i-2)*nx+j] + u_curr[(i+2)*nx+j]) +
                            3/2*(u_curr[(i-1)*nx+j] + u_curr[(i+1)*nx+j]) -
                            49/18 * u_curr[i*nx+j];
                //perform update of wave pressure
                float q = model->abs_model[i*nx+j];
                u_next[i*nx+j] = (dtdx2*pow(model->velocity[i*nx+j], 2)*(d2x+d2z)+
                                  (2-pow(q, 2))*u_curr[i*nx+j]-
                                  (1-q)*u_prev[i*nx+j])/(1+q);
            }
        }
        // reflect top 2 layers above the surface to simulate free surface condition
        memcpy(u_next, u_next+4*nx, nx*sizeof(float));
        memcpy(u_next+nx, u_next+3*nx, nx*sizeof(float));
        // copy wave pressures of receiver depth at this time step to result buffer
        memcpy(result+step*(nx-2*model->pad_num), u_next+model->receiver_depth*nx+model->pad_num,
               (nx-2*model->pad_num)*sizeof(float));
    }
    free (src);
    free(u_curr);
    free(u_prev);
    free(u_next);
    return 0;
}


float* ricker_source(float frequency, int step_num, float dt, float amplitude)
{
    float ts = MAGIC_SOURCE/frequency;
    int ns = (int) (ts/dt + 0.9999);
    ts = ns*dt;
    float a2 = pow(frequency * M_PI, 2);
    float t0 = ts/2 - dt/2;
    float* src_values = calloc(step_num, 4);
    for (int i = 0; i < ns; ++i)
    {
        float at2 = a2*pow(i*dt-t0, 2);
        src_values[i] = amplitude * (1-2*at2) * exp(-at2);
    }
    return src_values;
}
