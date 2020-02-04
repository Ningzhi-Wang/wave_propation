#include "propagation.h"

#define COURANT_NUMBER 0.7
#define EPSILON 0.00001
#define MAGIC_SOURCE 2.1

#define isZero(f) fabs(f) < EPSILON
#define notZero(f) fabs(f) > EPSILON

// helper function to simulate wave pressure at source location over time
// maybe changed to other kind of wavelet if needed
float* ricker_source(float frequency, int step_num, float dt, float amplitude);
void get_density(float* velocity, int nx, int nz, float water_vel, float water_den, float cutoff, float* density_buffer);

int propagate(WAVE_MODLE_2D* model, float* result)
{
    FILE* logger = fopen("logger.log", "w");
    int nx = model->nx;
    float dt = model->dt;
    // calucated souce index
    int src_loc = model->sz*nx+model->sx;
    // Create three arrays for time t-1, t and t
    float* u_curr = calloc(model->nx*model->nz, 4);
    float* u_prev = calloc(model->nx*model->nz, 4);
    float* u_next = calloc(model->nx*model->nz, 4);
    float* density = calloc(model->nx*model->nz, 4);
    get_density(model->velocity, nx, model->nz, model->water_vel, model->water_den, model->cutoff, density);
    // Value used muliple times in propagation simulation
    float dtdx2 = pow(dt, 2)/pow(model->dx, 2);
    int total_steps = model->total_time/dt;
    // Get inital pressures at souce location
    float* src = ricker_source(model->frequency, total_steps, dt, model->source_amplitude);
    // iteration over time steps
    int ind = 1;
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
                float d2x = 1.0/90.0*(u_curr[i*nx+j-3]*density[i*nx+j-3] + u_curr[i*nx+j+3]*density[i*nx+j+3]) -
                            3.0/20.0*(u_curr[i*nx+j-2]*density[i*nx+j-2]+ u_curr[i*nx+j+2]*density[i*nx+j+2]) +
                            3.0/2.0*(u_curr[i*nx+j-1]*density[i*nx+j-1]+ u_curr[i*nx+j+1]*density[i*nx+j+1]) -
                            49.0/18.0*u_curr[i*nx+j]*density[i*nx+j];
                float d2z = 1.0/90.0*(u_curr[(i-3)*nx+j]*density[(i-3)*nx+j] + u_curr[(i+3)*nx+j]*density[(i+3)*nx+j]) -
                            3.0/20.0*(u_curr[(i-2)*nx+j]*density[(i-2)*nx+j] + u_curr[(i+2)*nx+j]*density[(i+2)*nx+j]) +
                            3.0/2.0*(u_curr[(i-1)*nx+j]*density[(i-1)*nx+j-3] + u_curr[(i+1)*nx+j]*density[(i+1)*nx+j]) -
                            49.0/18.0*u_curr[i*nx+j]*density[i*nx+j];

                //perform update of wave pressure
                double q = model->abs_model[i*nx+j];
                u_next[i*nx+j] = (dtdx2*pow(model->velocity[i*nx+j], 2)*(d2x+d2z) / density[i*nx+j] +
                                  (2-pow(q, 2))*u_curr[i*nx+j]-(1-q)*u_prev[i*nx+j])/(1+q) ;
                if (u_next[i*nx+j] > 1 && ind) {
                    //fprintf(logger, "%d, %d, %d, %f, %f , %f, %f\n", step, i, j, d2x, d2x, (2-pow(q, 2))*u_curr[i*nx+j], dtdx2*pow(model->velocity[i*nx+j], 2));
                    ind = 0;
                }

            }
        }
        // reflect top 2 layers above the surface to simulate free surface condition
        memcpy(u_next, u_next+4*nx, nx*sizeof(float));
        memcpy(u_next+nx, u_next+3*nx, nx*sizeof(float));
        // copy wave pressures of receiver depth at this time step to result buffer
        memcpy(result+step*(nx-2*model->pad_num), u_next+model->receiver_depth*nx+model->pad_num,
               (nx-2*model->pad_num)*sizeof(float));
    }
    fclose(logger);
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

void get_density(float* velocity, int nx, int nz, float water_vel, float water_den, float cutoff, float* density_buffer)
{
    FILE* logger = fopen("den.log", "w");
    for (int i = 0; i < nz; ++i)
    {
        for (int j = 0; j < nx; ++j)
        {
            float vel = velocity[i*nx+j];
            if (vel > cutoff)
            {
                density_buffer[i*nx+j] = 1.0 / (0.23 * pow(vel, 0.25));
            } else if (vel < water_vel)
            {
                density_buffer[i*nx+j] = 1.0 / water_den;
            } else
            {
                float ratio = (vel-water_vel)/(cutoff-water_vel);
                density_buffer[i*nx+j] = 1.0 / (ratio * 0.23 * pow(vel, 0.25) + (1-ratio) * water_den);
            }
            if (density_buffer[i*nx+j] > 1) {
                fprintf(logger, "%d, %d, %f", i, j, 1/density_buffer[i*nx+j]);
            }
        }
    }
    fclose(logger);
}