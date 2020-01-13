from ctypes import *
import time
import math
import numpy as np
from matplotlib import pyplot as plt

class WaveModel2D(Structure):
    _fields_ = [
        ("nx", c_int), ("nz", c_int),
        ("dx", c_int), ("dt", c_float),
        ("pad_num", c_int), ("frequency", c_float),
        ("total_time", c_float),
        ("source_amplitude", c_float),
        ("sx", c_int), ("sz", c_int),
        ("receiver_depth", c_int),
        ("velocity", POINTER(c_float)),
        ("abs_model", POINTER(c_float))
    ]


def test_model(nx, nz):
    """
    Sample test velocity model for size over 100 x 100
    Not nx is the number of columns and nz is number of rows
    """
    c = np.full((nz,nx),1500.0)  # Note: 1500m/s is typical acoustic velocity of water
    for ix in range(110,nx-80):  # gradient starting at 111th grid-point down to 80th from bottom
        c[:,ix] = 1750.0+(ix-110)*10.0  # with values from 1750m/s to 3350m/s
    c[:,nx-80:] = c[0,nx-81]
    c[220:230,130:140]=3500.0  # small square which will show diffraction
    c = c.astype(np.float32)
    return c


def plot_at_receivers(r, nx, time, bounds):
    plt.figure(figsize=(10,7))
    plt.imshow(r.T, cmap='RdBu', interpolation='bilinear', aspect='auto',
               vmin=-bounds, vmax=bounds,   # set bounds for colourmap data
               extent=(0,nx,time,0))  # set bounds for axes
    plt.title('Receiver Data')
    plt.colorbar()
    plt.xlabel('Receiver number')
    plt.ylabel('Time / s')
    plt.show()

def create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz,
                      receiver_depth, courant_number=0.7, abs_layer_coefficient=4, abs_coefficient=0.2):
    nz, nx = velocity_model.shape
    dt = courant_number*dx/np.max(velocity_model)
    pad_size = int(np.ceil(dx*abs_layer_coefficient))
    new_vel_model = np.pad(velocity_model, ((3, pad_size), (pad_size, pad_size)), "symmetric").astype(np.float32)
    absorb_facts = np.fromfunction(
        lambda y, x: np.maximum(np.maximum(np.maximum(pad_size-x, x-pad_size-nx+1), np.zeros(new_vel_model.shape))**2,
                                np.maximum(y-nz-2, np.zeros(new_vel_model.shape))**2)/pad_size**2,
        new_vel_model.shape)
    absorb_facts = (absorb_facts * new_vel_model * dt/dx * abs_coefficient).astype(np.float32)
    nz, nx = new_vel_model.shape
    return WaveModel2D(nx=nx, nz=nz, dx=dx, dt=dt, pad_num=pad_size, frequency=frequency, total_time=total_time,
                       source_amplitude=source_amplitude, sx=sx+pad_size, sz=sz+3, receiver_depth=receiver_depth+3,
                       velocity=new_vel_model.ctypes.data_as(POINTER(c_float)),
                       abs_model=absorb_facts.ctypes.data_as(POINTER(c_float)))


def propagation(model):
    nx = model.nx-2*model.pad_num
    prop_lib = cdll.LoadLibrary("/home/miku/IC/Individual_Project/wave_propagation/build/libpropagation.so")
    step_num = round(model.total_time/model.dt)
    model.total_time = step_num*model.dt
    result_buffer = create_string_buffer(nx*step_num*4)
    propagate = prop_lib.propagate
    propagate.restype = c_int
    propagate.argtypes = [POINTER(WaveModel2D), POINTER(c_float)]
    propagate(byref(model), cast(result_buffer, POINTER(c_float)))
    return np.frombuffer(result_buffer, dtype=np.float32, count=step_num*nx).reshape((step_num, nx))


def run():
    nx, nz = 501, 351
    frequency = 10.0
    total_time = 2.0
    source_amplitude = 1.0
    sx, sz = 150, 80
    receiver_depth = 70
    dx = 7
    courant_number = 0.7
    abs_layer_coefficient = 4
    abs_fact = 0.2
    velocity_model = test_model(nx, nz)
    model = create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz,
                              receiver_depth, courant_number, abs_layer_coefficient, abs_fact)
    result = propagation(model)
    plot_at_receivers(result.T, nx, model.total_time, 0.06)


if __name__ == "__main__":
    run()
