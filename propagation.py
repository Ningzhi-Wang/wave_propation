from ctypes import *
import time
import os
import math
import numpy as np
from numpy.ctypeslib import ndpointer
from matplotlib import pyplot as plt
import segyio
from hicks_interpolation import interpolate
from loss import get_loss


class WaveModel2D(Structure):
    """
    Wave Model Struct as defined in C header file
    """
    _fields_ = [
        ("nx", c_int), ("nz", c_int),
        ("dx", c_int), ("dt", c_float),
        ("pad_num", c_int),
        ("total_time", c_float),
        ("sx", c_int), ("sz", c_int),
        ("receiver_depth", c_int),
        ("water_den", c_float),
        ("water_vel", c_float),
        ("cutoff", c_float)
   ]

def plot_model(c):
    """
    Plot Velocity model
    """
    plt.figure(figsize=(10,6))
    plt.imshow(c.T)
    plt.colorbar()
    plt.xlabel('x gridpoints')
    plt.ylabel('z gridpoints')
    plt.title('Velocity Model (m/s)')
    plt.show()


def get_vel(file_name):
    """
    Get 2D velocity
    Args:
        file_name (str): The path of segy file containing velocity file
    Return (2d numpy array): The numpy array of the velocity file
    """
    with segyio.open(file_name, "r") as f:
        f.mmap()
        nx = f.bin[segyio.BinField.Traces]
        nz = f.bin[segyio.BinField.Samples]
        vel = np.zeros((nx, nz))
        vel_gen = f.trace[:]
        for i, v in enumerate(vel_gen):
            vel[i] = v
        return vel.T


def get_sources(file_name, idx=1):
    """
    Get one souce location
    Args:
        file_name (str): The path of geometry (.geo) file containing source locations
        idx (int): The index of the souce to be retrieved
    Return (float, float): The real location of retrieved source location in meter
    """
    sources = np.fromfile(file_name, sep=" ").reshape((-1, 4))
    return sources[0, 0], sources[idx, 1], sources[idx, 3]

def get_signature(file_name, idx):
    with segyio.open(file_name, "r", ignore_geometry=True) as f:
        f.mmap()
        source = f.trace[idx]
    return source

def get_idx(file_name, idx):
    links = np.fromfile(file_name, dtype=np.int64)[3:].reshape(-1, 3)
    return links[links[:, 1] == idx, 2]

def get_receiver_depth(rev_file, idx_file, dx, souce_num, idx):
    """
    Get receiver depth
    Args:
        file_name (str): The path of geometry (.geo) file containing receiver locations
    Return (float): The real depth of receivers in meter
    """
    receivers = np.fromfile(rev_file, sep=" ").reshape((-1, 4))[1:]
    rev_idx = get_idx(idx_file, idx)
    rev_loc = receivers[np.isin(receivers[:, 0], rev_idx)][:, 1]
    #start_rev = receivers[idx*rev_per_src+1][1]
    #end_rev = receivers[(idx+1)*rev_per_src][1]
    #return receivers[1][-1]/dx, int(start_rev/dx), int(end_rev/dx)
    #print(receivers.shape)
    return int(receivers[0][-1]/dx), rev_loc/dx


def test_model(nx, nz):
    """
    Sample test velocity model for size over 100 x 100
    Note nx is the number of columns and nz is number of rows
    """
    c = np.full((nz,nx),1500.0)  # Note: 1500m/s is typical acoustic velocity of water
    for ix in range(110,nz-80):  # gradient starting at 111th grid-point down to 80th from bottom
        c[ix:, ] = 1750.0+(ix-110)*10.0  # with values from 1750m/s to 3350m/s
    c[nz-80:, :] = c[nz-81,0]
    c[130:140, 220:230]=3500.0  # small square which will show diffraction
    c = c.astype(np.float32)
    return c


def plot_at_receivers(r, nx, time, bounds):
    """
    Plot displacement as receiver over time time
    Args:
        r (2d array): Wave pressure values at receiver depth over given time
        nx (int): the number of cells in horizontal direction of the model
        time (float): Time legth
        bounds (float): The maximum absolute value for pressure for plotting
    """
    plt.figure(figsize=(10,7))
    plt.imshow(r.T, cmap='RdBu', interpolation='bilinear', aspect='auto',
               vmin=-bounds, vmax=bounds,   # set bounds for colourmap data
               extent=(0,nx,time,0))  # set bounds for axes
    plt.title('Receiver Data')
    plt.colorbar()
    plt.xlabel('Receiver number')
    plt.ylabel('Time / s')
    plt.savefig("./fpga.png")
    plt.show()


def create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz, receiver_depth,
                      dt, water_den, water_vel, cutoff, abs_layer_coefficient=1.6, abs_coefficient=0.1, source=None):
    """
    Create Wave model, adding paddings to velocity model and setup absorb factors for each cell
    Args:
        velocity_model (2d array): The velocity model for wave propagation
        dx (float): The width/height of one cell in the model in meter
        frequency (float): The heighest frequency for wave created by the source
        total_time (float): The duration for the wave propagation
        source_amplitude (float): The maximum amplitude of the wave created by the source
        sx (int): The horizontal location of source in cell number
        sz (int): The vertical location of source in cell number
        receiver_depth(int): The depth of receivers in cell number
        courant_number(int): Coefficient for time step to meet Courant–Friedrichs–Lewy condition
        abs_layer_coefficient(float): The number of absorbing layers needed per meter of dx
        abs_coefficient(float): The coefficient for pressure absorbing
    Return (WaveModel2D): 2D Wave model containing all information for propagation
    """
    nz, nx = velocity_model.shape
    pad_size = int(np.ceil(dx*abs_layer_coefficient))
    new_vel_model = np.pad(velocity_model, ((3, pad_size), (pad_size, pad_size)), "symmetric").astype(np.float32)
    absorb_facts = np.fromfunction(
        lambda y, x: np.maximum(np.maximum(np.maximum(pad_size-x, x-pad_size-nx+1), np.zeros(new_vel_model.shape))**2,
                                np.maximum(y-nz-2, np.zeros(new_vel_model.shape))**2)/pad_size**2,
        new_vel_model.shape)
    absorb_facts = (absorb_facts * new_vel_model * dt/dx * abs_coefficient).astype(np.float32)
    nz, nx = new_vel_model.shape
    sources = source.reshape(-1, 1).astype(np.float32)
    return (WaveModel2D(nx=nx, nz=nz, dx=dx, dt=dt, pad_num=pad_size, total_time=total_time,
                       sx=sx+pad_size, sz=sz+3, receiver_depth=receiver_depth+3,
                       water_den=water_den, water_vel=water_vel, cutoff=cutoff),
            new_vel_model, absorb_facts, sources)


def propagation(model, vel, abs, src):
    """
    Call to C functions to perform propagation
    Args:
        model(WaveModel2D): Wave model for propagation
    Return (2d array): The pressure received at receiver-depth over the propagation time
    """
    vel = np.ascontiguousarray(vel.astype(np.float32))
    abs = np.ascontiguousarray(abs.astype(np.float32))
    src = np.ascontiguousarray(src.astype(np.float32))
    nx = model.nx-2*model.pad_num
    dir_path = os.path.dirname(os.path.abspath(__file__))
    prop_lib = cdll.LoadLibrary(os.path.join(dir_path, "build/libpropagation.so"))
    step_num = round(model.total_time/model.dt)
    model.total_time = step_num*model.dt
    result = np.empty((nx*step_num)).astype(np.float32)
    prop_kernel = prop_lib.propagate
    prop_kernel.restype = c_int
    p_type = ndpointer(c_float, flags="C_CONTIGUOUS")
    prop_kernel.argtypes = [POINTER(WaveModel2D), p_type, p_type, p_type, p_type]
    prop_kernel(byref(model), vel, abs, src, result)
    return result.reshape(-1, nx)


def create_ricker(freq,dt,ampl, time_steps):
    ts = 2.1/freq  # desired length of wavelet in time is related to peak frequency
    ns = int(ts/dt+0.9999)  # figure out how many time-steps are needed to cover that time
    ts = ns*dt  # and now turn that back into a time that's exactly the required number of steps
    a2 = (freq*np.pi)**2  # a squared (see equation above)
    t0 = ts/2 - dt/2  # midpoint time of wavelet
    src = np.zeros(time_steps)
    # create Ricker wavelet (see equation above), offset by time t0 (so midpoint of wavelet is at time t=t0)
    for i in range(ns):
        src[i] = ampl*( (1 - 2*a2*((i*dt-t0)**2)) * np.exp(-a2*((i*dt-t0)**2)) )
    return src


def artificial_model_test():
    nx, nz = 501, 351
    frequency = 10.0
    total_time = 2.0
    source_amplitude = 1.0
    sx, sz = 150, 80
    receiver_depth = 70
    dx = 7
    courant_number = 0.3
    abs_layer_coefficient = 5
    abs_fact = 0.4
    velocity_model = test_model(nx, nz)
    dt = courant_number*dx/3500
    source = create_ricker(frequency, dt, source_amplitude, int(total_time/dt))
    model = create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz,
                              receiver_depth, dt, 1.0, 1500, 1650, abs_layer_coefficient, abs_fact, source)
    start = time.time()
    result = propagation(*model)
    end = time.time()
    plot_at_receivers(result.T, nx, model[0].total_time, 0.03)
    print(start-end)


def propagate(idx, water_den=1.0, water_vel=1500, cutoff=1650, vel_model=None, vel_file=None, sig_file=None, src_file=None,
              rev_file=None, idx_file=None, dx=0, freq=0, total_time=0, **kwargs):
    if vel_model is None:
        vel_model = get_vel(vel_file)
    source_amplitude = 1.0
    src_num, sx, sz = get_sources(src_file, idx)
    source = get_signature(sig_file, idx)
    sx, sz = sx/dx, sz/dx
    sx, sz = int(sx), int(sz)
    #receiver_depth, rev_s, rev_e = get_receiver_depth(rev_file, dx, src_num, idx)
    receiver_depth, receivers = get_receiver_depth(rev_file, idx_file, dx, src_num, idx)
    dt = total_time/len(source)
    # magic coefficients
    abs_layer_coefficient = 1.6
    abs_fact = 0.1

    model = create_test_model(vel_model, dx, freq, total_time, source_amplitude, sx, sz, receiver_depth,
                              dt, water_den, water_vel, cutoff, abs_layer_coefficient, abs_fact, source)

    result = propagation(*model)
    result = np.array([interpolate(result.T, r) for r in receivers])
    return result



if __name__ == "__main__":
    artificial_model_test()
    #nx = 501
    #nz = 351
    #vel_model = test_model(nx, nz)
    #plt.imshow(vel_model, cmap="jet")
    #plt.colorbar()
    #plt.xlabel('x gridpoints')
    #plt.ylabel('z gridpoints')
    #plt.title('Velocity Model (m/s)')
    #plt.show()


    #result = propagate(300, 1000, 1500, 1650, None, "../model/000-Template/inputs/J50-TrueVp.sgy",
    #                   "../model/000-Template/inputs/J50-Signature.sgy",
    #                   "../model/000-Template/inputs/J50-Sources.geo",
    #                   "../model/000-Template/inputs/J50-Receivers.geo",
    #                   "../model/000-Template/inputs/J50-Observed.idx",
    #                   50, 4.0, 6.144)
    #plot_at_receivers(result, 480, 6.144, result.max())

