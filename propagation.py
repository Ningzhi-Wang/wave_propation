from ctypes import *
import time
import math
import numpy as np
from matplotlib import pyplot as plt
import segyio
from hicks_interpolation import interpolate

class WaveModel2D(Structure):
    """
    Wave Model Struct as defined in C header file
    """
    _fields_ = [
        ("nx", c_int), ("nz", c_int),
        ("dx", c_int), ("dt", c_float),
        ("pad_num", c_int), ("frequency", c_float),
        ("total_time", c_float),
        ("source_amplitude", c_float),
        ("sx", c_int), ("sz", c_int),
        ("receiver_depth", c_int),
        ("water_den", c_float),
        ("water_vel", c_float),
        ("cutoff", c_float),
        ("source", POINTER(c_float)),
        ("velocity", POINTER(c_float)),
        ("abs_model", POINTER(c_float))
    ]

def plot_model(c):
    """
    Plot Velocity model
    """
    plt.figure(figsize=(10,6))
    plt.imshow(c.T, cmap=plt.get_cmap(name="jet"))
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
    plt.show()


def create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz, receiver_depth,
                      dt, abs_layer_coefficient=1.6, abs_coefficient=0.1, source=None):
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
        abs_coefficient(flaot): The coefficient for pressure absorbing
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
    if source is not None:
        source = source.ctypes.data_as(POINTER(c_float))
    return WaveModel2D(nx=nx, nz=nz, dx=dx, dt=dt, pad_num=pad_size, frequency=frequency, total_time=total_time,
                       source_amplitude=source_amplitude, sx=sx+pad_size, sz=sz+3, receiver_depth=receiver_depth+3,
                       water_den = 1.0, water_vel = 1500, cutoff=1650, source=source,
                       velocity=new_vel_model.ctypes.data_as(POINTER(c_float)),
                       abs_model=absorb_facts.ctypes.data_as(POINTER(c_float)))


def propagation(model):
    """
    Call to C functions to perform propagation
    Args:
        model(WaveModel2D): Wave model for propagation
    Return (2d array): The pressure received at receiver-depth over the propagation time
    """
    nx = model.nx-2*model.pad_num
    prop_lib = cdll.LoadLibrary("./build/libpropagation.so")
    step_num = round(model.total_time/model.dt)
    model.total_time = step_num*model.dt
    result_buffer = create_string_buffer(nx*step_num*4)
    prop_kernel = prop_lib.propagate
    prop_kernel.restype = c_int
    prop_kernel.argtypes = [POINTER(WaveModel2D), POINTER(c_float)]
    prop_kernel(byref(model), cast(result_buffer, POINTER(c_float)))
    return np.frombuffer(result_buffer, dtype=np.float32, count=step_num*nx).reshape((step_num, nx))


def artificial_model_test():
    nx, nz = 501, 351
    frequency = 10.0
    total_time = 2.0
    source_amplitude = 1.0
    sx, sz = 150, 0
    receiver_depth = 0
    dx = 7
    courant_number = 0.3
    abs_layer_coefficient = 5
    abs_fact = 0.2
    velocity_model = test_model(nx, nz)
    dt = courant_number*dx/np.max(velocity_model)
    model = create_test_model(velocity_model, dx, frequency, total_time, source_amplitude, sx, sz,
                              receiver_depth, dt, abs_layer_coefficient, abs_fact)
    result = propagation(model)
    plot_at_receivers(result.T, nx, model.total_time, 0.03)


def propagate(idx, vel_file=None, sig_file=None, src_file=None, rev_file=None,
              idx_file=None, dx=0, freq=0, total_time=0, **kwargs):
    vel_model = get_vel(vel_file)
    nz, nx = vel_model.shape
    source_amplitude = 1.0
    src_num, sx, sz = get_sources(src_file, idx)
    source = get_signature(sig_file, idx)
    sx, sz = int(round(sx/dx)), int(round(sz/dx))
    #receiver_depth, rev_s, rev_e = get_receiver_depth(rev_file, dx, src_num, idx)
    receiver_depth, receivers = get_receiver_depth(rev_file, idx_file, dx, src_num, idx)
    dt = total_time/len(source)
    # magic coefficients
    abs_layer_coefficient = 1.6
    abs_fact = 0.1
    velocity_model = test_model(nx, nz)
    model = create_test_model(velocity_model, dx, freq, total_time, source_amplitude, sx, sz,
                              receiver_depth, dt, abs_layer_coefficient, abs_fact, source)
    result = propagation(model)
    result = np.array([interpolate(result.T, r) for r in receivers])
    #result = result[:, rev_s:rev_e+1]
    return result



if __name__ == "__main__":
    result = propagate(300, "000-Template/inputs/J50-TrueVp.sgy", "000-Template/inputs/J50-Signature.sgy",
                       "000-Template/inputs/J50-Sources.geo", "000-Template/inputs/J50-Receivers.geo",
                       "000-Template/inputs/J50-Observed.idx",
                       50, 4.0, 6.144)
    plot_at_receivers(result, 480, 6.144, 0.01)

