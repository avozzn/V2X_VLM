import numpy as np
import random
import math
def sensor_corruption_mm(sensor_corruption_level=None,velocity=None,accel=None,steering=None):
    if sensor_corruption_level=='low':
        Cxy=0; Cxz=0;Cyx=0;Cyz=0; Czx=0;Czy=0;mean=0;std_dev=0.388;bias_1=random.uniform(0, 0.1);bias_2=random.uniform(0, 0.1);bias_3=random.uniform(0, 0.1)
    elif sensor_corruption_level=='mid':
        Cxy=random.uniform(0, 0.1);Cxz=random.uniform(0, 0.1);Cyx=random.uniform(0, 0.1);Cyz=random.uniform(0, 0.1);Czx=random.uniform(0, 0.1);Czy=random.uniform(0, 0.1);mean=0;std_dev=0.388;bias_1=random.uniform(0.1, 1);bias_2=random.uniform(0.1, 1);bias_3=random.uniform(0.1, 1)
    elif sensor_corruption_level=='high':
        Cxy=random.uniform(0, 1);Cxz=random.uniform(0, 1);Cyx=random.uniform(0, 1);Cyz=random.uniform(0, 1);Czx=random.uniform(0, 1);Czy=random.uniform(0, 1);mean=0;std_dev=0.388;bias_1=random.uniform(1, 9);bias_2=random.uniform(1, 9);bias_3=random.uniform(1, 9)
    ###create corruption method
    bias_vector = np.array([bias_1, bias_2, bias_3])
    noise_vector = np.abs(np.random.normal(mean, std_dev, 3))
    Gauss=np.random.normal(mean, std_dev, 1)
    coefficient_matrix=np.array([
        [1, Cxy, Cxz],
        [Cyx, 1, Cyz],
        [Czx, Czy, 1]
    ])
    accel_c=np.dot(accel,coefficient_matrix) + bias_vector + noise_vector
    ax_c,ay_c=accel_c[:2]
    ax,ay=accel[:2]
    velocity_c = velocity+math.sqrt((ax_c-ax)**2+(ay_c-ay)**2)
    steering_c = steering+float(Gauss[0])
    if steering_c>6.3:
        steering_c=6.3
    elif steering_c<-7.7:
        steering_c=-7.7
    steering_c=-1 * steering_c/ 2.588
    return accel_c[0],accel_c[1],velocity_c,steering_c