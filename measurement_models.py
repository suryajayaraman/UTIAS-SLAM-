#!/usr/bin/env python
# coding: utf-8

import numpy as np
from helper_functions import conBear

def lidar_2d_model(x_, l_data, deltaT):

    #temporary terms
    x,y,theta   = x_[0,0], x_[1,0], x_[2,0]
    lx, ly, lid = l_data[0], l_data[1], l_data[2]
    delx = x - lx
    dely = y - ly
    dist = np.square(dely) + np.square(delx)

    zHat = np.array([
                [np.sqrt(dist)],
                [np.arctan2(-dely, -delx) - theta],
                [lid]
                ])
    zHat[1,0] = conBear(zHat[1,0])

##    zHat = np.array([np.sqrt(dist), np.arctan2(-dely, -delx) - theta, lid])
##    zHat[1] = conBear(zHat[1])

    Hk = np.array([
                [ delx / np.sqrt(dist), dely / np.sqrt(dist),  0.0],
                [-dely / dist         , delx / dist         , -1.0],
                [0.0                  , 0.0                 ,  0.0]
                ])
    
    return Hk, zHat
