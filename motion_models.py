#!/usr/bin/env python
# coding: utf-8

import numpy as np
from helper_functions import conBear

def diff_drive_motion_model(x_, u, deltaT):
    state_ip_shape = x_.shape

    #temporary term
    x,y,theta    = x_[0,0], x_[1,0], x_[2,0]
    v_t, omega_t = u[0], u[1]
    t1 = theta + (omega_t * deltaT * 0.5)

    x_new = np.array([
                    [x     + (v_t     * deltaT * np.cos(t1))],
                    [y     + (v_t     * deltaT * np.sin(t1))],
                    [theta + (omega_t * deltaT)]
                    ])

    x_new[2, 0]   = conBear(x_new[2,0])
    
    #jacobians
    Fk = np.array([
                [1.0, 0.0, -(v_t * deltaT * np.sin(t1))],
                [0.0, 1.0,  (v_t * deltaT * np.cos(t1))],
                [0.0, 0.0,  1.0]
                ])
    
    Lk = np.array([
                [deltaT * np.cos(t1), -0.5 * np.square(deltaT) * v_t * np.sin(t1)],
                [deltaT * np.sin(t1),  0.5 * np.square(deltaT) * v_t * np.cos(t1)],
                [0.0                 ,  deltaT]
                ])
    x_new = x_new.reshape(state_ip_shape)
    return x_new, Fk, Lk
