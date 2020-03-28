#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm

"""

LIST OF FUNCTIONS
-----------------
1. sigmaEllipse2D( mu, Sigma, level = 1, npoints=32 )
2. plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs)
3. plot_robot_pose(pose, cov, **kwargs)

"""


def sigmaEllipse2D( mu, Sigma, level = 1, npoints=32 ):

    """
    function SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
    a sigma level in the Gaussian density defined by mean and covariance.
    
    Input
    -----
    MU       - [2 x 1] Mean of the Gaussian density
    SIGMA    - [2 x 2] Covariance matrix of the Gaussian density
    LEVEL    - Which sigma level curve to plot. Can take any positive value,
               but common choices are 1, 2 or 3. Default = 3.
    NPOINTS  - Number of points on the ellipse to generate. Default = 32.
    
    Output
    ------
    XY       - [2 x npoints] matrix.First and last columns should be the same point, to create a closed curve.
    """
    
    level = abs(level)
    Sigma = abs(Sigma)
    
    #initialise xy to zero array
    xy = np.zeros((2, npoints))

    #linspace of theta range
    angle_samples = np.linspace(0.0, 2* np.pi, npoints)
    polar_tf = np.column_stack(( np.cos(angle_samples),  np.sin(angle_samples) ))
    print(polar_tf.shape)
    #using equation (2)
    xy = mu + level * (np.dot(sqrtm(Sigma) ,polar_tf.T ))
    return xy.T


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Input
    -----
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Output
    ------
        A matplotlib ellipse artist
        
        
    Source Reference 
    ----------------
    #https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    #ax.set_xlim(pos[0] - 1.0, pos[0] + 1.0)
    #ax.set_ylim(pos[1]- 1.0, pos[1] + 1.0)
    ax.add_artist(ellip)
    return ellip


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def test_plot_point_cov():
    """ Example usage of plot_point_cov(points, nstd=2, ax=None, **kwargs) function """
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[0.4, 9],[9, 10]], size=1000
            )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')
    plt.show()
    return None


def plot_robot_pose(pose, cov, legend_name, **kwargs):
    """
    
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Input
    -----
        pose - The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0, theta].
        cov - The 3x3 covariance matrix to base the ellipse on
        legend_name - name of the axis in plot
        
        Additional keyword arguments 
        
        nstd - The radius of the ellipse in numbers of standard deviations.
               Defaults to 1 standard deviations.
               
        ax   - The axis that the ellipse will be plotted on. Defaults to the 
             current axis.
        plot_var - Boolean to speicfy if variance is to be plotted 
        plot_dir - Boolean to specify if direction arrow is to be plotted
    
    Output
    ------
        None
        
    References
    ----------
        https://www.geeksforgeeks.org/args-kwargs-python/
        
        
    """
    
    def set_param(param, param_dict, default_value):
        if param in param_dict.items():
            value = kwargs[param]
        else:
            value = default_value        
        return value
    
    ax       = set_param('ax', kwargs, plt.gca())
    nstd     = set_param('nstd', kwargs, 1)
    plot_dir = set_param('plot_dir', kwargs, True)
    plot_var = set_param('plot_var', kwargs, False)
    r_robot  = set_param('r_robot', kwargs, 0.15)

    print(plot_var)
    
    #print(pose.shape)
    pose = pose.reshape(3,1)
    x,y,theta = pose[0,0], pose[1,0], pose[2,0]
    plt.plot(x,y, 'ro', label = legend_name)

    #plot arrow to indicate orientation
    if plot_dir:
        dx, dy = (r_robot + 0.05) * np.cos(theta), (r_robot + 0.05) * np.sin(theta) 
        plt.arrow(x,y,dx,dy)

    #plot covariance ellipse
    if plot_var:
        plot_cov_ellipse(cov[0:2, 0:2], pose[0:2].reshape(2,1), nstd, alpha=0.25, color='green')
    return None
