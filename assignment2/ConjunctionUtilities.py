###############################################################################
# This file contains code developed from the open source NASA CARA Analysis 
# Tools, provided under the NASA Open Source Software Agreement.
#
# Copyright © 2020 United States Government as represented by the Administrator 
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Modified (port to python) by Steve Gehly Feb 27, 2024
#
# References:
#
#  [1] Denenberg, E., "Satellite Closest Approach Calculation Through 
#      Chebyshev Proxy Polynomials," Acta Astronautica, 2020.
#
#  [2] Hall, D.T., Hejduk, M.D., and Johnson, L.C., "Remediating Non-Positive
#      Definite State Covariances for Collision Probability Estimation," 2017.
#
#  [3] https://github.com/nasa/CARA_Analysis_Tools
#
#  [4] Foster, J., Estes, H., "A parametric analysis of orbital debris collision 
#      probability and maneuver rate for space vehicles," Tech Report, 1992.
#
#  [5] Alfano, S., "Review of Conjunction Probability Methods for Short-term 
#      Encounters," Advances in the Astronautical Sciences, Vol. 127, Jan 2007, 
#      pp 719-746.
#
#  [6] Alfano, S., "Satellite Conjuction Monte Carlo Analysis," AAS Spaceflight
#      Mechanics Meeting (AAS-09-233), 2009.
#  
#
#
###############################################################################

import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import inspect
from scipy.integrate import dblquad
from scipy.special import erfcinv
import json




def read_json_file(json_file):
    '''
    This function reads a json file containing data for Resident Space Objects
    (RSOs) and returns a dictionary containing the same data, indexed by 
    5 digit NORAD ID.
    
    Parameters
    ------
    json_file : string
        path and filename of json file containing RSO data
    
    Returns
    ------
    rso_dict : dictionary
        RSO data indexed by 5 digit NORAD ID
        The following data are provided for each object:
            UTC : datetime object corresponding to state and covar 
            state : 6x1 numpy array, Cartesian position and velocity in ECI 
            covar : 6x6 numpy array, covariance matrix associated with state
            
            Units of m and m/s
            
    '''
    
    f = open(json_file)
    rso_dict0 = json.load(f)
    f.close()
    
    # Create output
    rso_dict = {}    
    for obj_id in rso_dict0:
        
        # Convert dictionary keys to integers
        obj_id_int = int(obj_id)
        rso_dict[obj_id_int] = {}
        
        # Convert UTC to datetime object
        UTC_dt = datetime.strptime(rso_dict0[obj_id]['UTC'], '%Y-%m-%d %H:%M:%S')
        rso_dict[obj_id_int]['UTC'] = UTC_dt
        
        # Convert state and covariance to numpy arrays
        rso_dict[obj_id_int]['state'] = np.asarray(rso_dict0[obj_id]['state'])
        rso_dict[obj_id_int]['covar'] = np.asarray(rso_dict0[obj_id]['covar'])
    
    
    return rso_dict


###############################################################################
# 2D Probability of Collision (Pc) Functions
###############################################################################

def Pc2D_Foster(X1, P1, X2, P2, HBR, rtol=1e-8, HBR_type='circle'):
    '''
    This function computes the probability of collision (Pc) in the 2D 
    encounter plane following the method of Foster. The code has been ported
    from the MATLAB library developed by the NASA CARA team, listed in Ref 3.
    The function supports 3 types of hard body regions: circle, square, and 
    square equivalent to the area of the circle. The input covariance may be
    either 3x3 or 6x6, but only the 3x3 position covariance will be used in
    the calculation of Pc.
    
    
    Parameters
    ------
    X1 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 1 in ECI [m, m/s]
    P1 : 6x6 numpy array
        Estimated covariance of Object 1 in ECI [m^2, m^2/s^2]
    X2 : 6x1 numpy array
        Estimated mean state vector
        Cartesian position and velocity of Object 2 in ECI [m, m/s]
    P2 : 6x6 numpy array
        Estimated covariance of Object 2 in ECI [m^2, m^2/s^2]
    HBR : float
        hard-body region (e.g. radius for spherical object) [m]
    rtol : float, optional
        relative tolerance for numerical quadrature (default=1e-8)
    HBR_type : string, optional
        type of hard body region ('circle', 'square', or 'squareEqArea')
        (default='circle')
    
    Returns
    ------
    Pc : float
        probability of collision
    
    '''
    
    # Retrieve and combine the position covariance
    Peci = P1[0:3,0:3] + P2[0:3,0:3]
    
    # Construct the relative encounter frame
    r1 = np.reshape(X1[0:3], (3,1))
    v1 = np.reshape(X1[3:6], (3,1))
    r2 = np.reshape(X2[0:3], (3,1))
    v2 = np.reshape(X2[3:6], (3,1))
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v, axis=0)
    
    # Unit vectors of relative encounter frame
    yhat = v/np.linalg.norm(v)
    zhat = h/np.linalg.norm(h)
    xhat = np.cross(yhat, zhat, axis=0)
    
    # Transformation matrix
    eci2xyz = np.concatenate((xhat.T, yhat.T, zhat.T))
    
    # Transform combined covariance to relative encounter frame (xyz)
    Pxyz = np.dot(eci2xyz, np.dot(Peci, eci2xyz.T))
    
    # 2D Projected covariance on the x-z plane of the relative encounter frame
    red = np.array([[1., 0., 0.], [0., 0., 1.]])
    Pxz = np.dot(red, np.dot(Pxyz, red.T))

    # Exception Handling
    # Remediate non-positive definite covariances
    Lclip = (1e-4*HBR)**2.
    Pxz_rem, Pxz_det, Pxz_inv, posdef_status, clip_status = remediate_covariance(Pxz, Lclip)
    
    
    # Calculate Double Integral
    x0 = np.linalg.norm(r)
    z0 = 0.
    
    # Set up quadrature
    atol = 1e-13
    Integrand = lambda z, x: math.exp(-0.5*(Pxz_inv[0,0]*x**2. + Pxz_inv[0,1]*x*z + Pxz_inv[1,0]*x*z + Pxz_inv[1,1]*z**2.))

    if HBR_type == 'circle':
        lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'square':
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR, x0+HBR, z0-HBR, z0+HBR, epsabs=atol, epsrel=rtol)[0])
        
    elif HBR_type == 'squareEqArea':
        HBR_eq = HBR*np.sqrt(math.pi)/2.
        Pc = (1./(2.*math.pi))*(1./np.sqrt(Pxz_det))*float(dblquad(Integrand, x0-HBR_eq, x0+HBR_eq, z0-HBR_eq, z0+HBR_eq, epsabs=atol, epsrel=rtol)[0])
    
    else:
        print('Error: HBR type is not supported! Must be circle, square, or squareEqArea')
        print(HBR_type)
    
    return Pc



def remediate_covariance(Praw, Lclip, Lraw=[], Vraw=[]):
    '''
    This function provides a level of exception handling by detecting and 
    remediating non-positive definite covariances in the collision probability
    calculation, following the procedure in Hall et al. (Ref 2). This code has
    been ported from the MATLAB library developed by the NASA CARA team, 
    listed in Ref 3.
    
    The function employs an eigenvalue clipping method, such that eigenvalues
    below the specified Lclip value are reset to Lclip. The covariance matrix,
    determinant, and inverse are then recomputed using the original 
    eigenvectors and reset eigenvalues to ensure the output is positive (semi)
    definite. An input of Lclip = 0 will result in the output being positive
    semi-definite.
    
    Parameters
    ------
    Praw : nxn numpy array
        unremediated covariance matrix
    
    
    
    Returns
    ------
    
    
    '''
    
    # Ensure the covariance has all real elements
    if not np.all(np.isreal(Praw)):
        print('Error: input Praw is not real!')
        print(Praw)
        return
    
    # Calculate eigenvectors and eigenvalues if not input
    if len(Lraw) == 0 and len(Vraw) == 0:
        Lraw, Vraw = np.linalg.eig(Praw)
        
    # Define the positive definite status of Praw
    posdef_status = np.sign(min(Lraw))
    
    # Clip eigenvalues if needed, and record clipping status
    Lrem = Lraw.copy()
    if min(Lraw) < Lclip:
        clip_status = True
        Lrem[Lraw < Lclip] = Lclip
    else:
        clip_status = False
        
    # Determinant of remediated covariance
    Pdet = np.prod(Lrem)
    
    # Inverse of remediated covariance
    Pinv = np.dot(Vraw, np.dot(np.diag(1./Lrem), Vraw.T))
    
    # Remediated covariance
    if clip_status:
        Prem = np.dot(Vraw, np.dot(np.diag(Lrem), Vraw.T))
    else:
        Prem = Praw.copy()
    
    
    return Prem, Pdet, Pinv, posdef_status, clip_status



