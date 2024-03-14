import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle


import TudatPropagator as prop


###############################################################################
#
# This directory contains data files for coordinate frame transformations
# eop_alldata.pkl, IAU1980_nutation.csv, IAU2006_XYs.csv
#
# Update at your discretion based on where you put the data files
#
###############################################################################

data_dir = 'data'


###############################################################################
# Basic I/O
###############################################################################

def read_truth_file(truth_file):
    '''
    This function reads a pickle file containing truth data for state 
    estimation.
    
    Parameters
    ------
    truth_file : string
        path and filename of pickle file containing truth data
    
    Returns
    ------
    t_truth : N element numpy array
        time in seconds since J2000
    X_truth : Nxn numpy array
        each row X_truth[k,:] corresponds to Cartesian state at time t_truth[k]
    state_params : dictionary
        propagator params
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    '''
    
    # Load truth data
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    X_truth = data[1]
    state_params = data[2]
    pklFile.close()
    
    return t_truth, X_truth, state_params


def read_measurement_file(meas_file):
    '''
    This function reads a pickle file containing measurement data for state 
    estimation.
    
    Parameters
    ------
    meas_file : string
        path and filename of pickle file containing measurement data
    
    Returns
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            UTC: datetime object for epoch of state/covar
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
            
    '''

    # Load measurement data
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    state_params = data[0]
    sensor_params = data[1]
    meas_dict = data[2]
    pklFile.close()
    
    return state_params, meas_dict, sensor_params



###############################################################################
# Unscented Kalman Filter
###############################################################################


def ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

    Parameters
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            UTC: datetime object for epoch of state/covar
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    int_params : dictionary
        numerical integration parameters
        
    filter_params : dictionary
        fields:
            Qeci: 3x3 numpy array of SNC accelerations in ECI [m/s^2]
            Qric: 3x3 numpy array of SNC accelerations in RIC [m/s^2]
            alpha: float, UKF sigma point spread parameter, should be in range [1e-4, 1]
            gap_seconds: float, time in seconds between measurements for which SNC should be zeroed out, i.e., if tk-tk_prior > gap_seconds, set Q=0
            
    bodies : tudat object
        contains parameters for the environment bodies used in propagation

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
        
        indexed first by tk, then contains fields:
            state: nx1 numpy array, estimated Cartesian state vector at tk [m, m/s]
            covar: nxn numpy array, estimated covariance at tk [m^2, m^2/s^2]
            resids: px1 numpy array, measurement residuals at tk [meters and/or radians]
        
    '''
        
    # Retrieve data from input parameters
    UTC0 = state_params['UTC']
    t0 = (UTC0 - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    Xo = state_params['state']
    Po = state_params['covar']    
    Qeci = filter_params['Qeci']
    Qric = filter_params['Qric']
    alpha = filter_params['alpha']
    gap_seconds = filter_params['gap_seconds']

    n = len(Xo)
    q = int(Qeci.shape[0])
    
    # Prior information about the distribution
    beta = 2.
    kappa = 3. - float(n)
    
    # Compute sigma point weights    
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']
    
    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    Xk = Xo.copy()
    Pk = Po.copy()
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        # Propagate state and covariance
        # No prediction needed if measurement time is same as current state
        if tk_prior == tk:
            Xbar = Xk.copy()
            Pbar = Pk.copy()
        else:
            tvec = np.array([tk_prior, tk])
            dum, Xbar, Pbar = prop.propagate_state_and_covar(Xk, Pk, tvec, state_params, int_params, bodies, alpha)
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        delta_t = tk - tk_prior
        if delta_t > gap_seconds:    
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        # Combined Q matrix (ECI and RIC components)
        # Rotate RIC to ECI and add
        rc_vect = Xbar[0:3].reshape(3,1)
        vc_vect = Xbar[3:6].reshape(3,1)
        Q = Qeci + ric2eci(rc_vect, vc_vect, Qric)
                
        # Add Process Noise to Pbar
        Pbar += np.dot(Gamma, np.dot(Q, Gamma.T))

        # Recompute sigma points to incorporate process noise
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk       
        # Retrieve measurement data
        Yk = Yk_list[kk]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = unscented_meas(tk, chi_bar, sensor_params)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Joseph form of covariance update
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)        
        gamma_til_post, dum = unscented_meas(tk, chi_k, sensor_params)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        print('')
        print('kk', kk)
        print('Yk', Yk)
        print('ybar', ybar)     
        print('resids', resids)
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['state'] = Xk
        filter_output[tk]['covar'] = P
        filter_output[tk]['resids'] = resids

    
    return filter_output


###############################################################################
# Sensors and Measurements
###############################################################################


def unscented_meas(tk, chi, sensor_params):
    '''
    This function computes the measurement sigma point matrix.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    chi : nx(2n+1) numpy array
        state sigma point matrix
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    gamma_til : px(2n+1) numpy array
        measurement sigma point matrix
    Rk : pxp numpy array
        measurement noise covariance
        
    '''
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    eop_alldata = sensor_params['eop_alldata']
    XYs_df = sensor_params['XYs_df']
    UTC = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=tk)
    EOP_data = get_eop_data(eop_alldata, UTC)
    
    sensor_itrf = sensor_params['sensor_itrf']
    sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data, XYs_df)
    
    # Measurement information    
    meas_types = sensor_params['meas_types']
    sigma_dict = sensor_params['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in GCRF
        r_gcrf = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_gcrf = r_gcrf - sensor_gcrf
        rg = np.linalg.norm(rho_gcrf)
        rho_hat_gcrf = rho_gcrf/rg
        
        # Rotate to ENU frame
        rho_hat_itrf, dum = gcrf2itrf(rho_hat_gcrf, np.zeros((3,1)), UTC, EOP_data,
                                      XYs_df)
        rho_hat_enu = ecef2enu(rho_hat_itrf, sensor_itrf)
        
        if 'rg' in meas_types:
            rg_ind = meas_types.index('rg')
            gamma_til[rg_ind,jj] = rg
            
        if 'ra' in meas_types:
        
            ra = math.atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) # rad        
        
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                if ra > np.pi/2. and ra < np.pi:
                    quad = 2
                if ra < -np.pi/2. and ra > -np.pi:
                    quad = 3
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 2 and ra < 0.:
                    ra += 2.*np.pi
                if quad == 3 and ra > 0.:
                    ra -= 2.*np.pi
                    
            ra_ind = meas_types.index('ra')
            gamma_til[ra_ind,jj] = ra
                
        if 'dec' in meas_types:        
            dec = math.asin(rho_hat_gcrf[2])  # rad
            dec_ind = meas_types.index('dec')
            gamma_til[dec_ind,jj] = dec
            
        if 'az' in meas_types:
            az = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad 
            
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                if az > np.pi/2. and az < np.pi:
                    quad = 2
                if az < -np.pi/2. and az > -np.pi:
                    quad = 3
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 2 and az < 0.:
                    az += 2.*np.pi
                if quad == 3 and az > 0.:
                    az -= 2.*np.pi
                    
            az_ind = meas_types.index('az')
            gamma_til[az_ind,jj] = az
            
        if 'el' in meas_types:
            el = math.asin(rho_hat_enu[2])  # rad
            el_ind = meas_types.index('el')
            gamma_til[el_ind,jj] = el


    return gamma_til, Rk


def compute_measurement(tk, X, sensor_params):
    '''
    This function be used to compute a measurement given an input state vector
    and time.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    X : nx1 numpy array
        Cartesian state vector [m, m/s]
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    Y : px1 numpy array
        computed measurements for given state and sensor
    
    '''
    
    # Retrieve EOP data
    UTC = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=tk)
    eop_alldata = sensor_params['eop_alldata']
    EOP_data = get_eop_data(eop_alldata, UTC)
    XYs_df = sensor_params['XYs_df']    
    
    # Retrieve measurement types
    meas_types = sensor_params['meas_types']
    
    # Compute station location in GCRF
    sensor_itrf = sensor_params['sensor_itrf']
    sensor_gcrf, dum = itrf2gcrf(sensor_itrf, np.zeros((3,1)), UTC, EOP_data,
                                 XYs_df)
    
    # Object location in GCRF
    r_gcrf = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_gcrf - sensor_gcrf)
    rho_hat_gcrf = (r_gcrf - sensor_gcrf)/rg
    
    # Rotate to ENU frame
    rho_hat_itrf, dum = gcrf2itrf(rho_hat_gcrf, np.zeros((3,1)), UTC, EOP_data,
                                  XYs_df)
    rho_hat_enu = ecef2enu(rho_hat_itrf, sensor_itrf)
    
    # Loop over measurement types
    Y = np.zeros((len(meas_types),1))
    ii = 0
    for mtype in meas_types:
        
        if mtype == 'rg':
            Y[ii] = rg  # m
            
        elif mtype == 'ra':
            Y[ii] = math.atan2(rho_hat_gcrf[1], rho_hat_gcrf[0]) # rad
            
        elif mtype == 'dec':
            Y[ii] = math.asin(rho_hat_gcrf[2])  # rad
    
        elif mtype == 'az':
            Y[ii] = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            # if Y[ii] < 0.:
            #     Y[ii] += 2.*np.pi
            
        elif mtype == 'el':
            Y[ii] = math.asin(rho_hat_enu[2])  # rad
            
        ii += 1
            
            
    return Y



def define_radar_sensor(latitude_rad, longitude_rad, height_m):
    '''
    This function will generate the sensor parameters dictionary for a radar
    sensor provided the location in latitude, longitude, height.
    
    It is pre-filled with constraint and noise parameters per assignment
    description.

    Parameters
    ----------
    latitude_rad : float
        geodetic latitude of sensor [rad]
    longitude_rad : float
        geodetic longitude of sensor [rad]
    height_m : float
        geodetic height of sensor [m]

    Returns
    -------
    sensor_params : dictionary
        location, constraint, noise parameters of sensor

    '''
    
    # EOP data
    eop_file = os.path.join(data_dir, 'eop_alldata.pkl')
    pklFile = open(eop_file, 'rb' )
    data = pickle.load( pklFile )
    eop_alldata = data[0]
    pklFile.close()
    
    XYs_df = get_XYs2006_alldata()
    
        
    # Compute sensor location in ECEF/ITRF
    sensor_itrf = latlonht2ecef(latitude_rad, longitude_rad, height_m)
        
    # FOV dimensions
    LAM_deg = 10.   # deg
    PHI_deg = 10.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_deg*np.pi/180
    PHI_half = 0.5*PHI_deg*np.pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [5.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., 5000.*1000.]   # m
    sun_el_mask = -np.pi  # rad
    
    # Measurement types and noise
    meas_types = ['rg', 'az', 'el']
    sigma_dict = {}
    sigma_dict['rg'] = 10.              # m
    sigma_dict['az'] = 0.1*np.pi/180.   # rad
    sigma_dict['el'] = 0.1*np.pi/180.   # rad
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_itrf'] = sensor_itrf
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['FOV_hlim'] = FOV_hlim
    sensor_params['FOV_vlim'] = FOV_vlim
    sensor_params['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_params['meas_types'] = meas_types
    sensor_params['sigma_dict'] = sigma_dict
    
    # EOP data
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    return sensor_params


def define_optical_sensor(latitude_rad, longitude_rad, height_m):
    '''
    This function will generate the sensor parameters dictionary for an optical
    sensor provided the location in latitude, longitude, height.
    
    It is pre-filled with constraint and noise parameters per assignment
    description.

    Parameters
    ----------
    latitude_rad : float
        geodetic latitude of sensor [rad]
    longitude_rad : float
        geodetic longitude of sensor [rad]
    height_m : float
        geodetic height of sensor [m]

    Returns
    -------
    sensor_params : dictionary
        location, constraint, noise parameters of sensor

    '''
    
    arcsec2rad = (1./3600.)*np.pi/180.
    
    # EOP data
    eop_file = os.path.join(data_dir, 'eop_alldata.pkl')
    pklFile = open(eop_file, 'rb' )
    data = pickle.load( pklFile )
    eop_alldata = data[0]
    pklFile.close()
    
    XYs_df = get_XYs2006_alldata()
        
    # Compute sensor location in ECEF/ITRF
    sensor_itrf = latlonht2ecef(latitude_rad, longitude_rad, height_m)
        
    # FOV dimensions
    LAM_deg = 4.   # deg
    PHI_deg = 4.   # deg
    
    # Convert to radians
    LAM_half = 0.5*LAM_deg*np.pi/180
    PHI_half = 0.5*PHI_deg*np.pi/180
    FOV_hlim = [-LAM_half, LAM_half]
    FOV_vlim = [-PHI_half, PHI_half]
    
    # Constraints/Limits
    az_lim = [0., 2.*np.pi]  # rad
    el_lim = [15.*np.pi/180., np.pi/2.]  # rad
    rg_lim = [0., np.inf]   # m
    sun_el_mask = -12.*np.pi/180.  # rad (Nautical twilight)
    
    # Measurement types and noise
    meas_types = ['ra', 'dec']
    sigma_dict = {}
    sigma_dict['ra'] = arcsec2rad    # rad
    sigma_dict['dec'] = arcsec2rad   # rad
        
    # Location and constraints
    sensor_params = {}
    sensor_params['sensor_itrf'] = sensor_itrf
    sensor_params['el_lim'] = el_lim
    sensor_params['az_lim'] = az_lim
    sensor_params['rg_lim'] = rg_lim
    sensor_params['FOV_hlim'] = FOV_hlim
    sensor_params['FOV_vlim'] = FOV_vlim
    sensor_params['sun_elmask'] = sun_el_mask
    
    # Measurements and noise
    sensor_params['meas_types'] = meas_types
    sensor_params['sigma_dict'] = sigma_dict
    
    # EOP data
    sensor_params['eop_alldata'] = eop_alldata
    sensor_params['XYs_df'] = XYs_df
    
    return sensor_params









###############################################################################
#
# Functions past this point are used for coordinate frames and time systems
# and should not be edited.
#
###############################################################################


###############################################################################
# Coordinate Frames
###############################################################################


def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]

    Returns
    ------
    lat : float
      latitude [rad] [-pi/2,pi/2]
    lon : float
      longitude [rad] [-pi,pi]
    ht : float
      height [m]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    a = 6378137.0   # m
    rec_f = 298.257223563

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x)

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # m
    lat = 0.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = float(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # km
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0


    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [rad]
    lon : float
      geodetic longitude [rad]
    ht : float
      geodetic height [m]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [m]
    '''
    
    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378137.0   # m
    rec_f = 298.257223563

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric


def ric2eci(rc_vect, vc_vect, Q_ric=[]):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Q_ric (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T

    # Rotate Qin as appropriate for vector or matrix
    if len(Q_ric) == 0:
        Q_eci = NO
    elif np.size(Q_ric) == 3:
        Q_eci = np.dot(NO, Q_ric)
    else:
        Q_eci = np.dot(np.dot(NO, Q_ric), NO.T)

    return Q_eci


def get_eop_data(data_text, UTC):
    '''
    This function retrieves the EOP data for a specific time by computing
    a linear interpolation of parameters from the two closest times.
    
    Parameters
    ------
    data_text : string
        string containing observed and predicted EOP data, no header
        information
    UTC : datetime object
        time in UTC
    
    Returns
    ------
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day
    '''    
        
    # Compute MJD for desired time
    MJD = dt2mjd(UTC)
    MJD_int = int(MJD)
    
    # Find EOP data lines around time of interest
    nchar = 102
    nskip = 1
    nlines = 0
    for ii in range(len(data_text)):
        start = ii + nlines*(nchar+nskip)
        stop = ii + nlines*(nchar+nskip) + nchar
        line = data_text[start:stop]
        nlines += 1
        
        MJD_line = int(line[11:16])
        
        if MJD_line == MJD_int:
            line0 = line
        if MJD_line == MJD_int+1:
            line1 = line
            break
    
    # Compute EOP data at desired time by interpolating
    EOP_data = eop_linear_interpolate(line0, line1, MJD)
    
    return EOP_data


def eop_linear_interpolate(line0, line1, MJD):
    '''
    This function computes the linear interpolation of EOP parameters between
    two lines of the EOP data file.
    
    Parameters
    ------
    line0 : string
        EOP data line from time before desired MJD
    line1 : string
        EOP data line from time after desired MJD
    MJD : float
        fractional days since 1858-11-17 in UTC
    
    Returns
    ------
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day        
    '''    
    
    # Initialize output
    EOP_data = {}
    
    # Leap seconds do not interpolate
    EOP_data['TAI_UTC'] = int(line0[99:102])
    
    # Retrieve values
    line0_array = eop_read_line(line0)
    line1_array = eop_read_line(line1)
    
    # Adjust UT1-UTC column in case leap second occurs between lines
    line0_array[3] -= line0_array[9]
    line1_array[3] -= line1_array[9]
    
    # Linear interpolation
    dt = MJD - line0_array[0]
    interp = (line1_array[1:] - line0_array[1:])/ \
        (line1_array[0] - line0_array[0]) * dt + line0_array[1:]

    # Convert final output
    arcsec2rad = (1./3600.) * math.pi/180.
    EOP_data['xp'] = interp[0]*arcsec2rad
    EOP_data['yp'] = interp[1]*arcsec2rad
    EOP_data['UT1_UTC'] = interp[2] + EOP_data['TAI_UTC']
    EOP_data['LOD'] = interp[3]
    EOP_data['ddPsi'] = interp[4]*arcsec2rad
    EOP_data['ddEps'] = interp[5]*arcsec2rad
    EOP_data['dX'] = interp[6]*arcsec2rad
    EOP_data['dY'] = interp[7]*arcsec2rad
    

    return EOP_data


def eop_read_line(line):
    '''
    This function reads a single line of the EOP data file and returns the
    floating point values of each parameter per the format given below.
    
    http://celestrak.com/SpaceData/EOP-format.asp
    
    Columns   Description
    001-004	Year
    006-007	Month (01-12)
    009-010	Day
    012-016	Modified Julian Date (Julian Date at 0h UT minus 2400000.5)
    018-026	x (arc seconds)
    028-036	y (arc seconds)
    038-047	UT1-UTC (seconds)
    049-058	Length of Day (seconds)
    060-068	delta-Delta-psi (arc seconds)
    070-078	delta-Delta-epsilon (arc seconds)
    080-088	delta-X (arc seconds)
    090-098	delta-Y (arc seconds)
    100-102	Delta Atomic Time, TAI-UTC (seconds)
    
    Parameters
    ------
    line : string
        single line from EOP data file (format as specified)
    
    Returns
    ------
    line_array : 1D numpy array
        EOP parameters in 1D array    
    '''
    
    MJD = float(line[11:16])
    xp = float(line[17:26])
    yp = float(line[27:36])
    UT1_UTC = float(line[37:47])
    LOD = float(line[48:58])
    ddPsi = float(line[59:68])
    ddEps = float(line[69:78])
    dX = float(line[79:88])
    dY = float(line[89:98])
    TAI_UTC = float(line[99:102])
    
    line_array = np.array([MJD, xp, yp, UT1_UTC, LOD, ddPsi, ddEps, dX, dY,
                           TAI_UTC])
    
    return line_array

        
def get_nutation_data(TEME_flag=True):
    '''
    This function retrieves nutation data from the IAU 1980 CSV file included
    in this distribution, compiled from Reference [2].  For the conversion
    from TEME to GCRF, it is recommended to reduce the coefficients to the
    four largest terms, which can be done using the optional input flag.
    
    Parameters
    ------
    TEME_flag : boolean, optional
        flag to determine whether to reduce coefficient array to four largest
        rows (default=True)
    
    Returns
    ------
    IAU1980_nutation : 2D numpy array
        array of nutation coefficients    
    '''
    
    df = pd.read_csv(os.path.join(data_dir, 'IAU1980_nutation.csv'))
    
    # For TEME-GCRF conversion, reduce to 4 largest entries      
    if TEME_flag:          
        df = df.loc[np.abs(df['dPsi']) > 2000.]
        
    IAU1980_nutation = df.values
    
    return IAU1980_nutation


def get_XYs2006_alldata():
    
    # Load data
    XYs_df = pd.read_csv(os.path.join(data_dir, 'IAU2006_XYs.csv'))
    
    return XYs_df


def init_XYs2006(TT1, TT2, XYs_df=[]):
    '''
    This loads the data file containing CIP coordinates, X and Y, as well as 
    the CIO locator, s. The data file is named IAU2006_XYs.csv.
    X, Y, and s are tabulated from 1980 to 2050 every day at 0h 
    Terrestrial Time (TT). 

    The data is loaded into a single matrix and then trimmed. The resulting 
    XYsdata matrix contains data from 8 days before TT1 to 8 days after TT2 
    for interpolation purposes.

    NOTE: although TT is used for input, UTC can also be used without any
        issues.  The difference between TT and UTC is about 60 seconds. 
        Since data is trimmed for +/- 8 days on either side of the input
        times, UTC is fine.  The resulting data matrix will still contain
        X,Y,s data for 0h of TT though.
        
    Parameters
    ------
    TT1 : datetime object
        start time in TT
    TT2 : datetime object
        final time in TT
        
    Returns
    ------
    XYs_data : nx7 numpy array
        each row contains data for 0h TT for consecutive days
        [yr, mo, day, MJD, X, Y, s]
    
    '''
    
    # Load data if needed
    if len(XYs_df) == 0:        
        XYs_df = pd.read_csv(os.path.join(data_dir, 'IAU2006_XYs.csv'))        
        
    XYs_alldata = XYs_df.values
    
    # Compute MJD and round to nearest whole day
    MJD1 = int(round(dt2mjd(TT1)))
    MJD2 = int(round(dt2mjd(TT2)))
    
    # Number of additional data points to include on either side
    num = 10
    
    # Find rows
    MJD_data = XYs_df['MJD (0h TT)'].tolist()
    
    if MJD1 < MJD_data[0]:
        print('Error: init_XYs2006 start date before first XYs time')
    elif MJD1 <= MJD_data[0] + num:
        row1 = 0
    elif MJD1 > MJD_data[-1]:
        print('Error: init_XYs2006 start date after last XYs time')
    else:
        row1 = MJD_data.index(MJD1) - num
    
    if MJD2 < MJD_data[0]:
        print('Error: init_XYs2006 final date before first XYs time')
    elif MJD2 >= MJD_data[0] - num:
        row2 = -1
    elif MJD2 > MJD_data[-1]:
        print('Error: init_XYs2006 final date after last XYs time')
    else:
        row2 = MJD_data.index(MJD2) + num
        
    if row2 == -1:
        XYs_data = XYs_alldata[row1:, :]
    else:
        XYs_data = XYs_alldata[row1:row2, :]
    
    return XYs_data


def get_XYs(XYs_data, TT_JD):
    '''
    Interpolates X,Y, and s loaded by init_XYs2006.m using an 11th-order 
    Lagrange interpolation method. The init_XYsdata function must be 
    called before get_XYs is used.  This function uses the XYs data set that 
    has been loaded as a matrix. Each of the three values listed below are 
    tabulated at 0h TT of each day.
    
    Parameters
    ------
    XYs_data : nx7 numpy array
        each row contains data for 0h TT for consecutive days
        [yr, mo, day, MJD, X, Y, s]
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT
 
    
    Returns
    ------
    X : float
        x-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    Y : float
        y-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    s : float
        Celestial Intermediate Origin (CIO) locator [rad]
    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (math.pi/180.)
    
    # Compute MJD
    TT_MJD = TT_JD - 2400000.5
    
    # Compute interpolation
    XYs = interp_lagrange(XYs_data[:,3], XYs_data[:,4:], TT_MJD, 11)
    
    X = float(XYs[0,0])*arcsec2rad
    Y = float(XYs[0,1])*arcsec2rad
    s = float(XYs[0,2])*arcsec2rad
    
    return X, Y, s


def interp_lagrange(X, Y, xx, p):
    '''
    This function interpolates data using Lagrange method of order P
    
    Parameters
    ------
    X : 1D numpy array
        x-values of data to interpolate
    Y : 2D numpy array
        y-values of data to interpolate
    xx : float
        single x value to interpolate at
    p : int
        order of interpolation
    
    Returns
    ------
    yy : 1D numpy array
        interpolated y-value(s)
        
    References
    ------
    [1] Kharab, A., An Introduction to Numerical Methods: A MATLAB 
        Approach, 2nd ed., 2005.
            
    '''
    
    # Number of data points to use for interpolation (e.g. 8,9,10...)
    N = p + 1

    if (len(X) < N):
        print('Not enough data points for desired Lagrange interpolation!')
        
    # Compute number of elements on either side of middle element to grab
    No2 = 0.5*N
    nn  = int(math.floor(No2))
    
    # Find index such that X[row0] < xx < X[row0+1]
    row0 = list(np.where(X < xx)[0])[-1]
    
    # Trim data set
    # N is even (p is odd)    
    if (No2-nn == 0): 
        
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn-1):
            row0 = nn-1
        elif (row0 > (len(X)-nn)):  # (row0 == length(X))            
            row0 = len(X) - nn - 1        
    
        # Trim to relevant data points
        X = X[row0-nn+1 : row0+nn+1]
        Y = Y[row0-nn+1 : row0+nn+1, :]


    # N is odd (p is even)
    else:
    
        # adjust row0 in case near data set endpoints
        if (N == len(X)) or (row0 < nn):
            row0 = nn
        elif (row0 > len(X)-nn):
            row0 = len(X) - nn - 1
        else:
            if (xx-X(row0) > 0.5) and (row0+1+nn < len(X)):
                row0 = row0 + 1
    
        # Trim to relevant data points
        X = X[row0-nn:row0+nn+1]
        Y = Y[row0-nn:row0+nn+1, :]
        
    # Compute coefficients
    Pj = np.ones((1,N))
    
    for jj in range(N):
        for ii in range(N):
            
            if jj != ii:
                Pj[0, jj] = Pj[0, jj] * (-xx+X[ii])/(-X[jj]+X[ii])
    
    
    yy = np.dot(Pj, Y)
    
    return yy


def compute_precession_IAU1976(TT_cent):
    '''    
    This function computes the IAU1976 precession matrix required for the 
    frame transformation between GCRF and Mean of Date (MOD).
    
    r_GCRF = P76 * r_MOD
    
    Parameters
    ------
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    
    Returns
    ------
    P76 : 3x3 numpy array
        precession matrix to compute frame rotation    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (math.pi/180.)

    # Table values in arcseconds
    Pcoef = np.array([[2306.2181,   0.30188,   0.017998], 
                      [2004.3109,  -0.42665,  -0.041833],
                      [2306.2181,   1.09468,   0.018203]])

    
 
    # Multiply by [TT, TT**2, TT**3]^T  (creates column vector)
    # M[0] = zeta, M[1] = theta, M[2] = z
    vec = np.array([[TT_cent], [TT_cent**2.], [TT_cent**3.]])
    M = np.dot(Pcoef, vec) * arcsec2rad;

    # Construct IAU 1976 Precession Matrix
    # P76 = ROT3(zeta) * ROT2(-theta) * ROT3(z);    
    czet = math.cos(M[0])
    szet = math.sin(M[0])
    cth  = math.cos(M[1])
    sth  = math.sin(M[1])
    cz   = math.cos(M[2])
    sz   = math.sin(M[2])
    
    
    P76 = np.array([[cth*cz*czet-sz*szet,   sz*cth*czet+szet*cz,  sth*czet],
                    [-szet*cth*cz-sz*czet, -sz*szet*cth+cz*czet, -sth*szet],
                    [-sth*cz,              -sth*sz,                    cth]])
    
    
    return P76


def compute_nutation_IAU1980(IAU1980nut, TT_cent, ddPsi, ddEps):
    '''
    This function computes the IAU1980 nutation matrix required for the 
    frame transformation between Mean of Date (MOD) and True of Date (TOD).
    
    r_MOD = N80 * r_TOD
    
    Parameters
    ------
    IAU1980nut : 2D numpy array
        array of nutation coefficients  
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    ddPsi : float
        EOP parameter for correction to nutation in longitude [rad]
    ddEps : float
        EOP parameter for correction to nutation in obliquity [rad]
    
    Returns
    ------  
    N80 : 3x3 numpy array
        nutation matrix to compute frame rotation
    FA : 5x1 numpy array
        fundamental arguments of nutation (Delauney arguments)
    Eps0 : float
        mean obliquity of the ecliptic [rad]
    Eps_true : float
        true obliquity of the ecliptic [rad]
    dPsi : float
        nutation in longitude [rad]
    dEps : float
        nutation in obliquity [rad]    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (math.pi/180.)
    
    # Compute fundamental arguments of nutation
    FA = compute_fundarg_IAU1980(TT_cent)
    
    # Compute Nutation in longitude and obliquity  
    phi = np.dot(IAU1980nut[:,0:5], FA)  # column vector
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # Calculate Nutation in Longitude, rad    
    dPsi_vec = IAU1980nut[:,5] + IAU1980nut[:,6]*TT_cent
    dPsi_sum = float(np.dot(dPsi_vec.T, sphi))
    dPsi = ddPsi + dPsi_sum*0.0001*arcsec2rad

    # Calculate Nutation in Obliquity, rad
    dEps_vec = IAU1980nut[:,7] + IAU1980nut[:,8]*TT_cent
    dEps_sum = float(np.dot(dEps_vec.T, cphi))
    dEps = ddEps + dEps_sum*0.0001*arcsec2rad
        
    # Mean Obliquity of the Ecliptic, rad
    Eps0 = (((0.001813*TT_cent - 0.00059)*TT_cent - 46.8150)*TT_cent + 
             84381.448)*arcsec2rad
    
    # True Obliquity of the Ecliptic, rad
    Eps_true = Eps0 + dEps
    
    # Construct Nutation matrix
    # N = ROT1(-Eps_0 * ROT3(dPsi) * ROT1(Eps_true)
    cep  = math.cos(Eps0)
    sep  = math.sin(Eps0)
    cPsi = math.cos(dPsi)
    sPsi = math.sin(dPsi)
    cept = math.cos(Eps_true)
    sept = math.sin(Eps_true)
    
    N80 = \
        np.array([[ cPsi,     sPsi*cept,              sept*sPsi             ],
                  [-sPsi*cep, cept*cPsi*cep+sept*sep, sept*cPsi*cep-sep*cept],
                  [-sPsi*sep, sep*cept*cPsi-sept*cep, sept*sep*cPsi+cept*cep]])
    
    return N80, FA, Eps0, Eps_true, dPsi, dEps


def compute_fundarg_IAU1980(TT_cent):
    '''
    This function computes the fundamental arguments (Delauney arguments) due
    to luni-solar forces.
    
    Parameters
    ------
    TT_cent : float
        Terrestrial Time (TT) since J2000 in Julian centuries
    
    Returns
    ------
    DA_vec : 5x1 numpy array
        fundamental arguments of nutation (Delauney arguments) [rad]
    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (math.pi/180.)
    arcsec360 = 3600.*360.
    
    # Construct table for fundamental arguments of nutation
    #  Units: col 1,   degrees
    #         col 2-5, arcseconds
    #  Note: These values come from page 23 of [3].
    
    # Delauney Arguments
    DA = np.array([[134.96340251,  1717915923.2178,  31.8792,  0.051635, -0.00024470], # M_moon (l)
                   [357.52910918,  129596581.04810, -0.55320,  0.000136, -0.00001149], # M_sun (l')
                   [93.27209062,   1739527262.8478, -12.7512, -0.001037,  0.00000417], # u_Mmoon (F)
                   [297.85019547,  1602961601.2090, -6.37060,  0.006593, -0.00003169], # D_sun (D)
                   [125.04455501, -6962890.2665000,  7.47220,  0.007702, -0.00005939]]) # Om_moon (Omega)

    # Mulitply by [3600., TT, TT**2, TT**3, TT**4]^T to get column vector 
    # in arcseconds
    vec = np.array([[3600.], [TT_cent], [TT_cent**2.], [TT_cent**3],
                    [TT_cent**4]])
    DA_vec = np.dot(DA, vec)
    
    # Get fractional part of circle and convert to radians
    DA_vec = np.mod(DA_vec, arcsec360) * arcsec2rad
    
    return DA_vec


def eqnequinox_IAU1982_simple(dPsi, Eps0):
    '''
    This function computes the IAU1982 equation of the equinoxes matrix 
    required for the frame transformation between True of Date (TOD) and
    True Equator Mean Equinox (TEME).    
    
    r_TOD = R * r_TEME
    
    Parameters
    ------
    dPsi : float
        nutation in longitude [rad]
    Eps0 : float
        mean obliquity of the ecliptic [rad]
    
    Returns
    ------
    R : 3x3 numpy array
        matrix to compute frame rotation    
    '''
    
    # Equation of the Equinoxes (simple form for use with TEME) (see [1])
    Eq1982 = dPsi*math.cos(Eps0) # rad
    
    # Construct Rotation matrix
    # R  = ROT3(-Eq1982) (Eq. 3-80 in [1])
    cEq = math.cos(Eq1982)
    sEq = math.sin(Eq1982)

    R = np.array([[cEq,    -sEq,    0.],
                  [sEq,     cEq,    0.],
                  [0.,      0.,     1.]])
    

    return R


def compute_polarmotion(xp, yp, TT_cent):
    '''
    This function computes the polar motion transformation matrix required
    for the frame transformation between TIRS and ITRF.    
    
    r_TIRS = W * r_ITRF
    
    Parameters
    ------
    xp : float
        x-coordinate of the CIP unit vector [rad]
    yp : float
        y-coordinate of the CIP unit vector [rad]
    TT_cent : float
        Julian centuries since J2000 TT
    
    Returns
    ------
    W : 3x3 numpy array
        matrix to compute frame rotation    
    '''
    
    # Conversion
    arcsec2rad  = (1./3600.) * (math.pi/180.)
    
    # Calcuate the Terrestrial Intermediate Origin (TIO) locator 
    # Eq 5.13 in [5]
    sp = -0.000047 * TT_cent * arcsec2rad
    
    # Construct rotation matrix
    # W = ROT3(-sp)*ROT2(xp)*ROT1(yp) (Eq. 5.3 in [5])
    cx = math.cos(xp)
    sx = math.sin(xp)
    cy = math.cos(yp)
    sy = math.sin(yp)
    cs = math.cos(sp)
    ss = math.sin(sp)
    
    W = np.array([[cx*cs,  -cy*ss + sy*sx*cs,  -sy*ss - cy*sx*cs],
                  [cx*ss,   cy*cs + sy*sx*ss,   sy*cs - cy*sx*ss],
                  [   sx,             -sy*cx,              cy*cx]])
    
    
    return W


def compute_ERA(UT1_JD):
    '''
    This function computes the Earth Rotation Angle (ERA) and the ERA rotation
    matrix based on UT1 time. The ERA is modulated to lie within [0,2*pi] and 
    is computed using the precise equation given by Eq. 5.15 in [5].

    The ERA is the angle between the Celestial Intermediate Origin, CIO, and 
    Terrestrial Intermediate Origin, TIO (a reference meridian 100m offset 
    from Greenwich meridian).
    
    r_CIRS = R * r_TIRS
   
    Parameters
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
        
    Returns
    ------
    R : 3x3 numpy array
        matrix to compute frame rotation
    
    '''
    
    # Compute ERA based on Eq. 5.15 of [5]
    d,i = math.modf(UT1_JD)
    ERA = 2.*math.pi*(d + 0.7790572732640 + 0.00273781191135448*(UT1_JD - 2451545.))
    
    # Compute ERA between [0, 2*pi]
    ERA = ERA % (2.*math.pi)
    if ERA < 0.:
        ERA += 2*math.pi
        
#    print(ERA)
    
    # Construct rotation matrix
    # R = ROT3(-ERA)
    ct = math.cos(ERA)
    st = math.sin(ERA)
    R = np.array([[ct, -st, 0.],
                  [st,  ct, 0.],
                  [0.,  0., 1.]])

    return R


def compute_BPN(X, Y, s):
    '''
    This function computes the Bias-Precession-Nutation matrix required for the 
    CIO-based transformation between the GCRF/ITRF frames.
    
    r_GCRS = BPN * r_CIRS
    
    Parameters
    ------
    X : float
        x-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    Y : float
        y-coordinate of the Celestial Intermediate Pole (CIP) [rad]
    s : float
        Celestial Intermediate Origin (CIO) locator [rad]
    
    Returns
    ------
    BPN : 3x3 numpy array
        matrix to compute frame rotation
    
    '''
    
    # Compute z-coordinate of CIP
    Z  = np.sqrt(1 - X*X - Y*Y)
    aa = 1./(1. + Z)
    
    # Construct BPN (Bias-Precession-Nutation Matrix) 
    # Eq. 5.1 in [5]:  BPN = [f(X,Y)]*ROT3(s)
    cs = math.cos(s)
    ss = math.sin(s)
    
    f = np.array([[1-aa*X*X,    -aa*X*Y,                X],
                  [ -aa*X*Y,   1-aa*Y*Y,                Y], 
                  [      -X,         -Y,   1-aa*(X*X+Y*Y)]])
    
    R3 = np.array([[ cs,  ss,  0.],
                   [-ss,  cs,  0.],
                   [ 0.,  0.,  1.]])
    
    BPN = np.dot(f, R3)
    
    return BPN


def gcrf2itrf(r_GCRF, v_GCRF, UTC, EOP_data, XYs_df=[]):
    '''
    This function converts a position and velocity vector in the GCRF(ECI)
    frame to the ITRF(ECEF) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    
    '''
    
    # Form column vectors
    r_GCRF = np.reshape(r_GCRF, (3,1))
    v_GCRF = np.reshape(v_GCRF, (3,1))
        
    # Compute UT1 in JD format
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = init_XYs2006(UTC, UTC, XYs_df)
    
    X, Y, s = get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = compute_BPN(X, Y, s)
    
    # Transform position vector
    eci2ecef = np.dot(W.T, np.dot(R.T, BPN.T))
    r_ITRF = np.dot(eci2ecef, r_GCRF)

    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
        
    v_ITRF = np.dot(W.T, (np.dot(R.T, np.dot(BPN.T, v_GCRF)) - 
                          np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0)))
    
    
    
    return r_ITRF, v_ITRF


def itrf2gcrf(r_ITRF, v_ITRF, UTC, EOP_data, XYs_df=[]):
    '''
    This function converts a position and velocity vector in the ITRF(ECEF)
    frame to the GCRF(ECI) frame using the IAU 2006 precession and 
    IAU 2000A_R06 nutation theories. This routine employs a hybrid of the 
    "Full Theory" using Fukushima-Williams angles and the CIO-based method.  

    Specifically, this routine interpolates a table of X,Y,s values and then
    uses them to construct the BPN matrix directly.  The X,Y,s values in the 
    data table were generated using Fukushima-Williams angles and the 
    IAU 2000A_R06 nutation theory.  This general scheme is outlined in [3]
    and [4].
    
    Parameters
    ------
    r_ITRF : 3x1 numpy array
        position vector in ITRF
    v_ITRF : 3x1 numpy array
        velocity vector in ITRF
    UTC : datetime object
        time in UTC
    EOP_data : dictionary
        EOP data for the given time including pole coordinates and offsets,
        time offsets, and length of day  
    
    Returns
    ------
    r_GCRF : 3x1 numpy array
        position vector in GCRF
    v_GCRF : 3x1 numpy array
        velocity vector in GCRF
    
    '''
    
    # Form column vectors
    r_ITRF = np.reshape(r_ITRF, (3,1))
    v_ITRF = np.reshape(v_ITRF, (3,1))
    
    # Compute UT1 in JD format
    UT1_JD = utcdt2ut1jd(UTC, EOP_data['UT1_UTC'])
    
    # Compute TT in JD format
    TT_JD = utcdt2ttjd(UTC, EOP_data['TAI_UTC'])
    
    # Compute TT in centuries since J2000 epoch
    TT_cent = jd2cent(TT_JD)
    
    # Construct polar motion matrix (ITRS to TIRS)
    W = compute_polarmotion(EOP_data['xp'], EOP_data['yp'], TT_cent)
    
    # Contruct Earth rotaion angle matrix (TIRS to CIRS)
    R = compute_ERA(UT1_JD)
    
    # Construct Bias-Precessino-Nutation matrix (CIRS to GCRS/ICRS)
    XYs_data = init_XYs2006(UTC, UTC, XYs_df)
    X, Y, s = get_XYs(XYs_data, TT_JD)
    
    # Add in Free Core Nutation (FCN) correction
    X = EOP_data['dX'] + X  # rad
    Y = EOP_data['dY'] + Y  # rad
    
    # Compute Bias-Precssion-Nutation (BPN) matrix
    BPN = compute_BPN(X, Y, s)
    
    # Transform position vector
    ecef2eci = np.dot(BPN, np.dot(R, W))
    r_GCRF = np.dot(ecef2eci, r_ITRF)
    
    # Transform velocity vector
    # Calculate Earth rotation rate, rad/s (Vallado p227)
    wE = 7.29211514670639e-5*(1 - EOP_data['LOD']/86400)                    
    r_TIRS = np.dot(W, r_ITRF)
    
    v_GCRF = np.dot(BPN, np.dot(R, (np.dot(W, v_ITRF) + 
                    np.cross(np.array([[0.],[0.],[wE]]), r_TIRS, axis=0))))  
    
    return r_GCRF, v_GCRF


def dt2jd(dt):
    '''
    This function converts a calendar time to Julian Date (JD) fractional days
    since 12:00:00 Jan 1 4713 BC.  No conversion between time systems is 
    performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    '''
    
    MJD = dt2mjd(dt)
    JD = MJD + 2400000.5
    
    return JD


def dt2mjd(dt):
    '''
    This function converts a calendar time to Modified Julian Date (MJD)
    fractional days since 1858-11-17 00:00:00.  No conversion between time
    systems is performed.
    
    Parameters
    ------
    dt : datetime object
        time in calendar format
    
    Returns
    ------
    MJD : float
        fractional days since 1858-11-17 00:00:00
    '''
    
    MJD_datetime = datetime(1858, 11, 17, 0, 0, 0)
    delta = dt - MJD_datetime
    MJD = delta.total_seconds()/timedelta(days=1).total_seconds()
    
    return MJD


def utcdt2ut1jd(UTC, UT1_UTC):
    '''
    This function converts a UTC time to UT1 in Julian Date (JD) format.
    
    UT1_UTC = UT1 - UTC
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    UT1_UTC : float
        EOP parameter, time offset between UT1 and UTC 
    
    Returns
    ------
    UT1_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC UT1
    
    '''
    
    UTC_JD = dt2jd(UTC)
    UT1_JD = UTC_JD + (UT1_UTC/86400.)
    
    return UT1_JD


def utcdt2ttjd(UTC, TAI_UTC):
    '''
    This function converts a UTC time to Terrestrial Time (TT) in Julian Date
    (JD) format.
    
    UTC = TAI - TAI_UTC
    TT = TAI + 32.184
    
    Parameters
    ------
    UTC : datetime object
        time in UTC
    TAI_UTC : float
        EOP parameter, time offset between atomic time (TAI) and UTC 
        (10 + leap seconds)        
    
    Returns
    ------
    TT_JD : float
        fractional days since 12:00:00 Jan 1 4713 BC TT
    
    '''
    
    UTC_JD = dt2jd(UTC)
    TT_JD = UTC_JD + (TAI_UTC + 32.184)/86400.
    
    return TT_JD


def jd2cent(JD):
    '''
    This function computes Julian centuries since J2000. No conversion between
    time systems is performed.
    
    Parameters
    ------
    JD : float
        fractional days since 12:00:00 Jan 1 4713 BC
    
    Returns
    ------
    cent : float
        fractional centuries since 12:00:00 Jan 1 2000
    '''
    
    cent = (JD - 2451545.)/36525.
    
    return cent