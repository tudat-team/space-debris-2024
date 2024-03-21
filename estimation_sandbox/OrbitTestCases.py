import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Estimators



###############################################################################
# Test Case Setup and Execution - Orbit Model
###############################################################################


def generate_orbit_inputs(setup_file, m, orbit_regime='LEO'):
    
    arcsec2rad = (1./3600.)*np.pi/180.
    
    params = {}
    params['Po'] = np.diag([1e8, 1e8, 1e8, 100., 100., 100.])
    
    if orbit_regime == 'LEO':
        params['Xo_true'] = np.reshape([757700., 5222607., 4851500., 2213.21,
                                        4678.34, -5371.30], (6,1))
    if orbit_regime == 'GEO':
        params['Xo_true'] = np.reshape([3.57133892e7, 2.23987784e7, -6.69976699e4,
                                        -1.63445654e3, 2.60499214e3, 6.88190044], (6,1))
    
    if m == 3:
        params['Rk'] = np.diag([1., (360*arcsec2rad)**2., (360*arcsec2rad)**2.])
    elif m == 2:
        params['Rk'] = np.diag([arcsec2rad**2., arcsec2rad**2.])
        
    params['theta0'] = 0.
    params['dtheta'] = 7.2921158553e-5
    params['GM'] = 3.986004e14
    params['Re'] = 6378136.3
    params['J2'] = 1.082626683e-3
    params['sensor_ecef'] = np.reshape([-5466071., -2403990., 2242473.], (3,1))
    params['Qeci'] = 1e-10*np.diag([1., 1., 1.])
    params['Qric'] = 0*np.diag([1., 1., 1.])
    params['gap_seconds'] = 1e6
    params['alpha'] = 1.
        
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [params], pklFile, -1 )
    pklFile.close()
    
    
    return


def generate_orbit_truth(setup_file, truth_file, intfcn):
    
    # Load data
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    params = data[0]
    pklFile.close()
    
    # True initial state
    Xo = params['Xo_true']
    
    # Setup integrator
    step = 60.
    tvec = np.arange(0., 8.*3600.+1., step)
    
    # RK4
    # params['step'] = step
    # t_truth, Xt_mat, fcalls = Estimators.rk4(intfcn, tvec, Xo, params)    
    
    
    # Solve IVP
    method = 'DOP853'
    rtol = 1e-12
    atol = 1e-12
    tin = (tvec[0], tvec[-1])
    t_truth = tvec
    output = solve_ivp(intfcn,tin,Xo.flatten(),method=method,args=(params,),rtol=rtol,atol=atol,t_eval=tvec)
    
    Xt_mat = output['y'].T
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [t_truth, Xt_mat], pklFile, -1 )
    pklFile.close()
    
    return


def generate_orbit_meas(setup_file, truth_file, meas_file):
    
    rad2arcsec = 3600.*180./np.pi
    
    # Load data
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    params = data[0]
    pklFile.close()
    
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    Xt_mat = data[1]
    pklFile.close()
    
    # Random seed
    np.random.seed(1)
    
    # Retrieve data from params
    theta0 = params['theta0']
    dtheta = params['dtheta']
    sensor_ecef = params['sensor_ecef']   
    
    # Measurement noise standard deviation
    Rk = params['Rk']
    m = int(Rk.shape[0])
    if m == 3:
        sig_rho = np.sqrt(Rk[0,0])
        sig_ra = np.sqrt(Rk[1,1])
        sig_dec = np.sqrt(Rk[2,2])
    elif m == 2:
        sig_ra = np.sqrt(Rk[0,0])
        sig_dec = np.sqrt(Rk[1,1])
        
        
    # Loop over times
    t_obs = t_truth
    obs_data = np.zeros((len(t_obs), m))
    resids = np.zeros((len(t_obs), m))
    for kk in range(len(t_obs)):
        
        # Current time and true position states
        tk = t_obs[kk]
        r_eci = Xt_mat[kk,0:3].reshape(3,1)
        
        # Compute earth rotation and sensor position in ECI
        theta = theta0 + dtheta*tk
        sensor_eci = Estimators.GMST_ecef2eci(sensor_ecef, theta)

        # Compute range and line of sight unit vector
        rho_eci = r_eci - sensor_eci
        rho = np.linalg.norm(rho_eci)
        rho_hat_eci = rho_eci/rho
        
        # Compute topocentric right ascension and declination
        ra = math.atan2(rho_hat_eci[1,0], rho_hat_eci[0,0]) 
        dec = math.asin(rho_hat_eci[2,0])
        
        if m == 3:
            obs_data[kk,0] = rho + np.random.randn()*sig_rho
            obs_data[kk,1] = ra + np.random.randn()*sig_ra
            obs_data[kk,2] = dec + np.random.randn()*sig_dec
            
            resids[kk,0] = obs_data[kk,0] - rho
            resids[kk,1] = obs_data[kk,1] - ra
            resids[kk,2] = obs_data[kk,2] - dec
            
        elif m == 2:
            obs_data[kk,0] = ra + np.random.randn()*sig_ra
            obs_data[kk,1] = dec + np.random.randn()*sig_dec
            
            resids[kk,0] = obs_data[kk,0] - ra
            resids[kk,1] = obs_data[kk,1] - dec
        
    
    
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [t_obs, obs_data], pklFile, -1 )
    pklFile.close()
    
    if m == 3:
    
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_obs/3600., resids[:,0], 'k.')
        plt.yticks([-3, -1, 1, 3])
        plt.ylabel('Range [m]')
        plt.title('Measurement Noise')
        plt.subplot(3,1,2)
        plt.plot(t_obs/3600., resids[:,1]*rad2arcsec, 'k.')
        plt.yticks([-720, 0, 720])
        plt.ylabel('RA [arcsec]')
        plt.subplot(3,1,3)
        plt.plot(t_obs/3600., resids[:,2]*rad2arcsec, 'k.')
        plt.yticks([-720, 0, 720])
        plt.ylabel('DEC [arcsec]')
        plt.xlabel('Time [hours]')
        
    elif m == 2:
    
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t_obs/3600., resids[:,0]*rad2arcsec, 'k.')
        plt.yticks([-3, -1, 1, 3])
        plt.title('Measurement Noise')
        plt.ylabel('RA [arcsec]')
        plt.subplot(2,1,2)
        plt.plot(t_obs/3600., resids[:,1]*rad2arcsec, 'k.')
        plt.yticks([-3, -1, 1, 3])
        plt.ylabel('DEC [arcsec]')
        plt.xlabel('Time [hours]')
           
    
    plt.show()
    
    
    return


def run_estimator_orbit_model(setup_file, truth_file, meas_file, output_file,
                              intfcn, meas_fcn, estimator, Qeci=None, Qric=None):
    
    # Load measurement and truth data
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    params = data[0]
    pklFile.close()
    
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    Xt_mat = data[1]
    pklFile.close()
    
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    t_obs = data[0]
    obs_data = data[1]
    pklFile.close()    
    
    # Update process noise
    if Qeci is not None:
        params['Qeci'] = Qeci
    if Qric is not None:
        params['Qric'] = Qric
    
    
    # Orbit Model Params
    params['step'] = 60.
    pert_vect = np.multiply(np.sqrt(np.diag(params['Po'])), np.random.randn(6,))
    Xo_ref = params['Xo_true'] + pert_vect.reshape(6,1)
    Po = params['Po']
    
    # Execute batch estimator
    if estimator == 'batch':
        t_output = t_truth
        t_output, Xref_mat, P_mat, resids = Estimators.ls_batch(Xo_ref, Po, t_obs, t_output, obs_data, intfcn, meas_fcn, params)
    
    # Execute UKF
    if estimator == 'ukf':
        t_output, Xref_mat, P_mat, resids = Estimators.ukf(Xo_ref, Po, t_obs, obs_data, intfcn, meas_fcn, params)
    
    
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump( [t_output, Xref_mat, P_mat, resids], pklFile, -1 )
    pklFile.close()
    
    return





if __name__ == '__main__':
    
    plt.close('all')
    
    
    if not os.path.exists('orbit_model'):
        os.makedirs('orbit_model')
    
    # Setup and Run TwoBody Orbit with Range, RA, DEC measurements
    datadir = 'orbit_model'
    setup_file = os.path.join(datadir, 'orbit_model_setup_rgradec.pkl')
    truth_file = os.path.join(datadir, 'orbit_model_truth.pkl')
    meas_file = os.path.join(datadir, 'orbit_model_meas_rgradec.pkl')
    output_file = os.path.join(datadir, 'orbit_model_batch_output_rgradec.pkl')
    
    # Choose dynamics function for truth model, can include a J2 perturbation not modeled in filter
    intfcn = Estimators.int_twobody    
    # intfcn = Estimators.int_twobody_j2
    
    generate_orbit_inputs(setup_file, 3, 'LEO')
    generate_orbit_truth(setup_file, truth_file, intfcn)
    generate_orbit_meas(setup_file, truth_file, meas_file)
    
    
    # Batch estimator
    intfcn = Estimators.int_twobody_stm
    meas_fcn = Estimators.H_rgradec
    run_estimator_orbit_model(setup_file, truth_file, meas_file, output_file,
                              intfcn, meas_fcn, 'batch')
    
    # UKF
    output_file = os.path.join(datadir, 'orbit_model_ukf_output_rgradec.pkl')
    intfcn = Estimators.int_twobody_ukf
    meas_fcn = Estimators.unscented_rgradec
    
    # Set process noise
    Qeci = 1e-12*np.diag([1., 1., 1.])
    Qric = 0*np.diag([1., 1., 1.])
    run_estimator_orbit_model(setup_file, truth_file, meas_file, output_file,
                              intfcn, meas_fcn, 'ukf', Qeci=Qeci, Qric=Qric)
    
    
    
    # Setup and Run TwoBody Orbit with only RA, DEC measurements
    datadir = 'orbit_model'
    setup_file = os.path.join(datadir, 'orbit_model_setup_radec.pkl')
    truth_file = os.path.join(datadir, 'orbit_model_truth.pkl')
    meas_file = os.path.join(datadir, 'orbit_model_meas_radec.pkl')
    output_file = os.path.join(datadir, 'orbit_model_batch_output_radec.pkl')
    
    # Choose dynamics function for truth model, can include a J2 perturbation not modeled in filter
    intfcn = Estimators.int_twobody    
    # intfcn = Estimators.int_twobody_j2

    generate_orbit_inputs(setup_file, 2, 'GEO')
    generate_orbit_truth(setup_file, truth_file, intfcn)
    generate_orbit_meas(setup_file, truth_file, meas_file)
    
    
    # Batch    
    intfcn = Estimators.int_twobody_stm
    meas_fcn = Estimators.H_radec
    run_estimator_orbit_model(setup_file, truth_file, meas_file, output_file,
                              intfcn, meas_fcn, 'batch')
    
    # UKF
    output_file = os.path.join(datadir, 'orbit_model_ukf_output_radec.pkl')
    intfcn = Estimators.int_twobody_ukf
    meas_fcn = Estimators.unscented_radec
    
    # Set process noise
    Qeci = 1e-12*np.diag([1., 1., 1.])
    Qric = 0*np.diag([1., 1., 1.])
    run_estimator_orbit_model(setup_file, truth_file, meas_file, output_file,
                              intfcn, meas_fcn, 'ukf', Qeci=Qeci, Qric=Qric)