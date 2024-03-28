import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import Estimators



###############################################################################
# Test Case Setup and Execution - Linear Model
###############################################################################

def generate_linear_inputs(setup_file):
    
    params = {}
    params['Po'] = np.diag([10000., 25.])
    params['Xo_true'] = np.reshape([0., 5.], (2,1))
    params['Rk'] = np.diag([1.])
    params['Q'] = np.diag([1e-10])
    params['alpha'] = 1.
    params['gap_seconds'] = 1000.
    
    # Can model a constant acceleration (e.g. friction)
    params['acc'] = -0.001 
        
    pklFile = open( setup_file, 'wb' )
    pickle.dump( [params], pklFile, -1 )
    pklFile.close()
    
    
    return


def generate_linear_truth(setup_file, truth_file, intfcn):
    
    # Load data
    pklFile = open(setup_file, 'rb' )
    data = pickle.load( pklFile )
    params = data[0]
    pklFile.close()
    
    # True initial state
    Xo = params['Xo_true']
    
    # Setup integrator
    step = 10.
    tvec = np.arange(0., 901., step)
    
    # RK4
    params['step'] = step
    t_truth, Xt_mat, fcalls = Estimators.rk4(intfcn, tvec, Xo, params)    
    
    
    # # Solve IVP
    # method = 'DOP853'
    # rtol = 1e-12
    # atol = 1e-12
    # tin = (tvec[0], tvec[-1])
    # t_truth = tvec
    # output = solve_ivp(intfcn,tin,Xo.flatten(),method=method,args=(params,),rtol=rtol,atol=atol,t_eval=tvec)
    
    # Xt_mat = output['y'].T
    
    pklFile = open( truth_file, 'wb' )
    pickle.dump( [t_truth, Xt_mat], pklFile, -1 )
    pklFile.close()
    
    return


def generate_linear_meas(setup_file, truth_file, meas_file):
    
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
    
    # Measurement noise standard deviation
    std = np.sqrt(params['Rk'][0,0])
    
    # Compute noisy range measurements from truth data
    t_obs = t_truth
    rho = Xt_mat[:,0] + std*np.random.randn(len(t_obs),)
    obs_data = np.reshape(rho, (len(t_obs), 1))
    
    pklFile = open( meas_file, 'wb' )
    pickle.dump( [t_obs, obs_data], pklFile, -1 )
    pklFile.close()
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t_obs/60., Xt_mat[:,0], 'k')
    plt.xticks([0, 5, 10, 15])
    plt.yticks([0, 5000])
    plt.ylabel('x [m]')
    plt.title('Truth and Measurement Residuals')
    plt.subplot(3,1,2)
    plt.plot(t_obs/60., Xt_mat[:,1], 'k')
    plt.xticks([0, 5, 10, 15])
    plt.yticks([4, 5, 6])
    plt.ylabel('dx [m/s]')
    plt.subplot(3,1,3)
    plt.plot(t_obs/60., obs_data[:,0]-Xt_mat[:,0], 'k.')
    plt.ylabel('Range Resids [m]')
    plt.xlabel('Time [min]')
    plt.xticks([0, 5, 10, 15])
    plt.yticks([-2, 0, 2])
    
    plt.show()
    
    
    return


def run_estimator_linear_model(setup_file, truth_file, meas_file, output_file,
                               intfcn, meas_fcn, estimator, Q=None):
    
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
    
    
    # Linear Model Params
    params['step'] = 10.
    Xo_ref = np.reshape([100., 0], (2,1))
    Po = params['Po']
    
    # Update process noise if specified
    if Q is not None:
        params['Q'] = Q
    
    # Execute batch estimator
    if estimator == 'batch':
        t_output = t_truth
        t_output, Xref_mat, P_mat, resids = \
            Estimators.ls_batch(Xo_ref, Po, t_obs, t_output, obs_data, intfcn,
                                meas_fcn, params)
            
    if estimator == 'ukf':
        t_output, Xref_mat, P_mat, resids = \
            Estimators.ukf(Xo_ref, Po, t_obs, obs_data, intfcn, meas_fcn, params)
    
    # Save output
    pklFile = open( output_file, 'wb' )
    pickle.dump( [t_output, Xref_mat, P_mat, resids], pklFile, -1 )
    pklFile.close()
    
    return


def compute_linear_errors(setup_file, truth_file, output_file):
    
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
    
    pklFile = open(output_file, 'rb' )
    data = pickle.load( pklFile )
    t_output = data[0]
    Xref_mat = data[1]
    P_mat = data[2]
    resids = data[3]
    pklFile.close()
    
    # Compute state errors and standard deviations
    n = int(Xref_mat.shape[1])
    L = int(Xref_mat.shape[0])
    Xerr_mat = np.zeros(Xref_mat.shape)
    sig_mat = np.zeros((L,n))
    for kk in range(L):
        
        tk = t_output[kk]
        truth_ind = list(t_truth).index(tk)
        
        # Compute state error
        Xk_hat = Xref_mat[kk,:].reshape(n,1)
        Xk_true = Xt_mat[truth_ind,:].reshape(n,1)        
        Xerr_mat[kk,:] = (Xk_hat - Xk_true).flatten()
        
        # Compute standard deviations
        Pk = P_mat[:,:,kk]
        sig_mat[kk,:] = np.sqrt(np.diag(Pk)).flatten()
        
        
    # Compute RMS errors and resids
    RMS_pos = np.sqrt(np.mean(Xerr_mat[20:,0]**2.))
    RMS_vel = np.sqrt(np.mean(Xerr_mat[20:,1]**2.))
    RMS_rg = np.sqrt(np.mean(resids**2.))
        
    # Generate plots        
    plt.figure()
    
    plt.subplot(3,1,1)
    plt.plot(t_output/60.,   Xerr_mat[:,0], 'k.')
    plt.plot(t_output/60.,  3*sig_mat[:,0], 'k--')
    plt.plot(t_output/60., -3*sig_mat[:,0], 'k--')
    plt.xticks([0, 5, 10, 15])
    plt.ylabel('Pos Err [m]')
    
    plt.subplot(3,1,2)
    plt.plot(t_output/60.,   Xerr_mat[:,1], 'k.')
    plt.plot(t_output/60.,  3*sig_mat[:,1], 'k--')
    plt.plot(t_output/60., -3*sig_mat[:,1], 'k--')
    plt.xticks([0, 5, 10, 15])
    plt.ylabel('Vel Err [m/s]')
    
    plt.subplot(3,1,3)
    plt.plot(t_output/60., resids[:,0], 'k.')
    plt.xticks([0, 5, 10, 15])
    plt.yticks([-2, 2])
    plt.xlabel('Time [min]')
    plt.ylabel('Range Resids [m]')
        
        
    plt.show()    
    
    
    return RMS_pos, RMS_vel, RMS_rg


def snc_tuning(intfcn):
    
    # Setup and Run Linear Batch with Range Measurements
    datadir = 'linear_model'
    setup_file = os.path.join(datadir, 'linear_model_snc_setup.pkl')
    truth_file = os.path.join(datadir, 'linear_model_snc_truth.pkl')
    meas_file = os.path.join(datadir, 'linear_model_snc_meas.pkl')
    
    generate_linear_inputs(setup_file)
    generate_linear_truth(setup_file, truth_file, intfcn)
    generate_linear_meas(setup_file, truth_file, meas_file)
    
    
    Qmag = np.logspace(-2, -15, 14)
    RMS_pos_plot = []
    RMS_vel_plot = []
    RMS_rg_plot = []
    for ii in range(len(Qmag)):
        
        plt.close('all')
        
        output_file = os.path.join(datadir, 'linear_model_ukf_snc_output_' + str(ii) + '.pkl')
        intfcn = Estimators.int_constant_vel_ukf
        meas_fcn = Estimators.unscented_linear1d_rg
        
        Q = Qmag[ii]*np.array([[1.]])
        run_estimator_linear_model(setup_file, truth_file, meas_file, output_file,
                                   intfcn, meas_fcn, 'ukf', Q)
        
        RMS_pos, RMS_vel, RMS_rg = compute_linear_errors(setup_file, truth_file, output_file)
        
        RMS_pos_plot.append(RMS_pos)
        RMS_vel_plot.append(RMS_vel)
        RMS_rg_plot.append(RMS_rg)
        
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.semilogx(Qmag, RMS_pos_plot, 'ko-')
    plt.ylabel('RMS Pos Err [m]')
    plt.subplot(3,1,2)
    plt.semilogx(Qmag, RMS_vel_plot, 'ko-')
    plt.ylabel('RMS Vel Err [m/s]')
    plt.subplot(3,1,3)
    plt.semilogx(Qmag, RMS_rg_plot, 'ko-')
    plt.ylabel('RMS Rg Resids [m]')
    plt.xlabel('Q variance [$m^2/s^4$]')
    
    plt.show()
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    if not os.path.exists('linear_model'):
        os.makedirs('linear_model')
    
    # # Setup Linear Model Range Measurements
    # datadir = 'linear_model'
    # setup_file = os.path.join(datadir, 'linear_model_setup.pkl')
    # truth_file = os.path.join(datadir, 'linear_model_truth.pkl')
    # meas_file = os.path.join(datadir, 'linear_model_meas.pkl')
    # output_file = os.path.join(datadir, 'linear_model_batch_output.pkl')
    
    # intfcn = Estimators.int_constant_vel
    # generate_linear_inputs(setup_file)
    # generate_linear_truth(setup_file, truth_file, intfcn)
    # generate_linear_meas(setup_file, truth_file, meas_file)
    
    
    # # Run Batch Estimator
    # intfcn = Estimators.int_constant_vel_stm
    # meas_fcn = Estimators.H_linear_range
    # run_estimator_linear_model(setup_file, truth_file, meas_file, output_file,
    #                             intfcn, meas_fcn, 'batch')
    
    # compute_linear_errors(setup_file, truth_file, output_file)
    
    # # Run Linear UKF with Range Measurements
    # output_file = os.path.join(datadir, 'linear_model_ukf_output.pkl')
    # intfcn = Estimators.int_constant_vel_ukf
    # meas_fcn = Estimators.unscented_linear1d_rg
    
    # run_estimator_linear_model(setup_file, truth_file, meas_file, output_file,
    #                             intfcn, meas_fcn, 'ukf')
    
    # compute_linear_errors(setup_file, truth_file, output_file)
    
    
    # # Run UKF Process Noise Tuning for Filter = Truth
    # intfcn = Estimators.int_constant_vel
    # snc_tuning(intfcn)
    
    # Run UKF Process Noise Tuning for Filter not equal Truth
    # Setup file contains small negative acceleration like friction
    intfcn = Estimators.int_constant_acc
    snc_tuning(intfcn)
