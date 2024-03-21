import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



###############################################################################
# Batch Estimation
###############################################################################

def ls_batch(Xo_ref, Po_bar, t_obs, t_output, obs_data, intfcn, meas_fcn, params):
    '''
    This function implements the linearized batch estimator for the least
    squares cost function.

    Parameters
    ------
    Xo_ref : nx1 numpy array
        initial state vector
    Po_bar : nxn numpy array
        initial covariance matrix
    t_obs : list or 1D numpy array
        times of observations
    t_output : list or 1D numpy array
        times desired for output state and covariance
        (can be same or different from t_obs)
    obs_data : lxm numpy array
        each row corresponds to measurement vector Yk at time t_obs[k]
    intfcn : function reference
        integration function (dynamics model)
    meas_fcn : function reference
        measurement function to compute Hk_til and Gk
    params : dictionary
        contains sensor data, measurement noise covariance, etc    

    Returns
    ------
    t_output : list or 1D numpy array
        times desired for output state and covariance
        (can be same or different from t_obs)
    Xref_mat : Nxn numpy array
        each row corresponds to estimated state vector Xk at time t_output[k]
    P_mat : nxnxN numpy array
        each nxn matrix corresponds to estimated covar Pk at time t_output[k]
    resids : mxL numpy array
        each row corresponds to residual vector eps_k at time t_obs[k]
        (effectively post-fit after covergence)
        
    '''
    
    # Initial covariances - compute inverse once to use later
    Rk = params['Rk']
    cholPo = np.linalg.inv(np.linalg.cholesky(Po_bar))
    invPo_bar = np.dot(cholPo.T, cholPo)
    
    # Number of states, measurements, epochs
    n = len(Xo_ref)
    m = int(obs_data.shape[1])
    L = len(t_obs)

    # Initialize
    rtol = 1e-12
    atol = 1e-12
    maxiters = 10
    xo_bar = np.zeros((n, 1))
    xo_hat = np.zeros((n, 1))
    phi0 = np.identity(n)
    phi0_v = np.reshape(phi0, (n**2, 1))
    Xref_mat = np.zeros((L,n))
    P_mat = np.zeros((n,n,L))
    resids = np.zeros((L,m))

    # Begin Loop
    iters = 0
    xo_hat_mag = 1
    rms_prior = 1e6
    xhat_crit = 1e-5
    rms_crit = 1e-4
    conv_flag = False    
    while not conv_flag:

        # Increment loop counter and exit if necessary
        iters += 1
        if iters > maxiters:
            iters -= 1
            print('Solution did not converge in ', iters, ' iterations')
            print('Last xo_hat magnitude: ', xo_hat_mag)
            break

        # Initialze values for this iteration
        Xref_list = []
        phi_list = []
        # resids_list = []
        resids_sum = 0.
        phi_v = phi0_v.copy()
        Xref = Xo_ref.copy()
        Xo_ref_print = Xo_ref.copy()
        Lambda = invPo_bar.copy()
        N = np.dot(Lambda, xo_bar)

        # Loop over times        
        for kk in range(L):
            
            # Current and previous time
            if kk == 0:
                tk_prior = t_obs[0]
            else:
                tk_prior = t_obs[kk-1]

            tk = t_obs[kk]

            # Read the next observation
            Yk = np.reshape(obs_data[kk,:], (m,1))

            # Initialize
            Xref_prior = Xref.copy()

            # Initial Conditions for Integration Routine
            int0 = np.concatenate((Xref_prior, phi_v))

            # Integrate Xref and STM
            if tk_prior == tk:
                intout = int0.T
            else:
                int0 = int0.flatten()
                tin = [tk_prior, tk]
                
                # Integrate using RK4 or solve_ivp
                # tout, intout, fcalls = rk4(intfcn, tin, int0, params)

                output = solve_ivp(intfcn,tin,int0,method='DOP853',args=(params,),rtol=rtol,atol=atol)
                intout = output['y'].T

            # Extract values for later calculations
            xout = intout[-1,:]
            Xref = xout[0:n].reshape(n, 1)
            phi_v = xout[n:].reshape(n**2, 1)
            phi = np.reshape(phi_v, (n, n))

            # Accumulate the normal equations
            Hk_til, Gk = meas_fcn(tk, Xref, params)
            yk = Yk - Gk
            Hk = np.dot(Hk_til, phi)
            cholRk = np.linalg.inv(np.linalg.cholesky(Rk))
            invRk = np.dot(cholRk.T, cholRk)
                        
            Lambda += np.dot(Hk.T, np.dot(invRk, Hk))
            N += np.dot(Hk.T, np.dot(invRk, yk))
            
            # Store output
            resids[kk,:] = yk.flatten()
            Xref_list.append(Xref)
            phi_list.append(phi)
            resids_sum += float(np.dot(yk.T, np.dot(invRk, yk)))
            
            
        # Solve the normal equations
        cholLam_inv = np.linalg.inv(np.linalg.cholesky(Lambda))
        Po = np.dot(cholLam_inv.T, cholLam_inv)     
        xo_hat = np.dot(Po, N)
        xo_hat_mag = np.linalg.norm(xo_hat)

        # Update for next batch iteration
        Xo_ref = Xo_ref + xo_hat
        xo_bar = xo_bar - xo_hat
        
        # Evaluate xo_hat_mag and resids for convergence
#        if xo_hat_mag < xhat_crit:
#            conv_flag = True
            
        resids_rms = np.sqrt(resids_sum/(L*m))
        resids_diff = abs(resids_rms - rms_prior)/rms_prior
        if resids_diff < rms_crit:
            conv_flag = True
            
        rms_prior = float(resids_rms)
        
        print('')
        print('Iteration Number: ', iters)
        print('xo_hat_mag = ', xo_hat_mag)
        print('xo_hat = ', xo_hat)
        print('initial Xo_ref', Xo_ref_print)
        print('final Xo_ref', Xo_ref)
        print('resids_rms = ', resids_rms)
        print('resids_diff = ', resids_diff)
        
    
    # Use the initial state and covariance estimated by the final iteration
    # to propagate over time for output
    phi_v = phi0_v.copy()
    Xref = Xo_ref.copy()
    for kk in range(len(t_output)):
        
        # Current and previous time
        if kk == 0:
            tk_prior = t_output[0]
        else:
            tk_prior = t_output[kk-1]

        tk = t_output[kk]
        
        # Initial Conditions for Integration Routine
        Xref_prior = Xref.copy()
        int0 = np.concatenate((Xref_prior, phi_v))

        # Integrate Xref and STM
        if tk_prior == tk:
            intout = int0.T
        else:
            int0 = int0.flatten()
            tin = [tk_prior, tk]
            
            # Integrate using RK4 or solve_ivp
            # tout, intout, fcalls = rk4(intfcn, tin, int0, params)

            output = solve_ivp(intfcn,tin,int0,method='DOP853',args=(params,),rtol=rtol,atol=atol)
            intout = output['y'].T

        # Extract values for later calculations
        xout = intout[-1,:]
        Xref = xout[0:n].reshape(n, 1)
        phi_v = xout[n:].reshape(n**2, 1)
        phi = np.reshape(phi_v, (n, n))
        P = np.dot(phi, np.dot(Po, phi.T))
        
        Xref_mat[kk,:] = Xref.flatten()
        P_mat[:,:,kk] = P
    

    return t_output, Xref_mat, P_mat, resids


###############################################################################
# Unscented Kalman Filter
###############################################################################


def ukf(Xo, Po, t_obs, obs_data, intfcn, meas_fcn, params):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

    Parameters
    ------
    Xo : nx1 numpy array
        initial state vector
    Po : nxn numpy array
        initial covariance matrix
    t_obs : list or 1D numpy array
        times of observations
    obs_data : lxm numpy array
        each row corresponds to measurement vector Yk at time t_obs[k]
    intfcn : function reference
        integration function (dynamics model)
    meas_fcn : function reference
        measurement function to compute Hk_til and Gk
    params : dictionary
        contains sensor data, measurement noise covariance, etc    

    Returns
    ------
    t_obs : list or 1D numpy array
        times of observations
    Xk_mat : Nxn numpy array
        each row corresponds to estimated state vector Xk at time t_output[k]
    P_mat : nxnxN numpy array
        each nxn matrix corresponds to estimated covar Pk at time t_output[k]
    resids : mxL numpy array
        each row corresponds to residual vector eps_k at time t_obs[k]
        (effectively post-fit after covergence)
        
    '''
    
    # Initial covariances - compute inverse once to use later
    Rk = params['Rk']
    
    # Number of states, measurements, epochs
    n = len(Xo)
    m = int(obs_data.shape[1])
    L = len(t_obs)
        
    # Retrieve data from input parameters
    t0 = t_obs[0]  
    alpha = params['alpha']
    gap_seconds = params['gap_seconds']
    
    if 'Q' in params:
        Q = params['Q']
        q = int(Q.shape[0])
    
    if 'Qeci' in params:
        Qeci = params['Qeci']
        q = int(Qeci.shape[0])
        
    if 'Qric' in params:
        Qric = params['Qric']
    
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
  
    # Loop over times
    Xk = Xo.copy()
    Pk = Po.copy()
    rtol = 1e-12
    atol = 1e-12
    Xk_mat = np.zeros((L,n))
    P_mat = np.zeros((n,n,L))
    resids = np.zeros((L,m))
    for kk in range(L):
    
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = t_obs[kk-1]

        tk = t_obs[kk]
        
        # Read the next observation
        Yk = np.reshape(obs_data[kk,:], (m,1))
        
        # Compute sigma points matrix
        sqP = np.linalg.cholesky(Pk)
        Xrep = np.tile(Xk, (1, n))
        chi = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
        chi_v = np.reshape(chi, (n*(2*n+1), 1), order='F')
        
        # Propagate state and covariance
        # No prediction needed if measurement time is same as current state
        if tk_prior == tk:
            intout = chi_v.T
        else:
            int0 = chi_v.flatten()
            tin = np.array([tk_prior, tk])
            
            # Integrate using RK4 or solve_ivp
            # tout, intout, fcalls = rk4(intfcn, tin, int0, params)

            output = solve_ivp(intfcn,tin,int0,method='DOP853',args=(params,),rtol=rtol,atol=atol)
            intout = output['y'].T
       
            
        # Extract values for later calculations
        chi_v = intout[-1,:]
        chi = np.reshape(chi_v, (n, 2*n+1), order='F')
        
        # State Noise Compensation
        # Zero out SNC for long time gaps
        delta_t = tk - tk_prior
        if delta_t > gap_seconds:    
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        Xbar = np.dot(chi, Wm.T)
        Xbar = np.reshape(Xbar, (n, 1))
        chi_diff = chi - np.dot(Xbar, np.ones((1, (2*n+1))))        
        Pbar = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))

        if q == 3:
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
        # Computed measurements and covariance
        gamma_til_k = meas_fcn(tk, chi_bar, params)
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
        Pk = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(Pk)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)        
        gamma_til_post = meas_fcn(tk, chi_k, params)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids_k = Yk - ybar_post
        
        print('')
        print('kk', kk)
        print('Yk', Yk)
        print('ybar', ybar)     
        print('resids', resids_k)
        
        # Store output
        resids[kk,:] = resids_k.flatten()
        Xk_mat[kk,:] = Xk.flatten()
        P_mat[:,:,kk] = Pk

    
    return t_obs, Xk_mat, P_mat, resids


###############################################################################
# Dynamics Models
###############################################################################

def int_constant_vel(t, X, params):
    '''
    This function works with an ode integrator to propagate an object moving
    with no acceleration.

    Parameters
    ------
    t : float 
      current time in seconds
    X : 2 element array
      cartesian state vector     
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 2 element array 
      state derivative vector
    
    '''
    
    x = float(X[0])
    dx = float(X[1])
    
    dX = np.zeros(2,)
    dX[0] = dx
    dX[1] = 0.
    
    return dX


def int_constant_acc(t, X, params):
    '''
    This function works with an ode integrator to propagate an object moving
    with a constant.

    Parameters
    ------
    t : float 
      current time in seconds
    X : 2 element array
      cartesian state vector     
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 2 element array 
      state derivative vector
    
    '''
    
    acc = params['acc']
    
    x = float(X[0])
    dx = float(X[1])
    
    dX = np.zeros(2,)
    dX[0] = dx
    dX[1] = acc
    
    return dX


def int_constant_vel_stm(t, X, params):
    '''
    This function works with an ode integrator to propagate an object moving
    with no acceleration, and includes an augmented state vector including
    terms for the state transition matrix.

    Parameters
    ------
    t : float 
      current time in seconds
    X : 2 element array
      cartesian state vector including STM  
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 6 element array
      state derivative vector including STM
    
    '''

    # Number of states
    n = 2

    # State Vector
    x = float(X[0])
    dx = float(X[1])

    # Generate A matrix
    A = np.zeros((n, n))
    A[0,1] = 1.

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = 0.
    dX[n:] = dphi_v.flatten()

    return dX


def int_constant_vel_ukf(t, X, params):
    '''
    This function works with an ode integrator to propagate an object moving
    with no acceleration, and uses a state vector with all sigma points for
    use in the UKF.

    Parameters
    ------
    t : float 
      current time in seconds
    X : 10 element array
      cartesian state vector including sigma points 
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 10 element array
      state derivative vector including sigma points
    
    '''
    
    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        dx = float(X[ind*n + 1])

        # Set components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = 0.
    
    return dX


def int_twobody(t, X, params):
    '''
    This function works with an ode integrator to propagate object assuming
    simple two-body dynamics.  No perturbations included.

    Parameters
    ------
    t : float 
      current time in seconds
    X : 6 element array
      cartesian state vector (Inertial Frame)    
    params : dictionary
        additional arguments

    Returns
    ------
    dX : 6 element array
      state derivative vector
    '''
    

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    
    # Additional arguments
    GM = params['GM']

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Derivative vector
    dX = np.zeros(6,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    return dX


def int_twobody_stm(t, X, params):
    '''
    This function works with an ode integrator to propagate object assuming
    simple two-body dynamics.  No perturbations included.
    Partials for the STM dynamics are included.

    Parameters
    ------
    t : float 
      current time in seconds
    X : (n+n^2) element array
      initial condition vector of cartesian state and STM (Inertial Frame)    
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n+n^2) element array
      derivative vector
      
    '''

    # Additional arguments
    GM = params['GM']

    # Compute number of states
    n = int((-1 + np.sqrt(1 + 4*len(X)))/2)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])

    # Compute radius
    r = np.linalg.norm([x, y, z])

    # Find elements of A matrix
    xx_cf = -GM/r**3 + 3.*GM*x**2/r**5
    xy_cf = 3.*GM*x*y/r**5
    xz_cf = 3.*GM*x*z/r**5
    yy_cf = -GM/r**3 + 3.*GM*y**2/r**5
    yx_cf = xy_cf
    yz_cf = 3.*GM*y*z/r**5
    zz_cf = -GM/r**3 + 3.*GM*z**2/r**5
    zx_cf = xz_cf
    zy_cf = yz_cf

    # Generate A matrix
    A = np.zeros((n, n))

    A[0,3] = 1.
    A[1,4] = 1.
    A[2,5] = 1.

    A[3,0] = xx_cf
    A[3,1] = xy_cf
    A[3,2] = xz_cf

    A[4,0] = yx_cf
    A[4,1] = yy_cf
    A[4,2] = yz_cf

    A[5,0] = zx_cf
    A[5,1] = zy_cf
    A[5,2] = zz_cf

    # Compute STM components dphi = A*phi
    phi_mat = np.reshape(X[n:], (n, n))
    dphi_mat = np.dot(A, phi_mat)
    dphi_v = np.reshape(dphi_mat, (n**2, 1))

    # Derivative vector
    dX = np.zeros(n+n**2,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3
    dX[4] = -GM*y/r**3
    dX[5] = -GM*z/r**3

    dX[n:] = dphi_v.flatten()

    return dX


def int_twobody_ukf(t, X, params):
    '''
    This function works with ode to propagate object assuming
    simple two-body dynamics.  No perturbations included.  States for UKF
    sigma points included.

    Parameters
    ------
    X : (n*(2n+1)) element list
      initial condition vector of cartesian state and sigma points
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : (n*(2n+1)) element list
      derivative vector

    '''
    
    # Additional arguments
    GM = params['GM']
    
    # Initialize
    dX = np.zeros(len(X),)
    n = int((-1 + np.sqrt(1. + 8.*len(X)))/4.)

    for ind in range(0, 2*n+1):

        # Pull out relevant values from X
        x = float(X[ind*n])
        y = float(X[ind*n + 1])
        z = float(X[ind*n + 2])
        dx = float(X[ind*n + 3])
        dy = float(X[ind*n + 4])
        dz = float(X[ind*n + 5])

        # Compute radius
        r = np.linalg.norm([x, y, z])

        # Solve for components of dX
        dX[ind*n] = dx
        dX[ind*n + 1] = dy
        dX[ind*n + 2] = dz

        dX[ind*n + 3] = -GM*x/r**3
        dX[ind*n + 4] = -GM*y/r**3
        dX[ind*n + 5] = -GM*z/r**3

    return dX


def int_twobody_j2(t, X, params):
    '''
    This function works with ode to propagate object assuming
    two-body dynamics with J2 perturbations included.
    
    It should NOT be used for high fidelity orbit prediction.

    Parameters
    ------
    X : 6 element array
      cartesian state vector (Inertial Frame)
    t : float 
      current time in seconds
    params : dictionary
        additional arguments

    Returns
    ------
    dX : n element array array
      state derivative vector
    
    '''
    
    # Additional arguments
    GM = params['GM']
    J2 = params['J2']
    R = params['Re']

    
    # Number of states
    n = len(X)

    # State Vector
    x = float(X[0])
    y = float(X[1])
    z = float(X[2])
    dx = float(X[3])
    dy = float(X[4])
    dz = float(X[5])
    

    # Compute radius
    r_vect = np.array([[x], [y], [z]])
    r = np.linalg.norm(r_vect)    

    # Derivative vector
    dX = np.zeros(n,)

    dX[0] = dx
    dX[1] = dy
    dX[2] = dz

    dX[3] = -GM*x/r**3. - 1.5*J2*R**2.*GM*((x/r**5.) - (5.*x*z**2./r**7.))
    dX[4] = -GM*y/r**3. - 1.5*J2*R**2.*GM*((y/r**5.) - (5.*y*z**2./r**7.))
    dX[5] = -GM*z/r**3. - 1.5*J2*R**2.*GM*((3.*z/r**5.) - (5.*z**3./r**7.))
    
    return dX


###############################################################################
# Measurement Functions
###############################################################################


def H_linear_range(t, X, params):
    '''
    This function computes the observation mapping matrix Hk_til and computed
    measurement Gk from the reference state X

    Parameters
    ------
    t : float 
      current time in seconds
    X : 2 element array
      cartesian state vector    
    params : dictionary
        additional arguments

    Returns
    ------
    Hk_til : 1x2 numpy array
        observation mapping matrix
    Gk : 1x1 numpy array
        computed range observation
    
    '''
    
    # Break out state
    x = float(X[0])
    
    # Hk_til and Gk
    Hk_til = np.array([[1., 0.]])
    Gk = np.array([[x]])
    
    return Hk_til, Gk


def unscented_linear1d_rg(tk, chi, params):
    
    # Number of states
    n = int(chi.shape[0])
        
    # Compute transformed sigma points   
    gamma_til = np.zeros((1, (2*n+1)))
    for jj in range(2*n+1):
        
        rg = chi[0,jj]
        gamma_til[0,jj] = rg

    return gamma_til


def H_rgradec(t, X, params):
    
    # Retrieve data from params
    theta0 = params['theta0']
    dtheta = params['dtheta']
    sensor_ecef = params['sensor_ecef']    
    
    # Object location in ECI
    r_eci = X[0:3].reshape(3,1)
    
    # Sensor location in ECI
    theta = theta0 + dtheta*t
    sensor_eci = GMST_ecef2eci(sensor_ecef, theta)
    
    # Compute range and line of sight unit vector
    rho_eci = r_eci - sensor_eci
    rho = np.linalg.norm(rho_eci)
    rho_hat_eci = rho_eci/rho
    
    # Compute topocentric right ascension and declination
    ra = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) 
    dec = math.asin(rho_hat_eci[2])                 

    # Calculate partials of rho
    drho_dx = rho_hat_eci[0]
    drho_dy = rho_hat_eci[1]
    drho_dz = rho_hat_eci[2]
    
    # Calculate partials of right ascension
    d_atan = 1./(1. + (rho_eci[1]/rho_eci[0])**2.)
    dra_dx = d_atan*(-(rho_eci[1])/((rho_eci[0])**2.))
    dra_dy = d_atan*(1./(rho_eci[0]))
    
    # Calculate partials of declination
    d_asin = 1./np.sqrt(1. - ((rho_eci[2])/rho)**2.)
    ddec_dx = d_asin*(-(rho_eci[2])/rho**2.)*drho_dx
    ddec_dy = d_asin*(-(rho_eci[2])/rho**2.)*drho_dy
    ddec_dz = d_asin*(1./rho - ((rho_eci[2])/rho**2.)*drho_dz)

    # Hk_til and Gk
    Gk = np.reshape([rho, ra, dec], (3,1))
    
    Hk_til = np.zeros((3,len(X)))
    Hk_til[0,0] = drho_dx
    Hk_til[0,1] = drho_dy
    Hk_til[0,2] = drho_dz
    Hk_til[1,0] = dra_dx
    Hk_til[1,1] = dra_dy
    Hk_til[2,0] = ddec_dx
    Hk_til[2,1] = ddec_dy
    Hk_til[2,2] = ddec_dz    
    
    
    return Hk_til, Gk


def H_radec(t, X, params):
    
    # Retrieve data from params
    theta0 = params['theta0']
    dtheta = params['dtheta']
    sensor_ecef = params['sensor_ecef']    
    
    # Object location in ECI
    r_eci = X[0:3].reshape(3,1)
    
    # Sensor location in ECI
    theta = theta0 + dtheta*t
    sensor_eci = GMST_ecef2eci(sensor_ecef, theta)
    
    # Compute range and line of sight unit vector
    rho_eci = r_eci - sensor_eci
    rho = np.linalg.norm(rho_eci)
    rho_hat_eci = rho_eci/rho
    
    # Compute topocentric right ascension and declination
    ra = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) 
    dec = math.asin(rho_hat_eci[2])                 

    # Calculate partials of rho
    drho_dx = rho_hat_eci[0]
    drho_dy = rho_hat_eci[1]
    drho_dz = rho_hat_eci[2]
    
    # Calculate partials of right ascension
    d_atan = 1./(1. + (rho_eci[1]/rho_eci[0])**2.)
    dra_dx = d_atan*(-(rho_eci[1])/((rho_eci[0])**2.))
    dra_dy = d_atan*(1./(rho_eci[0]))
    
    # Calculate partials of declination
    d_asin = 1./np.sqrt(1. - ((rho_eci[2])/rho)**2.)
    ddec_dx = d_asin*(-(rho_eci[2])/rho**2.)*drho_dx
    ddec_dy = d_asin*(-(rho_eci[2])/rho**2.)*drho_dy
    ddec_dz = d_asin*(1./rho - ((rho_eci[2])/rho**2.)*drho_dz)

    # Hk_til and Gk
    Gk = np.reshape([ra, dec], (2,1))
    
    Hk_til = np.zeros((2,len(X)))
    Hk_til[0,0] = dra_dx
    Hk_til[0,1] = dra_dy
    Hk_til[1,0] = ddec_dx
    Hk_til[1,1] = ddec_dy
    Hk_til[1,2] = ddec_dz    
    
    
    return Hk_til, Gk


def unscented_rgradec(tk, chi, params):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    sensor_ecef = params['sensor_ecef']
    theta0 = params['theta0']
    dtheta = params['dtheta']
    theta = theta0 + dtheta*tk
    sensor_gcrf = GMST_ecef2eci(sensor_ecef, theta)
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((3, (2*n+1)))
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
        
        ra = math.atan2(rho_hat_gcrf[1,0], rho_hat_gcrf[0,0]) #rad
        dec = math.asin(rho_hat_gcrf[2,0])  #rad
        
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
                
        # Form Output
        gamma_til[0,jj] = rg
        gamma_til[1,jj] = ra
        gamma_til[2,jj] = dec


    return gamma_til


def unscented_radec(tk, chi, params):
    
    # Number of states
    n = int(chi.shape[0])
    
    # Compute sensor position in GCRF
    sensor_ecef = params['sensor_ecef']
    theta0 = params['theta0']
    dtheta = params['dtheta']
    theta = theta0 + dtheta*tk
    sensor_gcrf = GMST_ecef2eci(sensor_ecef, theta)    
    
    # Compute transformed sigma points   
    gamma_til = np.zeros((2, (2*n+1)))
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
        
        ra = math.atan2(rho_hat_gcrf[1,0], rho_hat_gcrf[0,0]) #rad
        dec = math.asin(rho_hat_gcrf[2,0])  #rad
        
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
                
        # Form Output
        
        gamma_til[0,jj] = ra
        gamma_til[1,jj] = dec

    return gamma_til


###############################################################################
# Utility Functions
###############################################################################

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


def GMST_eci2ecef(r_eci, theta):
    '''
    This function converts the coordinates of a position vector from
    the ECI to ECEF frame using simple z-axis rotation only.

    Parameters
    ------
    r_eci : 3x1 numpy array
      position vector in ECI
    theta : float
      earth rotation angle [rad]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    R3 = np.array([[ np.cos(theta),  np.sin(theta),     0.],
                   [-np.sin(theta),  np.cos(theta),     0.],
                   [            0.,             0.,     1.]])

    r_ecef = np.dot(R3, r_eci)

    return r_ecef


def GMST_ecef2eci(r_ecef, theta):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ECI frame using simple z-axis rotation only.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    theta : float
      earth rotation angle [rad]

    Returns
    ------
    r_eci : 3x1 numpy array
      position vector in ECI
    '''

    R3 = np.array([[np.cos(theta), -np.sin(theta),      0.],
                   [np.sin(theta),  np.cos(theta),      0.],
                   [           0.,             0.,      1.]])

    r_eci = np.dot(R3, r_ecef)

    return r_eci



def rk4(intfcn, tin, y0, params):
    '''
    This function implements the fixed-step, single-step, 4th order Runge-Kutta
    integrator.
    
    Parameters
    ------
    intfcn : function handle
        handle for function to integrate
    tin : 1D numpy array
        times to integrate over, [t0, tf] or [t0, t1, t2, ... , tf]
    y0 : numpy array
        initial state vector
    params : dictionary
        parameters for integration including step size and any additional
        variables needed for the integration function
    
    Returns
    ------
    tvec : 1D numpy array
        times corresponding to output states from t0 to tf
    yvec : (N+1)xn numpy array
        output state vectors at each time, each row is 1xn vector of state
        at corresponding time
    
    '''

    # Start and end times
    t0 = tin[0]
    tf = tin[-1]
    if len(tin) == 2:
        h = params['step']
        tvec = np.arange(t0, tf, h)
        tvec = np.append(tvec, tf)
    else:
        tvec = tin

    # Initial setup
    yn = y0.flatten()
    tn = t0
    yvec = y0.reshape(1, len(y0))
    fcalls = 0
    
    # Loop to end
    for ii in range(len(tvec)-1):
        
        # Step size
        h = tvec[ii+1] - tvec[ii]
        
        # Compute k values
        k1 = h * intfcn(tn,yn,params)
        k2 = h * intfcn(tn+h/2,yn+k1/2,params)
        k3 = h * intfcn(tn+h/2,yn+k2/2,params)
        k4 = h * intfcn(tn+h,yn+k3,params)
        
        # Compute solution
        yn += (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        
        # Increment function calls      
        fcalls += 4
        
        # Store output
        yvec = np.concatenate((yvec, yn.reshape(1,len(y0))), axis=0)

    return tvec, yvec, fcalls








