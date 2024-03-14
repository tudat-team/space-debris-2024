import numpy as np
import os
from datetime import datetime
import time

import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil

###############################################################################
# Basic I/O
###############################################################################

# Set directory for assignment data
assignment_data_directory = 'data/unit_test'

# Load RSO catalog file
rso_file = os.path.join(assignment_data_directory, 'estimated_rso_catalog.pkl')
rso_dict = ConjUtil.read_catalog_file(rso_file)
obj_id = 91260

print('')
print('RSO File contains', len(rso_dict), 'objects')
print('rso_dict contains the following objects:')
print(list(rso_dict.keys()))

print('')
print('Data for each object are stored as a dictionary and can be retrieved using the object ID')
print('The following fields are available')
print(list(rso_dict[obj_id].keys()))


# Load truth data for estimation case
truth_file = os.path.join(assignment_data_directory, 'test_sparse_truth_grav.pkl')
t_truth, X_truth, state_params = EstUtil.read_truth_file(truth_file)

print('')
print('The truth data file contains the following:')
print('t_truth shape', t_truth.shape)
print('X_truth shape', X_truth.shape)
print('X at t0', X_truth[0,:])
print('state_params')
print(state_params)


# Load measurement data for estimation case
meas_file = os.path.join(assignment_data_directory, 'test_sparse_radar_meas.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)

print('')
print('The measurement file contains an augmented state params dictionary that' 
      'includes the initial state and covariance for the filter to use')
print(state_params)
print('meas_dict fields:')
print(list(meas_dict.keys()))
print('tk_list length', len(meas_dict['tk_list']))
print('Y at t0', meas_dict['Yk_list'][0])
print('sensor_params fields:')
print(list(sensor_params.keys()))


###############################################################################
# Compute TCA Example

# This code performs a unit test of the compute_TCA function. The object
# parameters are such that a collision is expected 30 minutes after the
# initial epoch (zero miss distance).

# The TCA function is run twice, using a fixed step RK4 and variable step
# RKF78 to compare the results.
#
###############################################################################

print('\nBegin TCA Test')

# Initial time and state vectors
t0 = (datetime(2024, 3, 23, 5, 30, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()

X1 = np.array([[ 3.75944379e+05],
               [ 6.08137408e+06],
               [ 3.28340214e+06],
               [-5.32161464e+03],
               [-2.32172417e+03],
               [ 4.89152047e+03]])

X2 = np.array([[ 3.30312011e+06],
               [-2.69542170e+06],
               [-5.71365135e+06],
               [-4.06029364e+03],
               [-6.22037456e+03],
               [ 9.09217382e+02]])

# Basic setup parameters
bodies_to_create = ['Sun', 'Earth', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create) 

rso1_params = {}
rso1_params['mass'] = 260.
rso1_params['area'] = 17.5
rso1_params['Cd'] = 2.2
rso1_params['Cr'] = 1.3
rso1_params['sph_deg'] = 8
rso1_params['sph_ord'] = 8    
rso1_params['central_bodies'] = ['Earth']
rso1_params['bodies_to_create'] = bodies_to_create

rso2_params = {}
rso2_params['mass'] = 100.
rso2_params['area'] = 1.
rso2_params['Cd'] = 2.2
rso2_params['Cr'] = 1.3
rso2_params['sph_deg'] = 8
rso2_params['sph_ord'] = 8    
rso2_params['central_bodies'] = ['Earth']
rso2_params['bodies_to_create'] = bodies_to_create

int_params = {}
int_params['tudat_integrator'] = 'rk4'
int_params['step'] = 1.

# Expected result
TCA_true = 764445600.0  
rho_true = 0.

# Interval times
tf = t0 + 3600.
trange = np.array([t0, tf])

# RK4 test
start = time.time()
T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                        int_params, bodies)



print('')
print('RK4 TCA unit test runtime [seconds]:', time.time() - start)
print('RK4 TCA error [seconds]:', T_list[0]-TCA_true)
print('RK4 miss distance error [m]:', rho_list[0]-rho_true)


# RK78 test
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12


start = time.time()
T_list, rho_list = ConjUtil.compute_TCA(X1, X2, trange, rso1_params, rso2_params, 
                                        int_params, bodies)


print('')
print('RK78 TCA unit test runtime [seconds]:', time.time() - start)
print('RK78 TCA error [seconds]:', T_list[0]-TCA_true)
print('RK78 miss distance error [m]:', rho_list[0]-rho_true)



###############################################################################
# Run Unscented Kalman Filter Example
#
# This code performs a unit test of the UKF function. It does not compute
# errors or generate plots, just ensures that everything is present to run
# the filter without error.
#
###############################################################################

print('\nBegin UKF Test')

# Load measurement data
meas_file = os.path.join(assignment_data_directory, 'test_sparse_radar_meas.pkl')
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(meas_file)


# For the assignment, you should always set the filter to zero out Drag and SRP
# In the case truth/measurements are generated with a higher fidelity force 
# model, the SNC parameters of the filter should be tuned to account for this

# Set the Drag and SRP coefficient to zero for the filter acceleration model
state_params['Cd'] = 0.
state_params['Cr'] = 0.

# Setup filter parameters such as process noise
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

# Choose integration parameters
# int_params = {}
# int_params['tudat_integrator'] = 'rk4'
# int_params['step'] = 10.

int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

# Initialize tudat bodies
bodies_to_create = ['Earth', 'Sun', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)

# Run filter
filter_output = EstUtil.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)
















