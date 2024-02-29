import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
import pickle

import BreakupUtilities as util

plt.close('all')

###############################################################################
# DEFINE SIMULATION PARAMETERS
#
# Use meters, kilograms, seconds, radians for all units
###############################################################################

# Target orbit and epoch [UTC]
initial_epoch = datetime(2019, 7, 24, 0, 0, 0)

# Envisat
SMA_meters = 7143.4*1000.
ECC = 0.0001427
INC_rad = 98.1478*np.pi/180.
RAAN_rad = 224.8380*np.pi/180.
AOP_rad = 93.2798*np.pi/180.
TA_rad = 0.*np.pi/180.

# Impact Parameters
mass_target = 8211.
mass_impactor = 3.4
impact_velocity = 14.*1000.

# Spacecraft drag and SRP parameters
Cd = 2.2
Cr = 1.3

# Propagation time in seconds
tfinal_sec = 86400.*90.

# Maximum number of objects to keep for propagation and plots
N_max = 10

# One at a time flag for MEO/GEO propagation
# Set to True will propagate objects one at a time and print status
# Set to False will propagate all objects together as a batch which is 2-3
# times faster, but no status updates
one_at_a_time = False

# Set random seed for reproducibility of results
np.random.seed(1)

# Output directory and filename
outdir = 'output'
fname = os.path.join(outdir, 'breakup_data.pkl')



###############################################################################
# Simulate Collision and Long Term Propagation
###############################################################################


# Check EMR
EMR = mass_impactor*impact_velocity**2./(2.*mass_target) 
if EMR < 40000.:
    print('Note: These impact parameters do not produce a catastrophic collision!')    
else:
    print('Note: These impact parameters produce a catastrophic collision!')

print('EMR [J/kg]: ', EMR)


# Initialize parameters for breakup model
# For a collision set the explosion flag to False.
# For a satellite set the rocket body flag to False.
# The parameter cs is only needed for the explosion model and will not 
# influence results of collision analysis, set to default value 1.0
expl_flag = False
rb_flag = False
cs = 1.

# Propagation time
t0 = (initial_epoch - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
tf = t0 + tfinal_sec
tin = np.array([t0, tf])

# Define cutoff for LEO: apogee altitude under 2000 km
# For MEO/GEO initialize tudat bodies (Earth, Sun, Moon)
if (SMA_meters*(1.+ECC) - util.Re) < 2000000.:
    LEO_flag = True
    bodies = None
else:
    LEO_flag = False
    bodies = util.tudat_initialize_bodies()
    
# Store parameters for propagator
params = {}
params['Nquad'] = 20


# Create an array of characteristic lengths from 1 cm to 10 meters
# The model will generate debris objects of these diameters using a statistical
# model. Use log spacing as the number of objects in each size class will decay
# exponentially.
lc_array = np.logspace(-2, 1, 1000)

# Apply the model to generate the full set of particles from the breakup
N_bin, N_cum, lc_list_full, A_list, A_m_list, m_list, dV_list = \
    util.nasa_stdbrkup_model(lc_array, impact_velocity, mass_target,
                             mass_impactor, expl_flag, rb_flag, cs)
    
print('Total Number of Debris Particles: ', sum(N_bin))
print('Total mass of breakup [kg]: ', sum(m_list))


# Generate reduced list of objects that have not re-entered
# Breakup model generates lists of objects from smallest to largest
# Start from the end and count backwards to retain the N_max largest objects

# Initial state in Cartesian
Xo_cart = util.kep2cart(SMA_meters, ECC, INC_rad, RAAN_rad, AOP_rad, TA_rad)

# Array of debris states, adding the delta-V's computed by the breakup model
kep_array_initial = np.empty((0,6))
reduced_area_list = []
reduced_mass_list = []
Cr_list = []
Cd_list = []
nobj = 0
for ii in range(len(dV_list)-1, -1, -1):
    Xi = Xo_cart.flatten()
    Xi[3:6] += dV_list[ii].flatten()
    a, e, i, RAAN, w, theta = util.cart2kep(Xi)
    rp = a*(1.-e)
    hp = rp - util.Re
    
    # Only keep object if above minimum altitude and not hyperbolic
    if hp > 100000. and a > 0.:
        kep = np.array([[a, e, i, RAAN, w, theta]])
        kep_array_initial = np.concatenate((kep_array_initial, kep), axis=0)
        reduced_area_list.append(A_list[ii])
        reduced_mass_list.append(m_list[ii])
        Cr_list.append(Cr)
        Cd_list.append(Cd)
        nobj += 1
        
    # Exit after N_max objects stored
    if nobj >= N_max:
        break


print('Total mass of propagated objects [kg]: ', sum(reduced_mass_list))
print('')


    
# Plot the output of the model at time of breakup
util.plot_breakup_stats(lc_array, N_bin, N_cum, lc_list_full, A_list, A_m_list, m_list, dV_list)


# Propagate the orbits of the nobj largest objects from the breakup
start_prop = time.time()

# Initialize output arrays
# Note that latlon and kep arrays do not store data at intermediate times
# only the beginning and end of the propagation (kep_array_initial already
# generated above)

# latlon array rows correspond to each object's latitude and longitude at
# initial or final time
latlon_array_initial = np.empty((0,2))
latlon_array_final = np.empty((0,2))

# kep_array_final each row corresponds to the Keplerian elements not 
# including anomaly angle at final time, using the order
# [SMA, ECC, INC, RAAN, AOP]
kep_array_final = np.empty((0,5))

# reentry_times is a list of time in seconds since the start of the propagation
# at which objects reenter. If no objects reenter by final time, the list will
# be empty.
reentry_times = []

# Determine if the objects are propagated individually
if LEO_flag or one_at_a_time:
    
    # Loop over objects and propagate to final time
    for ii in range(nobj):
        
        print('Processing Object:', ii)
        kep = kep_array_initial[ii,:].flatten()
        params['area'] = reduced_area_list[ii]
        params['mass'] = reduced_mass_list[ii]
        params['Cd'] = Cd
        params['Cr'] = Cr
    
        # The propagator function will handle LEO, MEO, or GEO using the 
        # appropriate propagator as determined by LEO_flag
        output_dict = util.long_term_propagator(tin, kep, params, LEO_flag, bodies)
        tsec = output_dict['tsec']
        sma_array = output_dict['SMA']
        ecc_array = output_dict['ECC']
        
        # Check for re-entry
        reentry_flag = False
        for ii in range(len(tsec)):
            ai = float(sma_array[ii])
            ei = float(ecc_array[ii])
            rp = ai*(1.-ei)
            hp = rp - util.Re
            if hp < 100000.:
                reentry_times.append(tsec[ii]-tin[0])
                reentry_flag = True
                break
        
        # Store orbit elements of objects that do not re-enter by final time
        if not reentry_flag:
            
            sma_f = sma_array[-1]
            ecc_f = ecc_array[-1]
            inc_f = output_dict['INC'][-1]
            raan_f = output_dict['RAAN'][-1]
            aop_f = output_dict['AOP'][-1]
            
            kep_out = np.array([[sma_f, ecc_f, inc_f, raan_f, aop_f]])
            kep_array_final = np.concatenate((kep_array_final, kep_out), axis=0)
            
            if 'lat' in output_dict:
                lat_0 = output_dict['lat'][0]
                lon_0 = output_dict['lon'][0]
                lat_f = output_dict['lat'][-1]
                lon_f = output_dict['lon'][-1]
                latlon_array_initial = np.concatenate((latlon_array_initial, np.array([[lat_0, lon_0]])), axis=0)        
                latlon_array_final = np.concatenate((latlon_array_final, np.array([[lat_f, lon_f]])), axis=0)
            

# MEO/GEO objects can be propagated in a batch by setting one_at_a_time to False
# This will be 2-3 faster computation
else:

    # Propagate the objects
    params['mass_list'] = reduced_mass_list
    params['area_list'] = reduced_area_list
    params['Cd_list'] = Cd_list
    params['Cr_list'] = Cr_list
    
    tout, yout = util.long_term_propagator_all(tin, kep_array_initial, params, bodies)

    # Extract output for plots
    for jj in range(nobj):
        
        # Retrieve full time history of dependent variables for this object
        sma = yout[:,jj*8]
        ecc = yout[:,jj*8+1]
        inc = yout[:,jj*8+2]
        aop = yout[:,jj*8+3]
        raan = yout[:,jj*8+4]
        lat = yout[:,jj*8+6]
        lon = yout[:,jj*8+7]
        
        latlon_array_initial = np.concatenate((latlon_array_initial, np.array([[lat[0], lon[0]]])))
        
        # Check for re-entry
        reentry_flag = False
        for ii in range(len(tout)):
            ai = float(sma[ii])
            ei = float(ecc[ii])
            rp = ai*(1.-ei)
            hp = rp - util.Re
            if hp < 100000.:
                reentry_times.append(tout[ii]-tin[0])
                reentry_flag = True
                break
        
        # Store orbit elements of objects that do not re-enter by final time
        if not reentry_flag:
            kep_out = np.array([[sma[-1], ecc[-1], inc[-1], raan[-1], aop[-1]]])
            kep_array_final = np.concatenate((kep_array_final, kep_out), axis=0)
            latlon_array_final = np.concatenate((latlon_array_final, np.array([[lat[-1], lon[-1]]])))

    

print('Propagation Time: ', time.time()-start_prop)

print('Number of objects on orbit at final time:', int(kep_array_final.shape[0]))
print('reentry times [years]', [ti/(86400.*365.25) for ti in reentry_times])

# Save data, create new directory if necessary    
try:
    pklFile = open(fname, 'wb')
    pickle.dump([kep_array_initial, kep_array_final, reentry_times,
                 latlon_array_initial, latlon_array_final], pklFile, -1)
    pklFile.close()
except:
    os.mkdir(outdir)
    pklFile = open(fname, 'wb')
    pickle.dump([kep_array_initial, kep_array_final, reentry_times,
                 latlon_array_initial, latlon_array_final], pklFile, -1)
    pklFile.close()

