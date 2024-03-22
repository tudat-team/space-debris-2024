import numpy as np


# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array

# Load spice kernels
spice.load_standard_kernels()




def tudat_initialize_bodies(bodies_to_create=[]):
    '''
    This function initializes the bodies object for use in the Tudat 
    propagator. For the cases considered, only Earth, Sun, and Moon are needed,
    with Earth as the frame origin.
    
    Parameters
    ------
    bodies_to_create : list, optional (default=[])
        list of bodies to create, if empty, will use default Earth, Sun, Moon
    
    Returns
    ------
    bodies : tudat object
    
    '''

    # Define string names for bodies to be created from default.
    if len(bodies_to_create) == 0:
        bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    return bodies


def propagate_orbit(Xo, tvec, state_params, int_params, bodies=None):
    '''
    This function propagates an orbit using physical parameters provided in 
    state_params and integration parameters provided in int_params.
    
    Parameters
    ------
    Xo : 6x1 numpy array
        Cartesian state vector [m, m/s]
    tvec : list or numpy array
        propagator will only use first and last terms to set the initial and
        final time of the propagation, intermediate times are ignored
        
        [t0, ..., tf] given as time in seconds since J2000
        
    state_params : dictionary
        propagator parameters
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    int_params : dictionary
        numerical integration parameters
        
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation
        if None, will initialize with bodies given in state_params
        
    Returns
    ------
    tout : N element numpy array
        times of propagation output in seconds since J2000
    Xout : Nxn numpy array
        each row Xout[k,:] corresponds to Cartesian state at time tout[k]        
    
    '''
    
    # Initial state
    initial_state = Xo.flatten()
    
    # Retrieve input parameters
    central_bodies = state_params['central_bodies']
    bodies_to_create = state_params['bodies_to_create']
    mass = state_params['mass']
    Cd = state_params['Cd']
    Cr = state_params['Cr']
    area = state_params['area']
    sph_deg = state_params['sph_deg']
    sph_ord = state_params['sph_ord']
    
    # Simulation start and end
    simulation_start_epoch = tvec[0]
    simulation_end_epoch = tvec[-1]
    
    # Setup bodies
    if bodies is None:
        bodies = tudat_initialize_bodies(bodies_to_create)
    
    
    # Create the bodies to propagate
    # TUDAT always uses 6 element state vector
    N = int(len(Xo)/6)
    central_bodies = central_bodies*N
    bodies_to_propagate = []
    for jj in range(N):
        jj_str = str(jj)
        bodies.create_empty_body(jj_str)
        bodies.get(jj_str).mass = mass
        bodies_to_propagate.append(jj_str)
        
        if Cd > 0.:
            aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [Cd, 0, 0]
            )
            environment_setup.add_aerodynamic_coefficient_interface(
                bodies, jj_str, aero_coefficient_settings)
            
        if Cr > 0.:
            # occulting_bodies = ['Earth']
            # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            #     'Sun', srp_area_m2, Cr, occulting_bodies
            # )
            # environment_setup.add_radiation_pressure_interface(
            #     bodies, jj_str, radiation_pressure_settings)
            
            occulting_bodies_dict = dict()
            occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
            
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, Cr, occulting_bodies_dict )
            
            environment_setup.add_radiation_pressure_target_model(
                bodies, jj_str, radiation_pressure_settings)
            

    acceleration_settings_setup = {}        
    if 'Earth' in bodies_to_create:
        
        # Gravity
        if sph_deg == 0 and sph_ord == 0:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]
        else:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.spherical_harmonic_gravity(sph_deg, sph_ord)]
        
        # Aerodynamic Drag
        if Cd > 0.:                
            acceleration_settings_setup['Earth'].append(propagation_setup.acceleration.aerodynamic())
        
    if 'Sun' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Sun'] = [propagation_setup.acceleration.point_mass_gravity()]
        
        # Solar Radiation Pressure
        if Cr > 0.:                
            #acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.cannonball_radiation_pressure())
            acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.radiation_pressure())
    
    if 'Moon' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
    

    acceleration_settings = {}
    for jj in range(N):
        acceleration_settings[str(jj)] = acceleration_settings_setup
        
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(
        simulation_end_epoch, terminate_exactly_on_final_condition=True
    )


    # Create numerical integrator settings
    if int_params['tudat_integrator'] == 'rk4':
        fixed_step_size = int_params['step']
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            fixed_step_size
        )
        
    elif int_params['tudat_integrator'] == 'rkf78':
        initial_step_size = int_params['step']
        maximum_step_size = int_params['max_step']
        minimum_step_size = int_params['min_step']
        rtol = int_params['rtol']
        atol = int_params['atol']
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            initial_step_size,
            propagation_setup.integrator.CoefficientSets.rkf_78,
            minimum_step_size,
            maximum_step_size,
            rtol,
            atol)
    
        
        
    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition
    )
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)        
    
    
    tout = states_array[:,0]
    Xout = states_array[:,1:6*N+1]
    
    
    return tout, Xout


def propagate_state_and_covar(Xo, Po, tvec, state_params, int_params, bodies=None, alpha=1e-4):
    '''
    This function propagates an orbit using physical parameters provided in 
    state_params and integration parameters provided in int_params. It also
    propagates the Gaussian covariance matrix using the Unscented transform.
    
    This function is suitable to use as the prediction step in the UKF. It can
    also be used as a stand-alone method to propagate a state and covariance.
    
    Note that this will only work for 6 element state vector, if drag or SRP 
    coefficient are included for estimation it will require modification.
        
    Parameters
    ------
    Xo : 6x1 numpy array
        Cartesian state vector [m, m/s]
    tvec : list or numpy array
        propagator will only use first and last terms to set the initial and
        final time of the propagation, intermediate times are ignored
        
        [t0, ..., tf] given as time in seconds since J2000
        
    state_params : dictionary
        propagator parameters
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    int_params : dictionary
        numerical integration parameters
        
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation
        if None, will initialize with bodies given in state_params
        
    alpha: float, optional (default=1e-4)
        UKF sigma point spread parameter, should be in range [1e-4, 1]
        
    Returns
    ------
    tf : float
        final time in seconds since J2000
    Xf : nx1 numpy array
        Cartesian state at final time [m, m/s]   
    Pf : nxn numpy array
        Gaussian covariance matrix at final time [m^2, m^2/s^2]
    
    '''
    
    
    # Compute sigma point weights
    n = len(Xo)
    beta = 2.   # Gaussian distribution/Least Squares cost
    kappa = 3. - n
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    
    # Initial state and sigma points
    sqP = np.linalg.cholesky(Po)
    Xrep = np.tile(Xo, (1, n))
    chi = np.concatenate((Xo, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    chi_v = np.reshape(chi, (n*(2*n+1), 1), order='F')
    
    # TODO: Note that this will work only for 6 element state vector, if other
    # parameters are included such as drag or SRP coefficient, this function
    # will require modification
    initial_state = chi_v.flatten()
    
    # Retrieve input parameters
    central_bodies = state_params['central_bodies']
    bodies_to_create = state_params['bodies_to_create']
    mass = state_params['mass']
    Cd = state_params['Cd']
    Cr = state_params['Cr']
    area = state_params['area']
    sph_deg = state_params['sph_deg']
    sph_ord = state_params['sph_ord']
    
    # Simulation start and end
    simulation_start_epoch = tvec[0]
    simulation_end_epoch = tvec[-1]
    
    # Setup bodies
    if bodies is None:
        bodies = tudat_initialize_bodies(bodies_to_create)
    
    
    # Create the bodies to propagate
    # TUDAT always uses 6 element state vector
    N = int(len(initial_state)/n)
    central_bodies = central_bodies*N
    bodies_to_propagate = []
    for jj in range(N):
        jj_str = str(jj)
        bodies.create_empty_body(jj_str)
        bodies.get(jj_str).mass = mass
        bodies_to_propagate.append(jj_str)
        
        if Cd > 0.:
            aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [Cd, 0, 0]
            )
            environment_setup.add_aerodynamic_coefficient_interface(
                bodies, jj_str, aero_coefficient_settings)
            
        if Cr > 0.:
            # occulting_bodies = ['Earth']
            # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            #     'Sun', srp_area_m2, Cr, occulting_bodies
            # )
            # environment_setup.add_radiation_pressure_interface(
            #     bodies, jj_str, radiation_pressure_settings)
            
            occulting_bodies_dict = dict()
            occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
            
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, Cr, occulting_bodies_dict )
            
            environment_setup.add_radiation_pressure_target_model(
                bodies, jj_str, radiation_pressure_settings)
            

    acceleration_settings_setup = {}        
    if 'Earth' in bodies_to_create:
        
        # Gravity
        if sph_deg == 0 and sph_ord == 0:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]
        else:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.spherical_harmonic_gravity(sph_deg, sph_ord)]
        
        # Aerodynamic Drag
        if Cd > 0.:                
            acceleration_settings_setup['Earth'].append(propagation_setup.acceleration.aerodynamic())
        
    if 'Sun' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Sun'] = [propagation_setup.acceleration.point_mass_gravity()]
        
        # Solar Radiation Pressure
        if Cr > 0.:                
            #acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.cannonball_radiation_pressure())
            acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.radiation_pressure())
    
    if 'Moon' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
    

    acceleration_settings = {}
    for jj in range(N):
        acceleration_settings[str(jj)] = acceleration_settings_setup
        
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(
        simulation_end_epoch, terminate_exactly_on_final_condition=True
    )


    # Create numerical integrator settings
    if int_params['tudat_integrator'] == 'rk4':
        fixed_step_size = int_params['step']
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            fixed_step_size
        )
        
    elif int_params['tudat_integrator'] == 'rkf78':
        initial_step_size = int_params['step']
        maximum_step_size = int_params['max_step']
        minimum_step_size = int_params['min_step']
        rtol = int_params['rtol']
        atol = int_params['atol']
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            initial_step_size,
            propagation_setup.integrator.CoefficientSets.rkf_78,
            minimum_step_size,
            maximum_step_size,
            rtol,
            atol)
    
        
        
    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition
    )
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)        
    
    
    tf = states_array[-1,0]
    chi_v = states_array[-1,1:]
    
    # Compute mean and covariance at final time using unscented transform
    chi = np.reshape(chi_v, (n, 2*n+1), order='F')
    mean = np.dot(chi, Wm.T)
    mean = np.reshape(mean, (n, 1))
    chi_diff = chi - np.dot(mean, np.ones((1, (2*n+1))))
    Pf = np.dot(chi_diff, np.dot(diagWc, chi_diff.T))
    
    # Output single mean state not computed average of sigma points
    Xf = chi_v[0:6].reshape(6,1)
    
    return tf, Xf, Pf