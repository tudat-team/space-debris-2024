import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Tudatpy imports
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.util import result2array



###############################################################################
#
# This file contains functions to implement the NASA Standard Breakup Model.
# Futhermore, functions to propagate debris orbits over long time
# periods, either using Tudat or the Semi-Analytic Liu Theory (SALT)
# general perturbations orbit propagator, depending on the orbit regima.  
# The SALT code models the secular and long period effects of J2, J3, and 
# atmospheric drag (using the standard atmosphere model), and is used for 
# long-term LEO propagation. MEO and GEO orbits are propagated using Tudat with
# a low fidelity force model (4x4 gravity field, LuniSolar, cannonball SRP).
#
# References
#
#  [1] Johnson, et al., "NASA's New Breakup Model of Evolve 4.0," Advances in 
#      Space Research, Vol. 28, No. 9, pp. 1377-1384, 2001.
# 
#  [2] Liu and Alford, "Semi-Analytic Theory for a Close-Earth Artificial 
#      Satellite," Guidance and Control, 1980.
#
#  [3] Vallado, "Fundamentals of Astrodynamics and Applications," 4th ed., 2013.
#
#  [4] Finkelman, et al., "Analysis of the Response of a Space Surveillance 
#      Network to Orbital Debris Events," AAS-08-227.
#
#  [5] Klinkrad, "Space Debris - Models and Risk Analysis," 2006.
#
#  [6] Tudat https://docs.tudat.space/en/latest/
#
#  [7] Krisko, P., "Proper Implementation of the 1998 NASA Breakup Model,"
#      NASA Orbit Debris Quarterly News, Vol. 15, No. 4, Oct 2011.
#
###############################################################################



###############################################################################
# Physical Constants
###############################################################################

# Earth parameters
wE = 7.2921158553e-5  # rad/s
GME = 398600.4415*1e9  # m^3/s^2
J2E = 1.082626683e-3
J3E = -2.5327e-6

# WGS84 Data 
Re = 6378137.0   # m
rec_f = 298.257223563


###############################################################################
# NASA Standard Breakup Model
###############################################################################

def compute_number(lc, v_imp, m_tar, m_imp, expl_flag=False, cs=1.):
    '''
    This function computes the number of objects with diameter d >= lc, the 
    characteristic length, per the NASA Standard Breakup Model.  By convention
    the larger primary object is designated the target and smaller secondary 
    object is designated the impactor.
    
    Parameters
    ------
    lc : float
        characteristic length [m]
    v_imp : float
        relative velocity of impact [m/s]
    m_tar : float
        mass of target (primary object) [kg]
    m_imp : float
        mass of impactor (secondary object) [kg]
    expl_flag : boolean, optional
        flag to select explosion (True) or collision (False) (default=False)
    cs : float, optional
        unitless scale factor used for explosion model (default=1.)
    
    Returns
    ------
    Nd : float
        number of debris objects with diameter d >= lc

    '''

    # Explosion Case
    if expl_flag:
        Nd = 6*cs*lc**(-1.6)
        
    # Collision Case
    else:            
        EMR_crit = 40000.    # J/kg
        EMR = m_imp*v_imp**2./(2.*m_tar)     
        
        if EMR >= EMR_crit:
            mc = m_tar + m_imp
        else:
            # mc = m_imp*v_imp/1000.
            
            # Krisko 2011 Eq 4 states this should be velocity squared for 
            # non-catastrophic case
            mc = m_imp*(v_imp/1000.)**2.
        
        Nd = 0.1*mc**(.75)*lc**(-1.71)
        
    return Nd


def compute_cs_henize(H):
    '''
    This function computes a scaling factor cs to be used in generating the
    number of particles from an explosion in the NASA standard breakup model.
    
    Parameters
    ------
    H : float
        Tracking Relevant Altitude (TRA), often set to periapsis altitude [m]
    
    Returns
    ------
    cs : float
        unitless scaling factor
        
    References
    ------
    [1] Klinkrad, H., "Space Debris - Models and Risk Analysis," 2006. 
        Eqs 3.30-3.31
    [2] MASTER-8 Final Report Section 2.2.6
    [3] Henize, K. and Stanley, J., "Optical Observations of Space Debris," 
        AIAA-90-1340.
    
    '''
    
    # Convert H to km
    H *= (1./1000.)
    
    if H <= 620.:
        d = 0.089
    elif H <= 1300.:
        d = 10**(-2.737 + 0.604*np.log10(H))
    elif H <= 3800.:
        d = 10**(-6.517 + 1.819*np.log10(H))
    else:
        d = 1.
        
    x = 0.5*np.exp(-2.464*(np.log10(d) + 1.22)**2.)
    cs = 10.**x
    
    return cs


def compute_area(lc):
    '''
    This function computes the average cross-sectional area of objects with
    diameter d = lc, the characteristic length, per the NASA Standard Breakup 
    Model.  
    
    Parameters
    ------
    lc : float
        characteristic length [m]
        
    Returns
    ------
    A : float
        average cross-section area [m^2]
    
    '''
    
    if lc >= 0.00167:
        A = 0.556945*lc**(2.0047077)
    else:
        A = 0.540424*lc**2.
    
    return A


def compute_A_m(lc, rb_flag=False):
    '''
    This function computes the area-to-mass ratio of objects with
    diameter d = lc, the characteristic length, per the NASA Standard Breakup 
    Model.  The model employs a bi-modal distribution for larger debris
    objects with different parameters depending if they originate from a 
    payload or rocket body.
    
    Parameters
    ------
    lc : float
        characteristic length [m]
    rb_flag : boolean, optional
        flag to set object type as Payload (False) or Rocket Body (True)
        (default=False)
        
    Returns
    ------
    A_m : float
        area-to-mass ratio [m^2/kg]
    
    '''
    
    # Bridge Function parameters
    zbar_rb = 10.*(math.log10(lc) + 1.76)
    zbar_pl = 10.*(math.log10(lc) + 1.05)
    
    zeta = np.random.rand()
    bimodal_flag = True
    
    # Rocket Body Case
    if rb_flag:
        d_min = 0.017
        if lc > d_min and lc < 0.11:
            if zeta < zbar_rb:
                bimodal_flag = False
        
    # Payload Case        
    else:
        d_min = 0.08
        if lc > d_min and lc < 0.11:
            if zeta < zbar_pl:
                bimodal_flag = False

    
    # Determine distribution parameters based on Johnson paper
    # Check condition and apply single mode distribution
    lam_c = math.log10(lc)    
    if lc <= d_min or not bimodal_flag:
        if lam_c <= -1.75:
            mu = -0.3
        elif lam_c < -1.25:
            mu = -0.3 - 1.4*(lam_c + 1.75)
        else:
            mu = -1.
        
        if lam_c <= -3.5:
            sig = 0.2
        else:
            sig = 0.2 + 0.1333*(lam_c + 3.5)
        
        mu_1 = mu
        mu_2 = mu
        sig_1 = sig
        sig_2 = sig
        xi = 1.
    
    # Condition not met, use the bi-modal distribution (also depends on 
    # object type)        
    else:

        # Rocket Body
        if rb_flag:
            if lam_c <= -1.4:
                xi = 1.
            elif lam_c < 0.:
                xi = 1. - 0.3571*(lam_c + 1.4)
            else:
                xi = 0.5

            if lam_c <= -0.5:
                mu_1 = -0.45
            elif lam_c < 0.:
                mu_1 = -0.45 - 0.9*(lam_c + 0.5)
            else:
                mu_1 = -0.9

            sig_1 = 0.55

            mu_2 = -0.9

            if lam_c <= -1.:
                sig_2 = 0.28
            elif lam_c < 0.1:
                sig_2 = 0.28 - 0.1636*(lam_c + 1.)
            else:
                sig_2 = 0.1
        
        # Payload
        else:
            if lam_c <= -1.95:
                xi = 0.
            elif lam_c < 0.55:
                xi = 0.3 + 0.4*(lam_c + 1.2)
            else:
                xi = 1.

            if lam_c <= -1.1:
                mu_1 = -0.6
            elif lam_c < 0.:
                mu_1 = -0.6 - 0.318*(lam_c + 1.1)
            else:
                mu_1 = -0.95

            if lam_c <= -1.3:
                sig_1 = 0.1
            elif lam_c < -0.3:
                sig_1 = 0.1 + 0.2*(lam_c + 1.3)
            else:
                sig_1 = 0.3

            if lam_c <= -0.7:
                mu_2 = -1.2
            elif lam_c < -0.1:
                mu_2 = -1.2 - 1.333*(lam_c + 0.7)
            else:
                mu_2 = -2.

            if lam_c <= -0.5:
                sig_2 = 0.5
            elif lam_c < -0.3:
                sig_2 = 0.5 - (lam_c + 0.5)
            else:
                sig_2 = 0.3

    
    # Determine which distribution to use by drawing from uniform distribution
    y = np.random.rand()
    if y < xi:
        mu = mu_1
        sig = sig_1
    else:
        mu = mu_2
        sig = sig_2
        
    # Draw sample using appropriate Gaussian parameters
    x = np.random.normal(mu, sig)
    
    # Solve for area to mass ratio
    A_m = 10**x
    
    return A_m


def compute_dV_mag(A_m, expl_flag=False):
    '''
    This function computes the distribution of object delta-V magnitudes for
    the NASA Standard Breakup Model.
    
    Parameters
    ------
    A_m : float
        area-to-mass ratio [m^2/kg]    
    expl_flag : boolean, optional
        flag to select explosion (True) or collision (False) (default=False)
    
    Returns
    ------
    dV_mag : float
        delta-V magnitude [m/s]
    
    '''
        
    x = math.log10(A_m)

    # Explosion
    if expl_flag:
        mu = 0.2*x + 1.85
    
    # Collision
    else:
        mu = 0.9*x + 2.9   

    sig = 0.4
    
    # Draw sample and compute dv
    v = np.random.normal(mu, sig)    
    dV_mag = 10.**v

    return dV_mag


def nasa_stdbrkup_model(lc_array, v_imp, m_tar, m_imp, expl_flag=False, 
                        rb_flag=False, cs=1.):
    '''
    This function implements the NASA Standard Breakup Model to compute the
    distribution of objects (number, A/m, and dV) following a breakup event
    (collision or explosion).
    
    Parameters
    ------
    lc_array : 1D numpy array
        array of characteristic length values to compute number of objects [m]
    v_imp : float
        relative velocity of impact [m/s]
    m_tar : float
        mass of target (primary object) [kg]
    m_imp : float
        mass of impactor (secondary object) [kg]
    expl_flag : boolean, optional
        flag to select explosion (True) or collision (False) (default=False)
    rb_flag : boolean, optional
        flag to set object type as Payload (False) or Rocket Body (True)
        (default=False)
    cs : float, optional
        unitless scale factor used for explosion model (default=1.)
        
    Returns
    ------
    N_bin : list
        number of objects in each characteristic length bin
    N_cum : 1D numpy array
        cumulative number of objects with d > lc in each bin
    lc_list_full : list
        full list of characteristic length values to generate A/m plot [m]
    A_list : list
        list of areas corresponding to each object [m^2]
    A_m_list :
        list of A/m values corresponding to each object [m^2/kg]
    m_list :
        list of mass values corresponding to each object [kg]
    dV_list :
        list of (3x1) delta-V vectors corresponding to each object [m/s]
    
    '''
    
    # Generate number of objects in each size bin
    # Note that formulas used in compute_number() are cumulative so it is 
    # necessary to start at upper end and work down through size bins and
    # subtract number of objects in all larger size bins to get just the 
    # number in each given bin.
    total = 0
    N_bin = np.zeros(lc_array.shape)
    N_cum = np.zeros(lc_array.shape)
    for ii in range(len(lc_array)-1, -1, -1):
        lc = lc_array[ii]
        N_lc = compute_number(lc, v_imp, m_tar, m_imp, expl_flag, cs)

        N_bin[ii] = math.floor(N_lc - total)
        N_cum[ii] = N_lc
        total += N_bin[ii]
    
    N_bin = [int(N) for N in N_bin]
        
    
    # Loop over characteristic lengths, draw N_bin[ii] samples for A/m and dV
    # to build up full set of debris objects
    lc_list_full = []
    A_list = []
    A_m_list = []
    m_list = []
    dV_list = []
    for ii in range(len(lc_array)):
        
        # Current value of lc
        lc = lc_array[ii]
        
        # Compute A
        A = compute_area(lc)
    
        for jj in range(N_bin[ii]):
            
            # Compute A/m        
            A_m = compute_A_m(lc, rb_flag)
            
            # Compute mass
            m = A/A_m
            
            # Compute dV magnitude
            dV_mag = compute_dV_mag(A_m, expl_flag)  # m/s
            
            # Compute a dV direction from uniform distribution
            x1 = np.random.rand() - 0.5
            x2 = np.random.rand() - 0.5
            x3 = np.random.rand() - 0.5
            
            vect = np.array([x1,x2,x3]).reshape(3,1)
            v_hat = vect/np.linalg.norm(vect)
            dV = dV_mag*v_hat
            
            # Store for output
            lc_list_full.append(lc)
            A_list.append(A)
            A_m_list.append(A_m)
            m_list.append(m)
            dV_list.append(dV)
            
    
    return N_bin, N_cum, lc_list_full, A_list, A_m_list, m_list, dV_list


###############################################################################
# Long-Term Orbit Propagator
###############################################################################

def long_term_propagator(tin, kep, params, LEO_flag, bodies=None):
    '''
    This function performs long-term orbit propagation (on the order of months
    or years) for objects in Earth orbit. The function will propagate orbits
    from LEO-GEO, applying the Semi-Analytic Liu Theory (SALT) propagator for
    objects in LEO and the Tudat propagator for all other orbits.
    
    SALT (LEO) uses the odeint integrator and Keplerian elements not including
    anomaly angle. The force model includes secular and long-periodic effects 
    from J2, J3, and atmospheric drag, using an exponential atmospheric density 
    model (Ref 3 Vallado Table 8-4). 
    
    Tudat (MEO-GEO) uses variable step RK45 and modified equinoctial elements.
    The force model includes a 4x4 gravity field, luni-solar gravity, and 
    a cannonball SRP model.

    Parameters
    ----------
    tin : 2 element numpy array
        initial and final times [t0, tf] for the propagation, given in seconds
        since J2000 epoch
        
    kep : 6 element numpy array
        initial state vector in Keplerian orbit elements as ordered below
        [SMA, ECC, INC, RAAN, AOP, TA] in meters and radians
        
        Note: Tudat assumes order of AOP and RAAN is switched, this is handled
        internally in this function, the input kep array should use the order
        specified here
        
    params : dictionary
        additional parameters for the propagator, such as object data
        (mass, area) or integrator (number of quadrature nodes)
        
    LEO_flag : boolean
        flag to set whether object is LEO (True) or higher orbit (False)
        
    bodies : Tudat object, optional
        Tudat object containing celestial bodies (Earth, Sun, Moon) to which
        the object to propagate is added. The default is None, in which case
        the function will initialize bodies.

    Returns
    -------
    output_dict : dictionary
        output contains a variable amount of keys depending on the propagated
        orbit type.
        
        For LEO, the SALT propagator will output arrays of
        tsec : time in seconds since J2000
        SMA : semi-major axis [m]
        ECC : eccentricity
        INC : inclination [rad]
        RAAN : right ascension of ascending node [rad]
        AOP : argument of periapsis [rad]
        
        For other orbit, Tudat will output the same and additionally
        TA : true anomaly [rad]
        lat : geodetic latitude [rad]
        lon : geodetic longitude [rad]

    '''

    # For LEO objects use SALT propagator
    if LEO_flag:

        # Setup integrator
        intfcn = int_salt_grav_drag
        int_tol = 1e-8
           
        P = 2.*np.pi*np.sqrt(float(kep[0])**3./GME)
        
        # Retain only first 5 orbital elements for initial state
        Xo = kep[0:5].flatten()
        
        # ODEINT
        tin_full = np.arange(tin[0], tin[1]+P/2., P)
        yout = odeint(intfcn, Xo, tin_full, args=(params,), rtol=int_tol,
                      atol=int_tol, tfirst=True)
        tout = tin_full
        
        output_dict = {}
        output_dict['tsec'] = tout
        output_dict['SMA'] = yout[:,0]
        output_dict['ECC'] = yout[:,1]
        output_dict['INC'] = yout[:,2]
        output_dict['RAAN'] = yout[:,3]
        output_dict['AOP'] = yout[:,4]
        
        
        
    # All other cases use Tudat
    else:
        
        # Setup bodies
        if bodies is None:
            bodies = tudat_initialize_bodies()
            
        # Update order of elements - [SMA, ECC, INC, AOP, RAAN, TA]
        Xo = kep.copy()
        Xo[3] = kep[4]
        Xo[4] = kep[3]
            
        # Simulation start and end times
        simulation_start_epoch = tin[0]
        simulation_end_epoch = tin[-1]
            
        # Create satellite object to propagate
        bodies.create_empty_body("Satellite")

        bodies.get("Satellite").mass = params['mass']


        # Create radiation pressure settings, and add to vehicle
        # Radiation pressure is set up assuming tudatpy version >= 0.8
        # Code for earlier tudatpy is commented out below
        reference_area = params['area']
        radiation_pressure_coefficient = params['Cr']
        
        occulting_bodies_dict = dict()
        occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
        
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
            reference_area, radiation_pressure_coefficient, occulting_bodies_dict )
        
        # Radiation pressure setup for tudatpy < 0.8
        # occulting_bodies = ["Earth"]
        # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        #     "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
        # )               
    
        # body_settings.get( "Satellite" ).radiation_pressure_target_settings = radiation_pressure_settings
        
        environment_setup.add_radiation_pressure_target_model(
            bodies, "Satellite", radiation_pressure_settings)


        # Define bodies that are propagated
        bodies_to_propagate = ["Satellite"]

        # Define central bodies of propagation
        central_bodies = ["Earth"]

        # Define accelerations acting on satellite by Sun and Earth.
        accelerations_settings_sat = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure(),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            Earth=[
                propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)
            ],
            Moon=[
                propagation_setup.acceleration.point_mass_gravity()
            ]
        )

        # Create global accelerations settings dictionary.
        acceleration_settings = {"Satellite": accelerations_settings_sat}

        # Create acceleration models.
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)

        # Convert initial orbit elements to Cartesian
        initial_state = element_conversion.keplerian_to_cartesian(Xo, GME)

        # Define list of dependent variables to save
        dependent_variables_to_save = [
            propagation_setup.dependent_variable.total_acceleration("Satellite"),
            propagation_setup.dependent_variable.keplerian_state("Satellite", "Earth"),
            propagation_setup.dependent_variable.latitude("Satellite", "Earth"),
            propagation_setup.dependent_variable.longitude("Satellite", "Earth") 
        ]

        # Create termination settings        
        termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)


        current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45
        current_tolerance = 10.0 ** (-8)

        integrator = propagation_setup.integrator
        integrator_settings = integrator.runge_kutta_variable_step_size(
            1000.0,
            current_coefficient_set,
            np.finfo(float).eps,
            np.inf,
            current_tolerance,
            current_tolerance)


        current_propagator = propagation_setup.propagator.gauss_modified_equinoctial


        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state,
            simulation_start_epoch,
            integrator_settings,
            termination_condition,
            current_propagator,
            output_variables=dependent_variables_to_save
        )

        # Create simulation object and propagate the dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings
        )

        # Extract the resulting state and depedent variable history and convert it to an ndarray
        # states = dynamics_simulator.state_history
        # states_array = result2array(states)
        dep_vars = dynamics_simulator.dependent_variable_history
        dep_vars_array = result2array(dep_vars)
        
        # Store output 
        output_dict = {}
        output_dict['tsec'] = dep_vars_array[:,0]
        output_dict['SMA'] = dep_vars_array[:,4]
        output_dict['ECC'] = dep_vars_array[:,5]
        output_dict['INC'] = dep_vars_array[:,6]
        output_dict['AOP'] = dep_vars_array[:,7]
        output_dict['RAAN'] = dep_vars_array[:,8]
        output_dict['TA'] = dep_vars_array[:,9]
        output_dict['lat'] = dep_vars_array[:,10]
        output_dict['lon'] = dep_vars_array[:,11]        
        
        
    return output_dict
    

def long_term_propagator_all(tin, kep_array, params, bodies=None):
    '''
    This function performs long-term orbit propagation for a batch of objects
    all at once. It only propagates objects in MEO-GEO using Tudat and is not
    set up for use with LEO objects.
    
    Tudat (MEO-GEO) uses variable step RK45 and modified equinoctial elements.
    The force model includes a 4x4 gravity field, luni-solar gravity, and 
    a cannonball SRP model.

    Parameters
    ----------
    tin : 2 element numpy array
        initial and final times [t0, tf] for the propagation, given in seconds
        since J2000 epoch
        
    kep_array : Nx6 numpy array
        each row corresponds to an initial state vector in Keplerian orbit 
        elements as ordered below, for N total objects
        [SMA, ECC, INC, RAAN, AOP, TA] in meters in radians
        
        Note: Tudat assumes order of AOP and RAAN is switched, this is handled
        internally in this function, the input kep array should use the order
        specified here
        
    params : dictionary
        additional parameters for the propagator, such as object data
        (mass, area) or integrator (number of quadrature nodes)

    bodies : Tudat object, optional
        Tudat object containing celestial bodies (Earth, Sun, Moon) to which
        the object to propagate is added. The default is None, in which case
        the function will initialize bodies.

    Returns
    -------
    tout : numpy array
        time in seconds since J2000 epoch
    yout : 2D numpy array
        Each row corresponds to the Keplerian orbit elements for all N objects
        as well as latitude and longitude angles, total of 8*N columns.
        
        For each object, the column order is
        [SMA, ECC, INC, AOP, RAAN, TA, Lat, Lon]

    '''
        
    # Setup bodies
    if bodies is None:
        bodies = tudat_initialize_bodies()
        
    # Simulation start and end times
    simulation_start_epoch = tin[0]
    simulation_end_epoch = tin[-1]
    
    # Retrieve physical parameter data
    mass_list = params['mass_list']
    area_list = params['area_list']
    Cd_list = params['Cd_list']
    Cr_list = params['Cr_list']
    
    # Define central bodies of propagation
    central_bodies = ["Earth"]
    
    # Define occulting bodies for SRP
    #occulting_bodies = ['Earth']
    occulting_bodies_dict = dict()
    occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
    
    # Loop over objects and add to bodies
    Nobj = int(kep_array.shape[0])
    central_bodies = central_bodies*Nobj
    bodies_to_propagate = []
    initial_state = np.array([])
    dependent_variables_to_save = []
    for jj in range(Nobj):
        
        # Initialize empty body
        jj_str = str(jj)
        bodies.create_empty_body(jj_str)
        bodies_to_propagate.append(jj_str)
        
        # Initial State            
        # Update order of elements - [SMA, ECC, INC, AOP, RAAN, TA]
        kep = kep_array[jj,:]
        Xo = kep.copy()
        Xo[3] = kep[4]
        Xo[4] = kep[3]
        
        Xo_cart = element_conversion.keplerian_to_cartesian(Xo, GME)
        initial_state = np.append(initial_state, Xo_cart)
        
        # Physical parameters
        mass = mass_list[jj]
        area = area_list[jj]
        Cd = Cd_list[jj]
        Cr = Cr_list[jj]
        
        # Set mass
        bodies.get(jj_str).mass = mass
        
        # Radiation pressure
        # Radiation pressure is set up assuming tudatpy version >= 0.8
        # Code for earlier tudatpy is commented out below
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
            area, Cr, occulting_bodies_dict )
    
        environment_setup.add_radiation_pressure_target_model(
            bodies, jj_str, radiation_pressure_settings)
        
        # Radiation pressure for tudatpy < 0.8
        # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        #     'Sun', area_list[jj], Cr, occulting_bodies
        # )
        # environment_setup.add_radiation_pressure_interface(
        #     bodies, jj_str, radiation_pressure_settings)
            
        
        # Define list of dependent variables to save
        dependent_variables_to_save.append(propagation_setup.dependent_variable.keplerian_state(jj_str, "Earth"))
        dependent_variables_to_save.append(propagation_setup.dependent_variable.latitude(jj_str, "Earth"))
        dependent_variables_to_save.append(propagation_setup.dependent_variable.longitude(jj_str, "Earth"))


    # Define accelerations acting on object by Sun and Earth.
    acceleration_settings_setup = dict(
        Sun=[
            # propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)
        ],
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ]
    )

    # Create global accelerations settings dictionary.
    acceleration_settings = {}
    for jj in range(Nobj):
        acceleration_settings[str(jj)] = acceleration_settings_setup
        
        
    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Create termination settings        
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)


    current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_45
    current_tolerance = 10.0 ** (-8)

    integrator = propagation_setup.integrator
    integrator_settings = integrator.runge_kutta_variable_step_size(
        1000.0,
        current_coefficient_set,
        np.finfo(float).eps,
        np.inf,
        current_tolerance,
        current_tolerance)


    current_propagator = propagation_setup.propagator.gauss_modified_equinoctial


    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,
        current_propagator,
        output_variables=dependent_variables_to_save
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Extract the resulting state and depedent variable history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)
    dep_vars = dynamics_simulator.dependent_variable_history
    dep_vars_array = result2array(dep_vars)
    
    # Store output 
    tout = states_array[:,0]
    yout = dep_vars_array[:,1:]        
    
    
    return tout, yout



###############################################################################
# SALT Propagator Functions
###############################################################################

def atmosphere_lookup(h):
    '''
    This function acts as a lookup table for atmospheric density reference
    values, reference heights, and scale heights for a range of different 
    altitudes from 100 - 1000+ km.  Values from Vallado 4th ed. Table 8-4.
    
    Parameters
    ------
    h : float
        altitude [m]
    
    Returns
    ------
    rho0 : float
        reference density [kg/m^3]
    h0 : float
        reference altitude [m]
    H : float
        scale height [m]

    '''
        
    # Input h in meters, convert to km to read from table
    h = h/1000.
        
    if h <= 100:
        # Assume at this height we have re-entered atmosphere
        rho0 = 0
        h0 = 1
        H = 1
    elif h < 110:
        rho0 = 5.297e-7 * 1e9  # kg/km^3
        h0 = 100.    # km
        H = 5.877    # km    
    elif h < 120:
        rho0 = 9.661e-8 * 1e9  # kg/km^3
        h0 = 110.    # km
        H = 7.263    # km   
    elif h < 130:
        rho0 = 2.438e-8 * 1e9  # kg/km^3
        h0 = 120.    # km
        H = 9.473    # km   
    elif h < 140: 
        rho0 = 8.484e-9 * 1e9  # kg/km^3
        h0 = 130.    # km
        H = 12.636   # km       
    elif h < 150:
        rho0 = 3.845e-9 * 1e9  # kg/km^3
        h0 = 140.    # km
        H = 16.149   # km       
    elif h < 180:
        rho0 = 2.070e-9 * 1e9  # kg/km^3
        h0 = 150.    # km
        H = 22.523   # km       
    elif h < 200:
        rho0 = 5.464e-10 * 1e9  # kg/km^3
        h0 = 180.    # km
        H = 29.740   # km     
    elif h < 250:
        rho0 = 2.789e-10 * 1e9  # kg/km^3
        h0 = 200.    # km
        H = 37.105   # km   
    elif h < 300:
        rho0 = 7.248e-11 * 1e9  # kg/km^3
        h0 = 250.    # km
        H = 45.546   # km       
    elif h < 350:
        rho0 = 2.418e-11 * 1e9  # kg/km^3
        h0 = 300.    # km
        H = 53.628   # km       
    elif h < 400:
        rho0 = 9.518e-12 * 1e9  # kg/km^3
        h0 = 350.    # km
        H = 53.298   # km       
    elif h < 450:
        rho0 = 3.725e-12 * 1e9   # kg/km^3
        h0 = 400.    # km
        H = 58.515   # km     
    elif h < 500:
        rho0 = 1.585e-12 * 1e9   # kg/km^3
        h0 = 450.    # km
        H = 60.828   # km   
    elif h < 600:
        rho0 = 6.967e-13 * 1e9   # kg/km^3
        h0 = 500.    # km
        H = 63.822   # km
    elif h < 700:
        rho0 = 1.454e-13 * 1e9   # kg/km^3
        h0 = 600.    # km
        H = 71.835   # km
    elif h < 800:
        rho0 = 3.614e-14 * 1e9   # kg/km^3
        h0 = 700.    # km
        H = 88.667   # km       
    elif h < 900:
        rho0 = 1.17e-14 * 1e9    # kg/km^3
        h0 = 800.    # km
        H = 124.64   # km       
    elif h < 1000:
        rho0 = 5.245e-15 * 1e9   # kg/km^3
        h0 = 900.    # km
        H = 181.05   # km       
    else:
        rho0 = 3.019e-15 * 1e9   # kg/km^3
        h0 = 1000.   # km
        H = 268.00   # km
    
    
    # Table parameters in km, convert to meters for output
    rho0 *= (1./1e9)
    h0 *= 1000.
    H *= 1000.
    
    
    return rho0, h0, H


def lgwt(N,a,b):
    '''
    This function returns the locations and weights of nodes to use for
    Gauss-Legendre Quadrature for numerical integration.
    
    Adapted from MATLAB code by Greg von Winckel
    
    Parameters
    ------
    N : int
        number of nodes
    a : float
        lower limit of integral
    b : float
        upper limit of integral
    
    Returns
    ------
    x_vect : 1D numpy array
        node locations
    w_vect : 1D numpy array
        node weights
    
    '''
    
    xu = np.linspace(-1, 1, N)
    
    # Initial Guess
    y=np.cos((2*np.arange(0,N)+1)*np.pi/(2*(N-1)+2))+(0.27/N)*np.sin(np.pi*xu*(N-1)/(N+1))
    y=y.reshape(len(y),1)
    
    # Legendre-Gauss Vandermonde Matrix
    L=np.zeros((N,N+1))
    
    # Derivative of LGVM
    
    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method
    y0=2.
    eps = np.finfo(float).eps
    
    # Iterate until new points are uniformly within epsilon of old points
    while max(abs(y-y0)) > eps:
        
        L[:,0] = 1.
        
        L[:,1] = y.flatten()
        
        for k in range(1,N):
            
            L1 = (2*(k+1)-1)*y.flatten()
            L2 = L[:,k].flatten()
            L3 = L[:,k-1].flatten()
            
            L[:,k+1] = (np.multiply(L1, L2) - k*L3)/(k+1)

        y2 = np.multiply(y.flatten(), y.flatten())
        Lp1=(N+1)*( L[:,N-1]- np.multiply(y.flatten(), L[:,N].flatten() ))  
        Lp = np.multiply(Lp1, 1./(1-y2))

        y0 = y.copy()
        y = y0 - np.reshape(np.multiply(L[:,N].flatten(), 1./Lp), (len(y0), 1))
        
    # Linear map from[-1,1] to [a,b]
    x_vect = (a*(1-y)+b*(1+y))/2
    x_vect = x_vect.flatten()
    
    # Compute the weights
    y2 = np.multiply(y, y)
    Lp2 = np.multiply(Lp, Lp)
    w_vect = (b-a)/(np.multiply((1-y2.flatten()), Lp2.flatten()))*((N+1)/N)**2.
    
    return x_vect, w_vect


def int_salt_grav_drag(t, X, params):
    '''
    This function computes the derivatives for the Semi-Analytic Liu Theory
    orbit propagator, to be used with a numerical integrator such as RK4.

    Parameters
    ------
    t : float
        current time
    X : nx1 numpy array
        state vector
    params : dictionary
        extra parameters for integrator

    Returns
    -------
    dX : nx1 numpy array
        derivative vector
    
    '''
    
    # Retrieve parameters
    J2 = J2E
    J3 = J3E
    GM = GME
        
    # Retrieve states
    a = float(X[0])
    e = float(X[1])
    i = float(X[2])
    RAAN = float(X[3])
    w = float(X[4])
    
    # Check re-entry condition
    rp = a*(1. - e)
    hp = rp - Re
    if hp < 100000:
        dX = np.zeros(5,)
        return dX
    
    # Compute orbit params
    n = np.sqrt(GM/a**3.)
    p = a*(1.-e**2.)

    # Compute dadt and dedt for drag using Gauss Quadrature
    dadt_drag, dedt_drag = compute_gauss_quad_drag(X, params)
    
    # Compute dadt and dedt for gravity perturbations
    dadt_grav = 0.

    dedt_grav = -(3./8.)*n*J3*(Re/p)**3. * \
        (4. - 5.*np.sin(i)**2.)*(1.-e**2.)*np.sin(i)*np.cos(w)

    didt_grav = (3./8.)*n*J3*(Re/p)**3. * e * \
        (4. - 5.*np.sin(i)**2.)*np.cos(i)*np.cos(w)
    
    dRAANdt_grav = -(3./2.)*n*(Re/p)**2. * (J2*np.cos(i) + (J3/4.)*(Re/p) * 
                     (15.*np.sin(i)**2. - 4.)*(e*(1./np.tan(i))*np.sin(w)))
    
    dwdt_grav = (3./4.)*n*J2*(Re/p)**2.*(4. - 5.*np.sin(i)**2.) + \
        (3./8.)*n*J3*(Re/p)**3.*np.sin(w) * ((4. - 5.*np.sin(i)**2.) * 
         ((np.sin(i)**2. - e**2.*np.cos(i)**2.)/(e*np.sin(i))) + 
         2.*np.sin(i)*(13. - 15.*np.sin(i)**2.)*e)


    # Set up final derivative vector
    dX = np.zeros(5,)
    
    dX[0] = dadt_drag + dadt_grav    
    dX[1] = dedt_drag + dedt_grav    
    dX[2] = didt_grav    
    dX[3] = dRAANdt_grav    
    dX[4] = dwdt_grav
   
    
    return dX


def compute_gauss_quad_drag(X, params):
    '''
    This function computes the derivatives of semi-major axis and eccentricity
    resulting from drag forces as modeled by SALT.  The derivative requires
    numerical evaluation of an integral, which is done using Gauss-Legendre
    quadrature.
    
    Parameters
    ------
    X : 5x1 numpy array
        state vector [a, e, i, RAAN, w]
        units of distance in m, angles in radians
    params : dictionary
        additional input parameters including number of nodes to use in
        quadrature and physical parameters such as Cd, A/m ratio, etc.
    
    Returns
    ------
    dadt_drag : float
        instantaneous change in SMA wrt time [m/s]
    dedt_drag : float
        instantaneous change in eccentricity wrt time [1/s]
        
    '''
    
    # Retrieve values from state vector and params
    a = X[0]
    e = X[1]
    i = X[2]
    
    GM = GME
    Cd = params['Cd']
    A_m = params['area']/params['mass']
    N = params['Nquad']
    
    # Compute orbit params
    n = np.sqrt(GM/a**3.)
    p = a*(1.-e**2.)
    B = Cd*A_m
    
    # Compute locations and weights of nodes
    theta_vect, w_vect = lgwt(N, 0., 2.*np.pi)

    # Compute function value at node locations
    dadt_vect = np.array([])
    dedt_vect = np.array([])
    for theta_i in theta_vect:
        
        # Compute orbit and density parameters for this node
        r = p/(1. + e*np.cos(theta_i))  # orbit radius
        h = r - Re
        rho0, h0, H = atmosphere_lookup(h)
        rho = rho0*np.exp(-(h-h0)/H)
        eterm = 1. + e**2. + 2.*e*np.cos(theta_i)
        V = ((GM/p)*eterm)**(1./2.) * \
            (1.-((1.-e**2.)**(3./2.)/eterm)*(wE/n)*np.cos(i))
        
        # Compute function values at current node location
        dadt_i = -(B/(2.*np.pi))*rho*V*(r**2./(a*(1.-e**2)**(3./2.))) * \
            (eterm - wE*np.sqrt(a**3.*(1.-e**2.)/GM)*np.cos(i))
            
        dedt_i = -(B/(2.*np.pi))*rho*V*(e + np.cos(theta_i) - 
                   r**2.*wE*np.cos(i)/(2.*np.sqrt(GM*a*(1.-e)**2.)) * 
                   (2.*(e + np.cos(theta_i)) - e*np.sin(theta_i)**2.)*(r/a)**2. * 
                   (1.-e**2.)**(-1./2.))
    
        
        # Store in vector
        dadt_vect = np.append(dadt_vect,dadt_i)
        dedt_vect = np.append(dedt_vect,dedt_i)
        
    # Compute weighted output
    dadt_drag = np.dot(dadt_vect, w_vect)
    dedt_drag = np.dot(dedt_vect, w_vect)
        
    return dadt_drag, dedt_drag



###############################################################################
# Tudat Functions
###############################################################################

def tudat_initialize_bodies():
    '''
    This function initializes the bodies object for use in the Tudat 
    propagator. For the cases considered, only Earth, Sun, and Moon are needed,
    with Earth as the frame origin.
    
    Parameters
    ------
    None
    
    Returns
    ------
    bodies : tudat object
    
    '''
    
    # Load spice kernels
    spice_interface.load_standard_kernels()

    # Define string names for bodies to be created from default.
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


def kep2cart(SMA, ECC, INC, RAAN, AOP, TA):
    '''
    This function converts Keplerian elements to Cartesian position and 
    velocity. The function acts as a wrapper for the tudat element conversion
    function, in order to be explicit about the order of elements as tudat and
    SALT have RAAN and AOP swapped as a default.

    Parameters
    ----------
    SMA : float
        semi-major axis [m]
    ECC : float
        eccentricity
    INC : float
        inclination [rad]
    RAAN : float
        right ascension of ascending node [rad]
    AOP : float
        argument of periapsis [rad]
    TA : float
        true anomaly [rad]

    Returns
    -------
    Xf : 6 element numpy array
        Cartesian position and velocity vector [m, m/s]

    '''
    
    Xo = np.array([SMA, ECC, INC, AOP, RAAN, TA])    
    Xf = element_conversion.keplerian_to_cartesian(Xo, GME)
    
    return Xf


def cart2kep(X):
    '''
    This function converts Cartesian position and velocity to Keplerian 
    elements. The function acts as a wrapper for the tudat element conversion
    function, in order to be explicit about the order of elements as tudat and
    SALT have RAAN and AOP swapped as a default.

    Parameters
    ----------
    X : 6 element numpy array
        Cartesian position and velocity vector [m, m/s]

    Returns
    -------
    SMA : float
        semi-major axis [m]
    ECC : float
        eccentricity
    INC : float
        inclination [rad]
    RAAN : float
        right ascension of ascending node [rad]
    AOP : float
        argument of periapsis [rad]
    TA : float
        true anomaly [rad]

    '''
    
    kep = element_conversion.cartesian_to_keplerian(X, GME)
    SMA = kep[0]
    ECC = kep[1]
    INC = kep[2]
    AOP = kep[3]
    RAAN = kep[4]
    TA = kep[5]
    
    return SMA, ECC, INC, RAAN, AOP, TA



###############################################################################
# Unit Test Functions
###############################################################################

def unit_test_henize():
    '''
    Test case for scaling parameter to use in explosion model.
    
    '''
    
    H_vect = np.arange(200., 4000., 10.)*1000.   # meters
    cs_vect = np.zeros(H_vect.shape)
    for ii in range(len(H_vect)):
        H = H_vect[ii]
        cs = compute_cs_henize(H)
        cs_vect[ii] = cs
        
    
    plt.figure()
    plt.plot(H_vect/1000., cs_vect, 'k-')
    plt.xlabel('H [km]')
    plt.ylabel('cs')
    
    return


def unit_test_N(expl_flag):
    '''
    Test case to compute and plot number of objects generated in breakup

    Parameters
    ----------
    expl_flag : boolean
        flag to determine if using explosion (True) or collision (False) model

    Returns
    -------
    None.

    '''
    
    # Setup OneWeb case per Radtke 2017
    lc_array = np.logspace(-2, 1, 1000)
    m_tar = 150.
    m_imp = 0.058
    v_imp = 14.5*1000.
    cs = 1.
    
    
    total = 0
    N_bin = np.zeros(lc_array.shape)
    N_cum = np.zeros(lc_array.shape)
    for ii in range(len(lc_array)-1, -1, -1):
        lc = lc_array[ii]
        N_lc = compute_number(lc, v_imp, m_tar, m_imp, expl_flag, cs)

        N_bin[ii] = math.floor(N_lc - total)
        N_cum[ii] = N_lc
        total += N_bin[ii]
    
    N_bin = [int(N) for N in N_bin]
    
    print('Total Number of Debris Particles: ', sum(N_bin))
    print('Check N cumulative', N_cum[0])
    
    plt.figure()
    plt.semilogx(lc_array, N_cum, 'k.')
    plt.xlabel('Characteristic Length [m]')
    plt.ylabel('Cumulative Number of Objects')

    plt.figure()
    plt.semilogx(lc_array, N_bin, 'k.')
    plt.xlabel('Characteristic Length [m]')
    plt.ylabel('Number of Objects Per Bin')
    
    plt.show()
    
    
    return



def unit_test_a_m(rb_flag):
    '''
    Test case to generate plot of area-to-mass distribution as a function of
    characteristic length. Plots should match Figs 2-3 in reference.

    Parameters
    ----------
    rb_flag : boolean
        flag to determine if using rocket body (True) or payload (False) model

    Returns
    -------
    None.
    
    Reference
    ------
    [1] Finkelman, D., Oltrogge, D.L., Faulds, A., Gerber, J., "Analysis of the
        Response of a Space Surveillance Network to Orbital Debris Events,"
        AAS-08-227.

    '''

    N = 10
    lc_array = np.logspace(-4,1,1000)
    
    # Setup for plot
    if rb_flag:
        d_min = 0.017
        plot_title = 'Rocket Body'
    else:
        d_min = 0.08
        plot_title = 'Payload'
    
    # Setup output
    A_m_array = np.zeros((N,len(lc_array)))
    lc_plot1 = []
    mu_plot1 = []
    sig_plot1pos = []
    sig_plot1neg = []
    lc_plot2 = []
    mu_plot2 = []
    sig_plot2pos = []
    sig_plot2neg = []
    lc_plot3 = []
    mu_plot3 = []
    sig_plot3pos = []
    sig_plot3neg = []
    for ii in range(len(lc_array)):
        
        lc = lc_array[ii]
        lam_c = math.log10(lc) 
        
        # Compute deterministic mu and sigma values
        if lc < 0.11:
            if lam_c <= -1.75:
                mu = -0.3
            elif lam_c < -1.25:
                mu = -0.3 - 1.4*(lam_c + 1.75)
            else:
                mu = -1.
            
            if lam_c <= -3.5:
                sig = 0.2
            else:
                sig = 0.2 + 0.1333*(lam_c + 3.5)
                
            lc_plot1.append(lc)
            mu_plot1.append(10**mu)
            sig_plot1pos.append(10**(mu + 2*sig))
            sig_plot1neg.append(10**(mu - 2*sig))
            
        if lc > d_min:
            
            # Rocket Body
            if rb_flag:

                if lam_c <= -0.5:
                    mu_1 = -0.45
                elif lam_c < 0.:
                    mu_1 = -0.45 - 0.9*(lam_c + 0.5)
                else:
                    mu_1 = -0.9

                sig_1 = 0.55

                mu_2 = -0.9

                if lam_c <= -1.:
                    sig_2 = 0.28
                elif lam_c < 0.1:
                    sig_2 = 0.28 - 0.1636*(lam_c + 1.)
                else:
                    sig_2 = 0.1
                
                
            # Payload
            else:

                if lam_c <= -1.1:
                    mu_1 = -0.6
                elif lam_c < 0.:
                    mu_1 = -0.6 - 0.318*(lam_c + 1.1)
                else:
                    mu_1 = -0.95

                if lam_c <= -1.3:
                    sig_1 = 0.1
                elif lam_c < -0.3:
                    sig_1 = 0.1 + 0.2*(lam_c + 1.3)
                else:
                    sig_1 = 0.3

                if lam_c <= -0.7:
                    mu_2 = -1.2
                elif lam_c < -0.1:
                    mu_2 = -1.2 - 1.333*(lam_c + 0.7)
                else:
                    mu_2 = -2.

                if lam_c <= -0.5:
                    sig_2 = 0.5
                elif lam_c < -0.3:
                    sig_2 = 0.5 - (lam_c + 0.5)
                else:
                    sig_2 = 0.3
                    
            
            lc_plot2.append(lc)
            mu_plot2.append(10**mu_1)
            sig_plot2pos.append(10**(mu_1 + 2*sig_1))
            sig_plot2neg.append(10**(mu_1 - 2*sig_1))
            
            lc_plot3.append(lc)
            mu_plot3.append(10**mu_2)
            sig_plot3pos.append(10**(mu_2 + 2*sig_2))
            sig_plot3neg.append(10**(mu_2 - 2*sig_2))
        
        
        # Compute sampled A_m values
        for jj in range(N):
        
            A_m = compute_A_m(lc, rb_flag)            
            A_m_array[jj,ii] = float(A_m)
            

            
    plt.figure()
    plt.loglog(lc_array, A_m_array.T, 'k.')
    plt.loglog(lc_plot1, mu_plot1, 'r', lw=3, label='$\mu_1$')
    plt.loglog(lc_plot1, sig_plot1pos, 'r--', lw=3, label='2$\sigma_1$')
    plt.loglog(lc_plot1, sig_plot1neg, 'r--', lw=3)
    plt.loglog(lc_plot2, mu_plot2, 'c', lw=3, label='$\mu_2$')
    plt.loglog(lc_plot2, sig_plot2pos, 'c--', lw=3, label='2$\sigma_2$')
    plt.loglog(lc_plot2, sig_plot2neg, 'c--', lw=3)
    plt.loglog(lc_plot3, mu_plot3, 'm', lw=3, label='$\mu_3$')
    plt.loglog(lc_plot3, sig_plot3pos, 'm--', lw=3, label='2$\sigma_3$')
    plt.loglog(lc_plot3, sig_plot3neg, 'm--', lw=3)
    plt.title(plot_title)
    plt.xlabel('Characteristic Length [m]')
    plt.ylabel('Area-to-Mass Ratio [m$^2$/kg]')
    plt.legend()
    
    plt.show()
    
    
    return


def unit_test_dv(expl_flag):
    '''
    Test case to generate plot of delta-V distribution as a function of
    characteristic length. Plots should match Figs 4-5 in reference.

    Parameters
    ----------
    expl_flag : boolean
        flag to determine if using explosion (True) or collision (False) model

    Returns
    -------
    None.
    
    Reference
    ------
    [1] Finkelman, D., Oltrogge, D.L., Faulds, A., Gerber, J., "Analysis of the
        Response of a Space Surveillance Network to Orbital Debris Events,"
        AAS-08-227.

    '''
    
    if expl_flag:
        plot_title = 'Explosions'
    else:
        plot_title = 'Collisions'
    
    N = 10
    A_m_array = np.logspace(-3,2,1000)
    dv_array = np.zeros((N,len(A_m_array)))
    mu_plot = []
    sig_plot2pos = []
    sig_plot2neg = []
    for ii in range(len(A_m_array)):
        
        A_m = A_m_array[ii]
    
        # Compute deterministic mean and sigma
        x = math.log10(A_m)
    
        # Explosion
        if expl_flag:
            mu = 0.2*x + 1.85
        
        # Collision
        else:
            mu = 0.9*x + 2.9   
    
        sig = 0.4
        
        mu_plot.append(10**mu)
        sig_plot2pos.append(10**(mu + 2*sig))
        sig_plot2neg.append(10**(mu - 2*sig))
        
        # Compute sampled A_m values
        for jj in range(N):
        
            dv = compute_dV_mag(A_m, expl_flag)            
            dv_array[jj,ii] = float(dv)
            
    plt.figure()
    plt.loglog(A_m_array, dv_array.T, 'k.')
    plt.loglog(A_m_array, mu_plot, 'r', lw=3, label='$\mu$')
    plt.loglog(A_m_array, sig_plot2pos, 'r--', lw=3, label='2$\sigma$')
    plt.loglog(A_m_array, sig_plot2neg, 'r--', lw=3)
    plt.title(plot_title)
    plt.xlabel('Area-to-Mass Ratio [m$^2$/kg]')
    plt.ylabel('Delta-V Magnitude [m/s]')
    plt.legend()
    
    plt.show()
    
    
    return


def unit_test_SALT_prop():
    
    # Sample LEO object
    SMA_meters = 7000000.
    ECC = 0.001
    INC = 98.6*np.pi/180.
    RAAN = 0.1
    AOP = 0.2
    TA = 0.3
    kep = np.array([SMA_meters, ECC, INC, RAAN, AOP, TA])
    
    tin = np.array([0., 86400.*90.])
    
    params = {}
    params['Nquad'] = 20
    params['area'] = 1.
    params['mass'] = 100.
    params['Cd'] = 2.2
    params['Cr'] = 1.3
    
    LEO_flag = True
    
    output_dict = long_term_propagator(tin, kep, params, LEO_flag, bodies=None)
    
    tsec = output_dict['tsec']
    SMA = output_dict['SMA']
    
    plt.figure()
    plt.plot(tsec/86400., SMA/1000., 'k.')
    plt.xlabel('Time [days]')
    plt.ylabel('SMA [km]')
    
    return


def unit_test_tudat_prop():
    
    # Sample GEO object
    SMA_meters = 42164.1*1000.
    ECC = 0.001
    INC = 0.001
    RAAN = 0.1
    AOP = 0.2
    TA = 0.3
    kep = np.array([SMA_meters, ECC, INC, RAAN, AOP, TA])
    
    tin = np.array([0., 86400.*90.])
    
    params = {}
    params['area'] = 1.
    params['mass'] = 100.
    params['Cd'] = 2.2
    params['Cr'] = 1.3
    
    LEO_flag = False
    bodies = tudat_initialize_bodies()
    
    output_dict = long_term_propagator(tin, kep, params, LEO_flag, bodies)
    
    tsec = output_dict['tsec']
    SMA = output_dict['SMA']
    
    plt.figure()
    plt.plot(tsec/86400., SMA/1000., 'k.')
    plt.xlabel('Time [days]')
    plt.ylabel('SMA [km]')
    
    return



###############################################################################
# Plotting Functions
###############################################################################


def plot_breakup_stats(lc_array, N_bin, N_cum, lc_list_full, A_list, A_m_list, m_list, dV_list):
    
    plt.figure()
    plt.semilogx(lc_array, N_cum, 'k.')
    plt.xlabel('Diameter [m]')
    plt.ylabel('Cumulative Number of Objects')

    plt.figure()
    plt.semilogx(lc_array, N_bin, 'k.')
    plt.xlabel('Diameter [m]')
    plt.ylabel('Number of Objects Per Bin')
    
    plt.figure()
    plt.loglog(lc_list_full, A_m_list, 'k.')
    plt.xlabel('Diameter [m]')
    plt.ylabel('A/m Ratio [m$^2$/kg]')
    plt.xlim([0.01, 10.])

    dV_mag_list = [np.linalg.norm(dV) for dV in dV_list]
    plt.figure()
    plt.loglog(A_m_list, dV_mag_list, 'k.')
    plt.xlabel('A/m Ratio [m$^2$/kg]')
    plt.ylabel('delta-V [m/s]')
    
    
    return



if __name__ == '__main__':
    
    plt.close('all')
    
    unit_test_henize()
    
    unit_test_N(True)
    unit_test_N(False)
        
    unit_test_a_m(True)
    unit_test_a_m(False)
    
    unit_test_dv(True)
    unit_test_dv(False)
    
    unit_test_SALT_prop()
    unit_test_tudat_prop()