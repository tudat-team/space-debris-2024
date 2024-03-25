import numpy as np
import math
import matplotlib.pyplot as plt


###############################################################################
# USER INPUTS - CHOOSE VALUES
###############################################################################

def ramp_motion_input_parameters():
    
    # Initialize output
    params = {}
    
    # Choose the ramp incline angle
    params['ramp_incline'] = 40.        # deg
    
    # Choose an object shape (all options included)
    # Note that the cylinder can either slide or roll so 2 choices are listed
    params['shape'] = 'sphere'
    # params['shape'] = 'cylinder_slide'
    # params['shape'] = 'cylinder_roll'
    # params['shape'] = 'box'
    # params['shape'] = 'wedge'
    
    # Choose a material to select mu and rho (all options included)
    params['material'] = 'aluminium'
    # params['material'] = 'wood'
    # params['material'] = 'steel'
    
    # Choose whether to include drag force (True) or not (False)
    params['drag_flag'] = True
    # params['drag_flag'] = False
    
    # Dimensions
    # Specific choices depend on which shape is chosen above, make sure to 
    # fill in the correct entries. All dimensions given in units of centimeters
    # (maximum value is 10cm in all cases).
    
    if params['shape'] == 'sphere':
        
        # Choose diameter in centimeters
        params['diameter'] = 10.
        
    elif params['shape'][0:8] == 'cylinder':
        
        # Choose diameter and length in centimeters
        params['diameter'] = 10.
        params['length'] = 10.
        
    elif params['shape'] == 'box':
        
        # Choose length, width, and height in centimeters
        params['length'] = 10.
        params['width'] = 10.
        params['height'] = 10.
        
    elif params['shape'] == 'wedge':
        
        # Choose side and height in centimeters 
        # (assumes equilateral triangle so all sides measure the same)
        params['side'] = 10.
        params['height'] = 10.
        
    else:
        
        print('Error: invalid shape provided! Must choose from the options: '
              'sphere, cylinder_slide, cylinder_roll, box, wedge')
        print('Current selection: ', params['shape'])
        return
    

    return params



###############################################################################
# END USER INPUTS - REMAINING VALUES NEEDED ARE CALCULATED
###############################################################################


def compute_ramp_time():
    
    # Retrieve user-defined parameters
    params = ramp_motion_input_parameters()
    
    # Basic physics
    params['rho_air'] = 1.225e-6    # kg/cm^3
    params['g'] = 981               # cm/s^2
    
    # Ramp properties
    ramp_length = 300.              # cm
    
    # Material properties
    rho_aluminium = 2.7e-3          # kg/cm^3
    rho_wood = 0.7e-3               # kg/cm^3
    rho_steel = 8.0e-3              # kg/cm^3
    
    # Coefficient of kinetic friction (with steel ramp)
    mu_aluminium = 0.47    
    mu_wood = 0.30
    mu_steel = 0.42
    
    # Store rho and mu
    if params['material'] == 'aluminium':
        params['mu'] = mu_aluminium
        params['rho'] = rho_aluminium
        
    elif params['material'] == 'wood':
        params['mu'] = mu_wood
        params['rho'] = rho_wood
        
    elif params['material'] == 'steel':
        params['mu'] = mu_steel
        params['rho'] = rho_steel
        
    else:
        print('Error: invalid material provided! Must choose from the options: '
              'aluminium, wood, steel')
        print('Current selection: ', params['material'])
        return
        
        
    # Object dimensions, mass, etc.
    if params['shape'] == 'sphere':
        
        # Sphere must be rolling
        params['rolling_flag'] = True
        
        # Area, mass, and Cd
        radius = params['diameter']/2.
        params['area'] = np.pi*radius**2.                   # cm^2
        V = (4./3.)*np.pi*radius**3.                        # cm^3
        params['mass'] = params['rho']*V                    # kg
        params['Cd'] = 1.17
        params['radius'] = radius                           # cm
        params['J'] = (2./5.)*params['mass']*radius**2.     # kg*cm^2
        
    elif params['shape'][0:8] == 'cylinder':
        
        # Determine if rolling
        if params['shape'][9:] == 'slide':
            params['rolling_flag'] = False
        elif params['shape'][9:] == 'roll':
            params['rolling_flag'] = True
            
        # Area, mass, and Cd
        radius = params['diameter']/2.
        params['area'] = params['diameter']*params['length']    # cm^2
        V = np.pi*radius**2.*params['length']                   # cm^3
        params['mass'] = params['rho']*V                        # kg
        params['Cd'] = 1.17
        params['radius'] = radius                               # cm
        params['J'] = (1./2.)*params['mass']*radius**2.         # kg*cm^2
        
    elif params['shape'] == 'box':
        
        # Box must be sliding
        params['rolling_flag'] = False
        
        # Area, mass, and Cd
        params['area'] = params['width']*params['height']       # cm^2
        V = params['length']*params['area']                     # cm^3
        params['mass'] = params['rho']*V                        # kg
        params['Cd'] = 2.05
        params['radius'] = 0.
        params['J'] = 0.
        
        
    elif params['shape'] == 'wedge':
        
        # Wedge must be sliding
        params['rolling_flag'] = False
        
        # Area, mass, and Cd
        params['area'] = params['side']*params['height']        # cm^2
        V = 0.5*params['side']**2.*params['height']             # cm^3
        params['mass'] = params['rho']*V                        # kg
        params['Cd'] = 1.55
        params['radius'] = 0.
        params['J'] = 0.
        
        
    # Account for no drag case by zeroing out the density
    if not params['drag_flag']:
        params['rho_air'] = 0.
        
    # If rolling, check for no-slip condition
    if params['rolling_flag']:
        
        print('Rolling case')
        
        mu = params['mu']
        J = params['J']
        mass = params['mass']
        radius = params['radius']
        ramp_incline = params['ramp_incline']*np.pi/180.
        criteria = (1./(1. + (mass*radius**2.)/J))*np.tan(ramp_incline)
        
        print('mu', mu)
        print('no-slip criteria', criteria)
        
        if mu < criteria:
            
            print('Error: no-slip condition not met for rolling case')
            print('mu', mu, ' should be greater than ', criteria)
            
        else:
            
            print('Good! no-slip condition is met')
        
        
        
        
    # Compute the time to reach the bottom of the ramp by numerical integration
    # Set up RK4 parameters, set tin to exceed time needed to reach bottom
    intfcn = int_ramp
    tin = np.array([0., 10.])
    Xo = np.array([0., 0.])
    params['step'] = 0.001
    
    tout, xout, fcalls = rk4(intfcn, tin, Xo, params)
        
    # Find the first time for which the distance traveled is greater than the
    # ramp length
    ind = int(np.nonzero(xout[:,0] > ramp_length)[0][0])
    
    # Use linear interpolation to find the time the end of the ramp is reached
    frac = (xout[ind,0] - ramp_length)/(xout[ind,0] - xout[ind-1,0])
    t1 = tout[ind]
    
    t = t1 - frac*params['step']
    
    return t


def int_ramp(t, X, params):
    
    # Retrieve parameters from input data
    ramp_incline = params['ramp_incline']*np.pi/180.        # radians
    g = params['g']
    m = params['mass']
    mu = params['mu']
    rho_air = params['rho_air']
    Cd = params['Cd']
    A = params['area']
    J = params['J']
    r = params['radius']
    rolling_flag = params['rolling_flag']
    
    # Retrieve states
    x = X[0]
    dx = X[1]
    
    # Computations
    Fw = m*g*np.sin(ramp_incline)
    N = m*g*np.cos(ramp_incline)
    Ff = mu*N
    Fd = 0.5*rho_air*Cd*A*dx**2.

    # Output derivative vector
    dX = np.zeros(2,)
    dX[0] = dx
    
    if rolling_flag:
        dX[1] = (Fw - Fd)/(m + (J/r**2.))
    else:
        dX[1] = (Fw - Ff - Fd)/m
    
    
    return dX



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


if __name__ == '__main__':
    
    t = compute_ramp_time()
    
    print('\nTime to reach bottom of ramp [sec]: {:0.4f}'.format(t))

