import numpy as np
import scipy.stats
from scipy.integrate import dblquad


def compute_metrics(scenario_id):
    
    # Retrieve scenario data
    v_A0, v_B0, r_A0, r_B0, P_A, P_B, HBR = scenario_data(scenario_id)
    
    # Compute time of closest approach
    TCA = compute_TCA_simple(v_A0, v_B0, r_A0, r_B0)
    print('TCA', TCA)
    
    # Propagate states to TCA assuming linear dynamics
    r_A = r_A0 + v_A0*TCA
    r_B = r_B0 + v_B0*TCA
    
    # Compute metrics
    d2 = compute_euclidean_distance(r_A, r_B)
    d_M = compute_mahalanobis_distance(r_A, r_B, P_A, P_B)
    Pc_analytic = analytic_Pc_combined(r_A, r_B, P_A, P_B, TCA, HBR)
    Pc_MC = montecarlo_Pc_combined(r_A, r_B, P_A, P_B, TCA, HBR)
    
    
    # # These yield same result but are much slower
    # Pc_init = montecarlo_Pc_initial(v_A0, v_B0, r_A0, r_B0, P_A, P_B, TCA, HBR)
    # Pc_final = montecarlo_Pc_final(r_A, r_B, P_A, P_B, TCA, HBR)
    
    
    print('Euclidean miss distance [m]:', d2)
    print('Mahalanobis distance:', d_M)    
    print('Analytic Pc:', Pc_analytic)
    print('Monte Carlo Pc (combined):', Pc_MC)
    # print('Monte Carlo Pc (initial samples)', Pc_init)
    # print('Monte Carlo Pc (final samples)', Pc_final)
    
    return


def compute_TCA_simple(v_A0, v_B0, r_A0, r_B0):
    
    # Time for A to reach intersection
    dt_A = float(r_A0[0,0])/float(-v_A0[0,0])
    
    # Time for B to reach intersection
    dt_B = float(r_B0[1,0])/float(-v_B0[1,0])
    
    # Choose minimum for TCA
    TCA = min(dt_A, dt_B)    
    
    return TCA



def compute_euclidean_distance(r_A, r_B):
    
    d = np.linalg.norm(r_A - r_B)
    
    return d


def compute_mahalanobis_distance(r_A, r_B, P_A, P_B):    
    
    Psum = P_A + P_B
    invP = np.linalg.inv(Psum)
    diff = r_A - r_B
    M = float(np.sqrt(np.dot(diff.T, np.dot(invP, diff)))[0,0])
    
    return M


def montecarlo_Pc_initial(v_A0, v_B0, r_A0, r_B0, P_A, P_B, TCA, HBR):
    
    # Generate samples at initial time
    N = 10000
    samples_A0 = np.random.default_rng().multivariate_normal(r_A0.flatten(), P_A, N).T
    samples_B0 = np.random.default_rng().multivariate_normal(r_B0.flatten(), P_B, N).T

    # Propagate samples to TCA
    samples_A = samples_A0 + v_A0*TCA
    samples_B = samples_B0 + v_B0*TCA
    
    # Compute Pc
    hits = 0.
    total = 0.
    for ii in range(N):
        r_A = samples_A[:,ii].reshape(2,1)
        for jj in range(N):
            r_B = samples_B[:,ii].reshape(2,1)
            
            d = compute_euclidean_distance(r_A, r_B)
            total += 1.
            if d <= HBR:
                hits += 1.
    
    Pc = hits/total
    
    return Pc


def montecarlo_Pc_final(r_A, r_B, P_A, P_B, TCA, HBR):
    
    # Generate samples at initial time
    N = 10000
    samples_A = np.random.default_rng().multivariate_normal(r_A.flatten(), P_A, N).T
    samples_B = np.random.default_rng().multivariate_normal(r_B.flatten(), P_B, N).T

    # Compute Pc
    hits = 0.
    total = 0.
    for ii in range(N):
        r_A = samples_A[:,ii].reshape(2,1)
        for jj in range(N):
            r_B = samples_B[:,ii].reshape(2,1)
            
            d = compute_euclidean_distance(r_A, r_B)
            total += 1.
            if d <= HBR:
                hits += 1.
    
    Pc = hits/total
    
    return Pc


def montecarlo_Pc_combined(r_A, r_B, P_A, P_B, TCA, HBR):
    
    # Location of combined hardbody (center of object not at origin)
    r_HB = r_A
    
    # Compute combined P
    P = P_A + P_B
    
    # Draw samples from combined distribution
    N = 5000000
    mean = np.array([0., 0.])
    samples = np.random.default_rng().multivariate_normal(mean, P, N).T
    
    # Compute Pc
    hits = 0.
    total = 0.
    for ii in range(N):
        ri = samples[:,ii].reshape(2,1)
        d = compute_euclidean_distance(ri, r_HB)
        total += 1.
        if d <= HBR:
            hits += 1.
            
    Pc = hits/total
    
    
    return Pc


def analytic_Pc_combined(r_A, r_B, P_A, P_B, TCA, HBR):
    
    # Location of combined hardbody (center of object not at origin)
    # Note that this will set x0 to positive when actual scenario r_A is 
    # negative, but the distribution is symmetric so it doesn't affect result
    x0 = np.linalg.norm(r_A - r_B)
    y0 = 0.
    
    # Compute combined P
    P = P_A + P_B
    Pinv = np.linalg.inv(P)
    Pdet = np.linalg.det(P)
    
    # Set up quadrature    
    atol = 1e-13
    rtol = 1e-8
    Integrand = lambda y, x: np.exp(-0.5*(Pinv[0,0]*x**2. + Pinv[0,1]*x*y + Pinv[1,0]*x*y + Pinv[1,1]*y**2.))
    
    lower_semicircle = lambda x: -np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
    upper_semicircle = lambda x:  np.sqrt(HBR**2. - (x-x0)**2.)*(abs(x-x0)<=HBR)
    Pc = (1./(2.*np.pi))*(1./np.sqrt(Pdet))*float(dblquad(Integrand, x0-HBR, x0+HBR, lower_semicircle, upper_semicircle, epsabs=atol, epsrel=rtol)[0])
    
    
    
    return Pc



def scenario_data(scenario_id):
    
    if scenario_id == 1:
        
        v_A0 = np.array([[8.], [ 0.]])
        v_B0 = np.array([[0.], [10.]])
        
        r_A0 = np.array([[-20.], [  0.]])
        r_B0 = np.array([[  0.], [-10.]])
        
        P_A = np.array([[64., 0.], [0., 36.]])
        P_B = np.array([[36., 0.], [0., 64.]])
        
        HBR = 2.
        
    if scenario_id == 2:
        
        v_A0 = np.array([[19.], [ 0.]])
        v_B0 = np.array([[ 0.], [10.]])
        
        r_A0 = np.array([[-200.], [   0.]])
        r_B0 = np.array([[   0.], [-100.]])
        
        P_A = np.array([[6400., 0.], [0., 3600.]])
        P_B = np.array([[3600., 0.], [0., 6400.]])
        
        HBR = 2.
        
    if scenario_id == 3:
        
        v_A0 = np.array([[15.], [ 0.]])
        v_B0 = np.array([[ 0.], [10.]])
        
        r_A0 = np.array([[-20.], [  0.]])
        r_B0 = np.array([[  0.], [-10.]])
        
        P_A = np.array([[0.64, 0.], [0., 0.36]])
        P_B = np.array([[0.36, 0.], [0., 0.64]])
        
        HBR = 2.
    
    
    return v_A0, v_B0, r_A0, r_B0, P_A, P_B, HBR





if __name__ == '__main__':
    
    
    for scenario_id in range(1,4):
    
        print('')
        print('Scenario', scenario_id)
        compute_metrics(scenario_id)





