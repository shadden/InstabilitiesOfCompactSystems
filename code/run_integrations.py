import rebound as rb
import numpy as np
import celmech as cm
from celmech.lie_transformations import FirstOrderGeneratingFunction


def planar_cross_Q(alpha,e1,e2,delta_pomega):
    l = alpha**2 * (1-e1**2)     + (1 - e2**2) -  2 * alpha * (1 - e1 * e2 * np.cos(delta_pomega))
    return l<=0

def crossQ(pham):
    ps = pham.state.particles
    for i in range(1,pham.N-1):
        p1,p2 = ps[i],ps[i+1]
        if planar_cross_Q(p1.a/p2.a,p1.e,p2.e,p1.pomega-p2.pomega):
            return True
    return False

P_to_J = lambda P: 1 + 1/(P-1)
J_to_P = lambda J: J/(J-1)

def sim_setup(m,J,y):
    P = J_to_P(J)
    sim = rb.Simulation()
    sim.units = ('Msun','yr','AU')
    sim.add(m=1)
    periods = P**np.arange(3)
    alpha = P**(-2/3)
    ex = (P**(2/3)-1)/(P**(2/3)+1)
    e = y * ex
    for p in periods:
        sim.add(m=m,P=p,e = e,pomega = 'uniform',l='uniform')
    sim.move_to_com()

    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    jlo,jhi = int(np.floor(J)),int(np.floor(J)) + 1
    for i in (1,2):
        pham.add_MMR_terms(jlo,1,max_order=1,indexIn=i,indexOut=i+1)
        pham.add_MMR_terms(jlo + jhi,2,max_order=2,indexIn=i,indexOut=i+1,inclinations=False)
        pham.add_MMR_terms(jhi,1,max_order=1,indexIn=i,indexOut=i+1)
    return pham
if __name__=="__main__":

    import sys
    from scipy.integrate import solve_ivp

    I = int(sys.argv[1])
    J = np.linspace(7,8,40,endpoint=False)[I]
    pham = sim_setup(3e-6,J,0.25)
    rt_amd = np.sqrt(np.sum([p.Gamma for p in pham.particles[1:]]))
    pham.set_integrator('DOP853',rtol=1e-5,atol = 1e-5 * rt_amd)

    # integration
    Ntimes = 8 * 512
    tmax = 1e6
    times = np.linspace(0,tmax,Ntimes)
    
    # radau method
    f= lambda t,y: pham.flow_func(*y).reshape(-1)
    Df= lambda t,y: pham.jacobian_func(*y)

    rt_amd = np.sqrt(np.sum([p.Gamma for p in pham.particles[1:]]))
    soln = solve_ivp(f,(0,tmax),pham.state.values,t_eval=times,jac = Df, method="Radau", rtol=1e-6,atol=1e-6 * rt_amd)

    # values = np.zeros((Ntimes,pham.N_dim))
    # for i,t in enumerate(times):
    #     pham.integrate(t)
    #     values[i] = pham.state.values
    #     if crossQ(pham):
    #         tmax = t
    #         break
    
    values = np.zeros((Ntimes,pham.N_dim))
    E = np.zeros(Ntimes)
    amd = np.zeros(Ntimes)
    ecc = np.zeros((3,Ntimes))
    P = np.zeros((3,Ntimes))
    for i,val in enumerate(soln.y.T):
        values[i] = val
        pham.state.values=val
        E[i] = pham.calculate_energy()
        amd[i] = np.sum([p.Gamma for p in pham.particles[1:]])
        for j,p in enumerate(pham.particles[1:]):
            ecc[j,i] = p.e
            P[j,i] = p.P
        
    
    pham.state.values = values[0]
    pvars0 = pham.state.copy()
    chi = FirstOrderGeneratingFunction(pvars0)

    jlo,jhi = int(np.floor(J)),int(np.floor(J)) + 1
    for i in (1,2):
        chi.add_MMR_terms(jlo,1,max_order=1,indexIn=i,indexOut=i+1)
        chi.add_MMR_terms(jlo + jhi,2,max_order=2,indexIn=i,indexOut=i+1,inclinations=False)
        chi.add_MMR_terms(jhi,1,max_order=1,indexIn=i,indexOut=i+1)
    values_mean = np.array([chi.osculating_to_mean_state_vector(v) for v in values])
    ecc_mean = np.zeros((3,Ntimes))
    P_mean = np.zeros((3,Ntimes))
    amd_mean=np.zeros(Ntimes)
    Lambda_mean = np.zeros((3,Ntimes))
    lambda_mean = np.zeros((3,Ntimes))
    for i,val in enumerate(values_mean):
        pham.state.values=val
        amd_mean[i] = np.sum([p.Gamma for p in pham.particles[1:]])
        for j,p in enumerate(pham.particles[1:]):
            ecc_mean[j,i] = p.e
            P_mean[j,i] = p.P
            Lambda_mean[j,i] = p.Lambda
            lambda_mean[j,i] = p.l
    
    np.savez_compressed(
        "RADAU_chaotic_traj_J0_{:.3f}".format(J),
        times = times,
        P = P,
        ecc = ecc,
        P_mean = P_mean,
        e_mean = ecc_mean,
        amd_mean = amd_mean, 
        amd=amd,
        values = values,
        values_mean = values_mean
    )