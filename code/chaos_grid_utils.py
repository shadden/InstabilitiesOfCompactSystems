import numpy as np
import rebound as rb
from celmech.nbody_simulation_utilities import set_time_step

def get_sim(m1,m2,j1,j2,y1,y2):
    sim = rb.Simulation()
    sim.add(m=1)
    Pin = (j1-1)/j1
    Pout = j2/(j2-1)
    ex1 = Pin**(-2/3) - 1
    ex2 = 1 - Pout**(-2/3)
    sim.add(m=m1,P = Pin,l=np.pi/2, e = y1*ex1, pomega = 0.5*np.pi)
    sim.add(m=0,P=1,l=0,e = 0.)
    sim.add(m=m2,P = Pout,l=3*np.pi/2,e = y2 * ex2, pomega = 3*np.pi/2)
    sim.move_to_com()
    Rhill = np.max([p.rhill for p in sim.particles[1:]])
    sim.exit_min_distance = 5*Rhill 
    sim.integrator='whfast'
    set_time_step(sim,0.05)
    sim.init_megno()
    return sim

def run_sim(sim,Tmax,Ymax,Ncheck):
    times  = np.linspace(0,Tmax)
    for t in times:
        try:
            sim.integrate(t)
            Y = sim.megno()
            if Y>Ymax:
                break
        except rb.Encounter:
            return -10
    return Y
