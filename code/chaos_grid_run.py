from multiprocess import Pool
import sys
import numpy as np
from chaos_grid_utils import get_sim,run_sim


lims = list(zip(np.linspace(4,8,11)[:-1],np.linspace(4,8,11)[1:]))
Jlims = [(*j1,*j2) for j2 in lims for j1 in lims]
I = int(sys.argv[1])
J1low,J1hi,J2low,J2hi = Jlims[I]
Ngrid = int(sys.argv[2])

file = "./grid_{}".format(I)

j0 = 4
m0 = 0.5e-5
y = 0.25

Tmax = 5.e4
Ymax = 15.
Ncheck = 20

J1range = np.linspace(J1low,J1hi,Ngrid)
J2range = np.linspace(J2low,J2hi,Ngrid)

def run_func(j1_and_j2):
    j1,j2 = j1_and_j2
    m1 = m0 * (j0/j1)**4
    m2 = m0 * (j0/j2)**4 
    sim = get_sim(m1,m2,j1,j2,y,y)
    Y = run_sim(sim,Tmax,Ymax,Ncheck)
    return (j1,j2,Y)


with Pool() as pool:
    parameters = []
    for j2 in J2range:
        for j1 in J1range:
            parameters.append((j1,j2))
    results = pool.map(run_func,parameters)

results_arr = np.array(results).reshape(Ngrid,Ngrid,-1)
np.save(file,results_arr)