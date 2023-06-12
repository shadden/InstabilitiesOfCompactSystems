import celmech as cm
from celmech.miscellaneous import getOmegaMatrix
import rebound as rb
import numpy as np
import sympy as sp
import time
import sys
sys.path.append("code/")
from DisturbingFunctionSplittingMethod import H0soln,H1soln


# Generate a simulation of equal mass, evenly spaced planets.
def get_sim(m,Npl,pratio,exfrac):
    alpha = pratio**(-2/3)
    ex = np.min((1/alpha-1,1-alpha)) # orbit-crossing eccentricity
    ecc = exfrac * ex
    sim = rb.Simulation()
    sim.add(m=1)
    P = 1
    for i in range(Npl):
        sim.add(m=m,P=P,l=np.random.uniform(-np.pi,np.pi),e=ecc,pomega= np.random.uniform(-np.pi,np.pi))
        P*= pratio
    sim.move_to_com()
    return sim



sim = get_sim(1e-5,5,1.15354,0.2) # system of five planets

pvars = cm.Poincare.from_Simulation(sim)
pham = cm.PoincareHamiltonian(pvars)
Hkep = pham.H.copy()

pham_inits = pham.state.values

jlo,jhi = np.floor(1 + 1/(1.3456-1)),np.ceil(1 + 1/(1.3456-1))
jlo,jhi = int(jlo),int(jhi)


pham.add_MMR_terms(jlo,1,max_order=2,indexIn=1,indexOut=2,inclinations=False)
pham.add_MMR_terms(jlo,1,max_order=2,indexIn=2,indexOut=3,inclinations=False)
pham.add_MMR_terms(jlo,1,max_order=2,indexIn=3,indexOut=4,inclinations=False)
pham.add_MMR_terms(jlo,1,max_order=2,indexIn=4,indexOut=5,inclinations=False)

pham.add_MMR_terms(jhi,1,max_order=2,indexIn=1,indexOut=2,inclinations=False)
pham.add_MMR_terms(jhi,1,max_order=2,indexIn=2,indexOut=3,inclinations=False)
pham.add_MMR_terms(jhi,1,max_order=2,indexIn=3,indexOut=4,inclinations=False)
pham.add_MMR_terms(jhi,1,max_order=2,indexIn=4,indexOut=5,inclinations=False)

pham.add_MMR_terms(jlo+jhi,2,max_order=2,indexIn=1,indexOut=2,inclinations=False)
pham.add_MMR_terms(jlo+jhi,2,max_order=2,indexIn=2,indexOut=3,inclinations=False)
pham.add_MMR_terms(jlo+jhi,2,max_order=2,indexIn=3,indexOut=4,inclinations=False)
pham.add_MMR_terms(jlo+jhi,2,max_order=2,indexIn=4,indexOut=5,inclinations=False)

from celmech.canonical_transformations import reduce_hamiltonian
# Reduce to eliminate dependence on inclination variables...
Hpert = reduce_hamiltonian(pham)
# ...and subrtract Keplerian piece to create a Hamiltonian that is just the perturbation terms
Hpert.H += -1 * Hkep

Npl = 5

jac=Hpert.calculate_jacobian()

# indicies of lambda,Lambda, and x variables
l_indx = [2*j for j in range(Npl)]
L_indx = [2*Npl + 2*j for j in range(Npl)]
evar_indx = [1 + 2*j for j in range(Npl)] + [2*Npl+ 1 + 2*j for j in range(Npl)]

# Atilde matrix
Atilde = np.array([[jac[i,j] for j in evar_indx] for i in evar_indx])

# variable symbols
evarsymbols = [Hpert.qp_vars[j] for j in evar_indx]
L_symbols = [Hpert.qp_vars[j] for j in L_indx]
l_symbols = [Hpert.qp_vars[j] for j in l_indx]

# set ecc variables to zero to get btilde vector
ezero_rule={s:0 for s in evarsymbols}
btilde = np.array([Hpert.N_flow[i].xreplace(ezero_rule).subs(Hpert.qp) for i in evar_indx],dtype = float)

# Initial values
x0 = np.array([Hpert.state.values[i] for i in evar_indx])
l0 = np.array([Hpert.state.values[i] for i in l_indx])
L0 = np.array([Hpert.state.values[i] for i in L_indx])


# Omega matrix
OmegaN = getOmegaMatrix(5)

# get A as a function of lambdas
Amtrx = sp.Matrix(2*Npl,2*Npl,lambda i,j: sp.diff(Hpert.N_H,evarsymbols[i],evarsymbols[j]))
Afn = sp.lambdify(l_symbols,Amtrx)

# Get b as a function of lambdas
bvec = sp.Matrix([sp.diff(Hpert.N_H,evar).xreplace(ezero_rule) for evar in evarsymbols])
bfn = sp.lambdify(l_symbols,bvec)

# Derivatives of A and b as a function of lambda
grad_A = []
grad_b = sp.Matrix(Npl,2*Npl,lambda i,j: sp.diff(bvec[j],l_symbols[i]))
for l_symbol in l_symbols:
    grad_A.append(sp.diff(Amtrx,l_symbol))
grad_Afn = sp.lambdify(l_symbols,grad_A)
grad_bfn = sp.lambdify(l_symbols,grad_b)


h = 0.25 # time step
Niter = 100 # number of iterations
GMmvec = pvars.G * pvars.G * np.array([p.M**2 * p.mu**3 for p in pvars.particles[1:]])

# initial values
x,l,L = x0.copy(),l0.copy(),L0.copy()

print("start integration")
st = time.time()

# main loop
for i in range(Niter):
    x,l,L = H0soln(x,l,L,0.5*h,GMmvec)
    x,l,L = H1soln(x,l,L,h,Afn,bfn,grad_Afn,grad_bfn,getOmegaMatrix(5),5)
    x,l,L = H0soln(x,l,L,0.5*h,GMmvec)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# Start 0.18658876419067383s 
