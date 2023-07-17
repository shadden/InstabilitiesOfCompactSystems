import numpy as np
import celmech as cm
import rebound as rb
import sympy as sp

import sys
sys.path.append("../code/")
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
pratio = 4/3
sim = get_sim(1e-5,5,pratio,0.2) # system of five planets

pvars = cm.Poincare.from_Simulation(sim)
pham = cm.PoincareHamiltonian(pvars)
Hkep = pham.H.copy()

pham_inits = pham.state.values

jlo,jhi = np.floor(1 + 1/(pratio-1)),np.ceil(1 + 1/(pratio-1))
jlo,jhi = int(jlo),int(jhi)



Npl = pham.N - 1
for i in range(1,Npl):
    pham.add_MMR_terms(jlo,1,max_order = 2 , indexIn=i, indexOut=i+1,inclinations=False)
    

lmbda_vec = [pham.qp_vars[3*i] for i in range(Npl)]

T = 2 * sp.Matrix([
    [1-jlo,jlo,0,0,0],
    [0,1-jlo,jlo,0,0],
    [0,0,1-jlo,jlo,0],
    [0,0,0,1-jlo,jlo],
    [1,-1,0,0,0]
])
angvec = [*sp.symbols("phi(1:5)")] + [sp.symbols("psi")]
angrule = dict(zip(lmbda_vec,T.inv() * sp.Matrix(angvec)))

xvec = [pham.qp_vars[3*i+1] for i in range(Npl)] + [pham.qp_vars[Npl * 3 + 3*i+1] for i in range(Npl)]
xvec

phi2=sp.symbols("phi2",real=True)
A_mtrx=sp.Matrix(2*Npl,2*Npl, lambda i,j:sp.diff(pham.N_H,xvec[i],xvec[j]))
A_mtrx = A_mtrx.xreplace(angrule)

zero_rule = dict(zip(xvec,np.zeros(2*Npl)))
b_vec = [sp.diff(pham.N_H,xvec[i]).xreplace(zero_rule).xreplace(angrule) for i in range(2*Npl)]


from ResonantInteraction import EccentricityResonanceInteraction

interactions = []
for i in range(1,Npl):
    interaction = EccentricityResonanceInteraction(pvars,i,i+1,jlo * 2)
    interactions.append(interaction)


np.random.seed(123)
phi_vals = np.random.uniform(-np.pi,np.pi,size=4)


A = np.zeros((2*Npl,2*Npl))
b = np.zeros(2*Npl)
for i,interaction in enumerate(interactions):
    j=i+1
    phi = phi_vals[i]
    s,c = np.sin(phi/2),np.cos(phi/2)
    s2 = 2 * s * c
    c2 = c*c - s*s
    Aij = interaction.A(c2,s2)
    bij = interaction.b(c,s)
    b[[i,j,i+Npl,j+Npl]] += bij
    A[i:j+1,i:j+1] += Aij[:2,:2]
    A[i+Npl:j+Npl+1,i:j+1] += Aij[2:,:2]
    A[i:j+1,i+Npl:j+Npl+1] += Aij[:2,2:]
    A[i+Npl:j+Npl+1,i+Npl:j+Npl+1] += Aij[2:,2:]



Nangle_rule = dict(zip(angvec,phi_vals))
NA_mtrx = A_mtrx.xreplace(Nangle_rule)
NA_mtrx = np.array(NA_mtrx,dtype = float)


print(np.alltrue(np.isclose(NA_mtrx,A)))

Nb_vec = sp.Matrix(b_vec).xreplace(Nangle_rule)
Nb_vec = np.array(Nb_vec,dtype=float).reshape(-1)

print(np.alltrue(np.isclose(b,Nb_vec)))
