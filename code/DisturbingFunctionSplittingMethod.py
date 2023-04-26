# DisturbingFunctionSplittingMethod

import numpy as np

# Solve evolution under Keplerian Hamiltonian alone
# i.e., lambda -> lambda + n * dt
def H0soln(x0,l0,L0,h,GMmvec):
    Linv =1/L0
    Linv_cubed = Linv*Linv*Linv
    dl = h * Linv_cubed * GMmvec
    return x0,l0+dl,L0

# Solve evolution under perturbation Hamiltonian, H1, alone
def H1soln(x0,l0,L0,h,Afn,bfn,grad_Afn,grad_bfn,Omega,Npl):
    A = Afn(*l0)
    b = bfn(*l0)
    Atilde = Omega @ A
    btilde = Omega @ b
    assert np.alltrue(np.isclose(Atilde,Atilde.T))
    # diagonalize Atilde
    eigs,T = np.linalg.eigh(Atilde)
    D = np.diag(eigs)
    Dinv = np.diag(1/eigs)
    Atilde_inv = T @ Dinv @ T.T
    
    # forced eccentricity
    xf = -Atilde_inv @ btilde
    xf = xf.reshape(-1)
    
    # initial u
    u0 = T.T @ (x0 - xf)
    
    # final u
    uh = np.exp(eigs * h) * u0
    
    # x at time h
    xh = xf + T @ uh
    
    # integral of u from 0 to h
    Uh = (np.exp(h * eigs)  - 1)  * u0 / eigs
    # integral of u_i * u_j from 0 to h
    fn = lambda si,sj: h if np.isclose(si,-sj) else (np.exp((si + sj)*h) - 1) / (si + sj)
    UUh = np.outer(u0,u0)  * np.array([[fn(si,sj) for si in eigs] for sj in eigs])
    # integral of x from 0 to h
    Xh = xf * h + T @ Uh
    
    # integral of x_i * x_j from 0 to h
    XXh = h * np.outer(xf,xf)
    XXh += np.outer(xf,T @ Uh) + np.outer(T @ Uh, xf)
    XXh += T @ UUh @ T.T
    
    # Lambda solution
    dL = np.zeros(Npl)
    grad_A = grad_Afn(*l0)
    grad_b = grad_bfn(*l0)
    
    for i in range(Npl):
        dL[i] -= grad_b[i] @ Xh
        grad_Ai = grad_A[i]
        dL[i] -= 0.5 * np.sum([grad_Ai[j,k] * XXh[j,k] for j in range(2*Npl) for k in range(2*Npl)])
    return xh,l0,L0 + dL