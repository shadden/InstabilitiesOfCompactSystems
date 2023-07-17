import numpy as np
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict

from ctypes import cdll, c_int, c_double, Structure, POINTER, byref
clibH1soln = cdll.LoadLibrary("../code/H1soln.so")

class EccentricityResonanceInteraction(Structure):
    _fields_ = [
            ("order", c_int),
            ("C2_mtrx", POINTER(c_double)),
            ("C1_vec", POINTER(c_double)),
            ("b_vec", POINTER(c_double)),
            ]
    """
    A class for representing first- and second-order eccentricity resonance interactions between pairs of planets.
    """
    def __init__(self,pvars,indexIn,indexOut ,kres,Lambda0In = None, Lambda0Out = None):

        clibH1soln.init_interaction(byref(self))

        if kres%2==0:
            self.order = 1
        else:
            self.order = 2

        G = pvars.G
        pIn = pvars.particles[indexIn]
        pOut = pvars.particles[indexOut]
        if not Lambda0In:
            Lambda0In = pIn.Lambda        
        if not Lambda0Out:
            Lambda0Out = pOut.Lambda
        self.Lambda0s = (Lambda0In,Lambda0Out)
        mIn,muIn,MIn = pIn.m,pIn.mu,pIn.M
        mOut,muOut,MOut = pOut.m,pOut.mu,pOut.M
        a20 = (Lambda0Out / muOut)**2 / G / MOut
        Gmm_a2 = G * mIn * mOut / a20

        nu_vec = (0,0,0,0)
        alpha = (Lambda0In/Lambda0Out)**2 * (muOut/muIn)**2 * (MOut/MIn)
        

        if self.order==1:
            k1 = kres // 2
            for l in range(2):
                k_vec = (k1,1-k1,l-1,-l,0,0)
                C = df_coefficient_C(*k_vec,*nu_vec)
                NC = evaluate_df_coefficient_dict(C,alpha)
                self.C1_vec[l] = - Gmm_a2 * NC / np.sqrt(self.Lambda0s[l])

        for l in range(3):
            k_vec = (kres,2-kres,l-2,-l,0,0)
            C = df_coefficient_C(*k_vec,*nu_vec)
            NC = evaluate_df_coefficient_dict(C,alpha)
            if l==0:
                self.C2_mtrx[0*2+0] = - 2 * Gmm_a2 * NC / self.Lambda0s[0]
            if l==1:
                self.C2_mtrx[1*2+0] = self.C2_mtrx[0*2+1] = - Gmm_a2 * NC / np.sqrt(self.Lambda0s[0] * self.Lambda0s[1])
            if l==2:
                self.C2_mtrx[1*2+1] = - 2 * Gmm_a2 * NC / self.Lambda0s[1]
            
        # TODO can be optimized
        np_C2_mtrx = np.array([  [self.C2_mtrx[0*2+0],self.C2_mtrx[0*2+1]],
                                 [self.C2_mtrx[1*2+0],self.C2_mtrx[1*2+1]] ])
        self.C2_mtrx_inv = np.linalg.inv(np_C2_mtrx)

    def A(self,cos_phi2,sin_phi2):
        r"""
        Compute resonance contribution to the matrix terms

        .. math::
            \begin{pmatrix}
                A_{i,i}     & A_{i,j}   & A_{i,i+n}     & A_{i,j+n} \\
                A_{j,i}     & A_{j,j}   & A_{j,i+n}     & A_{j,j+n} \\
                A_{i+n,i}   & A_{i+n,j} & A_{i+n,i+n}   & A_{i+n,j+n} \\
                A_{j+n,i}   & A_{j+n,j} & A_{j+n,i+n}   & A_{j+n,j+n} 
            \end{pmatrix}
        
        Arguments
        ---------
        cos_phi2 : float
            Cosine of the angle :math:`k\lambda_j + (2-k)\lambda_i`
        sin_phi2 : float
            Sine of the angle :math:`k\lambda_j + (2-k)\lambda_i`
        
        Returns
        -------
        A : ndarray
            Matrix of DF coefficent values
        """
        
        # TODO can be optimized
        np_C2_mtrx = np.array([  [self.C2_mtrx[0*2+0],self.C2_mtrx[0*2+1]],
                                 [self.C2_mtrx[1*2+0],self.C2_mtrx[1*2+1]] ])
        
        return np.block(
            [
               [ -cos_phi2 * np_C2_mtrx, -sin_phi2 * np_C2_mtrx],
               [ -sin_phi2 * np_C2_mtrx,  cos_phi2 * np_C2_mtrx],
            ]
        )
    
    def b(self,cos_phi1,sin_phi1):
        clibH1soln.b(byref(self), c_double(cos_phi1), c_double(sin_phi1))
        return np.ctypeslib.as_array(self.b_vec, (4,))
    
    def Ainv(self,cos_phi2,sin_phi2):
        r"""
        Compute inverse of the A matrix (see EccentricityResonanceInteraction.A)
        
        Arguments
        ---------
        cos_phi2 : float
            Cosine of the angle :math:`k\lambda_j + (2-k)\lambda_i`
        sin_phi2 : float
            Sine of the angle :math:`k\lambda_j + (2-k)\lambda_i`
        
        Returns
        -------
        Ainv : ndarray
        """
        C2inv = self.C2_mtrx_inv
        Ainv =  np.block(
            [
                [-cos_phi2 * C2inv, -sin_phi2 * C2inv],
                [-sin_phi2 * C2inv,  cos_phi2 * C2inv],
            ]
        )
        print(Ainv)
        return Ainv
