import numpy as np
from celmech.disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict
class EccentricityResonanceInteraction(object):
    """
    A class for representing first- and second-order eccentricity resonance interactions between pairs of planets.
    """
    def __init__(self,pvars,indexIn,indexOut ,kres,Lambda0In = None, Lambda0Out = None):
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
        
        self.C2_mtrx = np.zeros((2,2))
        self.C1_vec = np.zeros(2)

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
                self.C2_mtrx[0,0] = - 2 * Gmm_a2 * NC / self.Lambda0s[0]
            if l==1:
                self.C2_mtrx[1,0] = self.C2_mtrx[0,1] = - Gmm_a2 * NC / np.sqrt(self.Lambda0s[0] * self.Lambda0s[1])
            if l==2:
                self.C2_mtrx[1,1] = - 2 * Gmm_a2 * NC / self.Lambda0s[1]
            
        self.C2_mtrx_inv = np.linalg.inv(self.C2_mtrx)

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
        
        return np.block(
            [
               [ -cos_phi2 * self.C2_mtrx, -sin_phi2 * self.C2_mtrx],
               [ -sin_phi2 * self.C2_mtrx,  cos_phi2 * self.C2_mtrx],
            ]
        )
    
    def b(self,sin_phi1,cos_phi1):
        C1vec = self.C1_vec
        return np.concatenate((-1 * C1vec * sin_phi1, C1vec * cos_phi1 ))
    
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
        return Ainv