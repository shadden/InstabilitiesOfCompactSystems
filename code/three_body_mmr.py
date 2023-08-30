import rebound as rb
import celmech as cm
import sympy as sp
import numpy as np
from celmech.poisson_series_manipulate import PoissonSeries, PSTerm, DFTerm_as_PSterms
from celmech.poisson_series_manipulate import PoissonSeries_to_GeneratingFunctionSeries
from celmech.miscellaneous import get_symbol
from celmech.poisson_series_manipulate import get_N_planet_poisson_series_symbols
from celmech.disturbing_function import list_resonance_terms

def term_order(k,nu):
    return np.sum(np.abs(k[2:])) + 2 * np.sum(nu)

def pair_mmr_lists(i1,i2,i3,jvec,sgn,inner_mmrs,outer_mmrs,max_order):
    """
    Pair combinations of disturbing function arguemnts that give rise to
    three-body MMR terms up to a given maximum order

    Parameters
    ----------
    i1 : int
        innermost planet index
    i2 : int
        middle planet index
    i3 : int
        outermost planet index
    jvec: ndarray
        1d array of coefficients multiplying desired three-body MMR
    sgn : int
        Integer ±1 depending on whether inner and outer MMR arguments
        are added or subtracted.
    inner_mmrs : list
        List of two-planet MMR terms, (k,nu), for the inner pair
    outer_mmrs : list
        List of two-planet MMR terms, (k,nu), for the outer pair
    max_order : int
        Maximum order of terms, in eccentricity and inclination,
        to retain in the three-body combinations of terms.  

    Returns
    -------
    list
        List of parings of individual terms in the format used by
        :func:`get_three_body_mmr_terms`
    """
    pairs = []
    l=(0,0)
    for k_inner,nu_inner in inner_mmrs:
        q_inner = np.array((0,*k_inner[:2]))
        order_inner = term_order(k_inner,nu_inner)
        res_inner = (i1,i2,k_inner,nu_inner,l)
        for k_outer,nu_outer in outer_mmrs:
            q_outer = np.array((*k_outer[:2],0))
            q_vec = q_outer + sgn * q_inner
            order_outer = term_order(k_outer,nu_outer)
            if order_inner + order_outer <= max_order+2:
                res_outer = (i2,i3,k_outer,nu_outer,l)

                pairs.append((res_inner,res_outer))
    return pairs

def get_three_body_mmr_terms(pham,res_pairs_list):
    # Frequency vector
    omega = pham.flow_func(*pham.state.values)[:pham.N_dof:3]
    omega = omega.reshape(-1)

    # Vector of freq. derivs, dn_i/dLambda_i
    domega = np.diag(pham.jacobian_func(*pham.state.values)[:pham.N_dof:3,pham.N_dof::3])

    # Initialize Poisson series to hold all the terms resulting from three-body interactions
    Npl  = pham.N-1
    symbol_kwargs = get_N_planet_poisson_series_symbols(Npl)
    h2_three_body_series = PoissonSeries(2*Npl,Npl,**symbol_kwargs)

    for res_in,res_out in res_pairs_list:
        i1_in,i2_in,k_in,nu_in,l_in = res_in
        i1_out,i2_out,k_out,nu_out,l_out = res_out
        terms_in  = DFTerm_as_PSterms(pham,i1_in,i2_in,k_in,nu_in,l_in)
        terms_out = DFTerm_as_PSterms(pham,i1_out,i2_out,k_out,nu_out,l_out)
        h1_in  = PoissonSeries.from_PSTerms(terms_in)
        h1_out = PoissonSeries.from_PSTerms(terms_out) 
        chi1_in  = PoissonSeries_to_GeneratingFunctionSeries(h1_in,omega,domega)
        chi1_out = PoissonSeries_to_GeneratingFunctionSeries(h1_out,omega,domega)

        # Add cross-terms to three-body hamiltonian series
        h2_three_body_series += 0.5 * (chi1_in.Lie_deriv(h1_out) + chi1_out.Lie_deriv(h1_in))

    return h2_three_body_series