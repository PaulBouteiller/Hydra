"""
Created on Wed Jan  8 18:07:35 2025

@author: bouteillerp
"""
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from ufl.algorithms.analysis import extract_arguments
from ufl import Form, derivative
from itertools import product
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

class SeparateSpaceFormSplitter(MultiFunction):

    def split(self, form, v, u=None):
        self.vu = tuple((v, u))
        return map_integrand_dags(self, form)

    def argument(self, obj):
        if obj not in self.vu:
            return Zero(shape=obj.ufl_shape)
        return obj

    expr = MultiFunction.reuse_if_untouched


def extract_rows(F, v):
    """
    Parameters
    ----------
    F : UFL residual formulation
    v : Ordered list of test functions

    Returns
    -------
    List of extracted block residuals ordered corresponding to each test
    function
    """
    vn = len(v)
    L = [None for _ in range(vn)]

    fs = SeparateSpaceFormSplitter()

    for vi in range(vn):
        # Do the initial split replacing testfunctions with zero
        L[vi] = fs.split(F, v[vi])

        # Now remove the empty forms. Why don't FFC/UFL do this already?
        L_reconstruct = []
        for integral in L[vi].integrals():
            arguments = extract_arguments(integral)
            if len(arguments) < 1:
                continue

            # Sanity checks: Should be only one test function, and it should
            # be the one we want to keep
            assert len(arguments) == 1
            assert arguments[0] is v[vi]
            L_reconstruct.append(integral)

        # Generate the new form with the removed zeroes
        L[vi] = Form(L_reconstruct)
    return L

def derivative_block(F, u, du):
    """
    Parameters
    ----------
    F : Block residual formulation
    u : Ordered solution functions
    du : Ordered trial functions
    coefficient_derivatives : Prescribed derivative map

    Returns
    -------
    Block matrix corresponding to the ordered components of the
    Gateaux/Frechet derivative.
    """
    m, n = len(u), len(F)
    J = [[None for _ in range(m)] for _ in range(n)]

    for (i, j) in product(range(n), range(m)):
        gateaux_derivative = derivative(F[i], u[j], du[j])
        gateaux_derivative = apply_derivatives(
            apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J
