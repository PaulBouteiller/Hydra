"""
Created on Wed Jan  8 18:07:35 2025

@author: bouteillerp
"""
"""
Block manipulation utilities for UFL forms and matrices
======================================================

This module provides utilities for manipulating UFL forms and matrices in a
block-structured manner. It includes functions for extracting and organizing
rows and derivatives of UFL forms, which is particularly useful for implementing
block-structured solvers for coupled PDEs.

The key functionalities include:
- Extraction of individual equation rows from a coupled system
- Computation of Jacobian matrices in block form
- Manipulation of UFL expressions with multifunction pattern

These utilities are essential for implementing solvers that exploit the block
structure of the discretized PDEs, such as block preconditioners and specialized
Newton methods for coupled systems.

Classes:
--------
SeparateSpaceFormSplitter : UFL multifunction for form splitting
    Implements pattern matching for UFL expressions
    Separates forms based on test function spaces

Functions:
----------
extract_rows(F, v) : Extract individual rows from a coupled system
    Splits the residual form F into separate forms for each test function in v
    Returns a list of forms corresponding to each equation

derivative_block(F, u, du) : Compute Jacobian blocks for a coupled system
    Calculates the Gateaux derivative of each row with respect to each solution component
    Returns a 2D list of forms representing the Jacobian blocks
------
These utilities operate on UFL forms before compilation, allowing for symbolic
manipulation of the forms. This enables specialized handling of block structures
in the resulting linear systems.
"""
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from ufl.algorithms.analysis import extract_arguments
from ufl import Form
import itertools
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl import derivative


class SeparateSpaceFormSplitter(MultiFunction):
    """
    UFL multifunction for splitting forms based on test function spaces.
    
    This class implements the pattern matching mechanism needed to
    separate a mixed variational form into individual forms for
    each test function space.
    """
    def split(self, form, v, u = None):
        """
        Split a UFL form based on a test function.
        
        Extracts the part of the form containing the specified test function,
        replacing other test functions with zeros.
        
        Parameters
        ----------
        form : UFL form Form to split
        v : TestFunction Test function to keep in the split form
        u : TrialFunction, optional Trial function (unused in current implementation)
            
        Returns
        -------
        UFL form The part of the form containing only the specified test function
        """
        self.vu = tuple((v, u))
        return map_integrand_dags(self, form)

    def argument(self, obj):
        """
        Handle UFL arguments (test/trial functions) during form splitting.
        
        Replaces arguments that don't match the target test function
        with zero expressions.
        
        Parameters
        ----------
        obj : UFL argument Test or trial function to check
            
        Returns
        -------
        UFL expression The original argument if it matches the target, zero otherwise
        """
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
    Compute block-wise derivatives of a variational form.
    
    Calculates the Gateaux derivative of each row of the residual
    with respect to each solution component, creating a block
    Jacobian structure.
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

    for (i, j) in itertools.product(range(n), range(m)):
        gateaux_derivative = derivative(F[i], u[j], du[j])
        gateaux_derivative = apply_derivatives(
            apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J
