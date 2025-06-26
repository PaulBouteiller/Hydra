#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:20:26 2025

@author: bouteillerp

"""

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from dolfinx.fem import petsc
import basix
from scipy.optimize import nnls
from basix.ufl import quadrature_element, blocked_element

def create_custom_quadrature_element(degree, equal_weight=True, el_type = "Scalar"):
    """ 
    Cette procédure est fondée sur l'article:
     Stable high-order quadrature rules with equidistant points   Daan Huybrechs ∗ ,1
    """ 
    liste = []
    n_sub_element = (degree + 1)
    n_points = (degree + 1)**2
    for i in range(degree + 1):
        for j in range(degree + 1):
            liste.append([(i + 1./2) * 1 / n_sub_element, (j + 1./2) * 1 / n_sub_element])
    points = np.array(liste)
    
    if equal_weight:
        # Poids uniformes
        weights = np.ones(n_points) * (1. / n_points)
    else:
        # Poids NNLS optimisés
        monomials = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                monomials.append((i, j))
        
        A = np.zeros((len(monomials), n_points))
        for k, (p, q) in enumerate(monomials):
            for i, pt in enumerate(points):
                A[k, i] = (pt[0]**p) * (pt[1]**q)
        
        b = np.array([1.0/((p+1)*(q+1)) for p, q in monomials])
        weights, _ = nnls(A, b)
        weights = weights / np.sum(weights)
    
    element = quadrature_element(
        cell="quadrilateral",
        points=points,
        weights=weights,
        degree = int(degree)
    )
    if el_type == "Scalar":
        return element
    elif el_type == "Vector":
        vector_element = blocked_element(element, shape=(2,))
        
        return vector_element

def init_L2_projection_on_V(u, source_term, bcs = []):
    V = u.function_space
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a = ufl.inner(u_trial, v_test) * ufl.dx
    L = ufl.inner(source_term, v_test) * ufl.dx

    return petsc.LinearProblem(a, L, u=u, bcs=bcs, 
                               petsc_options = {"ksp_type": "preonly", "pc_type": "lu"})


def map_indices(arr1, arr2, rtol=1e-5, atol=1e-8):
    """
    Mappe les indices de arr1 vers arr2 basé sur l'égalité des éléments avec tolérance.
    
    Args:
        arr1, arr2: numpy arrays contenant des numpy arrays
        rtol: tolérance relative (défaut: 1e-5)
        atol: tolérance absolue (défaut: 1e-8)
    
    Returns:
        numpy array où result[i] = j tel que arr1[i] ≈ arr2[j]
    """
    def arrays_close(a, b, rtol, atol):
        try:
            return np.allclose(a, b, rtol=rtol, atol=atol)
        except:
            # Fallback sur array_equal si allclose échoue
            return np.array_equal(a, b)
    assert len(arr1) == len(arr2)
    map_list = [np.where([arrays_close(arr1[i], x, rtol, atol) for x in arr2])[0][0] for i in range(len(arr1))]
    return np.array(map_list)

# def quadrature_space_jax_array_to_mapping(Q, jax_array):
#     quad_array = Q.tabulate_dof_coordinates()
#     return map_indices(quad_array, np.asarray(jax_array))