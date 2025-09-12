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

def create_custom_quadrature_element(ratio, equal_weight=True, el_type = "Scalar"):
    """ 
    Cette procédure est fondée sur l'article:
     Stable high-order quadrature rules with equidistant points   Daan Huybrechs ∗ ,1
    """ 
    liste = []
    n_points = ratio**2
    for i in range(ratio):
        for j in range(ratio):
            liste.append([(i + 1./2) * 1 / ratio, (j + 1./2) * 1 / ratio])
    points = np.array(liste)
    
    if equal_weight:
        # Poids uniformes
        weights = np.ones(n_points) * (1. / n_points)
    else:
        # Poids NNLS optimisés
        monomials = []
        for i in range(ratio):
            for j in range(ratio - i):
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
        degree = int(ratio)-1
    )
    if el_type == "Scalar":
        return element
    elif el_type == "Vector":
        vector_element = blocked_element(element, shape=(2,))
        
        return vector_element
    

def L2_proj(u, source_term, bcs = []):
    V = u.function_space
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a = ufl.inner(u_trial, v_test) * ufl.dx
    L = ufl.inner(source_term, v_test) * ufl.dx

    return petsc.LinearProblem(a, L, u=u, bcs=bcs, 
                               petsc_options = {"ksp_type": "preonly", "pc_type": "lu"})


def reordering_mapper(arr1, arr2, rtol=1e-5, atol=1e-8):
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


def map_indices(arr1, arr2, inactive_axes=None, rtol=1e-5, atol=1e-8):
    """
    Mappe chaque point de arr1 vers tous les points correspondants de arr2
    en ne comparant que les axes actifs.
    
    Returns: liste plate des indices de arr2 correspondant à chaque point de arr1
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if inactive_axes is None:
        active_indices = list(range(len(arr1[0])))
    else:
        if isinstance(inactive_axes, str):
            inactive_axes = [inactive_axes]
        inactive_idx = {axis_map[ax] for ax in inactive_axes if ax in axis_map}
        active_indices = [i for i in range(len(arr1[0])) if i not in inactive_idx]
    result = []
    source_indices = []
    for i, point1 in enumerate(arr1):
        coords1 = point1[active_indices]
        for j, point2 in enumerate(arr2):
            coords2 = point2[active_indices]
            if np.allclose(coords1, coords2, rtol=rtol, atol=atol):
                result.append(j)
                source_indices.append(i)  # Ajouter cette ligne
    
    return np.array(result), np.array(source_indices)

def create_vector_mapping(scalar_mapping, source_indices, dim=2):
    """
    Crée les mappings vectoriels à partir du mapping scalaire.
    
    Args:
        scalar_mapping: indices de mapping pour les scalaires
        source_indices: indices d'origine pour chaque correspondance
        dim: dimension du vecteur (2 pour 2D, 3 pour 3D)
    
    Returns:
        Dictionnaire avec les mappings pour chaque composante
    """
    # Mapping pour v_x: indices * 2
    vx_hydra_indices = scalar_mapping * 2
    
    # Mapping pour v_y: indices * 2 + 1
    vy_hydra_indices = scalar_mapping * 2 + 1
    
    mappings = {
        'vx': vx_hydra_indices,
        'vy': vy_hydra_indices,
        'source_indices': source_indices
    }
    
    if dim == 3:
        # Pour 3D, on aurait besoin d'une logique différente selon votre organisation
        # Par exemple: vz_hydra_indices = scalar_mapping * 3 + 2
        vz_hydra_indices = scalar_mapping * 2 + 2  # Adapter selon votre structure
        mappings['vz'] = vz_hydra_indices
    
    return mappings

def create_averaging_mapper(coords, tol=1e-8):
    """
    NB sur np.unique: Returns the sorted unique elements of an array. There are three optional outputs in addition to the unique elements:

    the indices of the input array that give the unique values

    the indices of the unique array that reconstruct the input array

    the number of times each unique value comes up in the input array

    """
    rounded_coords = np.round(coords / tol) * tol
    unique_coords, inverse_indices, counts = np.unique(
        rounded_coords, axis=0, 
        return_inverse=True, 
        return_counts=True
    )
    return {
        'unique_coords': unique_coords,
        'inverse_indices': inverse_indices,
        'counts': counts
    }

def apply_averaging(values, mapper):
    return np.bincount(mapper['inverse_indices'], weights=values) / mapper['counts']
    
def project_cell_to_facet(cell_values, a_cell, a_facet, unique_cell_to_unique_facet_mapper, rtol=1e-5, atol=1e-8):
    """Projette les valeurs cellules vers facettes"""
    averaged_values = apply_averaging(cell_values, a_cell)
    reordered_averaged = averaged_values[unique_cell_to_unique_facet_mapper]
    return reordered_averaged[a_facet['inverse_indices']]


def apply_averaging_vector(values, mapper, dim):
    """Moyennage vectoriel sans reshape"""
    n_unique = len(mapper['unique_coords'])
    result = np.zeros(n_unique * dim)
    for d in range(dim):
        avg_component = np.bincount(mapper['inverse_indices'], weights=values[d::dim]) / mapper['counts']
        result[d::dim] = avg_component
    
    return result

def project_cell_to_facet_vector(cell_values, a_cell, a_facet, unique_cell_to_unique_facet_mapper, dim=2):
    """Projette les valeurs vectorielles cellules vers facettes"""
    averaged_values = apply_averaging_vector(cell_values, a_cell, dim)
    reordered_averaged = averaged_values[unique_cell_to_unique_facet_mapper]
    vector_facet_indices = np.repeat(a_facet['inverse_indices'], dim) * dim + np.tile(range(dim), len(a_facet['inverse_indices']))
    
    a = reordered_averaged[vector_facet_indices]
    return a