"""
Created on Thu Mar 10 16:17:24 2022

@author: bouteillerp
Paramètres par défaut
"""

def default_fem_degree():
    """
    Degré d'interpolation par défaut du champ de déplacement
    """
    return 1

def default_fem_parameters():
    """
    Degré d'interpolation par défaut des champs.
    """
    fem={}
    fem.update({"u_degree" : default_fem_degree()})
    fem.update({"schema" : "default"})
    return fem

def default_damping_parameters():
    """
    Paramètres par défaut de la pseudo-viscosité.
    """
    damp = {}
    damp.update({"damping" : True})
    damp.update({"linear_coeff" : 0.1})
    damp.update({"quad_coeff" : 0.1})
    damp.update({"correction" : True})
    return damp

def default_Newton_parameters():
    solver_u = {}
    solver_u.update({"ksp_type": "preonly"})
    solver_u.update({"pc_type": "lu"})
    solver_u.update({"pc_factor_mat_solver_type" : "mumps"})
    solver_u.update({"relative_tolerance" : 1e-8})
    solver_u.update({"absolute_tolerance" : 1e-8})
    solver_u.update({"convergence_criterion" : "incremental"})
    solver_u.update({"maximum_iterations" : 50})
    return solver_u

def default_post_processing_parameters():
    post_processing = {}
    # post_processing.update({"writer" : "xdmf"})
    post_processing.update({"writer" : "VTK"})    
    if post_processing["writer"] == "xdmf":
        post_processing.update({"file_results": "results.xdmf"})
    elif post_processing["writer"] == "VTK":
        post_processing.update({"file_results": "results.pvd"})
    post_processing.update({"file_log": "log.txt"})
    return post_processing