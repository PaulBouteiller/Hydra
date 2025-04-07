"""
Created on Thu Mar 10 16:17:24 2022

@author: bouteillerp
Paramètres par défaut
"""

def default_fem_degree():
    """
    Degré d'interpolation par défaut du champ de déplacement
    """
    return 2

def default_fem_parameters():
    """
    Degré d'interpolation par défaut des champs.
    """
    fem={}
    fem.update({"u_degree" : default_fem_degree()})
    fem.update({"schema" : "default"})
    return fem

def default_Riemann_solver_parameters():
    """
    Type de flux numérique à utiliser. Options:
    - "Cockburn": Flux upwind simple avec pénalisation
    - "Rusanov": Flux de type Rusanov (Local Lax-Friedrichs)
    - "HLL": Approximation de Harten-Lax-van Leer
    - "HLLC": HLL avec restauration de l'onde de contact
    """
    riemann_solver = {}
    riemann_solver.update({"flux_type": "HLL"})
    riemann_solver.update({"signal_speed_type" : "davis"})
    return riemann_solver

def default_Newton_parameters():
    solver_u = {}
    solver_u.update({"ksp_type": "preonly"})
    solver_u.update({"pc_type": "lu"})
    solver_u.update({"pc_factor_mat_solver_type" : "mumps"})
    solver_u.update({"relative_tolerance" : 1e-8})
    solver_u.update({"absolute_tolerance" : 1e-8})
    solver_u.update({"convergence_criterion" : "incremental"})
    solver_u.update({"maximum_iterations" : 50})
    solver_u.update({"pc_factor_mat_solver_type": "mumps",
            # Activation de BLR (Block Low-Rank) pour MUMPS
            "mat_mumps_icntl_35": 1,
            "mat_mumps_cntl_7": 1e-8  # Tolérance BLR
        })
    structure_type = "block"
    debug = False
    
    return solver_u, structure_type, debug

def default_shock_capturing_parameters():
    shock_sensor = {}
    shock_sensor.update({"use_shock_capturing" : False})
    shock_sensor.update({"shock_sensor_type" : "ducros"})
    shock_sensor.update({"shock_threshold" : 0.95})
    return shock_sensor

def default_viscosity_parameters():
    viscosity = {}
    viscosity.update({"coefficient" : 1e-2})
    return viscosity
    

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