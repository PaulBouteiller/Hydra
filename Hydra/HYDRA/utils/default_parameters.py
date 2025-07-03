"""
Created on Thu Mar 10 16:17:24 2022

@author: bouteillerp
"""
"""
Default parameter settings for HYDRA simulations
===============================================

This module defines the default parameters for various components of the HYDRA simulation
framework. It provides consistent initialization values and configuration options for
numerical methods, physical models, and solver settings.

The default parameters cover:
- Finite element discretization (degree, scheme)
- Numerical flux methods for Riemann solvers
- Newton solver configuration
- Shock capturing methods
- Artificial viscosity settings
- Post-processing and visualization options

Functions:
----------
default_fem_degree() : Default polynomial degree for displacement field
    Returns the recommended polynomial degree for accurate solutions

default_fem_parameters() : Default finite element parameters
    Returns a dictionary with FEM configuration including degree and scheme

default_Riemann_solver_parameters() : Default Riemann solver configuration
    Returns a dictionary with flux type and signal speed estimation method

default_Newton_parameters() : Default Newton solver settings
    Returns a tuple with PETSc options, matrix structure type, and debug flag

default_shock_capturing_parameters() : Default shock capturing settings
    Returns a dictionary with shock sensor type and threshold values

default_viscosity_parameters() : Default artificial viscosity settings
    Returns a dictionary with viscosity coefficients for stabilization

default_post_processing_parameters() : Default export and visualization settings
    Returns a dictionary with output file formats and naming conventions
"""


def default_fem_degree():
    """
    Get the default polynomial degree for unknown fields.
    
    Returns
    -------
    int Default polynomial degree (2)
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
    Get the default Riemann solver parameters.
    
    Configures the numerical flux type for interfaces between elements.
    
    Returns
    -------
    dict
        Dictionary containing Riemann solver configuration:
        - flux_type: Type of numerical flux
        - signal_speed_type: Method for estimating wave speeds
        
    Notes
    -----
    Available flux types:
    - "Cockburn": Simple upwind flux with penalization
    - "Rusanov": Local Lax-Friedrichs flux
    - "HLL": Harten-Lax-van Leer approximation
    - "HLLC": HLL with restored contact wave
    """
    riemann_solver = {}
    riemann_solver.update({"flux_type": "HLL"})
    riemann_solver.update({"signal_speed_type" : "davis"})
    return riemann_solver

def default_Newton_parameters():
    """
    Get the default Newton solver parameters.
    
    Configures the nonlinear solver settings, including linear solver
    options, convergence criteria, and matrix structure.
    
    Returns
    -------
    tuple
        (solver_options, structure_type, debug)
        - solver_options: Dictionary of PETSc solver options
        - structure_type: Matrix structure ("block" or "nest")
        - debug: Whether to enable debug mode
    """
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
    debug = True
    
    return solver_u, structure_type, debug

def default_shock_capturing_parameters():
    """
    Get the default shock capturing parameters.
    
    Configures shock detection and stabilization settings for
    handling discontinuities in the solution.
    
    Returns
    -------
    dict
        Dictionary containing shock capturing configuration:
        - use_shock_capturing: Whether to enable shock capturing
        - shock_sensor_type: Method for detecting shocks
        - shock_threshold: Threshold value for shock detection
    """
    shock_sensor = {}
    shock_sensor.update({"use_shock_capturing" : False})
    shock_sensor.update({"shock_sensor_type" : "ducros"})
    shock_sensor.update({"shock_threshold" : 0.95})
    return shock_sensor

def default_viscosity_parameters():
    """
    Get the default artificial viscosity parameters.
    
    Configures the artificial viscosity used for stabilization
    near discontinuities.
    
    Returns
    -------
    dict
        Dictionary containing viscosity configuration:
        - coefficient: Scaling factor for artificial viscosity
    """
    viscosity = {}
    viscosity.update({"coefficient" : 1e-2})
    return viscosity
    

def default_post_processing_parameters():
    """
    Get the default post-processing parameters.
    
    Configures result export and visualization settings.
    
    Returns
    -------
    dict
        Dictionary containing post-processing configuration:
        - writer: Output format ("xdmf" or "VTK")
        - file_results: Results file name
        - file_log: Log file name
    """
    post_processing = {}
    # post_processing.update({"writer" : "xdmf"})
    post_processing.update({"writer" : "VTK"})    
    if post_processing["writer"] == "xdmf":
        post_processing.update({"file_results": "results.xdmf"})
    elif post_processing["writer"] == "VTK":
        post_processing.update({"file_results": "results.pvd"})
    post_processing.update({"file_log": "log.txt"})
    return post_processing