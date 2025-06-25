"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
"""

"""
Base problem definition framework for fluid dynamics simulations
==============================================================

This module provides the foundational infrastructure for defining and setting up
fluid dynamics problems in the HYDRA framework. It implements the common components
needed for all simulation types, including mesh management, function spaces,
measures, boundary conditions, and variational form construction.

The module consists of three main classes:
- BoundaryConditions: Manages boundary condition application
- MeshManager: Handles mesh creation, facet identification, and integration measures
- Problem: Base class for all fluid dynamics problems

Together, these classes provide a unified approach to problem definition, ensuring
consistent treatment across different equation types and numerical methods.

Classes:
--------
BoundaryConditions : Base class for boundary condition management
    Handles Dirichlet boundary condition application
    Provides interface for boundary residual terms
    Supports time-dependent boundary values

MeshManager : Mesh and integration domain handler
    Creates and manages the main mesh and facet submesh
    Identifies and marks boundaries based on coordinates
    Sets up integration measures for HDG formulation

Problem : Abstract base class for all fluid dynamics problems
    Initializes the problem components (mesh, spaces, variables)
    Sets up the EOS and material properties
    Manages the overall variational formulation construction
    Provides interface for initial and boundary conditions
"""

from ..ConstitutiveLaw.eos import EOS
from ..utils.default_parameters import default_fem_parameters, default_shock_capturing_parameters
from ..utils.MyExpression import MyConstant

from numpy import (hstack, argsort, finfo, full_like, arange, int32, unique, 
                    tile, repeat, vstack, full, zeros, array)

from dolfinx.fem import (compute_integration_domains, dirichletbc, locate_dofs_topological,
                          IntegralType, Constant, Expression, Function, functionspace)
from dolfinx.mesh import meshtags, create_submesh, locate_entities
from ufl import (Measure, inner, FacetNormal)

from mpi4py.MPI import COMM_WORLD
from dolfinx.cpp.mesh import cell_num_entities
from petsc4py.PETSc import ScalarType


from dolfinx.log import set_log_level
from dolfinx.cpp.log import LogLevel
set_log_level(LogLevel.WARNING)

class BoundaryConditions:
    """
    Base class for managing boundary conditions in HDG formulations.
    
    This class handles the application of boundary conditions in the
    Hybridizable Discontinuous Galerkin method, including:
    - Dirichlet boundary condition imposition
    - Boundary residual construction
    - Time-dependent boundary value management
    """
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag):
        """
        Initialize the boundary condition manager.
        
        Parameters
        ----------
        U : list of Functions Interior solution variables
        Ubar : list of Functions Facet solution variables
        Ubar_test : list of TestFunctions Facet test functions
        facet_mesh : Mesh Facet submesh
        S : EOS or Expression Stabilization parameter or EOS
        n : Expression Facet normal vector
        ds : Measure Integration measure for boundaries
        tdim : int Topological dimension
        entity_maps : dict Entity mappings between meshes
        facet_tag : MeshTags Tags for facets
        """
        self.U, self.Ubar, self.Ubar_test = U, Ubar, Ubar_test
        self.EOS = EOS
        self.ds = ds
        self.n = n
        self.boundary_residual = Constant(facet_mesh, ScalarType(0)) * Ubar_test[0] * ds
        self.mcl = []
        # self.S = S
        
        self.tdim = tdim
        self.facet_mesh = facet_mesh
        self.entity_maps = entity_maps
        self.facet_tag = facet_tag
        self.fdim = tdim - 1
        self.bcs = []   

    def add_component(self, space, isub, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition for a component.
        
        Parameters
        ----------
        space : FunctionSpace Function space for the boundary condition
        isub : int or None Subspace index (None for scalar fields)
        region : int Tag of the boundary region
        value : float, Constant, or Expression, optional Value to impose (default: 0)
    
        Notes
        -----
        This method handles the mapping between the facet mesh and the
        original mesh to impose conditions on the correct entities.
        """
        def current_space(space, isub):
            if isub is None:
                return space
            else:
                return space.sub(isub)
            
        def bc_value(value):
            if isinstance(value, float) or isinstance(value, Constant):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.constant
        msh_to_facet_mesh = self.entity_maps[self.facet_mesh]
        facets = msh_to_facet_mesh[self.facet_tag.indices[self.facet_tag.values == region]]
        
        dofs = locate_dofs_topological(current_space(space, isub), self.fdim, facets)
        self.bcs.append(dirichletbc(bc_value(value), dofs, current_space(space, isub)))
    
    def pressure_residual(self, value, tag):
        """
        Apply pressure boundary condition.
        
        Implements a pressure boundary condition using the HDG formulation,
        which enforces the prescribed pressure through the numerical flux.
        
        Parameters
        ----------
        value : float, Constant, or MyConstant Pressure value to impose
        tag : int Boundary tag for the pressure boundary
            
        Notes
        -----
        TODO: This is not exactly a pressure boundary condition, it should
        penalize the difference between the numerical flux and a Neumann flux.
        See equation (2.14) in the HDG reference."""
        if isinstance(value, MyConstant):
            p = value.Expression.constant
            self.mcl.append(value.Expression)
        else:
            p = value
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = -inner(p * self.n + self.S * (self.U[1]/self.U[0] - self.Ubar[1]/self.Ubar[0]), 
                       self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E
        
    def set_time_dependant_BCs(self, load_steps):
        """
        Set up time-dependent boundary condition values.
        
        Precomputes boundary values for all time steps to enable
        efficient time-dependent boundary conditions.
        
        Parameters
        ----------
        load_steps : array Time points for the simulation
        """
        for constant in self.mcl:
            constant.set_time_dependant_array(load_steps)
            
class MeshManager:
    """
    Manager for mesh operations and integration domains.
    
    This class handles all mesh-related operations, including:
    - Facet submesh creation for HDG
    - Boundary marking
    - Integration measure setup
    - Entity mapping between meshes
    """
    def __init__(self, mesh):
        """
        Initialize the mesh manager.
        
        Parameters
        ----------
        mesh : Mesh Main computational mesh
        """
        self.mesh = mesh
        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        self.n = FacetNormal(mesh)
        self.h = self.calculate_mesh_size()
        
        # Créer le sous-maillage des facettes et les mappages d'entités
        self.facet_mesh, self.entity_maps = self.create_facet_mesh()

    def create_facet_mesh(self):
        """
        Create a submesh containing all facets.
        
        Constructs a submesh of all facets from the main mesh and
        establishes mappings between entity indices in both meshes.
        
        Returns
        -------
        tuple (facet_mesh, entity_maps)
                - facet_mesh: Submesh of facets
                - entity_maps: Dictionary mapping entities between meshes
        """
        self.mesh.topology.create_entities(self.fdim)
        facet_imap = self.mesh.topology.index_map(self.fdim)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        facets = arange(num_facets, dtype=int32)
        facet_mesh, facet_mesh_to_msh, _, _ = create_submesh(self.mesh, self.fdim, facets)
        facet_mesh.topology.create_connectivity(self.fdim, self.fdim)
        
        # Créer les mappages d'entités
        msh_to_facet_mesh = full(num_facets, -1)
        msh_to_facet_mesh[facet_mesh_to_msh] = arange(len(facet_mesh_to_msh))
        entity_maps = {facet_mesh: msh_to_facet_mesh}
        
        return facet_mesh, entity_maps
    
    def mark_boundary(self, flag_list, coord_list, localisation_list, tol=finfo(float).eps):
        """
        Mark boundaries based on geometric criteria.
        
        Identifies and tags mesh boundaries based on their spatial coordinates,
        associating each with a numeric identifier for boundary condition application.
        
        Parameters
        ----------
        flag_list : list of int Numeric identifiers for the boundaries
        coord_list : list of str Coordinate directions ('x', 'y', 'z') for boundary detection
        localisation_list : list of float Coordinate values for boundary detection
        tol : float, optional Tolerance for coordinate matching
            
        Notes
        -----
        This method creates mesh tags that can be used to identify boundary
        regions when imposing boundary conditions or computing boundary integrals.
        """
        facets = []
        full_flag = []
        flags_found = []  # Pour suivre les flags qui ont effectivement trouvé des facettes
        
        for flag, coord, loc in zip(flag_list, coord_list, localisation_list):
            def boundary_condition(x):
                def index_coord(coord):
                    """Renvoie l'index correspondant à la variable spatiale"""
                    if coord == "x":
                        return 0
                    elif coord == "y":
                        return 1
                    elif coord == "z":
                        return 2
                    else:
                        raise ValueError("Standard coordinates must be chosen")
                return abs(x[index_coord(coord)] - loc) < tol
            
            bdry_facets = locate_entities(self.facet_mesh, self.fdim, boundary_condition)
            
            # Vérifier si des facettes ont été trouvées pour cette condition
            if len(bdry_facets) > 0:
                facets.append(bdry_facets)
                full_flag.append(full_like(bdry_facets, flag))
                flags_found.append(flag)
            else:
                print(f"Attention: Aucune facette trouvée pour flag={flag}, coord={coord}, loc={loc}")
        
        if len(facets) > 0:
            marked_facets = hstack(facets)
            marked_values = hstack(full_flag)
            sorted_facets = argsort(marked_facets)
            self.flag_list = unique(flags_found)  # Utiliser uniquement les flags qui ont trouvé des facettes
            
            # Créer les tags pour les facettes
            self.facet_tag = meshtags(self.facet_mesh, self.facet_mesh.topology.dim, 
                                      marked_facets[sorted_facets], 
                                      marked_values[sorted_facets])
            
            # Correspondance avec le maillage principal
            self.facet_to_mesh_tag = {}
            for tag in self.flag_list:
                tagged_facets = self.facet_tag.indices[self.facet_tag.values == tag]
                # Convertir en indices dans le maillage principal
                original_facets = array([self.entity_maps[self.facet_mesh][f] for f in tagged_facets])
                self.facet_to_mesh_tag[tag] = original_facets
        else:
            print("Attention: Aucune facette n'a été marquée pour aucune condition aux limites.")
            self.flag_list = []
            self.facet_to_mesh_tag = {}
    
    def compute_cell_boundary_facets(self):
        """
        Compute facets for integration around cell boundaries.
        
        Generates the integration entities needed for HDG formulation,
        identifying all facets around each cell.
        
        Returns
        -------
        numpy.ndarray Integration entities as (cell, local facet) pairs
        """
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local   
        # Crée toutes les paires (cellule, facette locale) pour toutes les cellules
        return vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
    
    def set_measures(self, deg):
        """
        Set up integration measures for HDG formulation.
        
        Creates the volume and surface integration measures required for
        the HDG formulation, with appropriate quadrature degrees and
        domain tags.
        
        Parameters
        ----------
        deg : int Polynomial degree of the approximation
            
        Notes
        -----
        This method defines dx_c for volume integrals, ds_c for external
        boundary integrals, and ds_tot for all facet integrals.
        """
        # Calculer toutes les facettes de cellules
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local
        all_facets = vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
        
        # Tags pour les différents types de frontières
        cell_boundaries_tag = 0  # Toutes les frontières
        
        # Initialiser les entités d'intégration
        facet_integration_entities = [(cell_boundaries_tag, all_facets)]
        
        # Ajouter les frontières externes marquées
        if hasattr(self, 'facet_to_mesh_tag'):
            for tag in self.flag_list:
                # Utiliser la correspondance précalculée
                entities = compute_integration_domains(
                    IntegralType.exterior_facet, self.mesh.topology, 
                    self.facet_to_mesh_tag[tag], self.fdim)
                facet_integration_entities.append((tag, entities))
        
        # Degrés de quadrature
        # quad_deg_facet = (deg + 1) ** (self.tdim - 1)
        quad_deg_facet = (2 * deg + 1) ** (self.tdim)
        quad_deg_volume = (deg + 1) ** self.tdim
        print(f"Nombre de points de Gauss: {quad_deg_volume}")
        
        # Définir les mesures
        self.dx_c = Measure("dx", domain=self.mesh, metadata={"quadrature_degree": quad_deg_volume})
        self.ds_c = Measure("ds", subdomain_data=facet_integration_entities,
                            domain=self.mesh, metadata={"quadrature_degree": quad_deg_facet})
        
        # Mesure spécifique pour toutes les facettes
        self.ds_tot = self.ds_c(cell_boundaries_tag)
        
    def calculate_mesh_size(self):
        """
        Calculate local mesh size.
        
        Computes the characteristic size of each element in the mesh,
        which is used for artificial viscosity scaling and error estimation.
        
        Returns
        -------
        Function Function containing the local mesh size for each element
        """
        h_loc = Function(functionspace(self.mesh, ("DG", 0)), name="MeshSize")
        num_cells = self.mesh.topology.index_map(self.tdim).size_local
        h_local = zeros(num_cells)
        for i in range(num_cells):
            h_local[i] = self.mesh.h(self.tdim, array([i]))
        
        h_loc.x.array[:] = h_local
        return h_loc

class Problem:
    """
    Base class for all fluid dynamics problems.
    
    This abstract class provides the framework for defining and solving
    fluid dynamics problems with the HDG method. It manages:
    - Mesh and function space setup
    - Material properties and equation of state
    - Boundary condition application
    - Variational formulation construction
    - Initial condition setting
    
    Derived classes implement specific equation types (Euler, Navier-Stokes)
    by extending and specializing this base framework.
    """
    def __init__(self, material, dt, initial_mesh = None, **kwargs):
        """
        Initialize the problem.
        
        Parameters
        ----------
        material : Material Material properties object
        dt : float Time step size
        initial_mesh : Mesh, optional Initial mesh (if None, define_mesh() will be called)
        **kwargs : dict Additional configuration parameters:
                            - analysis: "dynamic" or "static"
                            - isotherm: Whether to use isothermal model
        """
        if initial_mesh == None:
            self.mesh = self.define_mesh()
        else:
            self.mesh = initial_mesh
        if COMM_WORLD.Get_size()>1:
            print("Parallel computation")
            self.mpi_bool = True
        else:
            print("Serial computation")
            self.mpi_bool = False
        self.analysis = kwargs.get("analysis", "dynamic")
        self.iso_T = kwargs.get("isotherm", False)
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1
        self.dt = dt
        self.dt_factor = Constant(self.mesh, ScalarType(1.0 / self.dt))
        
        # Gestionnaire de maillage
        self.mesh_manager = MeshManager(self.mesh)
        self.facet_mesh = self.mesh_manager.facet_mesh
        self.entity_maps = self.mesh_manager.entity_maps
        self.n = self.mesh_manager.n
        self.h = self.mesh_manager.h

        # Set parameters and update from user
        self.fem_parameters()
        
        # Material properties
        self.material = material

        # MeshFunctions and Measures for different domains and boundaries
        self.set_boundary()
        self.set_measures()
        self.facet_tag = self.mesh_manager.facet_tag
        self.flag_list = self.mesh_manager.flag_list
        
        #Set function space and unknwown functions 
        self.set_function_space()
        self.set_functions()
        self.set_variable_to_solve()
        self.set_test_functions()
        self.shock_param = default_shock_capturing_parameters()
        self.shock_stabilization = self.shock_param.get("use_shock_capturing") and self.deg>0
        if self.shock_stabilization:
            self.set_stabilization_parameters(self.shock_param.get("shock_sensor_type"))

        # Constitutive Law
        self.EOS = EOS(None, None)
        # self.set_artifial_pressure()
        self.set_auxiliary_field()
        
        #Boundary conditions
        self.bc_class = self.boundary_conditions_class()(
                        self.U, self.Ubar, self.Ubar_test,
                        self.facet_mesh, self.EOS, self.n, self.ds_c,
                        self.tdim, self.entity_maps, self.facet_tag, self.dico_Vbar
                    )
        self.bcs = []
        self.set_boundary_conditions()
        
        # Set up variational formulation
        print("Starting setting up variational formulation")
        self.set_form()
        
    def set_output(self):
        return {}
    
    def query_output(self, t):
        return {}
    
    def final_output(self):
        pass
    
    def csv_output(self):
        return {}
    
    def prefix(self):
        return "problem"

    def set_measures(self):
        """
        Set up integration measures for the problem.
        
        Uses the mesh manager to create appropriate integration measures
        for the variational formulation.
        """
        self.mesh_manager.set_measures(self.deg)
        self.dx_c = self.mesh_manager.dx_c
        self.ds_c = self.mesh_manager.ds_c
        self.ds_tot = self.mesh_manager.ds_tot
            
    def set_auxiliary_field(self):
        """
        Initialize auxiliary fields for the problem.
        
        Sets up derived fields like pressure that are computed from
        the primary solution variables.
        """
        p_elem = self.EOS.set_eos(self.U, self.material)
        self.p_expr = Expression(p_elem, self.V_rho.element.interpolation_points())
        self.p_func = Function(self.V_rho, name = "Pression")      
        
    def fem_parameters(self):
        """
        Configure finite element parameters.
        
        Sets up polynomial degree and other FEM-specific configuration
        based on default settings or user overrides.
        """
        fem_parameters = default_fem_parameters()
        self.deg = fem_parameters.get("u_degree")
        
    def set_initial_conditions(self):
        pass
    
    def update_bcs(self, num_pas):
        pass
    
    def set_stabilization_parameters(self, shock_sensor_type):
        """
        Initialize shock capturing stabilization.
        
        Sets up the shock sensor for discontinuity detection and
        artificial viscosity application.
        
        Parameters
        ----------
        shock_sensor_type : str Type of shock sensor to use ("ducros" or "fernandez")
        """
        if shock_sensor_type == "ducros":
            from ..VariationalFormulation.shock_sensor import DucrosShockSensor
            self.shock_sensor = DucrosShockSensor(self)
        elif shock_sensor_type == "fernandez":
            from ..VariationalFormulation.shock_sensor import FernandezShockSensor
            self.shock_sensor = FernandezShockSensor(self)