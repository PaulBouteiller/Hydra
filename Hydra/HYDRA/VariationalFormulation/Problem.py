"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
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
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag):
        self.U, self.Ubar, self.Ubar_test = U, Ubar, Ubar_test
        self.EOS = EOS
        self.ds = ds
        self.n = n
        self.boundary_residual = Constant(facet_mesh, ScalarType(0)) * Ubar_test[0] * ds
        self.mcl = []
        
        self.tdim = tdim
        self.facet_mesh = facet_mesh
        self.entity_maps = entity_maps
        self.facet_tag = facet_tag
        self.fdim = tdim - 1
        self.bcs = []   

    def add_component(self, space, isub, region, value=ScalarType(0)):
        """
        Ajoute une condition aux limites de Dirichlet sur une composante
        
        Parameters
        ----------
        space : functionspace Espace fonctionnel pour la condition limite
        isub : int ou None Indice du sous-espace
        region : int Tag de la région pour la condition limite
        value : float, Constant ou expression, optional
            Valeur de la condition limite. Default is ScalarType(0)
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
        #ToDo en fait ce n'est pas ça du tout qu'il faut faire, il faut pénaliser la 
        # différence entre le flux numérique et un flux de Neumann voir équation (2.14) du poly HDG
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
        Définition de la liste donnant l'évolution temporelle du chargement.
        Parameters
        ----------
        load_steps : List, liste des pas de temps.
        """
        for constant in self.mcl:
            constant.set_time_dependant_array(load_steps)
            
class MeshManager:
    """Gestionnaire de maillage et de mesures d'intégration.
    
    Cette classe encapsule toutes les opérations liées au maillage,
    incluant la création de sous-maillages pour les facettes, le marquage des frontières,
    et la définition des mesures d'intégration pour la formulation HDG.
    """
    def __init__(self, mesh):
        """Initialise le gestionnaire de maillage.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh Maillage principal du problème.
        """
        self.mesh = mesh
        # self.comm = mesh.comm
        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        self.n = FacetNormal(mesh)
        self.h = self.calculate_mesh_size()
        
        # Créer le sous-maillage des facettes et les mappages d'entités
        self.facet_mesh, self.entity_maps = self.create_facet_mesh()

    def create_facet_mesh(self):
        """Crée un sous-maillage contenant toutes les facettes du maillage principal.
        Returns
        -------
        tuple (facet_mesh, entity_maps) où:
            - facet_mesh : dolfinx.mesh.Mesh Sous-maillage des facettes
            - entity_maps : dict ictionnaire de mappages entre indices d'entités
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
        """Marque les facettes qui correspondent aux frontières spécifiées.
        
        Cette méthode marque les frontières du maillage en fonction de leurs coordonnées
        spatiales et crée les meshtags correspondants pour les intégrations ultérieures.
        
        Parameters
        ----------
        flag_list : list[int] Liste des identifiants numériques pour les frontières.
        coord_list : list[str] Liste des coordonnées à utiliser ('x', 'y', 'z').
        localisation_list : list[float] Liste des positions des frontières.
        tol : float, optional Tolérance pour la détection des frontières.
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
        """Calcule les entités d'intégration pour les intégrales autour des
        frontières de toutes les cellules du maillage.
        
        Returns
        -------
        numpy.ndarray Facettes à intégrer, identifiées par des paires (cellule, indice local de facette).
        """
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local   
        # Crée toutes les paires (cellule, facette locale) pour toutes les cellules
        return vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
    
    def set_measures(self, deg):
        """Configure les mesures d'intégration et les domaines pour une formulation HDG.
        
        Cette méthode définit les mesures d'intégration nécessaires pour les formulations
        variationnelles utilisées dans la méthode HDG, en particulier les mesures sur les 
        volumes (dx) et les facettes (ds) avec leurs domaines d'intégration associés.
        
        Parameters
        ----------
        deg : int Degré polynomial des fonctions de forme, utilisé pour déterminer
                    le degré de quadrature.
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
        Calcule la taille locale des éléments du maillage.
        
        Returns
        -------
        Function  Fonction contenant la taille locale des éléments.
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
    Classe abstraite définissant un problème de mécanique des fluides compressibles.
    
    Cette classe est le point central pour la définition d'un problème physique
    à résoudre numériquement. Elle gère le maillage, les espaces fonctionnels,
    les conditions initiales et aux limites, et la formulation variationnelle."""
    def __init__(self, material, dt, initial_mesh = None, **kwargs):
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
        """Configure les mesures d'intégration pour le problème.
        
        Utilise le gestionnaire de maillage pour définir les mesures
        d'intégration avec le degré polynomial approprié.
        """
        self.mesh_manager.set_measures(self.deg)
        self.dx_c = self.mesh_manager.dx_c
        self.ds_c = self.mesh_manager.ds_c
        self.ds_tot = self.mesh_manager.ds_tot
            
    def set_auxiliary_field(self):
        """
        Initialise quelques champs auxiliaires qui permettent d'écrire de 
        manière plus concise le problème thermo-mécanique
        """
        p_elem = self.EOS.set_eos(self.U, self.material)
        self.p_expr = Expression(p_elem, self.V_rho.element.interpolation_points())
        self.p_func = Function(self.V_rho, name = "Pression")      
        
    def fem_parameters(self):
        fem_parameters = default_fem_parameters()
        self.deg = fem_parameters.get("u_degree")
        
    def set_initial_conditions(self):
        pass
    
    def update_bcs(self, num_pas):
        pass
    
    def set_stabilization_parameters(self, shock_sensor_type):
        """
        Initialise les paramètres pour la stabilisation par viscosité artificielle.
        """
        if shock_sensor_type == "ducros":
            from ..VariationalFormulation.shock_sensor import DucrosShockSensor
            self.shock_sensor = DucrosShockSensor(self)
        elif shock_sensor_type == "fernandez":
            from ..VariationalFormulation.shock_sensor import FernandezShockSensor
            self.shock_sensor = FernandezShockSensor(self)