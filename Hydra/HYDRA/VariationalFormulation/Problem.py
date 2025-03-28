"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
"""
from ..ConstitutiveLaw.eos import EOS
from ..utils.default_parameters import default_fem_parameters
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
        self.S = S
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
        self.n = FacetNormal(self.mesh)
        self.facet_mesh, self.entity_maps = self.create_facet_mesh()
        self.dt_factor = Constant(self.mesh, ScalarType(1.0 / self.dt))# Facteur par défaut pour backward Euler

        # Set parameters and update from user
        self.fem_parameters()
        
        # Material properties
        self.material = material

        # MeshFunctions and Measures for different domains and boundaries
        self.set_boundary()
        self.set_measures()
        
        #Set function space and unknwown functions 
        self.set_function_space()
        self.set_functions()
        self.set_variable_to_solve()
        self.set_test_functions()
        self.set_stabilization_parameters(**kwargs)

        # Constitutive Law
        self.EOS = EOS(None, None)
        self.material.c = self.EOS.set_celerity(self.U, self.material)
        self.material.c_bar = self.EOS.set_celerity(self.U, self.material)
        self.set_artifial_pressure()
        self.set_auxiliary_field()
        
        #Boundary conditions
        self.S = self.set_stabilization_matrix(self.U, self.Ubar, self.material.c, self.n)
        self.bc_class = self.boundary_conditions_class()(
                        self.U, self.Ubar, self.Ubar_test,
                        self.facet_mesh, self.S, self.n, self.ds_c,
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

    def create_facet_mesh(self):
        """
        Crée un sous-maillage contenant toutes les facettes du maillage principal.
        Parameters
        ----------
        msh : dolfinx.mesh.Mesh Maillage du domaine
        Returns
        -------
        tuple facet_mesh : dolfinx.mesh.Mesh Sous-maillage des facettes
              facet_mesh_to_msh : numpy.ndarray
                Tableau de mapping entre les indices des facettes du sous-maillage
                et ceux du maillage principal
        """
        self.mesh.topology.create_entities(self.fdim)
        facet_imap = self.mesh.topology.index_map(self.fdim)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        facets = arange(num_facets, dtype=int32)
        facet_mesh, facet_mesh_to_msh, _, _ = create_submesh(self.mesh, self.fdim, facets)
        facet_mesh.topology.create_connectivity(self.fdim, self.fdim)
        msh_to_facet_mesh = full(num_facets, -1)
        msh_to_facet_mesh[facet_mesh_to_msh] = arange(len(facet_mesh_to_msh))
        entity_maps = {facet_mesh: msh_to_facet_mesh}
        return facet_mesh, entity_maps
        
    def mark_boundary(self, flag_list, coord_list, localisation_list, tol=finfo(float).eps):
        """
        Marque directement les facettes qui correspondent aux frontières.
        """
        facets = []
        full_flag = []
        flags_found = []  # Pour suivre les flags qui ont effectivement trouvé des facettes
        for flag, coord, loc in zip(flag_list, coord_list, localisation_list):
            def boundary_condition(x):
                def index_coord(coord):
                    """
                    Renvoie l'index correspondant à la variable spatiale
                    """
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
        Calcule les entités d'intégration pour les intégrales autour des
        frontières de toutes les cellules du maillage.
        
        Returns:
            numpy.ndarray: Facettes à intégrer, identifiées par des paires (cellule, indice local de facette).
        """
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local
        
        # Crée toutes les paires (cellule, facette locale) pour toutes les cellules
        return vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
    
    # def identify_interior_facets(self):
    #     """
    #     Identifie correctement les facettes intérieures du maillage en excluant les facettes extérieures.
        
    #     Returns:
    #         numpy.ndarray: Tableau d'entités d'intégration pour les facettes intérieures.
    #     """
    #     # # S'assurer que toutes les connectivités nécessaires sont créées
    #     self.mesh.topology.create_connectivity(self.fdim, self.tdim)
    #     self.mesh.topology.create_connectivity(self.tdim, self.fdim)
        
    #     # Obtenir toutes les paires (cellule, facette locale)
    #     all_facets = self.compute_cell_boundary_facets()
        
    #     # Obtenir les facettes extérieures
    #     exterior_facets = exterior_facet_indices(self.mesh.topology)
    #     print(f"DEBUG: {len(exterior_facets)} facettes extérieures trouvées")
        
    #     # Calculer les entités d'intégration pour les facettes extérieures
    #     exterior_entities = compute_integration_domains(
    #         IntegralType.exterior_facet, self.mesh.topology, exterior_facets, self.fdim)
        
    #     # Calculer les indices à supprimer de all_facets
    #     n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
    #     remove_index = n_f * exterior_entities[::2] + exterior_entities[1::2]
        
    #     # Supprimer ces indices pour obtenir les facettes intérieures
    #     interior_facets = delete(all_facets.reshape(-1, 2), remove_index, axis=0).flatten()        
    #     return interior_facets

    def set_measures(self):
        """Configure les mesures d'intégration et les domaines pour une formulation HDG."""
        
        # Obtenir les entités d'intégration pour les facettes intérieures
        # interior_entities = self.identify_interior_facets()
        
        # Tags pour les différents types de frontières
        cell_boundaries_tag = 0       # Toutes les frontières
        # interior_boundaries_tag = 10  # Frontières intérieures
        
        # Calculer toutes les facettes de cellules
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local
        all_facets = vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
        
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
        quad_deg_facet = (2 * self.deg + 1) ** (self.tdim)
        quad_deg_volume = (self.deg + 1) ** self.tdim
        print(f"Nombre de points de Gauss: {quad_deg_volume}")
        
        # Définir les mesures
        self.dx_c = Measure("dx", domain=self.mesh, metadata={"quadrature_degree": quad_deg_volume})
        self.ds_c = Measure("ds", subdomain_data=facet_integration_entities,
                      domain=self.mesh, metadata={"quadrature_degree": quad_deg_facet})
        
        # Mesures spécifiques
        self.ds_tot = self.ds_c(cell_boundaries_tag)
        
        # # Créer une mesure spécifique pour les facettes intérieures
        # facet_integration_entities_int = [(interior_boundaries_tag, interior_entities)]
        # self.ds_int = Measure("ds", subdomain_data=facet_integration_entities_int,
        #                     domain=self.mesh, metadata={"quadrature_degree": quad_deg_facet})
        
        # # Vérification de l'intégration sur les facettes intérieures
        # print("\n==== VÉRIFICATION DE L'INTÉGRATION ====")
        
        # # Créer une constante égale à 1
        # one = Constant(self.mesh, ScalarType(1.0))
        
        # # Intégrer sur toutes les facettes
        # from dolfinx.fem import form, assemble_scalar
        # form_tot = form(one * self.ds_tot)
        # integral_tot = assemble_scalar(form_tot)
        # print(f"Intégrale de 1 sur toutes les facettes: {integral_tot}")
        
        # # Intégrer sur les facettes intérieures
        # form_int = form(one * self.ds_int(interior_boundaries_tag))
        # integral_int = assemble_scalar(form_int)
        # print(f"Intégrale de 1 sur les facettes intérieures: {integral_int}")
        
        # # Calcul de la valeur attendue
        # n_interior = len(interior_entities) // 2
        # print(f"Nombre de paires d'entités intérieures: {n_interior}")
        # print(f"Valeur attendue (longueur de chaque facette est 1): {n_interior}")
        # print(f"Test réussi? {abs(integral_int - n_interior) < 1e-10}")
            
    def set_auxiliary_field(self):
        """
        Initialise quelques champs auxiliaires qui permettent d'écrire de 
        manière plus concise le problème thermo-mécanique
        """
        p = self.EOS.set_eos(self.U, self.material)
        self.p_expr = Expression(p, self.V_rho.element.interpolation_points())
        self.p_func = Function(self.V_rho, name = "Pression")
        
        
    def fem_parameters(self):
        fem_parameters = default_fem_parameters()
        self.deg = fem_parameters.get("u_degree")
        
    def set_initial_conditions(self):
        pass
    
    def update_bcs(self, num_pas):
        pass
    
    def set_stabilization_parameters(self, **kwargs):
        """
        Initialise les paramètres pour la stabilisation par viscosité artificielle.
        
        Parameters
        ----------
        **kwargs : Dictionnaire de paramètres optionnels
            use_shock_capturing : bool Activer la capture de choc (défaut: True)
            shock_sensor_type : str Type de capteur de choc ('ducros', ou 'none') (défaut: 'ducros')
            shock_threshold : float Seuil pour le capteur de Ducros (défaut: 0.95)
            shock_viscosity_coeff : float Coefficient pour la viscosité artificielle (défaut: 0.5)
        """
        # Paramètres pour la viscosité artificielle
        self.use_shock_capturing = kwargs.get("use_shock_capturing", True)
        self.shock_sensor_type = kwargs.get("shock_sensor_type", "ducros")
        self.shock_threshold = kwargs.get("shock_threshold", 0.95)
        self.shock_viscosity_coeff = kwargs.get("shock_viscosity_coeff", 0.5)
        
        # Créer l'espace fonctionnel pour la viscosité artificielle
        self.V_art = functionspace(self.mesh, ("DG", self.deg))
        self.mu_art = Function(self.V_art, name="ArtificialViscosity")
        
        # Initialiser le capteur de choc approprié
        if self.use_shock_capturing:
            if self.shock_sensor_type.lower() == "ducros":
                from ..VariationalFormulation.shock_sensor import DucrosShockSensor
                self.shock_sensor = DucrosShockSensor(self, threshold=self.shock_threshold)
            elif self.shock_sensor_type.lower() == "fernandez":
                from ..VariationalFormulation.shock_sensor import FernandezShockSensor
                self.shock_sensor = FernandezShockSensor(self, threshold=self.shock_threshold)
            else:
                self.shock_sensor = None
                
    def calculate_mesh_size(self):
        """
        Calcule la taille locale des éléments du maillage.
        
        Returns
        -------
        Function  Fonction contenant la taille locale des éléments.
        """
        h_loc = Function(functionspace(self.mesh, ("DG", 0)), name="MeshSize")                
        
        # Calculer la taille locale des éléments
        tdim = self.tdim
        num_cells = self.mesh.topology.index_map(tdim).size_local
        h_local = zeros(num_cells)
        
        for i in range(num_cells):
            h_local[i] = self.mesh.h(tdim, array([i]))
        
        h_loc.x.array[:] = h_local
        return h_loc
    
    def set_artifial_pressure(self):
        from ..ConstitutiveLaw.artificial_pressure import ArtificialPressure
        h = self.calculate_mesh_size()
        c = self.EOS.set_celerity(self.U, self.material)
        self.artificial_pressure = ArtificialPressure(self.U, self.V_rho, h, c, self.deg, self.shock_sensor)