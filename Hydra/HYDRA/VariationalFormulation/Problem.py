"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
"""
from ..ConstitutiveLaw.eos import EOS
from ..utils.default_parameters import default_damping_parameters, default_fem_parameters

from numpy import (hstack, argsort, finfo, full_like, arange, int32, unique, 
                    tile, repeat, vstack, sort, full)

from dolfinx.fem import (compute_integration_domains, dirichletbc, locate_dofs_topological,
                          IntegralType, Constant, Expression, Function)
from dolfinx.mesh import locate_entities_boundary, meshtags, create_submesh
from ufl import (Measure, inner, FacetNormal)

from mpi4py import MPI
from dolfinx.cpp.mesh import cell_num_entities
from petsc4py.PETSc import ScalarType
try:
    from CharonX.utils.MyExpression import MyConstant
except:
    from .utils.MyExpression import MyConstant



class BoundaryConditions:
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag):
        self.U, self.Ubar, self.Ubar_test = U, Ubar, Ubar_test
        self.S = S
        self.ds = ds
        self.n = n
        self.boundary_residual = Constant(facet_mesh, ScalarType(0)) * Ubar_test[0] * ds
        self.mcl = []
        
        # Nouveaux attributs pour la gestion des conditions de Dirichlet
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
    La classe Problem est un des éléments principal du code Charon. C'est elle qui défini
    la formulation du problème en appelant les modèles mécaniques retenus. Les différents types 
    d'analyses (plastique, endommageable etc..) sont précisés en insérant des mots clés lors de 
    l'appel de cette classe. Voir la notice pour la définition des différents mots clés.
    """
    def __init__(self, material, dt, initial_mesh = None, **kwargs):
        if initial_mesh == None:
            self.mesh = self.define_mesh()
        else:
            self.mesh = initial_mesh
        if MPI.COMM_WORLD.Get_size()>1:
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

            
        # Constitutive Law
        self.EOS = EOS(None, None)
        self.set_auxiliary_field()
        
        #Boundary conditions
        self.S = self.set_stabilization_matrix(self.u, self.ubar, self.material.celerity, self.n)
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
        
    def mark_boundary(self, flag_list, coord_list, localisation_list, tol = finfo(float).eps):
        """
        Permet le marquage simple des bords pour des pavés/maillage 1D

        Parameters
        ----------
        flag_list : List de Int, drapeau qui vont être associés aux domaines.
        coord_list : List de String, variable permettant de repérer les domaines.
        localisation_list : List de Float, réel au voisinage duquel les poins de coordonnées
                            vont être récupéré.
        """
        facets, full_flag = self.set_facet_flags(flag_list, coord_list, localisation_list, tol)
        marked_facets = hstack(facets)
        marked_values = hstack(full_flag)
        sorted_facets = argsort(marked_facets)
        self.flag_list = unique(flag_list)#Contient les drapeaux sans les doublons
        self.facet_tag = meshtags(self.mesh, self.fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
        
    def set_facet_flags(self, flag_list, coord_list, localisation_list, tol):
        facets = []
        full_flag = []
        for variable in zip(flag_list, coord_list, localisation_list):
            def trivial_boundary(x):
                return abs(x[self.index_coord(variable[1])] - variable[2]) < tol
            facets.append(locate_entities_boundary(self.mesh, self.fdim, trivial_boundary))
            full_flag.append(full_like(facets[-1], variable[0]))
        return facets, full_flag
        
    def index_coord(self, coord):
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
            
    def set_measures(self):
        """Configure les mesures d'intégration et les domaines pour une formulation HDG.
        
        Cette fonction configure les mesures nécessaires pour l'intégration volumique et surfacique
        dans le contexte d'une méthode HDG. Elle gère à la fois
        les intégrations sur le volume des éléments et sur leurs facettes, en tenant compte des
        conditions aux limites.
        
        Parameters
        ----------
        msh : dolfinx.mesh.Mesh Le maillage sur lequel les mesures seront définies
        k : int L'ordre polynomial des éléments finis
        mt : dolfinx.mesh.MeshTags Les tags du maillage identifiant les différentes parties de la frontière
        boundaries : dict Dictionnaire associant les noms des frontières à leurs identifiants numériques
            
        Returns
        -------
        dx_c : ufl.Measure Mesure pour l'intégration volumique
        ds_c : ufl.Measure Mesure pour l'intégration surfacique, incluant les conditions aux limites
        cbt : int Tag pour les frontières internes des cellules (généralement 0)
        """
        n_f = cell_num_entities(self.mesh.topology.cell_type, self.fdim)
        n_c = self.mesh.topology.index_map(self.tdim).size_local
        cell_boundary_facets = vstack((repeat(arange(n_c), n_f), tile(arange(n_f), n_c))).T.flatten()
        cbt = 0  # cell_boundaries_tag
        
        facet_integration_entities = [(cbt, cell_boundary_facets)]
        facet_integration_entities += [
            (tag, compute_integration_domains(
                IntegralType.exterior_facet, self.mesh.topology, self.facet_tag.find(tag), self.fdim))
            for tag in sort(self.flag_list)]


        quad_deg_facet = (2 * self.deg + 1) ** (self.tdim)
        quad_deg_volume = (self.deg + 1) ** self.tdim
        print("Nombre de points de Gauss", quad_deg_volume)
        self.dx_c = Measure("dx", domain = self.mesh,  metadata = {"quadrature_degree": quad_deg_volume})
        self.ds_c = Measure("ds", subdomain_data = facet_integration_entities,
                      domain = self.mesh, metadata = {"quadrature_degree": quad_deg_facet})
        self.ds_tot = self.ds_c(cbt) #self.ds_c(0) contient l'ensembles des facettes extérieures du submesh
        
    def set_auxiliary_field(self):
        """
        Initialise quelques champs auxiliaires qui permettent d'écrire de 
        manière plus concise le problème thermo-mécanique
        """
        rho, u, E = self.U[0], self.U[1]/self.U[0], self.U[2]/self.U[0]
        p = self.EOS.set_eos(rho, u, E, self.material)
        self.p_expr = Expression(p, self.V_rho.element.interpolation_points())
        self.p_func = Function(self.V_rho, name = "Pression")
        
        
    def fem_parameters(self):
        fem_parameters = default_fem_parameters()
        self.deg = fem_parameters.get("u_degree")
        self.schema= fem_parameters.get("schema")
        
    def set_initial_conditions(self):
        pass
    
    def update_bcs(self, num_pas):
        pass