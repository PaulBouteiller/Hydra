"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
"""
Hybridizable Discontinuous Galerkin formulation for compressible Euler equations
==============================================================================

This module implements the Hybridizable Discontinuous Galerkin (HDG) formulation
for the compressible Euler equations. It provides the spatial discretization of
conservation laws for inviscid compressible flows, including the definition of
numerical fluxes, stabilization terms, and boundary conditions.

The formulation is based on the HDG method, which introduces hybrid variables on
element interfaces to reduce the global system size while maintaining the advantages
of discontinuous Galerkin methods for advection-dominated problems.

Key components:
- HDG formulation with interior/boundary numerical fluxes
- Various Riemann solvers for interface flux computation
- Specialized boundary condition implementations
- Shock capturing capabilities
- Conservation of mass, momentum, and energy

Classes:
--------
EulerBoundaryConditions : Boundary condition manager for Euler equations
    Implements various boundary types (wall, pressure, etc.)
    Handles the HDG-specific boundary terms
    Supports time-dependent boundary values

CompressibleEuler : Main HDG formulation for Euler equations
    Defines function spaces and solution variables
    Implements inviscid flux terms and numerical interface fluxes
    Manages shock capturing and stabilization
    Sets up the complete variational formulation
"""


from ufl import (outer, grad, Identity, inner, dot, MixedFunctionSpace, 
                 TestFunctions, TrialFunctions)
from petsc4py.PETSc import ScalarType
from dolfinx.fem import functionspace, Function

# from basix.ufl import quadrature_element
from numpy import zeros

from .Problem import Problem, BoundaryConditions
from ..utils.generic_functions import euler_eigenvalues_2D, max_abs_of_sequence
from..utils.generic_functions import extract_primitive_variables
from ..utils.default_parameters import default_Riemann_solver_parameters, facet_element_type

class EulerBoundaryConditions(BoundaryConditions):
    """
    Boundary condition manager for compressible Euler equations.
    
    This class implements various boundary condition types for the Euler
    equations in HDG formulation, including wall boundaries, inflow/outflow,
    and pressure boundaries.
    """
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, EOS, n, ds, tdim, entity_maps, facet_tag, dico_Vbar):
        """
     Initialize Euler boundary conditions.
     
     Parameters
     ----------
     U : list of Functions Interior solution variables
    Ubar : list of Functions Facet solution variables
    Ubar_test : list of TestFunctions Facet test functions
    facet_mesh : Mesh Facet submesh
    EOS : EOS Equation of state manager
    n : Expression Facet normal vector
    ds : Measure Integration measure for boundaries
    tdim : int Topological dimension
    entity_maps : dict Entity mappings between meshes
    facet_tag : MeshTags Tags for facets
    dico_Vbar : dict Function spaces for facet variables
    """  
        self.V_rhovbar, self.V_rhobar, self.V_rhoebar = dico_Vbar.get("Velocity"), dico_Vbar.get("Density"), dico_Vbar.get("Energy")
        super().__init__(U, Ubar, Ubar_test, facet_mesh, EOS, n, ds, tdim, entity_maps, facet_tag)
    
    def wall_residual(self, tag, normal):
        """
        Apply wall boundary condition (slip wall).
        
        Imposes zero normal velocity at the wall while allowing tangential
        flow (slip condition).
        
        Parameters
        ----------
        tag : int Boundary tag for the wall
        normal : str Direction of the wall normal ('x', 'y', or 'z')
        """
        if normal == None:
            self.add_component(self.V_rhovbar, None, tag, zeros((1,)))
        else:
            if normal == "x":
                sub = 0
            elif normal == "y":
                sub = 1
            elif normal == "z":
                sub = 2
            self.add_component(self.V_rhovbar, sub, tag, ScalarType(0))
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = - inner(self.U[1] - dot(self.U[1], self.n) * self.n 
                        - self.Ubar[1], self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E
        
    def wall_residual_with_rho(self, tag, normal, rho_value, rhoe_value):
        """
        Apply wall boundary with specified density and energy.
        
        Similar to wall_residual but with specified values for density
        and energy at the wall.
        
        Parameters
        ----------
        tag : int Boundary tag for the wall
        normal : str Direction of the wall normal ('x', 'y', or 'z')
        rho_value : float Density value at the wall
        rhoe_value : float Energy density value at the wall
        """
        self.wall_residual(tag, normal)
        self.add_component(self.V_rhobar, None, tag, ScalarType(rho_value))
        self.add_component(self.V_rhoebar, None, tag, ScalarType(rhoe_value))

class CompressibleEuler(Problem):
    """
    HDG formulation for compressible Euler equations.
    
    This class implements the Hybridizable Discontinuous Galerkin (HDG)
    spatial discretization for the compressible Euler equations, which
    describe inviscid compressible fluid flow.
    
    The implementation includes:
    - Function space and variable setup
    - Numerical flux formulation
    - Boundary condition handling
    - Variational form construction
    - Optional shock capturing
    """

    def set_stabilization_matrix(self, U, Ubar, c, n):
        """
        Compute the stabilization parameter for HDG formulation.
        
        Implements Local Lax-Friedrichs (Rusanov) stabilization based on
        the maximum wave speed at the interface.
        
        Parameters
        ----------
        U : list of UFL expressions Interior solution variables
        Ubar : list of UFL expressions Facet solution variables
        c : UFL expression Sound speed
        n : UFL expression Facet normal vector
            
        Returns
        -------
        UFL expression Stabilization parameter (maximum wave speed)
        """
        _, u, _ = extract_primitive_variables(U)
        _, ubar, _ = extract_primitive_variables(Ubar)
        evs_u = euler_eigenvalues_2D(u, n, c)
        evs_bar = euler_eigenvalues_2D(ubar, n, c)
        return max_abs_of_sequence([*evs_u, *evs_bar])
    
    def boundary_conditions_class(self):
        """
        Get the boundary condition class for Euler equations.
        """
        return EulerBoundaryConditions
        
    def set_function_space(self):  
        """
        Initialize function spaces for Euler equations.
        
        Sets up the discontinuous Galerkin spaces for interior variables
        and the trace spaces for facet variables.
        """
        self.V_rho = functionspace(self.mesh, ("DG", self.deg))
        self.V_rhov = functionspace(self.mesh, ("DG", self.deg, (self.tdim, )))
        self.V_rhoe = functionspace(self.mesh, ("DG", self.deg))
    
        # Créer l'espace fonctionnel
        facet_element = facet_element_type()
        self.V_rhobar = functionspace(self.facet_mesh, (facet_element, self.deg))
        self.V_rhovbar = functionspace(self.facet_mesh, (facet_element, self.deg, (self.tdim, )))
        self.V_rhoebar = functionspace(self.facet_mesh, (facet_element, self.deg))
           
       
    def set_functions(self):   
        """
        Initialize solution variables for Euler equations.
        
        Sets up the functions for density, momentum, and energy,
        both for interior and facet representations.
        """      
        self.rho = Function(self.V_rho, name = "Density")
        self.rhov = Function(self.V_rhov, name = "Momentum")
        self.rhoe = Function(self.V_rho, name = "Energy density")
        self.rho_n, self.rhov_n, self.rhoe_n = Function(self.V_rho), Function(self.V_rhov), Function(self.V_rhoe)
        
        self.s_rho, self.s_rhov, self.s_rhoe = Function(self.V_rho), Function(self.V_rhov), Function(self.V_rhoe)

        self.rhobar, self.rhovbar, self.rhoebar = Function(self.V_rhobar), Function(self.V_rhovbar), Function(self.V_rhoebar)

        self.rho.x.petsc_vec.set(self.material.rho_0)
        self.rho_n.x.petsc_vec.set(self.material.rho_0)
        self.rhobar.x.petsc_vec.set(self.material.rho_0)    
        
        self.U = [self.rho, self.rhov, self.rhoe]
        self.Ubar = [self.rhobar, self.rhovbar, self.rhoebar]
        self.U_n = [self.rho_n, self.rhov_n, self.rhoe_n]
        self.s = [self.s_rho, self.s_rhov, self.s_rhoe]
        
        self.dico_Vbar = {"Density" : self.V_rhobar, "Velocity" : self.V_rhovbar, "Energy" : self.V_rhoebar}
        
    def set_variable_to_solve(self):
        """
        Set the variables to be solved.
        
        Defines the list of variables that will be passed to the solver,
        with special handling for isothermal cases.
        """
        if self.iso_T:
            self.u_list = self.U[:2] + self.Ubar[:2]
        else:
            self.u_list = self.U + self.Ubar
        
    def set_test_functions(self):
        """
        Initialize test and trial functions.
        
        Sets up the test and trial functions for both interior and facet
        variables, with special handling for isothermal cases.
        """
        MFS = MixedFunctionSpace(self.V_rho, self.V_rhov, self.V_rhoe,
                                 self.V_rhobar, self.V_rhovbar, self.V_rhoebar)  
        rho_, rhov_, rhoe_, rhobar_, rhovbar_, rhoebar_ = TestFunctions(MFS)
        drho, drhov, drhoe, drhobar, drhovbar, drhoebar = TrialFunctions(MFS)
        self.U_test = [rho_, rhov_, rhoe_]
        self.Ubar_test = [rhobar_, rhovbar_, rhoebar_]
        self.dU = [drho, drhov, drhoe]
        self.dUbar = [drhobar, drhovbar, drhoebar]
        if self.iso_T:
            self.du_list = self.dU[:2] + self.dUbar[:2]
            self.u_test_list = self.U_test[:2] + self.Ubar_test[:2]
        else:
            self.du_list = self.dU + self.dUbar
            self.u_test_list = self.U_test + self.Ubar_test
            
    def total_flux(self):
        """
        Compute the total flux including artificial viscosity if enabled.
        
        Returns
        -------
        tuple (U_flux, Ubar_flux) - Fluxes for interior and facet variables
        """
        U_flux = self.inviscid_flux(self.U)
        Ubar_flux = self.inviscid_flux(self.Ubar)
        if self.shock_stabilization:
            self.p_star_U = Function(self.V_rho)
            self.p_star_U_expr = self.EOS.set_artifial_pressure(self.U, self.V_rho, self.material, 
                                                            self.h, self.deg, self.shock_sensor)
            self.p_star_U.interpolate(self.p_star_U_expr)
            U_flux[1]+= self.p_star_U * Identity(self.tdim)
        return U_flux, Ubar_flux

    def inviscid_flux(self, U):
        """
        Compute the inviscid flux for Euler equations.
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables [ρ, ρu, ρE]
            
        Returns
        -------
        list Fluxes for each conservation equation: [mass flux, momentum flux, energy flux]
        """
        p = self.EOS.set_eos(U, self.material)
        return [self.mass_flux(U),
                self.momentum_flux(U, p),
                self.energy_flux(U, p)]
    
    def mass_flux(self, U):
        """
        Compute the mass flux (ρu).
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables
            
        Returns
        -------
        UFL expression Mass flux vector
        """
        return U[1]
    
    def momentum_flux(self, U, p):
        """
        Compute the momentum flux ((ρu ⊗ u) + pI).
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables
        p : UFL expression Pressure
            
        Returns
        -------
        UFL expression Momentum flux tensor
        """
        return  outer(U[1], U[1]) / U[0] + p * Identity(self.tdim)
    
    def energy_flux(self, U, p):
        """
        Compute the energy flux ((ρE + p)u).
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables
        p : UFL expression Pressure
            
        Returns
        -------
        UFL expression Energy flux vector
        """
        return (U[2] + p) * U[1]/U[0]
    
    def set_dynamic_lhs(self):
        """
        Set up the left-hand side terms for dynamic simulation.
        
        Returns
        -------
        UFL form Time derivative terms for the variational formulation
        """
        return sum(inner(x * self.dt_factor, x_test) * self.dx_c 
                   for x, x_test in zip(self.U, self.U_test))
    
    def set_dynamic_source(self):
        """
        Set up the source terms for dynamic simulation.
        
        Returns
        -------
        UFL form Source terms for the variational formulation
        """
        return sum(inner(s, x_test) * self.dx_c for s, x_test in zip(self.s, self.U_test))
    
    def set_volume_residual(self, U_flux):
        """
        Set up the volume residual terms.
        
        Implements the weak form of the divergence terms.
        
        Parameters
        ----------
        U_flux : list of UFL expressions Fluxes for each conservation equation
            
        Returns
        -------
        UFL form Volume residual terms
        """
        return -sum(inner(flux, grad(test_func)) * self.dx_c 
                    for flux, test_func in zip(U_flux, self.U_test))        
        
    def set_numerical_flux(self, U_flux, Ubar_flux):
        """
        Compute the numerical flux at interfaces.
        
        Implements various Riemann solver options for the numerical flux
        at element interfaces.
        
        Parameters
        ----------
        U_flux : list of UFL expressions Interior fluxes
        Ubar_flux : list of UFL expressions Facet fluxes
            
        Returns
        -------
        list Numerical fluxes for each conservation equation
        """
        riemann_solvers_dic = default_Riemann_solver_parameters() 
        flux_type = riemann_solvers_dic.get("flux_type")
        from .riemann_solvers import RiemannSolvers
        riemann = RiemannSolvers(self.EOS, self.material)
        if flux_type == "Cockburn":
            return [dot(flux_bar, self.n) + self.S * (x - x_bar) 
                    for flux_bar, x, x_bar in zip(Ubar_flux, self.U, self.Ubar)]
        elif flux_type == "Rusanov":
            return riemann.rusanov_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLL":
            return riemann.hll_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLLC":
            return riemann.hllc_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLLCLM":
            return riemann.hllclm_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        else:
            raise ValueError("Flux numérique inconnu")
    
    def numerical_flux_continuity_residual(self, flux_num):
        """
        Compute the residual enforcing numerical flux continuity.
        
        Implements the weak form of the numerical flux continuity
        across element interfaces.
        
        Parameters
        ----------
        flux_num : list of UFL expressions Numerical fluxes
            
        Returns
        -------
        UFL form Numerical flux continuity residual
        """
        # TODO: Change ASAP. Here we impose numerical flux continuity
        # on all facet_mesh boundaries including external ones!
        # Then we subtract external boundaries which are here the
        # external boundaries of mesh. We should instead directly remove
        # external facets from the submesh facets by matching them with
        # their counterparts on the mesh.
        continuity_residual = -sum(inner(f_num, test_func) * self.ds_tot 
                                    for f_num, test_func in zip(flux_num, self.Ubar_test))
        for tag in self.flag_list:
            continuity_residual += sum(inner(f_num, test_func) * self.ds_c(tag) 
                                    for f_num, test_func in zip(flux_num, self.Ubar_test))
        
        #En gros il faudrait faire ça mais le dsèint ne serait il pas plus un dS_int ?
        # continuity_residual = -sum(inner(f_num, test_func) * self.ds_int
        #                             for f_num, test_func in zip(flux_num, self.Ubar_test))
        return continuity_residual

    def surface_residual(self, flux_num):
        """
        Compute the surface residual from integration by parts.
        
        Implements the boundary terms arising from integration by parts
        of the divergence terms.
        
        Parameters
        ----------
        flux_num : list of UFL expressions Numerical fluxes
            
        Returns
        -------
        UFL form Surface residual terms
        """
        return sum(inner(f_num, test_func) * self.ds_tot 
                    for f_num, test_func in zip(flux_num, self.U_test))
    
    def total_surface_residual(self, U_flux, Ubar_flux):
        """
        Compute the complete surface residual.
        
        Combines the surface residual and numerical flux continuity
        residual into the complete surface terms.
        
        Parameters
        ----------
        U_flux : list of UFL expressions Interior fluxes
        Ubar_flux : list of UFL expressions Facet fluxes
            
        Returns
        -------
        UFL form Complete surface residual
        """
        numerical_flux = self.set_numerical_flux(U_flux, Ubar_flux)
        surface_residual = self.surface_residual(numerical_flux)
        continuity_residual = self.numerical_flux_continuity_residual(numerical_flux)
        return surface_residual + continuity_residual

    def set_form(self):
        """
        Set up the complete variational formulation.
        
        Combines volume residual, surface residual, boundary residual,
        and time derivative terms into the complete variational form.
        """
        U_flux, Ubar_flux = self.total_flux()
        vol_res = self.set_volume_residual(U_flux)
        surf_res = self.total_surface_residual(U_flux, Ubar_flux)
        boundary_res = self.bc_class.boundary_residual
        self.residual = vol_res + surf_res + boundary_res    
        if self.analysis == "dynamic":        
            self.residual+= self.set_dynamic_lhs()
            self.residual-= self.set_dynamic_source()