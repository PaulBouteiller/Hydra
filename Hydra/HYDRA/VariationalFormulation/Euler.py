"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from ufl import (outer, grad, Identity, inner, dot, MixedFunctionSpace, 
                 TestFunctions, TrialFunctions)
from petsc4py.PETSc import ScalarType
from dolfinx.fem import functionspace, Function

from .Problem import Problem, BoundaryConditions
from ..utils.generic_functions import euler_eigenvalues_2D, max_abs_of_sequence
from..utils.generic_functions import extract_primitive_variables
from ..utils.default_parameters import default_Riemann_solver_parameters

class EulerBoundaryConditions(BoundaryConditions):
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag, dico_Vbar):
        self.V_vbar, self.V_rhobar = dico_Vbar.get("Velocity"), dico_Vbar.get("Density")
        super().__init__(U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag)
    
    def wall_residual(self, tag, normal):
        if normal == "x":
            sub = 0
        elif normal == "y":
            sub = 1
        elif normal == "z":
            sub = 2
        self.add_component(self.V_vbar, sub, tag, ScalarType(0))
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = - inner(self.U[1] - dot(self.U[1], self.n) * self.n 
                        - self.Ubar[1], self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E
        
    def wall_residual_with_rho(self, tag, normal, value):
        if normal == "x":
            sub = 0
        elif normal == "y":
            sub = 1
        elif normal == "z":
            sub = 2
        self.add_component(self.V_vbar, sub, tag, ScalarType(0))
        self.add_component(self.V_rhobar, None, tag, ScalarType(value))
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = - inner(self.U[1] - dot(self.U[1], self.n) * self.n 
                        - self.Ubar[1], self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E

class CompressibleEuler(Problem):
    """ 
    Implémentation des équations d'Euler compressibles avec la méthode HDG.
    
    Cette classe définit la formulation variationnelle pour les équations d'Euler
    compressibles, incluant les flux numériques, les termes de stabilisation, et
    la gestion des conditions aux limites. Elle utilise une formulation HDG où les 
    variables sont définies à la fois sur les éléments et sur leurs faces.
    """

    def set_stabilization_matrix(self, U, Ubar, c, n):
        """
        Local Lax-Friedrichs (Rusanov) stabilization matrix 
        """
        _, u, _ = extract_primitive_variables(U)
        _, ubar, _ = extract_primitive_variables(Ubar)
        evs_u = euler_eigenvalues_2D(u, n, c)
        evs_bar = euler_eigenvalues_2D(ubar, n, c)
        return max_abs_of_sequence([*evs_u, *evs_bar])
    
    def boundary_conditions_class(self):
        return EulerBoundaryConditions
        
    def set_function_space(self):  
        """
        Initialise les espaces fonctionnels
        """
        self.V_rho = functionspace(self.mesh, ("DG", self.deg))
        self.V_v = functionspace(self.mesh, ("DG", self.deg, (self.tdim, )))
        self.V_E = self.V_rho
        self.V_rhobar = functionspace(self.facet_mesh, ("DG", self.deg))
        self.V_vbar = functionspace(self.facet_mesh, ("DG", self.deg, (self.tdim, )))
        self.V_Ebar = self.V_rhobar
       
    def set_functions(self):   
        """ 
        Initialise les champs inconnues du problème densité, vitesse, énergie
        """        
        self.rho = Function(self.V_rho, name = "Density")
        self.rhou = Function(self.V_v, name = "Momentum")
        self.rhoE = Function(self.V_rho, name = "Energy density")
        self.rho_n, self.rhou_n, self.rhoE_n = Function(self.V_rho), Function(self.V_v), Function(self.V_rho)
        
        self.s_rho, self.s_rhou, self.s_rhoE = Function(self.V_rho), Function(self.V_v), Function(self.V_rho)

        self.rhobar, self.rhoubar, self.rhoEbar = Function(self.V_rhobar), Function(self.V_vbar), Function(self.V_rhobar)

        self.rho.x.petsc_vec.set(self.material.rho_0)
        self.rho_n.x.petsc_vec.set(self.material.rho_0)
        self.rhobar.x.petsc_vec.set(self.material.rho_0)    
        
        self.U = [self.rho, self.rhou, self.rhoE]
        self.Ubar = [self.rhobar, self.rhoubar, self.rhoEbar]
        self.U_n = [self.rho_n, self.rhou_n, self.rhoE_n]
        self.s = [self.s_rho, self.s_rhou, self.s_rhoE]
        
        self.dico_Vbar = {"Density" : self.V_rhobar, "Velocity" : self.V_vbar}
        
    def set_variable_to_solve(self):
        if self.iso_T:
            self.u_list = self.U[:2] + self.Ubar[:2]
        else:
            self.u_list = self.U + self.Ubar
        
    def set_test_functions(self):
        """
        Initialise les fonctions test et d'essai.
        """
        MFS = MixedFunctionSpace(self.V_rho, self.V_v, self.V_E,
                                 self.V_rhobar, self.V_vbar, self.V_Ebar)  
        rho_, rhou_, rhoE_, rhobar_, rhoubar_, rhoEbar_ = TestFunctions(MFS)
        drho, drhou, drhoE, drhobar, drhoubar, drhoEbar = TrialFunctions(MFS)
        self.U_test = [rho_, rhou_, rhoE_]
        self.Ubar_test = [rhobar_, rhoubar_, rhoEbar_]
        self.dU = [drho, drhou, drhoE]
        self.dUbar = [drhobar, drhoubar, drhoEbar]
        if self.iso_T:
            self.du_list = self.dU[:2] + self.dUbar[:2]
            self.u_test_list = self.U_test[:2] + self.Ubar_test[:2]
        else:
            self.du_list = self.dU + self.dUbar
            self.u_test_list = self.U_test + self.Ubar_test

    def inviscid_flux(self, U):
        """
        Définit les flux non visqueux pour les équations d'Euler
        Parameters
        ----------
        U : Liste de fonctions [ρ, ρu, ρE] ρ = densité, u = vitesse, E = énergie totale
        Returns
        -------
        Liste des flux: [flux de masse, flux de quantité de mouvement, flux d'énergie]
        """
        p = self.EOS.set_eos(U, self.material)
        if self.use_shock_capturing:
            p+= self.artificial_pressure.p_star
        return [self.mass_flux(U),
                self.momentum_flux(U, p),
                self.energy_flux(U, p)]
    
    def mass_flux(self, U):
        """Renvoie le flux de masse ρu"""
        return U[1]
    
    def momentum_flux(self, U, p):
        """Renvoie le flux de quantité de mouvement (ρu ⊗ ρu) / ρ  + pI """
        return  outer(U[1], U[1]) / U[0] + p * Identity(self.tdim)
    
    def energy_flux(self, U, p):
        """Renvoie le flux d'energie massique (ρE + p)u"""
        return (U[2] + p) * U[1]/U[0]
    
    def set_dynamic_lhs(self):
        return sum(inner(x * self.dt_factor, x_test) * self.dx_c 
                   for x, x_test in zip(self.U, self.U_test))
    
    def set_dynamic_source(self):
        return sum(inner(s, x_test) * self.dx_c for s, x_test in zip(self.s, self.U_test))
    
    def set_volume_residual(self, U_flux):
        """Renvoie le résidu volumique"""
        return -sum(inner(flux, grad(test_func)) * self.dx_c 
                    for flux, test_func in zip(U_flux, self.U_test))        
        
    def set_numerical_flux(self, U_flux, Ubar_flux):
        """
        Calcule le flux numérique à travers les interfaces entre éléments.
        
        Cette méthode implémente différents types de flux numériques utilisés pour
        approximer la solution du problème de Riemann aux interfaces. Les options
        incluent Cockburn (flux upwind simple), Lax-Friedrichs (LF), Rusanov, 
        et les solveurs plus sophistiqués HLL et HLLC.
        
        Parameters
        ----------
        U_flux : list[ufl.Expr] Flux associés aux variables U à l'intérieur de l'élément
        Ubar_flux : list[ufl.Expr] Flux associés aux variables Ubar aux interfaces
        Returns
        -------
        list[ufl.Expr] Liste des flux numériques pour chaque équation de conservation
        """
        riemann_solvers_dic = default_Riemann_solver_parameters() 
        flux_type = riemann_solvers_dic.get("flux_type")
        if flux_type == "Cockburn":
            return [dot(flux_bar, self.n) + self.S * (x - x_bar) 
                    for flux_bar, x, x_bar in zip(Ubar_flux, self.U, self.Ubar)]
        elif flux_type == "Rusanov":
            from .riemann_solvers import RiemannSolvers
            riemann = RiemannSolvers(self.EOS, self.material)
            return riemann.rusanov_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLL":
            from .riemann_solvers import RiemannSolvers
            riemann = RiemannSolvers(self.EOS, self.material)
            return riemann.hll_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLLC":
            from .riemann_solvers import RiemannSolvers
            riemann = RiemannSolvers(self.EOS, self.material)
            return riemann.hllc_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        else:
            raise ValueError("Flux numérique inconnu")
    
    def numerical_flux_continuity_residual(self, flux_num):
        """
        Calcule le résidu assurant la continuité du flux numérique au sens faible
        Parameters
        ----------
        flux_num : Flux numérique
        """
        #TODO a changer ASAP. Ici on impose la continuité du flux numérique
        #sur toutes les frontières (de facet_mesh) y compris les frontières extérieures
        # puis on retire les frontières exterieures mais qui sont ici les frontières
        # extérieures du mesh. Il faudrait enlever les facettes exterieures directement
        # des facettes du submesh si on arrivait à les faire correspondre à leur alter-ego sur le mesh.
        continuity_residual = -sum(inner(f_num, test_func) * self.ds_tot 
                                    for f_num, test_func in zip(flux_num, self.Ubar_test))
        for tag in self.flag_list:
            continuity_residual += sum(inner(f_num, test_func) * self.ds_c(tag) 
                                    for f_num, test_func in zip(flux_num, self.Ubar_test))
            
        # continuity_residual = -sum(inner(f_num, test_func) * self.ds_int
        #                             for f_num, test_func in zip(flux_num, self.Ubar_test))
        return continuity_residual

    def surface_residual(self, flux_num):
        """
        Calcule le résidu surfacique issu de l'IPP de la forme faible sur chaque élément.
        Parameters
        ----------
        flux_num : Flux numérique
        """
        return sum(inner(f_num, test_func) * self.ds_tot 
                    for f_num, test_func in zip(flux_num, self.U_test))
    
    def total_surface_residual(self, U_flux, Ubar_flux):
        numerical_flux = self.set_numerical_flux(U_flux, Ubar_flux)
        surface_residual = self.surface_residual(numerical_flux)
        continuity_residual = self.numerical_flux_continuity_residual(numerical_flux)
        return surface_residual + continuity_residual

    def set_form(self):
        """
        Initialise le résidu total
        """
        U_flux = self.inviscid_flux(self.U)
        Ubar_flux = self.inviscid_flux(self.Ubar)
        vol_res = self.set_volume_residual(U_flux)
        surf_res = self.total_surface_residual(U_flux, Ubar_flux)
        boundary_res = self.bc_class.boundary_residual
        self.residual = vol_res + surf_res + boundary_res
        
        if self.analysis == "dynamic":        
            self.residual+= self.set_dynamic_lhs()
            self.residual-=self.set_dynamic_source()