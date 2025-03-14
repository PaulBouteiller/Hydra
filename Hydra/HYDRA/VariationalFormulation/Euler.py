"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from .Problem import Problem, BoundaryConditions
from ..utils.generic_functions import euler_eigenvalues_2D,  max_abs_of_sequence
from ufl import outer, grad, Identity, inner, dot, div, dev, sym
from ufl import MixedFunctionSpace, TestFunctions, TrialFunctions
from petsc4py.PETSc import ScalarType
from dolfinx.fem import functionspace, Function


class EulerBoundaryConditions(BoundaryConditions):
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag, dico_Vbar):
        self.V_vbar = dico_Vbar.get("Velocity")
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

class CompressibleEuler(Problem):
    """ Classe implémentant les équations d'Euler compressibles avec la méthode HDG.
    
     Cette classe définit la formulation variationnelle pour les équations d'Euler
     compressibles, incluant les flux numériques et les termes de stabilisation."""

    def set_stabilization_matrix(self, u, ubar, c, n):
        """
        Local Lax-Friedrichs (Rusanov) stabilization matrix 
        """
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
        self.V_v = functionspace(self.mesh, ("DG", self.deg, (self.tdim ,)))
        self.V_E = self.V_rho
        self.V_rhobar = functionspace(self.facet_mesh, ("DG", self.deg))
        self.V_vbar = functionspace(self.facet_mesh, ("DG", self.deg, (self.tdim ,)))
        self.V_Ebar = self.V_rhobar
       
    def set_functions(self):   
        """ 
        Initialise les champs inconnues du problème densité, vitesse, énergie
        """        
        self.rho = Function(self.V_rho, name = "Density")
        self.u = Function(self.V_v, name = "Velocity")   
        self.rho_n, self.u_n = Function(self.V_rho), Function(self.V_v)

        self.rhobar, self.ubar = Function(self.V_rhobar), Function(self.V_vbar)
        self.E = Function(self.V_rho, name = "Energy")
        self.E_n = Function(self.V_rho)
        self.Ebar = Function(self.V_rhobar)

        self.rho.x.petsc_vec.set(self.material.rho_0)
        self.rho_n.x.petsc_vec.set(self.material.rho_0)
        self.rhobar.x.petsc_vec.set(self.material.rho_0)    
        
        self.U_base = [self.rho, self.u, self.E]
        self.Ubar_base = [self.rhobar, self.ubar, self.Ebar]
        self.Un_base = [self.rho_n, self.u_n, self.E_n]
        
        self.U = [self.rho, self.rho * self.u, self.rho * self.E]
        self.Ubar = [self.rhobar, self.rhobar * self.ubar, self.rhobar * self.Ebar]
        self.U_n = [self.rho_n, self.rho_n * self.u_n, self.rho_n * self.E_n]
        
        self.dico_Vbar = {"Velocity" : self.V_vbar}
        
    def set_variable_to_solve(self):
        if self.iso_T:
            self.u_list = self.U_base[:2] + self.Ubar_base[:2]
        else:
            self.u_list = self.U_base + self.Ubar_base
        
    def set_test_functions(self):
        """
        Initialise les fonctions test et d'essai.
        """
        MFS = MixedFunctionSpace(self.V_rho, self.V_v, self.V_E,
                                 self.V_rhobar, self.V_vbar, self.V_Ebar)  
        rho_test, u_test, E_test, rhobar_test, ubar_test, Ebar_test = TestFunctions(MFS)
        drho, du, dE, drhobar, dubar, dEbar = TrialFunctions(MFS)
        self.U_test = [rho_test, u_test, E_test]
        self.Ubar_test = [rhobar_test, ubar_test, Ebar_test]
        self.dU = [drho, du, dE]
        self.dUbar = [drhobar, dubar, dEbar]
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
        rho, u, E = U[0], U[1]/U[0], U[2]/U[0]
        p = self.EOS.set_eos(rho, u, E, self.material)
        if self.use_shock_capturing:
            pass
            # p+= self.artificial_pressure.p_star
        return [self.mass_flux(U),
                self.momentum_flux(rho, u, p),
                self.energy_flux(U, p)]
    
    def mass_flux(self, U):
        """Renvoie le flux de masse ρu"""
        return U[1]
    
    def momentum_flux(self, rho, u, p):
        """Renvoie le flux de quantité de mouvement ρ u⊗u + pI """
        return rho * outer(u, u) + p * Identity(self.tdim)
    
    def energy_flux(self, U, p):
        """Renvoie le flux d'energie massique (ρE + p)u"""
        return (U[2] + p) * U[1]/U[0]
    
    def set_dynamic_residual(self):
        """Renvoie le résidu associé à la dynamique implicite avec facteur de temps ajustable"""
        return sum(inner((x - x_n) * self.dt_factor, x_test) * self.dx_c 
                   for x, x_n, x_test in zip(self.U, self.U_n, self.U_test))
    
    def set_volume_residual(self, U_flux):
        """Renvoie le résidu volumique"""
        return -sum(inner(flux, grad(test_func)) * self.dx_c 
                    for flux, test_func in zip(U_flux, self.U_test))        
        
    def set_numerical_flux(self, U_flux, Ubar_flux, flux_type="HLL"):
        """
        Calcule le flux numérique associé à chacune des équations de conservations
    
        Parameters
        ----------
        U_flux : Flux associés aux variables U dans l'élément
        Ubar_flux : Flux associés aux variables Ubar au bord de l'élément
        flux_type : string, optional le type de flux numérique.
                    Options: "Cockburn", "LF", "Rusanov", "HLL", "HLLC"
    
        Returns
        -------
        list la liste des flux numériques
        """
        if flux_type == "Cockburn":
            return [dot(flux_bar, self.n) + self.S * (x - x_bar) 
                    for flux_bar, x, x_bar in zip(Ubar_flux, self.U, self.Ubar)]
        elif flux_type == "LF":
            #Doit être redondant avec Rusanov
            return [dot(1./2 * (flux_bar + flux), self.n) + self.S * (x - x_bar) 
                    for flux_bar, flux, x, x_bar in zip(Ubar_flux, U_flux, self.U, self.Ubar)]
        elif flux_type == "Rusanov":
            from .riemann_solvers import RiemannSolvers
            riemann = RiemannSolvers(self.EOS, self.material)
            return riemann.rusanov_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLL":
            from .riemann_solvers import RiemannSolvers
            riemann = RiemannSolvers(self.EOS, self.material)
            return riemann.hll_flux(self.U, self.Ubar, U_flux, Ubar_flux, self.n)
        elif flux_type == "HLLC":
            raise ValueError("Currently buggued")
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
            self.residual += self.set_dynamic_residual()