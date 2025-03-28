"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from .Euler import CompressibleEuler, EulerBoundaryConditions
from dolfinx.fem import functionspace, Function
from ufl import (inner, dot, dev, div, MixedFunctionSpace, TestFunctions, TrialFunctions)
from petsc4py.PETSc import ScalarType

class NavierStokesBoundaryConditions(EulerBoundaryConditions):
    def sticky_wall_residual(self, tag):
        for i in range(self.tdim):
            self.add_component(self.V_vbar, i, tag, ScalarType(0))
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = - inner(self.U[1] - self.Ubar[1], self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E

class CompressibleNavierStokes(CompressibleEuler):
    def boundary_conditions_class(self):
        return NavierStokesBoundaryConditions
    
    def set_function_space(self):
        """
        Initialise les espaces fonctionnels
        """
        super().set_function_space()
        self.V_L = functionspace(self.mesh, ("DG", self.deg, (self.tdim ,self.tdim)))  
        
    def set_functions(self):   
        """ 
        Initialise les champs inconnues du problème thermo-mécanique
        """
        super().set_functions() 
        self.L = Function(self.V_L, name = "Velocity_gradient")
        
    def set_variable_to_solve(self):
        super().set_variable_to_solve()
        if self.iso_T:
            self.u_list.insert(2, self.L)
        else:
            self.u_list.insert(3, self.L)
        
    def set_test_functions(self):
        MFS = MixedFunctionSpace(self.V_rho, self.V_v, self.V_E, self.V_L, 
                                 self.V_rhobar, self.V_vbar, self.V_Ebar)  
        rho_, rhou_, rhoE_, L_, rhobar_, rhoubar_, rhoEbar_ = TestFunctions(MFS)
        drho, drhou, drhoE, dL, drhobar, drhoubar, drhoEbar = TrialFunctions(MFS)
        self.U_ = [rho_, rhou_, rhoE_]
        self.Ubar_ = [rhobar_, rhoubar_, rhoEbar_]
        self.dU = [drho, drhou, drhoE]
        self.dUbar = [drhobar, drhoubar, drhoEbar]
        self.L_ = L_
        self.dL = dL
        if self.iso_T:
            self.du_list = self.dU[:2] + [self.dL] + self.dUbar[:2]
            self.u_test_list = self.U_test[:2] + [self.L_] + self.Ubar_test[:2]
        else:
            self.du_list = self.dU + [self.dL] + self.dUbar
            self.u_test_list = self.U_test + [self.L_] + self.Ubar_test
    
    def flux(self, U):
        return self.inviscid_flux(U) + self.viscous_flux(U)
    
    def viscous_flux(self, U):
        s = self.material.devia.mu * dev(self.L + self.L.T)
        v = U[1]/U[0]
        return [0, -s, -dot(s, v)]
    
    def gradient_residual(self):
        u = self.U[1] / self.U[0]
        ubar = self.Ubar[1] / self.Ubar[0]
        vol_form = inner(self.L, self.L_) * self.dx_c
        vol_form += inner(u, div(self.L_)) * self.dx_c
        surf_form = -inner(ubar, dot(self.L_, self.n)) * self.ds_tot
        return vol_form + surf_form

    def set_form(self):
        """Initialise le résidu en réutilisant la logique de la classe parente"""
        # Appeler la version d'Euler pour la partie commune
        super().set_form()
        # Ajouter le terme de gradient spécifique à Navier-Stokes
        self.residual += self.gradient_residual()