"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from .Euler import CompressibleEuler, EulerBoundaryConditions
from ufl import inner, dot, dev, div
from dolfinx.fem import functionspace, Function
from ufl import MixedFunctionSpace, TestFunctions, TrialFunctions
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
        rho_test, u_test, E_test, L_test, rhobar_test, ubar_test, Ebar_test = TestFunctions(MFS)
        drho, du, dE, dL, drhobar, dubar, dEbar = TrialFunctions(MFS)
        self.U_test = [rho_test, u_test, E_test]
        self.Ubar_test = [rhobar_test, ubar_test, Ebar_test]
        self.dU = [drho, du, dE]
        self.dUbar = [drhobar, dubar, dEbar]
        self.L_test = L_test
        self.dL = dL
        if self.iso_T:
            self.du_list = self.dU[:2] + [self.dL] + self.dUbar[:2]
            self.u_test_list = self.U_test[:2] + [self.L_test] + self.Ubar_test[:2]
        else:
            self.du_list = self.dU + [self.dL] + self.dUbar
            self.u_test_list = self.U_test + [self.L_test] + self.Ubar_test
    
    def flux(self, U):
        return self.inviscid_flux(U) + self.viscous_flux(U)
    
    def viscous_flux(self, U):
        s = self.material.devia.mu * dev(self.L + self.L.T)
        v = U[1]/U[0]
        return [0, -s, -dot(s, v)]
    
    def gradient_residual(self):
        u = self.U[1] / self.U[0]
        ubar = self.Ubar[1] / self.Ubar[0]
        vol_form = inner(self.L, self.L_test) * self.dx_c
        vol_form += inner(u, div(self.L_test)) * self.dx_c
        surf_form = -inner(ubar, dot(self.L_test, self.n)) * self.ds_tot
        return vol_form + surf_form

    def set_form(self):
        """
        Initialise le résidu
        """
        U_flux = self.flux(self.U)
        Ubar_flux = self.flux(self.Ubar)
        vol_res = self.set_volume_residual(U_flux)
        surf_res = self.total_surface_residual(U_flux, Ubar_flux)
        boundary_res = self.bc_class.boundary_residual
        gradient_res = self.gradient_residual()
        self.residual = vol_res + surf_res + boundary_res + gradient_res
        if self.analysis == "dynamic":
            self.residual += self.set_dynamic_residual()