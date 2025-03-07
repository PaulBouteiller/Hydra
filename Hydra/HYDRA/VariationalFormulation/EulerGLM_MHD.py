"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from .Euler import CompressibleEuler
from ufl import outer, grad, Identity, inner, dot
from dolfinx.fem import functionspace, Function
from ufl import MixedFunctionSpace, TestFunctions, TrialFunctions, curl, cross

class CompressibleMHDEuler(CompressibleMHDEuler):
    def set_function_space(self):  
        """
        Initialise les espaces fonctionnels
        """
        CompressibleEuler().set_function_space()
        self.V_B = functionspace(self.mesh, ("DGs", self.deg, (self.tdim ,)))
        self.V_Bbar = functionspace(self.facet_mesh, ("DG", self.deg, (self.tdim ,)))
       
    def set_functions(self):   

        """ 
        Initialise les champs inconnues du problème thermo-mécanique
        """        
        super().set_functions()
        
    def set_test_functions(self):
        MFS = MixedFunctionSpace(self.V_rho, self.V_v, self.V_E,
                                      self.V_rhobar, self.V_vbar, self.V_Ebar)  
        rho_test, u_test, E_test, rhobar_test, ubar_test, Ebar_test = TestFunctions(MFS)
        drho, du, dE, drhobar, dubar, dEbar = TrialFunctions(MFS)
        self.U_test = [rho_test, u_test, E_test]
        self.Ubar_test = [rhobar_test, ubar_test, Ebar_test]
        self.dU = [drho, du, dE]
        self.dUbar = [drhobar, dubar, dEbar]
    
    
    def set_inviscid_flux(self, U):
        rho, u = U[0], U[1]/U[0]
        p = self.EOS.set_eos(self.material.rho_0/rho, self.material)
        return [self.mass_flux(U),
                self.momentum_flux(rho, u, p),
                self.energy_flux(U, p)]
    
    def mass_flux(self, U):
        return U[1]
    
    def momentum_flux(self, rho, u, p):
        return rho * outer(u, u) + p * Identity(self.tdim)
    
    def energy_flux(self, U, p):
        return (U[2] + p) * U[1]/U[0]
    
    def set_dynamic_residual(self):
        return sum(inner((x - x_n)/self.dt, x_test) * self.dx_c 
                   for x, x_n, x_test in zip(self.U, self.U_n, self.U_test))
    
    def set_volume_residual(self, U_flux):
        return -sum(inner(flux, grad(test_func)) * self.dx_c 
                    for flux, test_func in zip(U_flux, self.U_test))
    
    def set_numerical_flux(self, U_flux, Ubar_flux, flux_type = "Cockburn"):
        if flux_type == "Cockburn":
            return [dot(flux_bar, self.n) + self.S * (x - x_bar) 
                    for flux_bar, x, x_bar in zip(Ubar_flux, self.U, self.Ubar)]
        elif flux_type == "LF":
            return [dot(1./2 * (flux_bar + flux), self.n) + self.S * (x - x_bar) 
                    for flux_bar, flux, x, x_bar in zip(Ubar_flux, U_flux, self.U, self.Ubar)]
    
    def numerical_flux_continuity_residual(self, flux_num):
        #TODO a changer ASAP. Ici on impose la continuité du flux numérique
        #sur toutes les frontières (de facet_mesh) y compris les frontières extérieures
        # puis on retire les frontières exterieures mais qui sont ici les frontières
        # extérieures du mesh. Il faudrait enlever les facettes exterieures directement
        # des facettes du submesh si on arrivait à les faire correspondre
        continuity_residual = -sum(inner(f_num, test_func) * self.ds_tot 
                                   for f_num, test_func in zip(flux_num, self.Ubar_test))
        for tag in self.flag_list:
            continuity_residual += sum(inner(f_num, test_func) * self.ds_c(tag) 
                                    for f_num, test_func in zip(flux_num, self.Ubar_test))
        return continuity_residual

    def surface_residual(self, flux_num):
        return sum(inner(f_num, test_func) * self.ds_tot 
                    for f_num, test_func in zip(flux_num, self.U_test))
    
    def total_surface_residual(self, U_flux, Ubar_flux):
        numerical_flux = self.set_numerical_flux(U_flux, Ubar_flux)
        surface_residual = self.surface_residual(numerical_flux)
        continuity_residual = self.numerical_flux_continuity_residual(numerical_flux)
        return surface_residual + continuity_residual
    
    def maxwell_residual(self):
        #Il s'agit de la version residuelle de rot H = j
        mu = self.material.emag.mu
        sigma = self.material.emag.sigma
        v = self.U[1]/self.U[0]
        a_44 = inner(sigma * self.A / self.dt, self.A_test) * self.dx_c \
                + inner(1 / mu * curl(self.A), curl(self.A_test)) * self.dx_c \
                + inner(sigma * cross(curl(self.A), v), self.phi) * self.dx_c
                
    def exterior_magnetic_field_residual(self):
        pass
    
    def momentum_additional_magnetic_term(self):
        a_04 = inner(sigma * A / delta_t, cross(curl(A_n), v)) * dx_c
                        - inner(sigma * cross(u_n, curl(A)), cross(curl(A_n), v)) * dx_c
                        + inner(sigma * A / delta_t, cross(B_0, v)) * dx_c,
                        entity_maps={msh: sm_f_to_msh})
        L_0 = (inner(sigma * A_n / delta_t, cross(curl(A_n), v)) * dx_c
               + inner(sigma * A_n / delta_t, cross(B_0, v)) * dx_c
               + inner(sigma * cross(u_n, B_0), cross(B_0, v)) * dx_c)

    def set_form(self):
        """
        Initialise le résidu
        """
        U_flux = self.set_inviscid_flux(self.U)
        Ubar_flux = self.set_inviscid_flux(self.Ubar)
        vol_res = self.set_volume_residual(U_flux)
        surf_res = self.total_surface_residual(U_flux, Ubar_flux)
        boundary_res = self.bc_class.boundary_residual
        self.residual = vol_res + surf_res + boundary_res
        if self.analysis == "dynamic":
            self.residual += self.set_dynamic_residual()