"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
from .Euler import CompressibleEuler, EulerBoundaryConditions
from ufl import outer, Identity, dot, inner
from dolfinx.fem import functionspace, Function
from ufl import MixedFunctionSpace, TestFunctions, TrialFunctions, pi
from petsc4py.PETSc import ScalarType

class EulerIdealMHDBoundaryConditions(EulerBoundaryConditions):
    def __init__(self, U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag, dico_Vbar):
        self.V_Bbar = dico_Vbar.get("Magnetic")
        super().__init__(U, Ubar, Ubar_test, facet_mesh, S, n, ds, tdim, entity_maps, facet_tag, dico_Vbar)
    
    def wall_residual(self, tag, normal):
        super().wall_residual(tag, normal)
        if normal == "x":
            sub = 0
        elif normal == "y":
            sub = 1
        elif normal == "z":
            sub = 2      
        
        self.add_component(self.V_Bbar, sub, tag, ScalarType(0))
        res_B = - inner(self.U[3] - dot(self.U[3], self.n) * self.n 
                        - self.Ubar[3], self.Ubar_test[3]) * self.ds(tag)
        self.boundary_residual += res_B

class CompressibleMHDEuler(CompressibleEuler):
    def boundary_conditions_class(self):
        return EulerIdealMHDBoundaryConditions
        
    def set_function_space(self):  
        """
        Initialise les espaces fonctionnels
        """
        super().set_function_space()
        self.V_B = functionspace(self.mesh, ("Discontinuous Raviart-Thomas", self.deg + 1, (self.tdim ,)))
        self.V_Bbar = functionspace(self.facet_mesh, ("DG", self.deg, (self.tdim ,)))

       
    def set_functions(self):   
        """ 
        Initialise les champs inconnues du problème densité, vitesse, énergie
        """       
        super().set_functions()
        self.B = Function(self.V_B, name = "Magnetic_field")
        # self.B.x.petsc_vec.set(1e-11)
        self.U.append(self.B)
        self.U_base.append(self.B)
        self.B_n = Function(self.V_B)
        # self.B_n.x.petsc_vec.set(1e-11)
        self.U_n.append(self.B_n)
        self.Un_base.append(self.B_n)
        self.Bbar = Function(self.V_Bbar)
        # self.Bbar.x.petsc_vec.set(1e-11)
        self.Ubar.append(self.Bbar)
        self.Ubar_base.append(self.Bbar)
        self.dico_Vbar.update({"Magnetic" : self.V_Bbar})
        
    def set_variable_to_solve(self):
        if self.iso_T:
            self.u_list = self.U_base[:2] + [self.U_base[3]] + self.Ubar_base[:2] + [self.Ubar_base[3]]
        else:
            self.u_list = self.U_base + self.Ubar_base
        
    def set_test_functions(self):
        """
        Initialise les fonctions test et d'essai.
        """
        MFS = MixedFunctionSpace(self.V_rho, self.V_v, self.V_E, self.V_B,
                                 self.V_rhobar, self.V_vbar, self.V_Ebar, self.V_Bbar)  
        rho_test, u_test, E_test, B_test, rhobar_test, ubar_test, Ebar_test, Bbar_test = TestFunctions(MFS)
        drho, du, dE, dB, drhobar, dubar, dEbar, dBbar = TrialFunctions(MFS)
        self.U_test = [rho_test, u_test, E_test, B_test]
        self.Ubar_test = [rhobar_test, ubar_test, Ebar_test, Bbar_test]
        self.dU = [drho, du, dE, dB]
        self.dUbar = [drhobar, dubar, dEbar, dBbar]
        if self.iso_T:
            self.du_list = self.dU[:2] + [self.dU[3]] + self.dUbar[:2] + [self.dUbar[3]]
            self.u_test_list = self.U_test[:2] + [self.U_test[3]] + self.Ubar_test[:2] + [self.Ubar_test[3]]
        else:
            self.du_list = self.dU + self.dUbar
            self.u_test_list = self.U_test + self.Ubar_test
    
    def set_inviscid_flux(self, U):
        """
        Défini les flux non visqueux pour la MHD
        Parameters
        ----------
        U : Vecteur état (\rho, \rho u, \rho E, B)
        Returns
        -------
        list Liste des flux non visqueux
        """
        rho, u, B = U[0], U[1]/U[0], U[3]
        p = self.EOS.set_eos(self.material.rho_0/rho, self.material)
        sig_mag = self.magnetic_stress(B)
        return [self.mass_flux(U),
                self.momentum_flux(rho, u, p) - sig_mag,
                self.energy_flux(U, p) - dot(sig_mag, u),
                (outer(B, u) - outer(u, B))]
    
    def magnetic_stress(self, B):
        """Renvoie la contrainte magnétique"""
        mu0 = 4 * pi * 1e-7 #g.mm/ms²/A²
        return (outer(B, B) - 1./2 * dot(B, B) * Identity(self.tdim)) / mu0 
                
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