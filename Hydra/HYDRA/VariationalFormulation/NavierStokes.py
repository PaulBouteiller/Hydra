"""
Created on Mon Jan 27 18:22:27 2025

@author: bouteillerp
"""
"""
Hybridizable Discontinuous Galerkin formulation for compressible Navier-Stokes equations
======================================================================================

This module extends the Euler equations implementation to the compressible Navier-Stokes
equations by adding viscous flux terms. It implements the HDG formulation for viscous
compressible flows, including velocity gradient reconstruction for viscous stress computation.

The formulation handles:
- Viscous stresses in momentum equations
- Heat conduction in energy equation
- Local velocity gradient reconstruction
- Modified boundary conditions for viscous flows
- Consistent treatment of diffusive and advective fluxes

The implementation inherits and extends the HDG formulation from the Euler module,
adding the necessary components for viscous terms while maintaining the structure
and advantages of the HDG approach.

Classes:
--------
NavierStokesBoundaryConditions : Boundary condition manager for Navier-Stokes
    Extends Euler boundary conditions with viscous-specific types
    Implements no-slip wall boundary conditions
    Handles thermal boundary conditions

CompressibleNavierStokes : HDG formulation for Navier-Stokes equations
    Extends CompressibleEuler with viscous terms
    Implements velocity gradient reconstruction
    Adds viscous fluxes to momentum and energy equations
    Provides consistent formulation for the complete system

Methods:
--------
set_function_space : Set up function spaces for Navier-Stokes
    Adds spaces for velocity gradient reconstruction

viscous_flux : Compute viscous flux terms
    Implements stress tensor computation
    Adds viscous contributions to momentum and energy

gradient_residual : Residual for velocity gradient reconstruction
    Implements the HDG formulation for gradient computation
    Ensures consistent gradient approximation
"""

from .Euler import CompressibleEuler, EulerBoundaryConditions
from dolfinx.fem import functionspace, Function
from ufl import (inner, dot, dev, div, MixedFunctionSpace, TestFunctions, TrialFunctions)
from petsc4py.PETSc import ScalarType

class NavierStokesBoundaryConditions(EulerBoundaryConditions):
    """
    Boundary condition manager for compressible Navier-Stokes equations.
    
    This class extends Euler boundary conditions with additional types
    specific to viscous flows, such as no-slip wall conditions and
    thermal boundary conditions.
    """
    def sticky_wall_residual(self, tag):
        """
        Apply no-slip wall boundary condition.
        
        Imposes zero velocity at the wall (no-slip condition), suitable
        for viscous flows where fluid adheres to solid boundaries.
        
        Parameters
        ----------
        tag : int Boundary tag for the wall
        """
        for i in range(self.tdim):
            self.add_component(self.V_vbar, i, tag, ScalarType(0))
        res_rho = - inner(self.U[0] - self.Ubar[0], self.Ubar_test[0]) * self.ds(tag)
        res_u = - inner(self.U[1] - self.Ubar[1], self.Ubar_test[1]) * self.ds(tag)
        res_E = - inner(self.U[2] - self.Ubar[2], self.Ubar_test[2]) * self.ds(tag)
        self.boundary_residual += res_rho + res_u + res_E

class CompressibleNavierStokes(CompressibleEuler):
    """
    HDG formulation for compressible Navier-Stokes equations.
    
    This class extends the Euler equations formulation to include viscous
    effects, implementing the compressible Navier-Stokes equations with
    the Hybridizable Discontinuous Galerkin method.
    
    The implementation adds:
    - Velocity gradient reconstruction
    - Viscous flux terms
    """
    def boundary_conditions_class(self):
        """
        Get the boundary condition class for Navier-Stokes equations.
        
        Returns
        -------
        class NavierStokesBoundaryConditions class
        """
        return NavierStokesBoundaryConditions
    
    def set_function_space(self):
        """
        Initialize function spaces for Navier-Stokes equations.
        
        Extends the Euler function spaces with additional space for
        velocity gradient reconstruction.
        """
        super().set_function_space()
        self.V_L = functionspace(self.mesh, ("DG", self.deg, (self.tdim ,self.tdim)))  
        
    def set_functions(self):   
        """
        Initialize solution variables for Navier-Stokes equations.
        
        Extends the Euler variables with additional variable for
        velocity gradient tensor.
        """
        super().set_functions() 
        self.L = Function(self.V_L, name = "Velocity_gradient")
        
    def set_variable_to_solve(self):
        """
        Set the variables to be solved for Navier-Stokes.
        
        Extends the Euler variables list with velocity gradient.
        """
        super().set_variable_to_solve()
        if self.iso_T:
            self.u_list.insert(2, self.L)
        else:
            self.u_list.insert(3, self.L)
        
    def set_test_functions(self):
        """
        Initialize test and trial functions for Navier-Stokes.
        
        Extends the Euler test functions with additional function
        for velocity gradient.
        """
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
        """
        Compute the total flux including both inviscid and viscous terms.
        
        Parameters
        ----------
        U : list of UFL expressions
            Conservative variables
            
        Returns
        -------
        list
            Total fluxes (inviscid + viscous) for each conservation equation
        """
        return self.inviscid_flux(U) + self.viscous_flux(U)
    
    def viscous_flux(self, U):
        """
        Compute the viscous flux terms.
        
        Implements the viscous stress tensor and heat conduction terms
        for the Navier-Stokes equations.
        
        Parameters
        ----------
        U : list of UFL expressions
            Conservative variables
            
        Returns
        -------
        list Viscous fluxes for each conservation equation:
            [0, -s, -s·v]
        """
        s = self.material.devia.mu * dev(self.L + self.L.T)
        v = U[1]/U[0]
        return [0, -s, -dot(s, v)]
    
    def gradient_residual(self):
        """
        Compute the residual for velocity gradient reconstruction.
        
        Implements the HDG formulation for reconstructing the velocity
        gradient tensor from the velocity field.
        
        Returns
        -------
        UFL form Velocity gradient reconstruction residual
        """
        u = self.U[1] / self.U[0]
        ubar = self.Ubar[1] / self.Ubar[0]
        vol_form = inner(self.L, self.L_) * self.dx_c
        vol_form += inner(u, div(self.L_)) * self.dx_c
        surf_form = -inner(ubar, dot(self.L_, self.n)) * self.ds_tot
        return vol_form + surf_form

    def set_form(self):
        """
        Set up the complete variational formulation for Navier-Stokes.
        
        Extends the Euler formulation with the velocity gradient
        reconstruction residual.
        """
        # Call Euler version for common part
        super().set_form()
        # Add gradient term specific to Navier-Stokes
        self.residual += self.gradient_residual()