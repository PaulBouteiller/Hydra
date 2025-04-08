"""
Created on Thu Jun 23 10:47:33 2022

@author: bouteillerp
"""

"""
Equation of State (EOS) models for compressible fluid dynamics
==============================================================

This module provides implementations of various equations of state (EOS) for compressible
fluid dynamics simulations. The EOS models define the relationship between thermodynamic
variables such as pressure, density, internal energy, and temperature.

The module includes:
- U1 EOS: Single-parameter hyperelastic equation of state
- GP EOS: Ideal gas equation of state (gamma-law)

The EOS class serves as the central interface for all equation of state calculations,
including:
- Pressure computation based on conservative variables
- Wave speed (sound speed) calculation
- Artificial viscosity pressure for shock stabilization

Classes:
--------
EOS : Main class for equation of state calculations
    Handles selection and application of specific EOS models
    Provides methods for pressure and wave speed calculations
    Implements artificial viscosity pressure for shock capturing

Notes:
------
The implementation supports different EOS models with a unified interface.
Additional EOS models can be added by extending the set_eos and set_celerity methods.
"""

from ufl import div, inner
from dolfinx.fem import Expression
from ufl import dot, sqrt
from ..utils.generic_functions import extract_primitive_variables
from ..utils.default_parameters import default_viscosity_parameters

def npart(x):
    """
    Negative part function for scalar or array-like input.
    
    Computes the negative part of the input defined as (x - |x|)/2, which returns:
    - 0 if x ≥ 0
    - x if x < 0 (i.e., the negative part of x)
    
    This function is used in artificial viscosity formulations to apply
    dissipation only in compression regions (where div(u) < 0).
    """
    return (x - abs(x))/2

class EOS:
    def __init__(self, kinematic, quadrature):
        """
        Initialize the equation of state (EOS) manager.
        
        Creates an object that handles various equation of state models available
        in the HYDRA code. The specific EOS type is determined during the creation
        of a Material object.
        
        Parameters
        ----------
        kinematic : Kinematic Object containing kinematic model parameters
        quadrature : QuadratureScheme Quadrature scheme used for numerical integration
        """
        self.kin = kinematic
        self.quad = quadrature
    
    def set_eos(self, U, mat):
        """
        Calculate pressure based on the equation of state.
        
        Computes the pressure from conservative variables according to the
        specified material's equation of state. Supports multiple EOS types:
        - U1: Single-parameter hyperelastic model
        - GP: Ideal gas (gamma-law) model
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables [rho, rho*u, rho*E]
        mat : Material Material object containing EOS parameters
            
        Returns
        -------
        UFL expression
            Pressure field computed according to the material's EOS
        """
        eos_param = mat.eos
        rho, u, E = extract_primitive_variables(U)
        if mat.eos_type == "U1":
            p = -eos_param.kappa * (mat.rho_0/rho - 1)
        elif mat.eos_type == "GP":
            p = (eos_param.gamma - 1) * rho * (E - 1./2 * dot(u, u))
        else:
            raise ValueError("Unknwon eos")
        return p
    
    def set_celerity(self, U, mat):
        """
        Calculate the speed of sound (wave celerity).
        
        Computes the speed of sound based on the material's equation of state
        and the current state of the fluid.
        
        Parameters
        ----------
        see the method set_eos
            
        Returns
        -------
        UFL expression Sound speed field computed according to the material's EOS
        """
        rho, u, E = extract_primitive_variables(U)
        eos_param = mat.eos
        if mat.eos_type == "U1":
            mat.celerity
        elif mat.eos_type == "GP":
            c = sqrt(eos_param.gamma * (eos_param.gamma - 1) * (E - 1./2 * dot(u, u)))
        else:
            raise ValueError("Unknwon eos")
        return c
    
    def set_artifial_pressure(self, U, V_quad, mat, h, deg, sensor):
        """
        Calculate artificial pressure for shock stabilization.
        
        Computes an artificial pressure term used for shock capturing based on
        the velocity divergence and a shock sensor. This is typically used to
        add localized artificial viscosity near discontinuities.
        
        Parameters
        ----------
        U : list of UFL expressions Conservative variables [rho, rho*u, rho*E]
        V_quad : FunctionSpace Quadrature function space for evaluating the expression
        mat : Material Material object containing properties
        h : UFL expression Local mesh size
        deg : int Polynomial degree of the approximation
        sensor : ShockSensor Shock sensor object providing the shock indicator function
            
        Returns
        -------
        Expression Artificial pressure term for shock stabilization
        """
        rho, u, _ = extract_primitive_variables(U)
        c = self.set_celerity(U, mat)
        coeff = default_viscosity_parameters().get("coefficient")
        # p_star = coeff * rho * h / (deg + 1) * (inner(u, u) + c**2)**0.5 * sensor.sensor_expr * npart(div(u))
        p_star = coeff * rho * h / (deg + 1) * (inner(u, u) + c**2)**0.5 * sensor.sensor_expr * div(u)
        return Expression(p_star, V_quad.element.interpolation_points())