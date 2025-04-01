"""
Created on Thu Jun 23 10:47:33 2022

@author: bouteillerp
"""
from ufl import div, inner
from dolfinx.fem import Expression
from ufl import dot, sqrt
from ..utils.generic_functions import extract_primitive_variables
from ..utils.default_parameters import default_viscosity_parameters

def npart(x):
    return (x - abs(x))/2

class EOS:
    def __init__(self, kinematic, quadrature):
        """
        Classe contenant toutes les équations d'état disponible du code Hydra.
        Le type d'équation d'état est entièrement déterminé lors de la création
        d'un objet de la classe Material.

        Parameters
        ----------
        kinematic : Objet de la classe Kinematic
        """
        self.kin = kinematic
        self.quad = quadrature
    
    def set_eos(self, U, mat):
        """
        Renvoie l'expression de la pression

        Returns
        -------
        p : Pression du modèle
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
        Renvoie l'expression de la célérité des ondes
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
        rho, u, _ = extract_primitive_variables(U)
        c = self.set_celerity(U, mat)
        coeff = default_viscosity_parameters().get("coefficient")
        # p_star = coeff * rho * h / (deg + 1) * (inner(u, u) + c**2)**0.5 * sensor.sensor_expr * npart(div(u))
        p_star = coeff * rho * h / (deg + 1) * (inner(u, u) + c**2)**0.5 * sensor.sensor_expr * div(u)
        return Expression(p_star, V_quad.element.interpolation_points())