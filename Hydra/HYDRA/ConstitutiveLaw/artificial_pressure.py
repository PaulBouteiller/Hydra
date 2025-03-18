"""
Created on Tue Mar 11 16:01:20 2025

@author: bouteillerp
"""
from ufl import div, inner
from dolfinx.fem import Function, Expression
def npart(x):
    return (x - abs(x))/2
        
class ArtificialPressure:
    def __init__(self, U, V_p, h, c, degree, sensor):
        """
        Initialise le capteur de choc.
        
        Parameters
        ----------
        problem : Objet de la classe Problem
            Le problème pour lequel le capteur de choc est défini.
        """
        self.U = U
        self.V_p = V_p
        self.h = h
        self.c = c
        self.deg = degree
        self.p_star = Function(V_p, name="ShockIndicator")
        self.set_artifial_pressure(sensor)
        
    def set_artifial_pressure(self, sensor):
        rho, u = self.U[0], self.U[1]/self.U[0]
        coeff = 1.5e-3
        p_star = coeff * rho * self.h / (self.deg+1) * (inner(u, u) + self.c**2)**0.5 * sensor.sensor_expr * npart(div(u))
        self.p_star_expr = Expression(p_star, self.V_p.element.interpolation_points())
        self.p_star.interpolate(self.p_star_expr)
    
    
    def compute_artificial_pressure(self):
        """
        Calcule l'indicateur de choc de Ducros.
        
        Returns
        -------
        Function L'indicateur de choc (1 où des chocs sont détectés, 0 ailleurs).
        """
        self.p_star.interpolate(self.p_star_expr)