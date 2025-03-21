"""
Created on Tue Mar 11 16:01:20 2025

@author: bouteillerp
"""
from ufl import div, inner
from dolfinx.fem import Function, Expression

def npart(x):
    return (x - abs(x))/2
        
class ArtificialPressure:
    """
    Implémentation de la pression artificielle pour la stabilisation des chocs.
    
    Cette classe ajoute un terme de pression artificielle dans les régions de choc
    détectées par un capteur, afin de stabiliser la solution numérique et éviter
    les oscillations non physiques. La pression artificielle est proportionnelle
    à la densité, la taille de maille locale, et la vitesse des ondes.
    
    La formule utilisée est:
    p* = coeff * rho * h / (deg+1) * sqrt(|u|² + c²) * sensor * npart(div(u))
    
    où npart(x) = (x - |x|)/2 est la partie négative de x, permettant d'appliquer
    la pression uniquement dans les zones de compression (div(u) < 0).
    
    Parameters
    ----------
    U : list[dolfinx.fem.Function] Variables conservatives du problème
    V_p : dolfinx.fem.FunctionSpace Espace fonctionnel pour la pression artificielle
    h : dolfinx.fem.Function Taille locale des éléments
    c : float Vitesse du son
    degree : int Degré des fonctions de base
    sensor : ShockSensor Capteur de choc pour la détection des discontinuités
    """
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
        # self.set_artifial_pressure(sensor)
        
    def set_artifial_pressure(self, sensor):
        rho, u = self.U[0], self.U[1]/self.U[0]
        coeff = 1.5e-3
        p_star = coeff * rho * self.h / (self.deg + 1) * (inner(u, u) + self.c**2)**0.5 * sensor.sensor_expr * npart(div(u))
        self.p_star_expr = Expression(p_star, self.V_p.element.interpolation_points())
        self.p_star.interpolate(self.p_star_expr)
    
    
    def compute_artificial_pressure(self):
        self.p_star.interpolate(self.p_star_expr)