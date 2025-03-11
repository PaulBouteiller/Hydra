"""
Created on Fri Mar  7 11:10:20 2025

@author: bouteillerp
"""

from ufl import div, curl, sqrt, conditional, ge, inner
from dolfinx.fem import Function, functionspace

class ShockSensor:
    """
    Classe abstraite pour les capteurs de choc dans HYDRA.
    Les capteurs de choc indiquent la présence de discontinuités.
    """
    def __init__(self, problem):
        """
        Initialise le capteur de choc.
        
        Parameters
        ----------
        problem : Objet de la classe Problem
            Le problème pour lequel le capteur de choc est défini.
        """
        self.problem = problem
        self.mesh = problem.mesh
        self.tdim = problem.tdim
        
        self.epsilon = 1e-10
        self.V_shock = functionspace(self.mesh, ("DG", problem.deg))
        self.shock_indicator = Function(self.V_shock, name="ShockIndicator")
        self.p_star = Function(self.V_shock, name="ShockIndicator")
        
    def compute_sensor_function(self):
        """
        Calcule la fonction de détection de choc.
        Méthode abstraite à implémenter dans les classes dérivées.
        """
        pass

class DucrosShockSensor(ShockSensor):
    """
    Implémentation du capteur de choc de Ducros.
    
    Le capteur de Ducros utilise le rapport entre la divergence du champ de vitesse
    et la somme de la divergence et du rotationnel pour détecter les chocs.
    
    Dans les régions de choc, la divergence est élevée (compression forte),
    alors que dans les régions de turbulence, le rotationnel est élevé.
    
    fs = {1 si |div(v)| / (|div(v)| + |curl(v)| + ε) >= seuil
         0 sinon}
    
    Références:
    ----------
    Ducros, F., Ferrand, V., Nicoud, F., Weber, C., Darracq, D., Gacherieu, C., & Poinsot, T. (1999).
    "Large-Eddy Simulation of the Shock/Turbulence Interaction."
    Journal of Computational Physics, 152(2), 517-549.
    """
    
    def __init__(self, problem, threshold=0.95):
        """
        Initialise le capteur de choc de Ducros.
        
        Parameters
        ----------
        problem : Objet de la classe Problem Le problème pour lequel le capteur de choc est défini.
        threshold : float, optional Le seuil au-delà duquel un choc est détecté. La valeur par défaut est 0.95.
        """
        super().__init__(problem)
        self.threshold = threshold
        self.set_sensor_function()

    def set_sensor_function(self):
        """
        Calcule l'indicateur de choc de Ducros.
        
        Returns
        -------
        Function L'indicateur de choc (1 où des chocs sont détectés, 0 ailleurs).
        """
        u = self.problem.u
        velocity_div = div(u)
        velocity_curl = curl(u)
        
        # Calculer la norme du rotationnel (le curl en 3D est un vecteur)
        # curl_norm = sqrt(inner(velocity_curl, velocity_curl)) if self.tdim == 3 else abs(velocity_curl)
        curl_norm = sqrt(inner(velocity_curl, velocity_curl))
        s_vort = abs(velocity_div) / (abs(velocity_div) + curl_norm + self.epsilon)

        self.sensor_expr = conditional(ge(s_vort, self.threshold), 1.0, 0.0)
        
class FernandezShockSensor(ShockSensor):
    def __init__(self, problem, threshold=0.95):
        """
        Initialise le capteur de choc de Fernandez.
        
        Parameters
        ----------
        problem : Objet de la classe Problem Le problème pour lequel le capteur de choc est défini.
        threshold : float, optional Le seuil au-delà duquel un choc est détecté. La valeur par défaut est 0.95.
        """
        super().__init__(problem)
        self.threshold = threshold
        
    def set_sensor_function(self):
        """
        Calcule l'indicateur de choc de Fernandez.
        
        Returns
        -------
        Function L'indicateur de choc (1 où des chocs sont détectés, 0 ailleurs).
        """
        u = self.problem.u
        k = self.problem.deg
        velocity_div = div(u)
        velocity_curl = curl(u)
        
        curl_norm = sqrt(inner(velocity_curl, velocity_curl))
        s_vort = abs(velocity_div) / (abs(velocity_div) + curl_norm + self.epsilon)
        h = self.problem.calculate_mesh_size()
        c = self.problem.material.celerity
        s_div = -h * velocity_div / (k * c)
        hat_s = s_div * s_vort

        self.sensor_expr = conditional(ge(hat_s, self.threshold), 1.0, 0.0)
