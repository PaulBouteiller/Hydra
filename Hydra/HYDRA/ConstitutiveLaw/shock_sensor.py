"""
Created on Fri Mar  7 11:10:20 2025

@author: bouteillerp
"""

from ufl import div, curl, sqrt, conditional, ge, abs, inner
from dolfinx.fem import Function, functionspace
import numpy as np

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
        
        # Seuil pour éviter la division par zéro
        self.epsilon = 1e-10
        
        # Créer l'espace pour stocker l'indicateur de choc
        self.V_shock = functionspace(self.mesh, ("DG", problem.deg))
        self.shock_indicator = Function(self.V_shock, name="ShockIndicator")
        
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
        problem : Objet de la classe Problem
            Le problème pour lequel le capteur de choc est défini.
        threshold : float, optional
            Le seuil au-delà duquel un choc est détecté. La valeur par défaut est 0.95.
        """
        super().__init__(problem)
        self.threshold = threshold
    
    def compute_sensor_function(self):
        """
        Calcule l'indicateur de choc de Ducros.
        
        Returns
        -------
        Function L'indicateur de choc (1 où des chocs sont détectés, 0 ailleurs).
        """
        # Obtenir le champ de vitesse du problème
        u = self.problem.u
        # Calculer la divergence et le rotationnel du champ de vitesse
        velocity_div = div(u)
        velocity_curl = curl(u)
        
        # Calculer la norme du rotationnel (le curl en 3D est un vecteur)
        curl_norm = sqrt(inner(velocity_curl, velocity_curl)) if self.tdim == 3 else abs(velocity_curl)
        
        # Calculer l'indicateur de choc de Ducros
        sensor_expr = conditional(
            ge(abs(velocity_div) / (abs(velocity_div) + curl_norm + self.epsilon), self.threshold),
            1.0, 0.0
        )
        
        # Mettre à jour l'indicateur de choc
        self.shock_indicator.interpolate(sensor_expr)
        
        return self.shock_indicator
    
    def apply_shock_capturing(self, mu_artificial=1.0):
        """
        Applique une viscosité artificielle basée sur l'indicateur de choc.
        
        Parameters
        ----------
        mu_artificial : float, optional
            Coefficient de viscosité artificielle. La valeur par défaut est 1.0.
        
        Returns
        -------
        Function La viscosité artificielle à utiliser dans le schéma numérique.
        """
        mu_shock = Function(self.V_shock, name="ShockViscosity")
        
        # Calculer l'indicateur de choc si ce n'est pas déjà fait
        if np.all(self.shock_indicator.x.array == 0):
            self.compute_sensor_function()
        
        # Définir la viscosité artificielle en fonction de l'indicateur de choc
        # mu_shock = mu_artificial * shock_indicator
        mu_shock.x.array[:] = mu_artificial * self.shock_indicator.x.array
        
        return mu_shock