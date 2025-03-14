
"""
Created on Tue Mar 11 16:30:20 2025

@author: bouteillerp
"""
import numpy as np

class DIRKParameters:
    """
    Classe définissant les paramètres d'un schéma DIRK.
    Stocke les coefficients de Butcher et fournit des méthodes utilitaires.
    """
    def __init__(self, method="SDIRK2"):
        """
        Initialise les paramètres DIRK pour la méthode spécifiée.
        
        Parameters
        ----------
        method : str, optional
            Nom du schéma DIRK à utiliser. Options: "SDIRK2", "SDIRK3", "ESDIRK3", "ESDIRK4".
            Par défaut "SDIRK2".
        """
        if method == "SDIRK2":
            # SDIRK d'ordre 2 avec gamma = 1 - 1/sqrt(2)
            gamma = 1.0 - 1.0/np.sqrt(2.0)
            self.A = np.array([[gamma, 0.0], 
                               [1.0-gamma, gamma]])
            self.c = np.array([gamma, 1.0])
            self.b = np.array([1.0-gamma, gamma])
            self.order = 2
        elif method == "SDIRK3":
            # SDIRK d'ordre 3
            gamma = 0.4358665215  # Racine de x³ - 3x² + 3x - 1/2 = 0
            self.A = np.array([
                [gamma, 0.0, 0.0],
                [(1.0-gamma)/2.0, gamma, 0.0],
                [1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma]
            ])
            self.c = np.array([gamma, (1.0+gamma)/2.0, 1.0])
            self.b = np.array([1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma])
            self.order = 3
        elif method == "ESDIRK3":
            # ESDIRK d'ordre 3 (Explicit first stage)
            gamma = 0.435866521508
            self.A = np.array([
                [0.0, 0.0, 0.0],
                [0.87173304301691, gamma, 0.0],
                [0.84457060015369, -0.12990812375553, gamma]
            ])
            self.c = np.array([0.0, 0.87173304301691 + gamma, 0.84457060015369 - 0.12990812375553 + gamma])
            self.b = np.array([0.84457060015369, -0.12990812375553, gamma])
            self.order = 3
        elif method == "ESDIRK4":
            # ESDIRK d'ordre 4 (méthode L-stable)
            gamma = 0.25
            self.A = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.25, 0.0, 0.0, 0.0],
                [0.17, 0.5-gamma-0.17, gamma, 0.0, 0.0],
                [0.39, 0.25-0.39, 0.45-gamma, gamma, 0.0],
                [0.15, 0.2-0.15, 0.6-gamma-0.2, 0.25, gamma]
            ])
            self.c = np.array([0.0, 0.75, 0.5, 0.75, 1.0])
            self.b = np.array([0.15, 0.2-0.15, 0.6-gamma-0.2, 0.25, gamma])
            self.order = 4
        else:
            raise ValueError(f"Méthode DIRK inconnue: {method}")
        
        self.num_stages = len(self.b)
        self.method = method
    
    def __str__(self):
        """Représentation sous forme de chaîne de caractères"""
        return f"DIRK method: {self.method}, Order: {self.order}, Stages: {self.num_stages}"