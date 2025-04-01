import numpy as np
from numpy import sqrt, array
""" Pour plus de référence, voir la page de Sundials par le LNAL
https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#implicit-butcher-tables

Allez voir l'ouvrage: Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems'
"""
class DIRKParameters:
    """
    Classe définissant les paramètres d'un schéma DIRK (Diagonally Implicit Runge-Kutta).
    Stocke les coefficients de Butcher et fournit des méthodes utilitaires pour l'intégration
    temporelle des équations différentielles ordinaires, particulièrement pour les problèmes raides.

    Cette classe implémente diverses méthodes DIRK optimisées pour différentes propriétés
    de stabilité et précision, incluant des méthodes A-stables, L-stables, et algébriquement stables
    (BN-stables).
    """
    def __init__(self, method):
        """
        Initialise les paramètres DIRK pour la méthode spécifiée.
        
        Parameters
        ----------
        method : str, optional
            Nom du schéma DIRK à utiliser:
            
            Méthodes d'ordre 1:
            - "BDF1" : Backward Euler (L-stable)
            
            Méthodes d'ordre 2:
            - "SDIRK2" : SDIRK classique avec γ = 1-1/√2 (A-stable)
            - "SDIRK212" : TR-BDF2 (A-stable et B-stable), avec estimateur d'ordre 1
            
            Méthodes d'ordre 3:
            - "SDIRK3" : SDIRK classique (A-stable et L-stable)
            
            Méthodes d'ordre 4:
            - "SDIRK4" : Hairer & Wanner, 5 étages (L-stable)
            - "SDIRK4-2" : Variante de Hairer & Wanner avec γ=(2-√2)/2
            - "SDIRK4_BN_Stable_1_0" : Wang et al. 2024, algébriquement stable avec ρ∞=1.0
            - "SDIRK4_BN_Stable_0_0" : Wang et al. 2024, algébriquement stable avec ρ∞=0.0 (L-stable) 
            - "SDIRK534" : Méthode ARKode 5 étages (L-stable) avec estimateur d'ordre 3
            
            Méthodes d'ordre 5:
            - "SDIRK5" : Cooper & Sayfy, 6 étages (L-stable)
            - "SDIRK5-2" : Hairer & Wanner, 7 étages (L-stable)
            
        Références
        ----------
        [1] Hairer, E. & Wanner, G. (1996). "Solving Ordinary Differential Equations II: Stiff 
            and Differential-Algebraic Problems." Springer-Verlag.
        [2] Cooper, G. J., & Sayfy, A. (1983). "Additive methods for the numerical solution of 
            ordinary differential equations." Mathematics of Computation, 40(162), 559-571.
        [3] Wang, Y., Xue, X., Tamma, K.K., & Adams, N.A. (2024). "Algebraically stable SDIRK 
            methods with controllable numerical dissipation for first/second-order time-dependent 
            problems." Journal of Computational Physics, 508, 113032.
        [4] Carpenter, M.H., Kennedy, C.A., et al. "ARKode: A library of Runge-Kutta Methods for 
            ODEs with Support for Additive Methods", https://sundials.readthedocs.io/

        Notes
        -----
        Terminologie:
        - A-stable: Stabilité dans tout le demi-plan gauche (Re(z) < 0)
        - L-stable: A-stable + lim(z→∞) R(z) = 0 (dissipation totale des hautes fréquences)
        - B-stable: Stabilité pour problèmes non linéaires dissipatifs
        - BN-stable/Algébriquement stable: Stabilité pour problèmes non linéaires sans réduction d'ordre
        
        ρ∞ (rho-infinity) est le facteur d'amplification à l'infini, défini comme:
        ρ∞ = lim(Ω→∞) L(Ω), qui contrôle la dissipation numérique:
        - ρ∞ = 1.0: Aucune dissipation (plus précis mais moins robuste)
        - ρ∞ = 0.0: Dissipation maximale (L-stable, robuste pour problèmes raides)
        - 0 < ρ∞ < 1: Dissipation intermédiaire contrôlable
        """
        # Méthodes DIRK traditionnelles
        if method == "BDF1":
            # BDF1 = Backward Euler (schéma d'ordre 1)
            gamma = 1.0
            self.A = np.array([[gamma]])
            self.c = np.array([gamma])
            self.b = np.array([gamma])
            self.order = 1
        
        elif method == "SDIRK2":
            # SDIRK d'ordre 2 avec gamma = 1 - 1/sqrt(2)
            gamma = 1.0 - 1.0/sqrt(2.0)
            self.A = array([[gamma, 0.0], 
                               [1.0-gamma, gamma]])
            self.c = array([gamma, 1.0])
            self.b = array([1.0-gamma, gamma])
            self.order = 2
            
        elif method == "SDIRK212":
            # https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#implicit-butcher-tables
            # SDIRK-2-1-2: méthode à 2 étages d'ordre 2 avec estimation d'erreur d'ordre 1
            # A-stable et B-stable (TR-BDF2)
            self.A = np.zeros((2, 2))
            
            # Première ligne
            self.A[0, 0] = 1.0
            self.A[0, 1] = 0.0
            
            # Deuxième ligne
            self.A[1, 0] = -1.0
            self.A[1, 1] = 1.0
            
            # Abscisses c
            self.c = np.array([1.0, 0.0])
            
            # Coefficients b pour la méthode d'ordre 2
            self.b = np.array([0.5, 0.5])
            
            # Coefficients b̂ pour l'estimation d'erreur d'ordre 1
            self.bhat = np.array([1.0, 0.0])
            
            self.order = 2
            self.embedded_order = 1
            
        elif method == "SDIRK3":
            # SDIRK d'ordre 3
            gamma = 0.4358665215084589994160194  # Racine de x³ - 3x² + 3x - 1/2 = 0
            self.A = array([
                [gamma, 0.0, 0.0],
                [(1.0-gamma)/2.0, gamma, 0.0],
                [1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma]
            ])
            self.c = array([gamma, (1.0+gamma)/2.0, 1.0])
            self.b = array([1.0/(4.0*gamma), 1.0-1.0/(4.0*gamma), gamma])
            self.order = 3
            
        elif method == "SDIRK4":
            # SDIRK d'ordre 4 à 5 étages de Hairer & Wanner
            gamma = 1/4  # Valeur optimale pour la L-stabilité
            self.A = np.zeros((5, 5))
            
            # Coefficients de la matrice A
            self.A[0, 0] = gamma
            
            self.A[1, 0] = 1/2
            self.A[1, 1] = gamma
            
            self.A[2, 0] = 17/50
            self.A[2, 1] = -1/25
            self.A[2, 2] = gamma
            
            self.A[3, 0] = 371/1360
            self.A[3, 1] = -137/2720
            self.A[3, 2] = 15/544
            self.A[3, 3] = gamma
            
            self.A[4, 0] = 25/24
            self.A[4, 1] = -49/48
            self.A[4, 2] = 125/16
            self.A[4, 3] = -85/12
            self.A[4, 4] = gamma
            
            # Calcul des abscisses c
            self.c = np.array([gamma, 0.5 + gamma, 17/50 - 1/25 + gamma, 371/1360 - 137/2720 + 15/544 + gamma, 1.0])
            
            # Coefficients b (dernière ligne de A pour stiffly-accurate)
            self.b = self.A[4, :]
            
            self.order = 4
        
        elif method == "SDIRK4-2":
            # Variante alternative de SDIRK d'ordre 4 (Hairer-Wanner)
            gamma = (2-sqrt(2))/2  # (2-√2)/2
            self.A = np.zeros((5, 5))
            
            # Coefficients de la matrice A
            self.A[0, 0] = gamma
            
            self.A[1, 0] = 0.5 - gamma
            self.A[1, 1] = gamma
            
            self.A[2, 0] = 0.25
            self.A[2, 1] = 0.25
            self.A[2, 2] = gamma
            
            self.A[3, 0] = 1/6
            self.A[3, 1] = 1/6
            self.A[3, 2] = 1/6
            self.A[3, 3] = gamma
            
            self.A[4, 0] = 1/6
            self.A[4, 1] = 1/6
            self.A[4, 2] = 1/6
            self.A[4, 3] = 1/6
            self.A[4, 4] = gamma
            
            # Calcul des abscisses c
            self.c = np.array([gamma, 0.5, 0.5 + gamma, 0.5 + gamma, 0.5 + gamma])
            
            # Coefficients b (dernière ligne de A pour stiffly-accurate)
            self.b = self.A[4, :]
            
            self.order = 4
            
        elif method == "SDIRK4_BN_Stable_1_0":  # ρ∞ = 1.0 (Tableau A.1)
            # Méthode d'ordre 4 algébriquement stable (BN-stable) avec dissipation minimale tirée de 
            #Algebraically stable SDIRK methods with controllable numerical dissipation for first/second-order time-dependent problems
            gamma = 0.39433757
            self.A = np.zeros((4, 4))
            
            # Première ligne
            self.A[0, 0] = gamma
            
            # Deuxième ligne
            self.A[1, 0] = -0.28867513
            self.A[1, 1] = gamma
            
            # Troisième ligne
            self.A[2, 0] = -0.28867513
            self.A[2, 1] = 0.78867513
            self.A[2, 2] = gamma
            
            # Quatrième ligne
            self.A[3, 0] = 0.78867513
            self.A[3, 1] = -0.28867513
            self.A[3, 2] = -0.28867513
            self.A[3, 3] = gamma
            
            # Calcul des abscisses c
            self.c = np.array([0.39433757, 0.10566243, 0.89433757, 0.60566243])
            
            # Coefficients b (dernière ligne du tableau A.1)
            self.b = np.array([0.25, 0.25, 0.25, 0.25])
            
            self.order = 4
        
        elif method == "SDIRK4_BN_Stable_0_0":  # ρ∞ = 0.0 (Tableau A.6)
            # Méthode d'ordre 4 algébriquement stable (L-stable) tirée de 
            #Algebraically stable SDIRK methods with controllable numerical dissipation for first/second-order time-dependent problems
            gamma = 0.57281606
            self.A = np.zeros((4, 4))
            
            # Première ligne
            self.A[0, 0] = gamma
            
            # Deuxième ligne
            self.A[1, 0] = -0.54852714
            self.A[1, 1] = gamma
            
            # Troisième ligne
            self.A[2, 0] = -0.71695606
            self.A[2, 1] = 1.11985107
            self.A[2, 2] = gamma
            
            # Quatrième ligne
            self.A[3, 0] = 0.54506318
            self.A[3, 1] = -0.39131155
            self.A[3, 2] = -0.29938376
            self.A[3, 3] = gamma
            
            # Calcul des abscisses c
            self.c = np.array([0.57281606, 0.02428893, 0.97571107, 0.42718394])
            
            # Coefficients b (dernière ligne du tableau A.6)
            self.b = np.array([0.32345801, 0.17654199, 0.17654199, 0.32345801])
            
            self.order = 4
               
        elif method == "SDIRK534":
            # https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html#implicit-butcher-tables
            # SDIRK 5-stage méthode d'ordre 4 avec estimation d'erreur d'ordre 3
            # A- et L-stable
            gamma = 0.25  # Tous les coefficients diagonaux sont 1/4
            self.A = np.zeros((5, 5))
            
            # Première ligne
            self.A[0, 0] = gamma
            
            # Deuxième ligne
            self.A[1, 0] = 0.5
            self.A[1, 1] = gamma
            
            # Troisième ligne
            self.A[2, 0] = 17/50
            self.A[2, 1] = -1/25
            self.A[2, 2] = gamma
            
            # Quatrième ligne
            self.A[3, 0] = 371/1360
            self.A[3, 1] = -137/2720
            self.A[3, 2] = 15/544
            self.A[3, 3] = gamma
            
            # Cinquième ligne
            self.A[4, 0] = 25/24
            self.A[4, 1] = -49/48
            self.A[4, 2] = 125/16
            self.A[4, 3] = -85/12
            self.A[4, 4] = gamma
            
            # Calcul des abscisses c
            self.c = np.array([0.25, 0.75, 11/20, 0.5, 1.0])
            
            # Coefficients b pour la méthode d'ordre 4
            self.b = np.array([25/24, -49/48, 125/16, -85/12, 0.25])
            
            # Coefficients b̂ pour l'estimation d'erreur d'ordre 3
            self.bhat = np.array([59/48, -17/96, 225/32, -85/12, 0.0])
            
            self.order = 4
            self.embedded_order = 3
        
        elif method == "SDIRK5":
            # SDIRK d'ordre 5 à 6 étages (Cooper & Sayfy)
            gamma = 0.1840  # Valeur optimisée pour la L-stabilité
            self.A = np.zeros((6, 6))
            
            # Coefficients de la matrice A
            self.A[0, 0] = gamma
            
            self.A[1, 0] = 0.4
            self.A[1, 1] = gamma
            
            self.A[2, 0] = 0.376
            self.A[2, 1] = 0.1296
            self.A[2, 2] = gamma
            
            self.A[3, 0] = 0.0
            self.A[3, 1] = 0.0
            self.A[3, 2] = 0.8
            self.A[3, 3] = gamma
            
            self.A[4, 0] = 0.16
            self.A[4, 1] = 0.0
            self.A[4, 2] = 0.36
            self.A[4, 3] = 0.2
            self.A[4, 4] = gamma
            
            self.A[5, 0] = 0.1
            self.A[5, 1] = 0.0
            self.A[5, 2] = 0.3
            self.A[5, 3] = 0.2
            self.A[5, 4] = 0.2
            self.A[5, 5] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(6)
            for i in range(6):
                self.c[i] = np.sum(self.A[i, :])
            
            # Coefficients b (dernière ligne de A pour stiffly-accurate)
            self.b = self.A[5, :]
            
            self.order = 5
        
        elif method == "SDIRK5-2":
            # Méthode SDIRK d'ordre 5 à 7 étages (Hairer & Wanner variante)
            gamma = 1/7  # Valeur optimisée pour la L-stabilité
            self.A = np.zeros((7, 7))
            
            # Coefficients de la matrice A
            self.A[0, 0] = gamma
            
            self.A[1, 0] = 0.0
            self.A[1, 1] = gamma
            
            self.A[2, 0] = 1/10
            self.A[2, 1] = 3/10
            self.A[2, 2] = gamma
            
            self.A[3, 0] = -3/14
            self.A[3, 1] = 8/14
            self.A[3, 2] = 2/14
            self.A[3, 3] = gamma
            
            self.A[4, 0] = 19/50
            self.A[4, 1] = -21/100
            self.A[4, 2] = 1/4
            self.A[4, 3] = 3/50
            self.A[4, 4] = gamma
            
            self.A[5, 0] = 11/20
            self.A[5, 1] = -9/20
            self.A[5, 2] = 1/5
            self.A[5, 3] = 1/5
            self.A[5, 4] = 5/12
            self.A[5, 5] = gamma
            
            self.A[6, 0] = 89/300
            self.A[6, 1] = -61/225
            self.A[6, 2] = 37/300
            self.A[6, 3] = 29/300
            self.A[6, 4] = 7/120
            self.A[6, 5] = 119/120 - gamma
            self.A[6, 6] = gamma
            
            # Calcul des abscisses c
            self.c = np.zeros(7)
            for i in range(7):
                self.c[i] = np.sum(self.A[i, :])
            
            # Coefficients b (dernière ligne de A pour stiffly-accurate)
            self.b = self.A[6, :]
            
            self.order = 5
        else:
            raise ValueError("Unknown DIRK Method")
 
        self.num_stages = len(self.b)
        self.method = method

    def __str__(self):
        """Représentation sous forme de chaîne de caractères avec informations détaillées"""
        info = f"DIRK method: {self.method}\n"
        info += f"Order: {self.order}\n"
        info += f"Stages: {self.num_stages}\n"
        info += f"Diagonal coefficient γ: {self.A[0, 0]:.6f}\n"
        if self.method != "BDF1":  # Ajout d'une condition pour BDF1
            info += "L-stability: Yes (optimal for stiff problems)\n"
            info += "Stage-order: 2 (reduced order reduction)\n"
            info += "Stiffly-accurate: Yes (better for DAEs)\n"
        else:
            info += "L-stability: Yes\n"
            info += "Stage-order: 1\n"
            info += "Stiffly-accurate: Yes\n"
        return info