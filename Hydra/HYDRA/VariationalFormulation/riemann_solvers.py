"""
Created on Fri Mar  7 15:57:50 2025

@author: bouteillerp
"""
from ufl import sqrt, dot, conditional, gt, lt, And, max_value, min_value

class RiemannSolvers:
    """
    Implémentation de différents solveurs de Riemann pour les équations d'Euler et de Navier-Stokes compressibles.
    Cette classe fournit des méthodes pour calculer les flux numériques aux interfaces entre éléments,
    en utilisant différentes approximations du problème de Riemann.
    """

    def __init__(self, EOS, material):
        """
        Initialise le solveur de Riemann.
        
        Parameters
        ----------
        EOS : EOS Gestionnaire de l'EOS
        material : Material 
        """
        self.EOS = EOS
        self.c = material.celerity
        self.material = material
        self.eps = 1e-10  # Valeur epsilon pour éviter les divisions par zéro
        
    def signal_speed_davis(self, u_L, u_R, c_L, c_R, n):
        """
        Estimation de la vitesse du signal selon Davis. Voir Toro Eq. (10.48)
        
        Parameters
        ----------
        u_L : Array Vecteur vitesse à gauche de l'interface
        u_R : Array Vecteur vitesse à droite de l'interface
        c_L : Array Vitesse du son à gauche de l'interface
        c_R : Array Vitesse du son à droite de l'interface
        n : Array Vecteur normal à l'interface
            
        Returns
        -------
        S_L, S_R : Tuple Vitesses des ondes gauche et droite
        """
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        S_L = min_value(u_n_L - c_L, u_n_R - c_R)
        S_R = max_value(u_n_L + c_L, u_n_R + c_R)
        return S_L, S_R
    
    def signal_speed_einfeldt(self, u_L, u_R, c_L, c_R, rho_L, rho_R, n):
        """
        Estimation de la vitesse du signal selon Einfeldt (HLLE). Voir Toro Eqs. (10.52) - (10.54)
        
        Parameters
        ----------
        u_L, u_R : Array Vecteurs vitesse à gauche et à droite
        c_L, c_R : Array  Vitesses du son à gauche et à droite
        rho_L, rho_R : Array Densités à gauche et à droite
        n : Array Vecteur normal à l'interface
        Returns
        -------
        S_L, S_R : Tuple Vitesses des ondes gauche et droite
        """
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        rho_L_sqrt = sqrt(rho_L)
        rho_R_sqrt = sqrt(rho_R)
        
        one_dens = 1.0 / (rho_L_sqrt + rho_R_sqrt + self.eps)
        eta2 = 0.5 * rho_L_sqrt * rho_R_sqrt * one_dens * one_dens
        u_bar = (rho_L_sqrt * u_n_L + rho_R_sqrt * u_n_R) * one_dens
        d_bar = sqrt((rho_L_sqrt * c_L * c_L + rho_R_sqrt * c_R * c_R) * one_dens + eta2 * (u_n_R - u_n_L) * (u_n_R - u_n_L))
        
        S_L = min_value(u_bar - d_bar, u_n_L - c_L)
        S_R = max_value(u_bar + d_bar, u_n_R + c_R)
        
        return S_L, S_R
    
    def compute_star_speed(self, u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n):
        """
        Calcule la vitesse de l'onde intermédiaire (contact) pour le solveur HLLC.
        
        Parameters
        ----------
        u_L, u_R : Array Vecteurs vitesse à gauche et à droite
        p_L, p_R : Array Pressions à gauche et à droite
        rho_L, rho_R : Array Densités à gauche et à droite
        S_L, S_R : Array Vitesses des ondes gauche et droite
        n : Array Vecteur normal à l'interface
            
        Returns
        -------
        S_star : Array Vitesse de l'onde de contact
        """
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        delta_uL = S_L - u_n_L
        delta_uR = S_R - u_n_R
        one_rho_deltaSU = 1.0 / (rho_L * delta_uL - rho_R * delta_uR + self.eps)
        
        S_star = one_rho_deltaSU * (p_R - p_L + rho_L * u_n_L * delta_uL - rho_R * u_n_R * delta_uR)
        
        return S_star

    def rusanov_flux(self, U, Ubar, U_flux, Ubar_flux, n):
        """
        Calcule le flux numérique de Rusanov (Local Lax-Friedrichs).
        
        Parameters
        ----------
        U : Liste d'Arrays Variables conservatives dans la cellule
        Ubar : Liste d'Arrays Variables conservatives à la facette
        U_flux : Liste d'Arrays Flux dans la cellule
        Ubar_flux : Liste d'Arrays Flux à la facette
        n : Array Vecteur normal à l'interface
        Returns
        -------
        fluxes : Liste Flux numériques de Rusanov pour chaque variable
        """
        # Extraction des variables primitives
        rho_L = U[0]
        rho_R = Ubar[0]
        u_L = U[1] / rho_L
        u_R = Ubar[1] / rho_R
        
        # Composantes normales des vitesses
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        # Vitesse maximale des ondes pour Rusanov
        lambda_max = max_value(abs(u_n_L) + self.c, abs(u_n_R) + self.c)
        
        # Flux de Rusanov
        fluxes = []
        for i in range(len(U)):
            flux_L_n = dot(U_flux[i], n)
            flux_R_n = dot(Ubar_flux[i], n)
            flux_rusanov = 0.5 * (flux_L_n + flux_R_n) - 0.5 * lambda_max * (Ubar[i] - U[i])
            fluxes.append(flux_rusanov)
        
        return fluxes

    def hll_flux(self, U, Ubar, U_flux, Ubar_flux, n, signal_speed_type="davis"):
        """
        Calcule le flux numérique HLL (Harten-Lax-van Leer).
        
        Parameters
        ----------
        U : Liste d'Arrays Variables conservatives dans la cellule
        Ubar : Liste d'Arrays Variables conservatives à la facette
        U_flux : Liste d'Arrays Flux dans la cellule
        Ubar_flux : Liste d'Arrays Flux à la facette
        n : Array Vecteur normal à l'interface
        signal_speed_type : str, optional Type d'estimateur de vitesse d'onde à utiliser
        Returns
        -------
        fluxes : Liste  Flux numériques HLL pour chaque variable
        """
        # Extraction des variables primitives
        rho_L = U[0]
        rho_R = Ubar[0]
        u_L = U[1] / rho_L
        u_R = Ubar[1] / rho_R
        
        # Calcul des vitesses d'onde
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, self.c, self.c, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, self.c, self.c, rho_L, rho_R, n)
        else:
            raise ValueError(f"Type d'estimateur de vitesse d'onde inconnu: {signal_speed_type}")
        
        # S'assurer que S_L < 0 et S_R > 0
        S_L = min_value(S_L, 0.0)
        S_R = max_value(S_R, 0.0)
        
        # Flux HLL
        fluxes = []
        for i in range(len(U)):
            flux_L_n = dot(U_flux[i], n)
            flux_R_n = dot(Ubar_flux[i], n)
            flux_hll = (S_R * flux_L_n - S_L * flux_R_n + S_L * S_R * (Ubar[i] - U[i])) / (S_R - S_L + self.eps)
            fluxes.append(flux_hll)
        
        return fluxes

    def hllc_flux(self, U, Ubar, U_flux, Ubar_flux, n, signal_speed_type="davis"):
        """
        Calcule le flux numérique HLLC (Harten-Lax-van Leer-Contact).
        
        Parameters
        ----------
        U : Liste d'Arrays Variables conservatives dans la cellule
        Ubar : Liste d'Arrays Variables conservatives à la facette
        U_flux : Liste d'Arrays Flux dans la cellule
        Ubar_flux : Liste d'Arrays Flux à la facette
        n : Array Vecteur normal à l'interface
        signal_speed_type : str, optional Type d'estimateur de vitesse d'onde à utiliser
        Returns
        -------
        fluxes : Liste Flux numériques HLLC pour chaque variable
        """
        # Extraction des variables primitives
        rho_L = U[0]
        rho_R = Ubar[0]
        u_L = U[1] / rho_L
        u_R = Ubar[1] / rho_R
        E_L = U[2] / rho_L
        E_R = Ubar[2] / rho_R
        
        # Calcul des vitesses du son et des pressions
        p_L = self.EOS.set_eos(rho_L, u_L, E_L, self.material)
        p_R = self.EOS.set_eos(rho_R, u_R, E_R, self.material)
        
        # Calcul des vitesses d'onde
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, self.c, self.c, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, self.c, self.c, rho_L, rho_R, n)
        else:
            raise ValueError(f"Type d'estimateur de vitesse d'onde inconnu: {signal_speed_type}")
        
        # Calcul de la vitesse de l'onde de contact
        S_star = self.compute_star_speed(u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n)
        
        # Composantes normales des vitesses
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        # Calcul des états intermédiaires (star states)
        # Préfacteurs pour les états intermédiaires
        pre_factor_L = (S_L - u_n_L) / (S_L - S_star + self.eps)
        pre_factor_R = (S_R - u_n_R) / (S_R - S_star + self.eps)
        
        # Pour les flux HLLC, nous devons calculer les états intermédiaires pour chaque variable
        # Nous allons créer des listes pour stocker ces états
        U_star_L = []
        U_star_R = []
        
        # Pour l'équation de masse (première équation)
        U_star_L.append(pre_factor_L * rho_L)
        U_star_R.append(pre_factor_R * rho_R)
        
        # Pour l'équation de quantité de mouvement (deuxième équation)
        # On doit créer un nouveau vecteur vitesse avec la composante normale remplacée par S_star
        u_star_L = u_L + (S_star - u_n_L) * n
        u_star_R = u_R + (S_star - u_n_R) * n
        
        U_star_L.append(pre_factor_L * rho_L * u_star_L)
        U_star_R.append(pre_factor_R * rho_R * u_star_R)
        
        # Pour l'équation d'énergie (troisième équation)
        E_star_L = E_L + (S_star - u_n_L) * (S_star + p_L / (rho_L * (S_L - u_n_L + self.eps)))
        E_star_R = E_R + (S_star - u_n_R) * (S_star + p_R / (rho_R * (S_R - u_n_R + self.eps)))
        
        U_star_L.append(pre_factor_L * rho_L * E_star_L)
        U_star_R.append(pre_factor_R * rho_R * E_star_R)
        
        # Calcul des flux HLLC
        fluxes = []
        
        # Flux physiques
        F_L = [dot(f, n) for f in U_flux]
        F_R = [dot(f, n) for f in Ubar_flux]
        
        # Déterminer dans quelle région se trouve la solution
        cond_L = conditional(S_L >= 0, 1, 0)  # Si S_L ≥ 0, on est dans la région L
        cond_star_L = conditional(And(lt(S_L, 0), gt(S_star, 0)), 1, 0)  # Si S_L < 0 et S_star > 0, on est dans la région *L
        cond_star_R = conditional(And(lt(S_star, 0), gt(S_R, 0)), 1, 0)  # Si S_star < 0 et S_R > 0, on est dans la région *R
        cond_R = conditional(S_R <= 0, 1, 0)  # Si S_R ≤ 0, on est dans la région R
        
        # Calcul des flux HLLC pour chaque variable
        for i in range(len(U)):
            # Flux dans la région L
            flux_L = F_L[i]
            
            # Flux dans la région *L
            flux_star_L = F_L[i] + S_L * (U_star_L[i] - U[i])
            
            # Flux dans la région *R
            flux_star_R = F_R[i] + S_R * (U_star_R[i] - Ubar[i])
            
            # Flux dans la région R
            flux_R = F_R[i]
            
            # Sélection du flux approprié en fonction de la position de l'onde de contact
            flux_hllc = cond_L * flux_L + cond_star_L * flux_star_L + cond_star_R * flux_star_R + cond_R * flux_R
            
            fluxes.append(flux_hllc)
        
        return fluxes