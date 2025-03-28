"""
Created on Fri Mar  7 15:57:50 2025

@author: bouteillerp
"""
from ufl import sqrt, dot, conditional, gt, max_value, min_value, ge, lt
from ..utils.generic_functions import extract_primitive_variables

class RiemannSolvers:
    """
    Implémentation de différents solveurs de Riemann pour les équations d'Euler.
    
    Cette classe fournit des approximations numériques pour résoudre le problème
    de Riemann aux interfaces entre éléments. Elle inclut plusieurs types de solveurs
    avec différents niveaux de précision et de coût computationnel.
    
    Les solveurs implémentés sont:
    - Rusanov (Local Lax-Friedrichs): Simple et robuste mais diffusif
    - HLL (Harten-Lax-van Leer): Meilleure précision, utilise des estimateurs de vitesse d'onde
    - HLLC: Version améliorée de HLL qui restaure l'onde de contact
    
    Les estimateurs de vitesse d'onde incluent:
    - Davis: Estimateur simple basé sur les valeurs maximales locales
    - Einfeldt: Estimateur plus précis basé sur la moyenne de Roe
    
    References
    ----------
    Toro, E.F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics.
    Springer-Verlag, 3rd edition.
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
        self.c = material.c
        self.c_bar = material.c_bar
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
        rho_L, u_L, _ = extract_primitive_variables(U)
        rho_R, u_R, _ = extract_primitive_variables(Ubar)
        
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
        rho_L, u_L, _ = extract_primitive_variables(U)
        rho_R, u_R, _ = extract_primitive_variables(Ubar)

        # Calcul des vitesses d'onde
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, self.c, self.c_bar, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, self.c, self.c_bar, rho_L, rho_R, n)
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
        
    def hllc_flux(self, U, Ubar, U_flux, Ubar_flux, n):
        """
        Calcule le flux numérique HLLC (Harten-Lax-van Leer-Contact).
        Implémentation directe basée sur l'article original de Toro et al. (1994).
        """
        # Extraction des variables primitives
        rho_L, u_L, E_L = extract_primitive_variables(U)
        rho_R, u_R, E_R = extract_primitive_variables(Ubar)
        
        # Calcul des pressions
        p_L = self.EOS.set_eos(U, self.material)
        p_R = self.EOS.set_eos(Ubar, self.material)
        
        # Vitesses normales
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        # Estimation des vitesses d'onde (Davis)
        S_L = min_value(u_n_L - self.c, u_n_R - self.c_bar)
        S_R = max_value(u_n_L + self.c, u_n_R + self.c_bar)
        
        # Protection contre les cas pathologiques
        S_L = conditional(gt(S_L, 0), S_L, min_value(S_L, -1e-8))
        S_R = conditional(lt(S_R, 0), S_R, max_value(S_R, 1e-8))
        
        # Vitesse de l'onde de contact (eq. 10 de Toro 1994)
        S_M = (p_R - p_L + rho_L * u_n_L * (S_L - u_n_L) - rho_R * u_n_R * (S_R - u_n_R)) / \
              (rho_L * (S_L - u_n_L) - rho_R * (S_R - u_n_R) + 1e-15)
        
        # Flux standards en direction normale
        F_L = [dot(f, n) for f in U_flux]
        F_R = [dot(f, n) for f in Ubar_flux]
        
        # États intermédiaires selon Toro (1994)
        # Densités dans les régions intermédiaires (eqs. 22 et 26)
        rho_star_L = rho_L * (S_L - u_n_L) / (S_L - S_M + 1e-15)
        rho_star_R = rho_R * (S_R - u_n_R) / (S_R - S_M + 1e-15)
        
        # Pressions dans les régions intermédiaires (eqs. 23 et 27)
        p_star_L = p_L + rho_L * (S_L - u_n_L) * (S_M - u_n_L)
        p_star_R = p_R + rho_R * (S_R - u_n_R) * (S_M - u_n_R)
        
        # Prendre la moyenne des pressions pour être cohérent
        p_star = (p_star_L + p_star_R) / 2.0
        
        # Calcul des flux HLLC
        F_star_L = []
        F_star_R = []
        
        # Flux de masse
        F_star_L.append(F_L[0] + S_L * (rho_star_L - rho_L))
        F_star_R.append(F_R[0] + S_R * (rho_star_R - rho_R))
        
        # Flux de quantité de mouvement - utiliser p_star pour la cohérence
        mom_star_L = rho_star_L * (u_L + (S_M - u_n_L) * n)
        mom_star_R = rho_star_R * (u_R + (S_M - u_n_R) * n)
        
        # Correction : Utiliser p_star dans les termes de pression
        F_star_L.append(F_L[1] + S_L * (mom_star_L - U[1]))
        F_star_R.append(F_R[1] + S_R * (mom_star_R - Ubar[1]))
        
        # Flux d'énergie - utiliser p_star pour la cohérence
        # Calculer l'énergie totale dans les régions intermédiaires en utilisant p_star
        E_star_L = rho_star_L * (E_L + (S_M - u_n_L) * (S_M + p_star/(rho_L * (S_L - u_n_L) + 1e-15)))
        E_star_R = rho_star_R * (E_R + (S_M - u_n_R) * (S_M + p_star/(rho_R * (S_R - u_n_R) + 1e-15)))
        
        F_star_L.append(F_L[2] + S_L * (E_star_L - U[2]))
        F_star_R.append(F_R[2] + S_R * (E_star_R - Ubar[2]))
        
        # Détermination du flux HLLC en fonction de la position de l'onde
        fluxes = []
        for k in range(len(U)):
            flux = conditional(ge(S_L, 0),
                              F_L[k],
                              conditional(ge(S_M, 0),
                                         F_star_L[k],
                                         conditional(ge(S_R, 0),
                                                    F_star_R[k],
                                                    F_R[k])))
            fluxes.append(flux)
        
        return fluxes