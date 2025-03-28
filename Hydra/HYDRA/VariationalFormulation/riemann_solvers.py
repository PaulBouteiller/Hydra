"""
Created on Fri Mar  7 15:57:50 2025

@author: bouteillerp
"""
from ufl import sqrt, dot, conditional, gt, max_value, min_value
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
        
    def hllc_flux(self, U, Ubar, U_flux, Ubar_flux, n, signal_speed_type="davis"):
        """Calcule le flux numérique HLLC (Harten-Lax-van Leer-Contact).
    
        Le solveur HLLC est une amélioration du solveur HLL qui restaure la précision
        de la discontinuité de contact, ce qui est crucial pour les problèmes multi-matériaux
        et les interfaces entre différentes phases.
        
        L'idée principale du HLLC est de considérer une structure d'onde à trois ondes:
        - S_L : Vitesse de l'onde la plus à gauche
        - S_* : Vitesse de la discontinuité de contact
        - S_R : Vitesse de l'onde la plus à droite
        
        Cette structure permet de définir trois états intermédiaires, et le flux numérique
        est calculé en fonction de la position relative de zéro par rapport à ces vitesses.
        
        Parameters
        ----------
        U : list[ufl.core.expr.Expr] Variables conservatives à l'intérieur de l'élément [ρ, ρu, ρE].
        Ubar : list[ufl.core.expr.Expr] Variables conservatives à la facette [ρ, ρu, ρE].
        U_flux : list[ufl.core.expr.Expr] Flux physiques à l'intérieur de l'élément.
        Ubar_flux : list[ufl.core.expr.Expr] Flux physiques à la facette.
        n : ufl.core.expr.Expr Vecteur normal unitaire à la facette.
        signal_speed_type : str, optional
            Type d'estimateur de vitesse d'onde à utiliser:
            - "davis" : Estimateur simple basé sur les valeurs max/min locales
            - "einfeldt" : Estimateur plus précis basé sur la moyenne de Roe
            
        Returns
        -------
        list[ufl.core.expr.Expr] Flux numériques HLLC pour chaque variable conservative.
            
        References
        ----------
        .. [1] Toro, E.F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics."
               Springer-Verlag, 3rd edition, pp. 321-326.
        .. [2] Batten, P., Leschziner, M.A., & Goldberg, U.C. (1997). "Average-state Jacobians
               and implicit methods for compressible viscous and turbulent flows."
               Journal of Computational Physics, 137(1), 38-78.
        """
        rho_L, u_L, E_L = extract_primitive_variables(U)
        rho_R, u_R, E_R = extract_primitive_variables(Ubar)
        
        # Calcul des pressions
        p_L = self.EOS.set_eos(rho_L, u_L, E_L, self.material)
        p_R = self.EOS.set_eos(rho_R, u_R, E_R, self.material)

        # Calcul des vitesses d'onde
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, self.c, self.c_bar, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, self.c, self.c_bar, rho_L, rho_R, n)
        
        # Forcer S_L < 0 et S_R > 0 pour la stabilité
        S_L = min_value(S_L, -self.eps)
        S_R = max_value(S_R, self.eps)
        S_star = self.compute_star_speed(u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n)
        
        #S_star entre S_L et S_R
        S_star = min_value(max_value(S_star, S_L + self.eps), S_R - self.eps)
        
        # Flux standards
        F_L = [dot(f, n) for f in U_flux]
        F_R = [dot(f, n) for f in Ubar_flux]
        
        # Facteur de mélange augmenté légèrement
        blend_factor = 0.05  # Augmenté de 0.01 à 0.05
        energy_blend = 0.05  # Facteur plus conservateur pour l'énergie
        
        # Version améliorée du flux HLLC
        fluxes = []
        for i in range(len(U)):
            # Flux HLL standard
            flux_hll = (S_R * F_L[i] - S_L * F_R[i] + S_L * S_R * (Ubar[i] - U[i])) / (S_R - S_L)
            correction_dir = conditional(gt(S_star, 0), 1.0, -1.0)
            if i == 1:  # Quantité de mouvement
                # Correction scalaire basée sur la densité et la vitesse du son
                rho_avg = 0.5 * (rho_L + rho_R)
                rho_ratio = abs(rho_L - rho_R) / (rho_L + rho_R + self.eps)
                correction_scalar = blend_factor * rho_avg * self.c * self.c * rho_ratio
                
                # Appliquer la correction comme un vecteur dans la direction normale
                flux = flux_hll + correction_dir * correction_scalar * n
                
            elif i == 2:  # Énergie
                # Correction adaptée pour l'énergie
                p_avg = 0.5 * (p_L + p_R)
                energy_ratio = abs(E_L - E_R) / (abs(E_L) + abs(E_R) + self.eps)
                energy_correction = energy_blend * p_avg * energy_ratio
                
                # Appliquer la correction comme un scalaire directement à l'énergie
                flux = flux_hll + correction_dir * energy_correction
                
            else:
                flux = flux_hll
            
            fluxes.append(flux)
        
        return fluxes