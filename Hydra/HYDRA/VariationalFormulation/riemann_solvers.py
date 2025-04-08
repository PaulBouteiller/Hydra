"""
Created on Fri Mar  7 15:57:50 2025

@author: bouteillerp
"""
from ufl import sqrt, dot, conditional, gt, max_value, min_value, ge, lt, le
from ..utils.generic_functions import extract_primitive_variables
from ..utils.default_parameters import default_Riemann_solver_parameters

class RiemannSolvers:
    """
    Collection of Riemann solvers for numerical flux computation.
    
    This class implements various approximate Riemann solvers used to
    compute numerical fluxes at element interfaces in discontinuous Galerkin
    methods. The solvers range from simple and robust (Rusanov) to more
    accurate but complex (HLLC, HLLCLM) approaches.
    
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
        Initialize the Riemann solver collection.
        
        Parameters
        ----------
        EOS : EOS Equation of state manager
        material : Material Material properties
        """
        self.EOS = EOS
        self.material = material
        self.eps = 1e-8  # Pour éviter les divisions par zéro        
        
    def signal_speed_davis(self, u_L, u_R, c_L, c_R, n):
        """
        Estimation de la vitesse du signal selon Davis. Voir Toro Eq. (10.48) 
        (dépend de l'édition 10.37 pour la première)
        
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
        (10.46)--(10.48) dans la première édition
        
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
        d_bar = sqrt((rho_L_sqrt * c_L**2 + rho_R_sqrt * c_R**2) * one_dens + eta2 * (u_n_R - u_n_L)**2)
        
        S_L = min_value(u_bar - d_bar, u_n_L - c_L)
        S_R = max_value(u_bar + d_bar, u_n_R + c_R)
        
        return S_L, S_R
    
    def compute_star_speed(self, u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n):
        """
        Calcule la vitesse de l'onde intermédiaire (contact) pour le solveur HLLC. Equation 10.58 de Toro
        
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
        c = self.EOS.set_celerity(U, self.material)
        c_bar = self.EOS.set_celerity(Ubar, self.material)
        # Vitesse maximale des ondes pour Rusanov
        lambda_max = max_value(abs(u_n_L) + c, abs(u_n_R) + c_bar)
        
        # Flux de Rusanov
        fluxes = []
        for i in range(len(U)):
            flux_L_n = dot(U_flux[i], n)
            flux_R_n = dot(Ubar_flux[i], n)
            flux_rusanov = 0.5 * (flux_L_n + flux_R_n) - 0.5 * lambda_max * (Ubar[i] - U[i])
            fluxes.append(flux_rusanov)
        
        return fluxes

    def hll_flux(self, U, Ubar, U_flux, Ubar_flux, n):
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
        signal_speed_type = default_Riemann_solver_parameters().get("signal_speed_type")
        c = self.EOS.set_celerity(U, self.material)
        c_bar = self.EOS.set_celerity(Ubar, self.material)
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, c, c_bar, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, c, c_bar, rho_L, rho_R, n)
        else:
            raise ValueError(f"Type d'estimateur de vitesse d'onde inconnu: {signal_speed_type}")
        # Flux HLL
        fluxes = []
        for i in range(len(U)):
            flux_L_n = dot(U_flux[i], n)
            flux_R_n = dot(Ubar_flux[i], n)
            flux_hll = (S_R * flux_L_n - S_L * flux_R_n + S_L * S_R * (Ubar[i] - U[i])) / (S_R - S_L + self.eps)
            
            flux_hll_tot = conditional(ge(S_L, 0),
                                       flux_L_n,
                                       conditional(le(S_R, 0),
                                                   flux_R_n,
                                                   flux_hll))

            fluxes.append(flux_hll_tot)
        
        return fluxes
    
    def hllc_flux(self, U, Ubar, U_flux, Ubar_flux, n):
        """
        Calcule le flux numérique HLLC (Harten-Lax-van Leer-Contact).
        Implémentation directe basée sur l'article original de Toro et al. (1994).
        """
        rho_L, u_L, E_L = extract_primitive_variables(U)
        rho_R, u_R, E_R = extract_primitive_variables(Ubar)
        
        p_L = self.EOS.set_eos(U, self.material)
        p_R = self.EOS.set_eos(Ubar, self.material)
        
        c = self.EOS.set_celerity(U, self.material)
        c_bar = self.EOS.set_celerity(Ubar, self.material)

        signal_speed_type = default_Riemann_solver_parameters().get("signal_speed_type")
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, c, c_bar, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, c, c_bar, rho_L, rho_R, n)
        
        #Vitesses de l'onde de contact S_star = S_M
        S_M = self.compute_star_speed(u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n)
        
        # Flux standards en direction normale
        F_L = [dot(f, n) for f in U_flux]
        F_R = [dot(f, n) for f in Ubar_flux]
        
        def compute_star_variables(S_M, S, U, normal_flux, rho, u, p, E, n):
            """
            Calcule les variables dans les régions intermédiaires.
            """
            u_n = dot(u, n)
            u_tang = u - u_n * n

            # Densité dans la région intermédiaire (eq. 10.33)
            rho_star = rho * (S - u_n) / (S - S_M + self.eps)
            
            # Quantité de mouvement vectorielle complète (eq. 10.33)
            # Combinaison de la composante normale (S_M) et des composantes tangentielles préservées
            mom_star = rho_star * (S_M * n + u_tang)
            
            # Énergie totale dans la région intermédiaire (eq. 10.33)
            E_star = rho_star * (E/rho + (S_M - u_n) * (S_M + p/(rho * (S - u_n) + self.eps)))
            
            return [rho_star, mom_star, E_star]
        # Calcul des variables dans les régions intermédiaires
        U_star_L = compute_star_variables(S_M, S_L, U, F_L, rho_L, u_L, p_L, E_L, n)
        
        U_star_R = compute_star_variables(S_M, S_R, Ubar, F_R, rho_R, u_R, p_R, E_R, n)
        
        # Calcul des flux HLLC
        F_star_L = [f + S_L * (u_star - u) for f, u_star, u in zip(F_L, U_star_L, U)]
        F_star_R = [f + S_R * (u_star - u) for f, u_star, u in zip(F_R, U_star_R, Ubar)]
        
        
        HLL_flux = self.hll_flux(U, Ubar, U_flux, Ubar_flux, n)
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
            
        # for k in range(len(U)):
        #     flux = conditional(ge(S_L, 0),
        #                       F_L[k],
        #                       conditional(ge(S_M, 0),
        #                                   HLL_flux[k],
        #                                   conditional(ge(S_R, 0),
        #                                             F_star_R[k],
        #                                             F_R[k])))
            fluxes.append(flux)
        
        return fluxes
    
    def hllclm_flux(self, U, Ubar, U_flux, Ubar_flux, n):
        """
        Calcule le flux numérique HLLCLM (Harten-Lax-van Leer-Contact Low Mach).
        Implémentation basée sur Fleischmann et al. 2020.
        
        Parameters
        ----------
        U : Liste d'Arrays Variables conservatives dans la cellule
        Ubar : Liste d'Arrays Variables conservatives à la facette
        U_flux : Liste d'Arrays Flux dans la cellule
        Ubar_flux : Liste d'Arrays Flux à la facette
        n : Array Vecteur normal à l'interface
        
        Returns
        -------
        fluxes : Liste Flux numériques HLLCLM pour chaque variable
        """
        # Extraction des variables primitives
        rho_L, u_L, E_L = extract_primitive_variables(U)
        rho_R, u_R, E_R = extract_primitive_variables(Ubar)
        
        # Protection pour la stabilité numérique
        rho_L = max_value(rho_L, self.eps)
        rho_R = max_value(rho_R, self.eps)
        
        # Calcul des pressions
        p_L = self.EOS.set_eos(U, self.material)
        p_R = self.EOS.set_eos(Ubar, self.material)
        
        # Protection pour les pressions
        p_L = max_value(p_L, self.eps)
        p_R = max_value(p_R, self.eps)
        
        # Vitesses du son
        c_L = self.EOS.set_celerity(U, self.material)
        c_R = self.EOS.set_celerity(Ubar, self.material)
        
        # Protection pour les vitesses du son
        c_L = max_value(c_L, self.eps)
        c_R = max_value(c_R, self.eps)
        
        # Vitesses normales
        u_n_L = dot(u_L, n)
        u_n_R = dot(u_R, n)
        
        # Estimation des vitesses d'onde
        signal_speed_type = default_Riemann_solver_parameters().get("signal_speed_type")
        if signal_speed_type == "davis":
            S_L, S_R = self.signal_speed_davis(u_L, u_R, c_L, c_R, n)
        elif signal_speed_type == "einfeldt":
            S_L, S_R = self.signal_speed_einfeldt(u_L, u_R, c_L, c_R, rho_L, rho_R, n)
        
        # Protection pour éviter S_L = S_R
        delta_S = max_value(S_R - S_L, self.eps * 10)
        S_R = S_L + delta_S
        
        # Vitesse de l'onde de contact
        S_M = self.compute_star_speed(u_L, u_R, p_L, p_R, rho_L, rho_R, S_L, S_R, n)
        
        # Protection pour S_M
        S_M = min_value(max_value(S_M, S_L + self.eps), S_R - self.eps)
        
        # Calcul du limiteur basé sur le nombre de Mach (Fleischmann et al. 2020)
        # ----- NOUVEAU PAR RAPPORT À HLLC STANDARD -----
        Ma_limit = 0.1  # Seuil pour la correction de faible Mach
        Ma_L = abs(u_n_L) / c_L
        Ma_R = abs(u_n_R) / c_R
        Ma_local = max_value(Ma_L, Ma_R)
        
        # Fonction de transition douce (sin)
        from ufl import sin, pi, conditional
        Ma_ratio = min_value(Ma_local / Ma_limit, 1.0)
        phi = sin(Ma_ratio * pi * 0.5)
        
        # Application du limiteur aux vitesses d'onde
        S_L_lim = phi * S_L
        S_R_lim = phi * S_R
        
        # Flux standards en direction normale
        F_L = [dot(f, n) for f in U_flux]
        F_R = [dot(f, n) for f in Ubar_flux]
        
        # Préfacteurs pour les états intermédiaires
        pre_factor_L = (S_L - u_n_L) / (S_L - S_M + self.eps)
        pre_factor_R = (S_R - u_n_R) / (S_R - S_M + self.eps)
        
        # Protection pour les préfacteurs
        pre_factor_L = conditional(abs(pre_factor_L) > 100.0, 
                                   conditional(ge(pre_factor_L, 0), 100.0, -100.0), 
                                   pre_factor_L)
        pre_factor_R = conditional(abs(pre_factor_R) > 100.0, 
                                   conditional(ge(pre_factor_R, 0), 100.0, -100.0), 
                                   pre_factor_R)
        
        # Composantes tangentielles de la vitesse
        u_tang_L = u_L - u_n_L * n
        u_tang_R = u_R - u_n_R * n
        
        # États intermédiaires à gauche et à droite
        # Densité dans les régions intermédiaires
        rho_star_L = rho_L * pre_factor_L
        rho_star_R = rho_R * pre_factor_R
        
        # Quantité de mouvement dans les régions intermédiaires
        mom_star_L = rho_star_L * (S_M * n + u_tang_L)  # Formulation vectorielle correcte
        mom_star_R = rho_star_R * (S_M * n + u_tang_R)  # Formulation vectorielle correcte
        
        # Énergie totale dans les régions intermédiaires
        E_star_L = rho_star_L * (E_L/rho_L + (S_M - u_n_L) * (S_M + p_L/(rho_L * (S_L - u_n_L) + self.eps)))
        E_star_R = rho_star_R * (E_R/rho_R + (S_M - u_n_R) * (S_M + p_R/(rho_R * (S_R - u_n_R) + self.eps)))
        
        # États intermédiaires complets
        U_star_L = [rho_star_L, mom_star_L, E_star_L]
        U_star_R = [rho_star_R, mom_star_R, E_star_R]
        
        # ----- FORMULATION DE FLUX SPÉCIFIQUE À HLLCLM (Fleischmann et al. 2020) -----
        # Calcul du flux HLLCLM 
        # Équation (19) de Fleischmann et al. 2020
        flux_star = []
        for i in range(len(U)):
            # Flux moyen
            f_avg = 0.5 * (F_L[i] + F_R[i])
            
            # Terme de dissipation
            f_diss = 0.5 * (
                S_L_lim * (U_star_L[i] - U[i]) + 
                abs(S_M) * (U_star_L[i] - U_star_R[i]) + 
                S_R_lim * (U_star_R[i] - Ubar[i])
            )
            
            # Flux total
            flux_star.append(f_avg + f_diss)
        
        # Équation (18) de Fleischmann et al. 2020 - Pondération du flux basée sur la position des ondes
        fluxes = []
        for k in range(len(U)):
            # Terme gauche - remplacer sign(S_L) par conditional(ge(S_L, 0), 1.0, -1.0)
            f_L_term = 0.5 * (1.0 + conditional(ge(S_L, 0), 1.0, -1.0)) * F_L[k]
            
            # Terme droit - remplacer sign(S_R) par conditional(ge(S_R, 0), 1.0, -1.0)
            f_R_term = 0.5 * (1.0 - conditional(ge(S_R, 0), 1.0, -1.0)) * F_R[k]
            
            # Terme central - remplacer les sign par conditional
            f_star_term = 0.25 * (1.0 - conditional(ge(S_L, 0), 1.0, -1.0)) * (1.0 + conditional(ge(S_R, 0), 1.0, -1.0)) * flux_star[k]
            
            # Flux complet
            fluxes.append(f_L_term + f_R_term + f_star_term)
        
        return fluxes