"""
Intégration temporelle et résolution non-linéaire pour la simulation de fluides compressibles.

Ce module contient les classes responsables de l'intégration temporelle et de la résolution
des systèmes non-linéaires pour les simulations d'écoulements compressibles. Il implémente
principalement des méthodes DIRK (Diagonally Implicit Runge-Kutta) pour l'intégration
temporelle, couplées à des solveurs de Newton par blocs pour la résolution des systèmes
non-linéaires à chaque pas de temps.
"""

from .customNewton2 import create_newton_solver
from ..utils.block import extract_rows, derivative_block
from ..utils.default_parameters import default_Newton_parameters
from ..Export.export_result import ExportResults
from ..utils.dirk_parameters import DIRKParameters
from tqdm import tqdm
import numpy as np
from dolfinx.fem import form, Function

class Solve:
    """
    Résout le problème de mécanique des fluides compressibles avec intégration temporelle.
    
    Cette classe gère le processus complet de résolution incluant:
    - Initialisation du problème et des structures de données
    - Intégration temporelle avec des schémas DIRK
    - Résolution des systèmes non-linéaires à chaque pas de temps
    - Export des résultats
    
    La classe supporte différentes méthodes DIRK (de BDF1 jusqu'à des schémas d'ordre 5)
    et peut être configurée avec différentes options pour le solveur et l'export.
    """

    def __init__(self, problem, **kwargs):
        """
        Initialise le solveur pour l'intégration temporelle.
        
        Parameters
        ----------
        problem : Problem Le problème à résoudre.
        **kwargs : dict
            Arguments supplémentaires pour la configuration:
            - dt : float Pas de temps.
            - TFin : float Temps final de la simulation.
            - dirk_method : str, optional Méthode DIRK à utiliser (par défaut: "BDF1").
            - Prefix : str, optional Préfixe pour les fichiers de résultats.
            - compteur : int, optional Fréquence d'export des résultats (1 = à chaque pas de temps).
        """
        self.pb = problem
        self.initialize_solve(problem, **kwargs)
        print("Starting simulation")       
        self.run_simulation(**kwargs)

    def initialize_solve(self, problem, **kwargs):
        """
        Initialise le solveur avec les paramètres de base.
        
        Parameters
        ----------
        problem : Problem Le problème à résoudre.
        **kwargs : dict Arguments supplémentaires pour la configuration.
        """
        # Initialisation du problème
        self.pb.set_initial_conditions()
        self.t = 0
        
        # Configuration de l'export des résultats
        self.setup_export(**kwargs)
        
        # Configuration du solveur non-linéaire
        self.setup_solver()
        
        # Configuration des paramètres temporels
        self.setup_time_parameters(**kwargs)
        
        # Initialisation du schéma DIRK
        self.setup_dirk_scheme(**kwargs)
        
    def setup_export(self, **kwargs):
        """
        Configure l'export des résultats.
        
        Parameters
        ----------
        **kwargs : dict Arguments supplémentaires avec éventuellement 'Prefix'.
        """
        self.export = ExportResults(
            self.pb, 
            kwargs.get("Prefix", self.pb.prefix()), 
            self.pb.set_output(), 
            self.pb.csv_output()
        )
        self.export_results(self.t)
        
    def setup_solver(self):
        """
        Configure le solveur non-linéaire.
        """
        # Extraction du résidu et du jacobien
        Fr = extract_rows(self.pb.residual, self.pb.u_test_list)
        J = derivative_block(Fr, self.pb.u_list, self.pb.du_list)
        
        # Création des formulaires
        Fr_form = form(Fr, entity_maps=self.pb.entity_maps)
        J_form = form(J, entity_maps=self.pb.entity_maps)
        
        # Configuration des options PETSc
        petsc_options, structure_type, debug = default_Newton_parameters()
        self.solver = create_newton_solver(structure_type, Fr_form, self.pb.u_list, 
                                           J_form, petsc_options, debug, self.pb.bc_class.bcs)

    def setup_time_parameters(self, **kwargs):
        """
        Configure les paramètres temporels de la simulation.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments supplémentaires avec 'dt' et 'TFin'.
        """
        self.dt = kwargs.get("dt")
        self.Tfin = kwargs.get("TFin")
        self.num_time_steps = int(self.Tfin / self.dt)
        self.load_steps = np.linspace(0, self.Tfin, self.num_time_steps + 1)
        self.pb.bc_class.set_time_dependant_BCs(self.load_steps)
        
    def setup_dirk_scheme(self, **kwargs):
        """
        Configure le schéma DIRK pour l'intégration temporelle.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments supplémentaires avec éventuellement 'dirk_method'.
        """
        # Initialisation des paramètres DIRK
        self.dirk_method = kwargs.get("dirk_method", "BDF1")  # Par défaut: Backward Euler = BDF1
        self.dirk_params = DIRKParameters(self.dirk_method)
        print(f"Initializing with temporal scheme: {self.dirk_params}")
        self.initialize_stages()
        
    def initialize_stages(self):
        """
        Initialise les structures pour stocker les étapes intermédiaires du schéma DIRK.
        
        Crée des tableaux pour les solutions intermédiaires et les termes sources
        à chaque étape du schéma DIRK.
        """
        self.num_stages = self.dirk_params.num_stages
        
        # Création des structures pour les solutions intermédiaires
        self.stage_solutions = []
        self.stage_rhs = []
        
        for _ in range(self.num_stages):
            stage_sol = []
            stage_source = []
            
            for space_idx in range(len(self.pb.U)):
                # Déterminer l'espace fonctionnel approprié
                if space_idx == 0:  # Densité
                    space = self.pb.V_rho
                elif space_idx == 1:  # Vitesse
                    space = self.pb.V_rhov
                else:  # Énergie ou autres variables
                    space = self.pb.V_rhoE
                
                # Créer les fonctions pour cette étape
                stage_sol.append(Function(space))
                stage_source.append(Function(space))
            
            self.stage_solutions.append(stage_sol)
            self.stage_rhs.append(stage_source)
        
        # Initialisation des termes sources pour la première étape
        for i, s in enumerate(self.pb.s):
            s.x.array[:] = self.pb.U_n[i].x.array * self.pb.dt_factor.value
            self.stage_rhs[0][i].x.array[:] = s.x.array
        
    def export_results(self, t):
        """
        Exporte les résultats au temps spécifié.
        
        Parameters
        ----------
        t : float Temps auquel exporter les résultats.
        """
        self.pb.query_output(t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)

    def run_simulation(self, **kwargs):
        """
        Exécute la simulation complète.
        
        Réalise la boucle temporelle principale en effectuant les étapes suivantes
        à chaque pas de temps:
        1. Mise à jour du temps
        2. Résolution des étapes DIRK
        3. Calcul de la solution finale
        4. Export des résultats
        
        Parameters
        ----------
        **kwargs : dict Arguments supplémentaires avec éventuellement 'compteur'.
        """
        compteur_output = kwargs.get("compteur", 1)
        num_time_steps = self.num_time_steps
        
        # Configuration de la fréquence d'export
        if compteur_output != 1:
            self.is_compteur = True
            self.compteur = 0 
        else:
            self.is_compteur = False            
        
        # Boucle temporelle principale
        with tqdm(total=num_time_steps, desc="Simulation progress", unit="step") as pbar:
            j = 0
            while self.t < self.Tfin:
                # Mise à jour du temps
                self.update_time(j)
                
                # Résolution des étapes DIRK
                for stage in range(self.num_stages):
                    if self.num_stages > 1:
                        print(f"Solving DIRK stage {stage+1}/{self.num_stages} at time {self.stage_times[stage]:.6f}")
                    self.solve_stage(stage)
                
                # Calcul de la solution finale
                self.compute_final_solution()
                self.update_fields()
                
                # Incrémentation et export
                j += 1
                self.handle_output(compteur_output)
                pbar.update(1)
        
        # Finalisation
        self.export.csv.close_files()
        self.pb.final_output()
            
    def handle_output(self, compteur_output):
        """
        Gère l'export périodique des résultats.
        
        Parameters
        ----------
        compteur_output : int
            Fréquence d'export (nombre de pas de temps entre chaque export).
        """
        if self.is_compteur:
            if self.compteur == compteur_output:
                self.export_results(self.t)
                self.compteur = 0
            self.compteur += 1
        else:
            self.export_results(self.t)
            
    def update_time(self, j):
        """
        Met à jour le temps courant et calcule les temps intermédiaires.
        
        Parameters
        ----------
        j : int Indice du pas de temps courant.
        """
        self.t = self.load_steps[j]
        
        # Calcul des temps intermédiaires pour les étapes DIRK
        self.stage_times = [self.t + self.dirk_params.c[s] * self.dt for s in range(self.num_stages)]
    
    def configure_stage_residual(self, stage):
        """
        Configure le résidu pour une étape spécifique du schéma DIRK.
        
        Définit le facteur dt et les termes sources appropriés pour l'étape courante,
        puis met à jour les conditions aux limites pour le temps intermédiaire.
        
        Parameters
        ----------
        stage : int Numéro de l'étape.
        """
        # Configuration du facteur dt
        a_ii = self.dirk_params.A[stage, stage]
        self.pb.dt_factor.value = 1.0 / (self.dt * a_ii)
        
        # Calcul et configuration des termes sources
        stage_s = self.calculate_stage_source_term(stage)
        for i, s in enumerate(self.pb.s):
            s.x.array[:] = stage_s[i].x.array            
        
        # Mise à jour des conditions aux limites pour le temps intermédiaire
        stage_time = self.stage_times[stage]
        stage_step = int(stage_time / (self.Tfin/self.num_time_steps))
        self.pb.update_bcs(stage_step)
        
        # Mise à jour des valeurs des conditions aux limites dépendantes du temps
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[stage_step]
            
    def calculate_stage_source_term(self, stage):
        """
        Calcule les termes sources pour une étape spécifique du schéma DIRK.
        
        Implémente la formule mathématique pour calculer les termes sources s_h^{n,i}
        selon la théorie des schémas DIRK.
        
        Parameters
        ----------
        stage : int Numéro de l'étape.
            
        Returns
        -------
        list Liste des termes sources pour chaque variable.
        """
        # Pour la première étape, utiliser directement la solution précédente
        if stage == 0:
            for i, s in enumerate(self.pb.s):
                s.x.array[:] = self.pb.U_n[i].x.array * self.pb.dt_factor.value
                self.stage_rhs[0][i].x.array[:] = s.x.array
            return self.stage_rhs[0]
        
        # Pour les étapes suivantes, calculer les termes sources récursivement
        a_ii = self.dirk_params.A[stage, stage]
        for i, u_n in enumerate(self.pb.U_n):
            # Terme de base : U_h^n / (a_ii * dt)
            base_term = u_n.x.array.copy() / (a_ii * self.dt)
            self.stage_rhs[stage][i].x.array[:] = base_term
            
            # Ajouter les contributions des étapes précédentes
            for j in range(stage):
                a_ij = self.dirk_params.A[stage, j]
                if abs(a_ij) > 1e-14:  # Ignorer les contributions nulles
                    a_jj = self.dirk_params.A[j, j]
                    prev_u = self.stage_solutions[j][i]
                    prev_s = self.stage_rhs[j][i]
                    
                    # Calculer (U_h^{n,j}/(a_jj*dt) - s_h^{n,j})
                    correction = prev_u.x.array / (a_jj * self.dt) - prev_s.x.array
                    self.stage_rhs[stage][i].x.array[:] += (a_ij / a_ii) * correction
        
        return self.stage_rhs[stage]
            
    def solve_stage(self, stage):
        """
        Résout une étape spécifique du schéma DIRK.
        
        Configure le système pour l'étape courante, résout le système non-linéaire,
        puis stocke la solution.
        
        Parameters
        ----------
        stage : int Numéro de l'étape à résoudre.
        """
        # Configuration du système
        self.configure_stage_residual(stage)
        
        # Résolution du système non-linéaire
        self.solver.solve()
        
        # Stockage de la solution
        for i, u in enumerate(self.pb.U):
            self.stage_solutions[stage][i].x.array[:] = u.x.array
                
    def compute_final_solution(self):
        """
        Calcule la solution finale en utilisant la formule du schéma DIRK.
        
        Pour BDF1, utilise directement la solution de l'étape unique.
        Pour les méthodes d'ordre supérieur, applique la formule complète du schéma DIRK.
        """
        if self.dirk_method == "BDF1":
            # Pour BDF1, la solution est déjà correcte (méthode à une étape)
            pass
        else:
            # Calcul complet des z_h^{n,i} pour les méthodes DIRK d'ordre supérieur
            z_stage = []
            for stage in range(self.num_stages):
                z_current = []
                for i, u_n in enumerate(self.pb.U_n):
                    a_ii = self.dirk_params.A[stage, stage]
                    stage_u = self.stage_solutions[stage][i]
                    z_i = (stage_u.x.array - u_n.x.array) / (self.dt * a_ii)
                    
                    # Soustraire les contributions des étapes précédentes
                    if stage > 0:
                        for prev_stage in range(stage):
                            a_ij = self.dirk_params.A[stage, prev_stage]
                            if abs(a_ij) > 1e-14:  # Ignorer les contributions nulles
                                prev_z = z_stage[prev_stage][i]
                                z_i -= (a_ij / a_ii) * prev_z
                    
                    z_current.append(z_i)
                
                z_stage.append(z_current)
            
            # Calculer la solution finale
            for i, u_n in enumerate(self.pb.U_n):
                self.pb.U[i].x.array[:] = u_n.x.array
                
                # Ajouter les contributions de chaque étape
                for stage in range(self.num_stages):
                    b_s = self.dirk_params.b[stage]
                    if abs(b_s) < 1e-14:  # Ignorer les contributions nulles
                        continue                    
                    # U^{n+1} = U^n + dt * sum(b_i * z_h^{n,i})
                    self.pb.U[i].x.array[:] += self.dt * b_s * z_stage[stage][i]

    def update_fields(self):
        """
        Met à jour les champs pour le pas de temps suivant.
        
        Copie les solutions du pas de temps courant vers les champs "n"
        pour la prochaine itération.
        """
        for x, x_n in zip(self.pb.U, self.pb.U_n):
            x_n.x.array[:] = x.x.array
        
        # Mise à jour de la viscosité artificielle basée sur le nouvel état
        # Désactivée par défaut, décommenter si nécessaire
        if self.pb.shock_stabilization:
            # raise ValueError("Actuellement buggué")
            self.pb.p_star_U.interpolate(self.pb.p_star_U_expr)