"""
Created on Tue Mar 11 16:30:20 2025

@author: bouteillerp
"""
from .Solve import Solve
from ..utils.dirk_parameters_2 import DIRKParameters
from tqdm import tqdm
from dolfinx.fem import Function
import numpy as np

class DIRKSolve(Solve):
    """
    Solveur utilisant un schéma Diagonally Implicit Runge-Kutta (DIRK).
    """
    def __init__(self, problem, **kwargs):
        """
        Initialise le solveur DIRK.
        
        Parameters
        ----------
        problem : Problem Le problème à résoudre.
        dirk_method : str, optional La méthode DIRK à utiliser. Par défaut "SDIRK2".
        **kwargs : dict Arguments supplémentaires pour la classe parent Solve.
        """
        self.dirk_method = kwargs.pop("dirk_method", "SDIRK2")
        self.dirk_params = DIRKParameters(self.dirk_method)
        print(f"Initializing DIRK solver: {self.dirk_params}")
        self.pb = problem
        self.initialize_solve(problem, **kwargs)
        self.initialize_stages()
        
        print("Start solving with DIRK method")       
        self.iterative_solve(**kwargs)
    
    def initialize_stages(self):
        """
        Initialise les structures pour stocker les étapes intermédiaires DIRK.
        """
        
        self.num_stages = self.dirk_params.num_stages # Nombre d'étapes DIRK
        self.stage_solutions = [] #Fonctions pour chaque étape intermédiaire
        for s in range(self.num_stages):
            # Cloner les fonctions de solution pour chaque étape
            stage_u = []
            for u_base in self.pb.U_base:
                # Créer une fonction dans le même espace
                u_stage = Function(u_base.function_space)
                u_stage.x.array[:] = u_base.x.array
                stage_u.append(u_stage)
            
            self.stage_solutions.append(stage_u)
        
        # Conserver les solutions temporaires pour le calcul des étapes
        self.temp_solutions = []
        for u_base in self.pb.U_base:
            temp_u = Function(u_base.function_space)
            self.temp_solutions.append(temp_u)
    
    def update_time(self, j):
        """
        Met à jour le temps courant et les temps intermédiaires pour les étapes DIRK.
        
        Parameters
        ----------
        j : int Numéro du pas de temps.
        """
        super().update_time(j)
        self.stage_times = [self.t + self.dirk_params.c[s] * self.dt for s in range(self.num_stages)]
    

    def set_stage_system(self, stage):
        """
        Prépare le système pour une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int Numéro de l'étape DIRK (0 à num_stages-1).
        """
        self.save_original_state() # Sauvegarder les solutions originales
        self.calculate_stage_initial_approximation(stage) # Calculer l'approximation initiale pour cette étape
        self.configure_stage_residual(stage)# Configurer le résidu pour cette étape
    
    def save_original_state(self):
        """
        Sauvegarde l'état original du problème avant modification pour une étape DIRK.
        """
        # Sauvegarder les solutions actuelles
        for i, u_base in enumerate(self.pb.U_base):
            self.temp_solutions[i].x.array[:] = u_base.x.array
    
    def calculate_stage_initial_approximation(self, stage):
        """
        Version simplifiée qui utilise directement les coefficients A
        de Butcher pour calculer l'approximation initiale.
        """
        for i, u_n in enumerate(self.pb.Un_base):
            # Commencer par la solution au pas de temps précédent
            self.pb.U_base[i].x.array[:] = u_n.x.array.copy()
            # Ajouter les contributions des étapes précédentes (si applicable)
            for prev_stage in range(stage):
                a_ij = self.dirk_params.A[stage, prev_stage]
                if a_ij != 0.0:  # Éviter les calculs inutiles
                    prev_u = self.stage_solutions[prev_stage][i]
                    # K_j ≈ (prev_u - u_n) / dt
                    self.pb.U_base[i].x.array[:] += a_ij * (prev_u.x.array - u_n.x.array)
    
    def configure_stage_residual(self, stage):
        """
        Configure le résidu pour une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int
            Numéro de l'étape.
        """
        # Pour cette étape, on modifie le facteur du terme temporel
        a_ii = self.dirk_params.A[stage, stage]
        self.pb.dt_factor.value = 1.0 / (self.dt * a_ii)
        
        # Mettre à jour les conditions aux limites pour le temps intermédiaire
        stage_time = self.stage_times[stage]
        stage_step = int(stage_time / (self.Tfin/self.num_time_steps))
        self.pb.update_bcs(stage_step)
        
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[stage_step]
        
    def solve_stage(self, stage):
        """
        Résout une étape spécifique du schéma DIRK.
        """
        self.set_stage_system(stage)
        self.problem_solve()  # Résoudre le système pour cette étape
        
        # Stocker la solution pour cette étape
        for i, u in enumerate(self.pb.U_base):
            self.stage_solutions[stage][i].x.array[:] = u.x.array
        self.pb.dt_factor.value = 1.0 / self.dt  # Rétablir le facteur par défaut                
                
    def compute_final_solution(self):
        """
        Calcule la solution finale en utilisant la formule du schéma DIRK.
        Version modifiée pour gérer correctement les étapes explicites.
        """
        # Calculer les z_h^{n,i} en séquence
        z_stage = []
        
        for stage in range(self.num_stages):
            z_current = []
            for i, u_n in enumerate(self.pb.Un_base):
                a_ii = self.dirk_params.A[stage, stage]
                stage_u = self.stage_solutions[stage][i]
                
                # Calcul spécial pour l'étape explicite (a_ii = 0)
                if abs(a_ii) < 1e-14:  # Considéré comme zéro
                    if stage == 0:  # Première étape (explicite)
                        # Pour la première étape explicite, utiliser une approximation
                        # On peut sauter cette étape car elle ne contribue pas à la solution
                        # finale (le coefficient b[0] est généralement 0 pour ESDIRK)
                        z_i = np.zeros_like(stage_u.x.array)
                    else:
                        # Cela ne devrait pas arriver avec des méthodes ESDIRK standard
                        print(f"AVERTISSEMENT: a_ii proche de zéro à l'étape {stage}")
                        # Utiliser une approximation par différence finie
                        z_i = (stage_u.x.array - u_n.x.array) / self.dt
                else:
                    # Calcul normal pour z_h^{n,i}
                    z_i = (stage_u.x.array - u_n.x.array) / (self.dt * a_ii)
                    
                    # Soustraire les contributions des étapes précédentes
                    if stage > 0:
                        for prev_stage in range(stage):
                            a_ij = self.dirk_params.A[stage, prev_stage]
                            if abs(a_ij) > 1e-14:  # Ignorer les contributions nulles
                                prev_z = z_stage[prev_stage][i]
                                if abs(a_ii) > 1e-14:  # Éviter division par zéro
                                    z_i -= (a_ij / a_ii) * prev_z
                
                z_current.append(z_i)
            
            z_stage.append(z_current)
        
        # Calculer la solution finale
        for i, u_n in enumerate(self.pb.Un_base):
            # Commencer avec la solution au pas de temps précédent
            self.pb.U_base[i].x.array[:] = u_n.x.array.copy()
            
            # Ajouter les contributions de chaque étape
            for stage in range(self.num_stages):
                b_s = self.dirk_params.b[stage]
                if abs(b_s) < 1e-14:  # Ignorer les contributions nulles
                    continue
                    
                z_i = z_stage[stage][i]
                
                # U^{n+1} = U^n + dt * sum(b_i * z_h^{n,i})
                self.pb.U_base[i].x.array[:] += self.dt * b_s * z_i
                
        
    def iterative_solve(self, **kwargs):
        """
        Résout le problème en utilisant un schéma DIRK.
        
        Parameters
        ----------
        **kwargs : dict Arguments pour la méthode parent.
        """
        compteur_output = kwargs.get("compteur", 1)
        num_time_steps = self.num_time_steps
        if compteur_output != 1:
            self.is_compteur = True
            self.compteur = 0 
        else:
            self.is_compteur = False            
        
        with tqdm(total=num_time_steps, desc="Progression", unit="pas") as pbar:
            j = 0
            while self.t < self.Tfin:
                # Mettre à jour le temps
                self.update_time(j)
                
                # Sauvegarder la solution au pas de temps précédent
                for i, (x, x_n) in enumerate(zip(self.pb.U_base, self.pb.Un_base)):
                    x_n.x.array[:] = x.x.array
                
                # Résoudre chaque étape du schéma DIRK
                for stage in range(self.num_stages):
                    print(f"Solving DIRK stage {stage+1}/{self.num_stages} at time {self.stage_times[stage]:.6f}")
                    self.solve_stage(stage)
                
                # Calculer la solution finale
                self.compute_final_solution()
                
                # Mettre à jour les champs (pression artificielle, etc.)
                self.pb.artificial_pressure.compute_artificial_pressure()
                
                # Incrémenter et exporter
                j += 1
                self.output(compteur_output)
                pbar.update(1)
        
        self.export.csv.close_files()
        self.pb.final_output()