"""
Created on Tue Mar 11 16:30:20 2025

@author: bouteillerp
"""
from .Solve import Solve
from ..utils.dirk_parameters import DIRKParameters
from tqdm import tqdm
from dolfinx.fem import Function

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
        
        # Initialiser le solveur parent (sans lancer la résolution)
        self.pb = problem
        self.initialize_solve(problem, **kwargs)
        # Initialiser les structures pour les étapes intermédiaires
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
        
        # Calculer les temps intermédiaires pour chaque étape
        self.stage_times = [self.t + self.dirk_params.c[s] * self.dt for s in range(self.num_stages)]
    
    def set_stage_system(self, stage):
        """
        Prépare le système pour une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int Numéro de l'étape DIRK (0 à num_stages-1).
        """
        # Sauvegarder les solutions originales
        self.save_original_state()
        
        # Calculer l'approximation initiale pour cette étape
        self.calculate_stage_initial_approximation(stage)
        
        # Configurer le résidu pour cette étape
        self.configure_stage_residual(stage)
    
    def save_original_state(self):
        """
        Sauvegarde l'état original du problème avant modification pour une étape DIRK.
        """
        self._original_residual = self.pb.residual
        
        # Sauvegarder les solutions actuelles
        for i, u_base in enumerate(self.pb.U_base):
            self.temp_solutions[i].x.array[:] = u_base.x.array
    
    def calculate_stage_initial_approximation(self, stage):
        """
        Calcule l'approximation initiale pour une étape DIRK.
        
        Parameters
        ----------
        stage : int
            Numéro de l'étape.
        """
        # Pour chaque composante du vecteur solution
        for i, u_n in enumerate(self.pb.Un_base):
            # Partir de la solution au pas de temps précédent
            u_approx = u_n.x.array.copy()
            
            # Ajouter les contributions des étapes précédentes
            for prev_stage in range(stage):
                a_ij = self.dirk_params.A[stage, prev_stage]
                if a_ij != 0.0:
                    prev_u = self.stage_solutions[prev_stage][i]
                    a_jj = self.dirk_params.A[prev_stage, prev_stage]
                    # K_j = (prev_u - u_n) / (dt * a_jj)
                    u_approx += self.dt * a_ij * (prev_u.x.array - u_n.x.array) / (self.dt * a_jj)
            
            # Utiliser cette approximation comme point de départ pour la résolution
            self.pb.U_base[i].x.array[:] = u_approx
    
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
        self.pb.dt_factor = 1.0 / (self.dt * a_ii)
        
        # Mettre à jour les conditions aux limites pour le temps intermédiaire
        stage_time = self.stage_times[stage]
        stage_step = int(stage_time / (self.Tfin/self.num_time_steps))
        self.pb.update_bcs(stage_step)
        
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[stage_step]
        
        # Recalculer le résidu complet
        self.pb.set_form()
    
    def restore_original_state(self):
        """
        Restaure l'état original du problème après une étape DIRK.
        """
        self.pb.residual = self._original_residual
        self.pb.dt_factor = 1.0 / self.dt  # Rétablir le facteur par défaut
        
        # Restaurer les solutions
        for i, u_base in enumerate(self.pb.U_base):
            u_base.x.array[:] = self.temp_solutions[i].x.array
    
    def solve_stage(self, stage):
        """
        Résout une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int Numéro de l'étape DIRK à résoudre.
        """
        self.set_stage_system(stage)
        self.problem_solve() # Résoudre le système pour cette étape
        
        # Stocker la solution pour cette étape
        for i, u in enumerate(self.pb.U_base):
            self.stage_solutions[stage][i].x.array[:] = u.x.array
        
        # Restaurer l'état original
        self.restore_original_state()
    
    def compute_final_solution(self):
        """
        Calcule la solution finale en utilisant la formule du schéma DIRK.
        """
        # Pour chaque variable
        for i, u_n in enumerate(self.pb.Un_base):
            # Commencer avec la solution au pas de temps précédent
            self.pb.U_base[i].x.array[:] = u_n.x.array
            
            # Ajouter les contributions de chaque étape
            for stage in range(self.num_stages):
                b_s = self.dirk_params.b[stage]
                a_ss = self.dirk_params.A[stage, stage]
                stage_u = self.stage_solutions[stage][i]
                
                # K_s = (stage_u - u_n) / (dt * a_ss)
                # U^{n+1} = U^n + dt * sum(b_s * K_s)
                self.pb.U_base[i].x.array[:] += self.dt * b_s * (stage_u.x.array - u_n.x.array) / (self.dt * a_ss)
    
    def iterative_solve(self, **kwargs):
        """
        Résout le problème en utilisant un schéma DIRK.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments pour la méthode parent.
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