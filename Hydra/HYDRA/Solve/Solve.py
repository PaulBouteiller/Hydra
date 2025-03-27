"""
Created on Tue Mar  8 15:51:14 2022

@author: bouteillerp
"""
from ..utils.block import extract_rows, derivative_block
from ..utils.default_parameters import default_Newton_parameters
from ..Export.export_result import ExportResults
from .customblockedNewton import BlockedNewtonSolver
from ..utils.dirk_parameters import DIRKParameters
from tqdm import tqdm
from numpy import linspace, zeros_like
from dolfinx.fem import form, Function

class Solve:
    """
    Classe de base pour les méthodes de résolution temporelle.
    Implémente la méthode de base pour l'intégration temporelle.
    """
    def __init__(self, problem, **kwargs):
        """
        Initialise le solveur pour l'intégration temporelle.
        
        Parameters
        ----------
        problem : Problem Le problème à résoudre.
        **kwargs : dict Arguments supplémentaires pour la configuration.
        """
        self.pb = problem
        self.initialize_solve(problem, **kwargs)
        print("Start solving")       
        self.iterative_solve(**kwargs)

    def initialize_solve(self, problem, **kwargs):
        """
        Initialise le solveur avec les paramètres de base.
        
        Parameters
        ----------
        problem : Problem Le problème à résoudre.
        **kwargs : dict Arguments supplémentaires pour la configuration.
        """
        self.pb.set_initial_conditions()
        self.t = 0        
        self.export = ExportResults(problem, kwargs.get("Prefix", self.pb.prefix()), 
                                   self.pb.set_output(), self.pb.csv_output())
        self.set_solver()
        self.Tfin = kwargs.get("TFin")
        self.dt = kwargs.get("dt")
        self.num_time_steps = int(self.Tfin / self.dt)
        self.load_steps = linspace(0, self.Tfin, self.num_time_steps + 1)
        self.pb.bc_class.set_time_dependant_BCs(self.load_steps)
        
        # Initialisation des paramètres DIRK
        self.dirk_method = kwargs.get("dirk_method", "BDF1")  # Par défaut: Backward Euler = BDF1
        self.dirk_params = DIRKParameters(self.dirk_method)
        print(f"Initializing with temporal scheme: {self.dirk_params}")
        self.initialize_stages()
        
    def initialize_stages(self):
        """
        Initialise les structures pour stocker les étapes intermédiaires.
        Pour Backward Euler (BDF1), il n'y a qu'une seule étape.
        """
        self.num_stages = self.dirk_params.num_stages
        self.stage_solutions = []
        for s in range(self.num_stages):
            # Cloner les fonctions de solution pour chaque étape
            stage_u = []
            for u in self.pb.U:
                u_stage = u.copy()
                stage_u.append(u_stage)
            self.stage_solutions.append(stage_u)
        self.stage_rhs = [[Function(self.pb.V_rho), Function(self.pb.V_v), Function(self.pb.V_rho)] for _ in range(self.num_stages)]
        for i, s in enumerate(self.pb.s):
            s.x.array[:] = self.pb.U_n[i].x.array * self.pb.dt_factor.value
            self.stage_rhs[0][i].x.array[:] = s.x.array
        
    def set_solver(self):
        """
        Initialise les solveurs successivement appelés.
        """         
        Fr = extract_rows(self.pb.residual, self.pb.u_test_list)
        J = derivative_block(Fr, self.pb.u_list, self.pb.du_list)
        Fr_form = form(Fr, entity_maps=self.pb.entity_maps)
        J_form = form(J, entity_maps=self.pb.entity_maps)
        petsc_options = default_Newton_parameters()
        petsc_options.update({
            # Options générales pour MUMPS
            "pc_factor_mat_solver_type": "mumps",
            # Activer BLR (Block Low-Rank) pour MUMPS
            "mat_mumps_icntl_35": 1,
            "mat_mumps_cntl_7": 1e-8,
        })
        
        self.solver = BlockedNewtonSolver(Fr_form, self.pb.u_list, J_form, bcs = self.pb.bc_class.bcs, 
                                          petsc_options=petsc_options, 
                                          entity_maps = self.pb.entity_maps)


    def problem_solve(self):
        """
        Résout le problème non linéaire.
        """
        self.solver.solve()

    def update_time(self, j):
        """
        Actualise le temps courant et les temps intermédiaires pour les étapes DIRK.
        Parameters
        ----------
        j : int Numéro du pas de temps.
        """
        t = self.load_steps[j]
        self.t = t     
        # Calcul des temps intermédiaires pour les étapes DIRK
        self.stage_times = [self.t + self.dirk_params.c[s] * self.dt for s in range(self.num_stages)]

    def set_stage_system(self, stage):
        """
        Prépare le système pour une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int Numéro de l'étape DIRK (0 à num_stages-1).
        """
        self.configure_stage_residual(stage)  # Configurer le résidu pour cette étape
    
    def configure_stage_residual(self, stage):
        """
        Configure le résidu pour une étape spécifique du schéma DIRK.
        
        Parameters
        ----------
        stage : int  Numéro de l'étape.
        """
        # Pour cette étape, on modifie le facteur du terme temporel
        a_ii = self.dirk_params.A[stage, stage]
        self.pb.dt_factor.value = 1.0 / (self.dt * a_ii)
        stage_s = self.calculate_stage_source_term(stage)
        for i, s in enumerate(self.pb.s):
            s.x.array[:] = stage_s[i].x.array            
        
        # Mettre à jour les conditions aux limites pour le temps intermédiaire
        stage_time = self.stage_times[stage]
        stage_step = int(stage_time / (self.Tfin/self.num_time_steps))
        self.pb.update_bcs(stage_step)
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[stage_step]
            
    def calculate_stage_source_term(self, stage):
        """
        Calcule les termes sources s_h^{n,i} pour l'étape 'stage' selon la formulation théorique.
        """
        # Pour la première étape, les termes sources ont déjà été initialisés dans votre code
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
                if abs(a_ij) > 1e-14:  # Ignorer les coefficients proches de zéro
                    a_jj = self.dirk_params.A[j, j]
                    prev_u = self.stage_solutions[j][i]
                    prev_s = self.stage_rhs[j][i]
                    
                    # Calculer (U_h^{n,j}/(a_jj*dt) - s_h^{n,j})
                    correction = prev_u.x.array / (a_jj * self.dt) - prev_s.x.array
                    
                    # Ajouter (a_ij/a_ii) * correction
                    self.stage_rhs[stage][i].x.array[:] += (a_ij / a_ii) * correction
        
        return self.stage_rhs[stage]
            
    def solve_stage(self, stage):
        """
        Résout une étape spécifique du schéma DIRK.
        Parameters
        ----------
        stage : int Numéro de l'étape à résoudre.
        """
        self.set_stage_system(stage)
        self.problem_solve()        
        # Stocker la solution pour cette étape
        for i, u in enumerate(self.pb.U):
            self.stage_solutions[stage][i].x.array[:] = u.x.array
                
    def compute_final_solution(self):
        """
        Calcule la solution finale en utilisant la formule du schéma DIRK.
        Pour BDF1, cela revient à utiliser la solution de l'étape unique.
        """
        # if self.dirk_method == "BDF1":
        #     pass
        #     # Pour BDF1, la solution finale est celle de l'unique étape
        #     # for i, u in enumerate(self.pb.U):
        #     #     u.x.array[:] = self.stage_solutions[0][i].x.array
        # else:
        if self.dirk_method != "BDF1":
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
                # Commencer avec la solution au pas de temps précédent
                self.pb.U[i].x.array[:] = u_n.x.array.copy()
                
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
        """
        for x, x_n in zip(self.pb.U, self.pb.U_n):
            x_n.x.array[:] = x.x.array
            
        # Mettre à jour la viscosité artificielle basée sur le nouvel état
        # self.pb.artificial_pressure.compute_artificial_pressure()
        
    def update_bcs(self, num_pas):
        """
        Mise à jour des conditions aux limites de Dirichlet et de Neumann
        """
        self.pb.update_bcs(num_pas)          
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[num_pas]

    def iterative_solve(self, **kwargs):
        """
        Boucle temporelle principale.
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
                
                # Résoudre chaque étape du schéma DIRK
                for stage in range(self.num_stages):
                    if self.num_stages > 1:
                        print(f"Solving DIRK stage {stage+1}/{self.num_stages} at time {self.stage_times[stage]:.6f}")
                    self.solve_stage(stage)
                
                # Calculer la solution finale
                self.compute_final_solution()
                self.update_fields()
                
                # Incrémenter et exporter
                j += 1
                self.output(compteur_output)
                pbar.update(1)
        
        self.export.csv.close_files()
        self.pb.final_output()
            
    def output(self, compteur_output):
        """
        Permet l'export de résultats tout les 'compteur_output' - pas de temps
        ('compteur_output' doit être un entier), tout les pas de temps sinon.      
        Parameters
        ----------
        compteur_output : Int ou None Compteur donnant la fréuence d'export des résultats.
        """
        if self.is_compteur:
            if self.compteur == compteur_output:
                self.in_loop_export(self.t)
                self.compteur=0
            self.compteur+=1
        else:
            self.in_loop_export(self.t)
            
    def in_loop_export(self, t):
        """
        Exporte les résultats au temps t.
        
        Parameters
        ----------
        t : float Temps auquel exporter les résultats.
        """
        self.pb.query_output(t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)