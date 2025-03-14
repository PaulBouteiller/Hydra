"""
Created on Tue Mar  8 15:51:14 2022

@author: bouteillerp
"""
from ..utils.block import extract_rows, derivative_block
from ..Export.export_result import ExportResults
from .customblockedNewton import BlockedNewtonSolver
from .hybrid_blocked_newton import HybridBlockedNewtonSolver
from tqdm import tqdm
from numpy import linspace

class Solve:
    def __init__(self, problem, **kwargs):
        self.pb = problem
        self.initialize_solve(problem, **kwargs)
        print("Start solving")       
        self.iterative_solve(**kwargs)
        
    def initialize_solve(self, problem, **kwargs):
        self.pb.set_initial_conditions()
        self.t = 0        
        self.export = ExportResults(problem, kwargs.get("Prefix", self.pb.prefix()), \
                                    self.pb.set_output(), self.pb.csv_output())
        self.set_solver()
        self.Tfin = kwargs.get("TFin")
        self.dt = kwargs.get("dt")
        self.num_time_steps = int(self.Tfin / self.dt)
        self.load_steps = linspace(0, self.Tfin, self.num_time_steps + 1)
        self.pb.bc_class.set_time_dependant_BCs(self.load_steps)
        
    def set_solver(self):
        """
        Initialise les solveurs successivement appelés.
        """         
        Fr = extract_rows(self.pb.residual, self.pb.u_test_list)
        J = derivative_block(Fr, self.pb.u_list, self.pb.du_list)
        
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        self.solver = BlockedNewtonSolver(Fr, self.pb.u_list, J, bcs = self.pb.bc_class.bcs, 
                                          petsc_options=petsc_options, 
                                          entity_maps = self.pb.entity_maps)

        # self.solver = HybridBlockedNewtonSolver(Fr, self.pb.u_list, J, bcs = self.pb.bc_class.bcs, 
        #                                   entity_maps = self.pb.entity_maps)
    

    def problem_solve(self):
        self.solver.solve()

    def iterative_solve(self, **kwargs):
        """
        Boucle temporelle
        Parameters
        ----------
        **kwargs : Int, si un compteur donné par un int est spécifié
                        l'export aura lieu tout les compteur-pas de temps.
        """
        compteur_output = kwargs.get("compteur", 1)
        num_time_steps = self.num_time_steps
        if compteur_output !=1:
            self.is_compteur = True
            self.compteur=0 
        else:
            self.is_compteur = False            
        with tqdm(total=num_time_steps, desc="Progression", unit="pas") as pbar:
            j = 0
            while self.t < self.Tfin:
                self.update_time(j)
                self.update_bcs(j)
                self.problem_solve()
                self.update_fields()
                j += 1
                self.output(compteur_output)
                pbar.update(1)
    
        self.export.csv.close_files()
        self.pb.final_output()

            
    def output(self, compteur_output):
        """
        Permet l'export de résultats tout les 'compteur_output' - pas de temps
        ('compteur_output' doit être un entier), tout les pas de temps sinon        

        Parameters
        ----------
        compteur_output : Int ou None, compteur donnant la fréuence d'export des résultats.
        """
        if self.is_compteur:
            if self.compteur == compteur_output:
                self.in_loop_export(self.t)
                self.compteur=0
            self.compteur+=1
        else:
            self.in_loop_export(self.t)
            
    def in_loop_export(self, t):
        self.pb.query_output(t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)
            
    def update_time(self, j):
        """
        Actualise le temps courant

        Parameters
        ----------
        j : Int, numéro du pas de temps.
        """
        t = self.load_steps[j]
        self.t = t
        
    def update_fields(self):
        """
        Met a jour le déplacement généralisé U_n avec la nouvelle valeur U.
        """
        for x, x_n in zip(self.pb.U_base, self.pb.Un_base):
            x_n.x.array[:] = x.x.array
            # Mettre à jour la viscosité artificielle basée sur le nouvel état
        self.pb.artificial_pressure.compute_artificial_pressure()
        
    def update_bcs(self, num_pas):
        """
        Mise à jour des CL de Dirichlet et de Neumann
        """
        self.pb.update_bcs(num_pas)          
        for i in range(len(self.pb.bc_class.mcl)):            
            self.pb.bc_class.mcl[i].constant.value = self.pb.bc_class.mcl[i].value_array[num_pas]