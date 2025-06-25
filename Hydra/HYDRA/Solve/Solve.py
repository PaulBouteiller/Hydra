"""
Time integration and nonlinear solving for compressible fluid simulations
========================================================================

This module provides the main solving infrastructure for time-dependent compressible
fluid dynamics simulations. It implements high-order DIRK (Diagonally Implicit Runge-Kutta)
time integration schemes coupled with Newton-type nonlinear solvers.

The implementation offers:
- Multiple DIRK schemes of orders 1-5 with various stability properties
- Efficient handling of nonlinear systems at each time step or stage
- Flexible configuration of time stepping and solver parameters
Classes:
--------
Solve : Main solver class for time-dependent problems
    Manages the time integration process
    Coordinates nonlinear solvers for each time step/stage
    Handles result export and progress tracking
    Implements various DIRK schemes for time integration

Methods:
--------
initialize_solve : Initialize the solver with problem settings
    Sets up export, solver, and time integration parameters

run_simulation : Execute the complete time-dependent simulation
    Implements the main time stepping loop
    Manages DIRK stages and solution updates
    Coordinates result export and progress tracking

solve_stage : Solve a single DIRK stage
    Configures and solves the nonlinear system for a specific stage
    Updates stage solutions for multi-stage schemes

Notes:
------
The module supports various DIRK schemes with different orders and stability properties.
Advanced DIRK schemes (e.g., SDIRK4_BN_Stable_1_0) provide control over numerical
dissipation and are suitable for capturing complex physical phenomena.
"""

from .customNewton import create_newton_solver
# from .SNESBlock2 import BlockedSNESSolver
from ..utils.block import extract_rows, derivative_block
from ..utils.default_parameters import default_Newton_parameters
from ..Export.export_result import ExportResults
from ..utils.dirk_parameters import DIRKParameters
from tqdm import tqdm
import numpy as np
from dolfinx.fem import form, Function

class Solve:
    """
    Main time integration and solution manager for fluid dynamics simulations.
    
    This class handles the complete time-dependent simulation process, including:
    - Initialization of the problem and solution vectors
    - Time integration using DIRK (Diagonally Implicit Runge-Kutta) schemes
    - Nonlinear solution at each time step/stage
    - Result export and progress tracking
    
    It supports various DIRK schemes with different orders of accuracy and
    stability properties, from simple backward Euler to sophisticated
    algebraically stable high-order methods.
    """

    def __init__(self, problem, **kwargs):
        """
        Initialize the solver for time-dependent simulation.
        
        Parameters
        ----------
        problem : Problem The fluid dynamics problem to solve
        **kwargs : dict Additional configuration parameters:
            - dt : float Time step size
            - TFin : float Final simulation time
            - dirk_method : str, optional DIRK scheme to use (default: "BDF1")
            - Prefix : str, optional Prefix for result files
            - compteur : int, optional Export frequency (1 = every time step)
        """
        self.pb = problem
        self.initialize_solve(**kwargs)
        print("Starting simulation")       
        self.run_simulation(**kwargs)

    def initialize_solve(self, **kwargs):
        """
        Initialize the solver components and parameters.

        Parameters
        ----------
        **kwargs : dict Additional configuration parameters
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
        Configure the result export system.
        
        Sets up the export manager for VTK and CSV outputs based on the
        problem's export settings.
        
        Parameters
        ----------
        **kwargs : dict Additional parameters, potentially including 'Prefix'
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
        Configure the nonlinear solver.
        
        Sets up the Newton-type solver for the nonlinear systems arising
        at each time step or stage, including residual and Jacobian forms.
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
    
        # from dolfiny import snesblockproblem
        # self.solver = snesblockproblem.SNESBlockProblem(Fr_form, self.pb.u_list, bcs = self.pb.bc_class.bcs, J_form = J_form)

    def setup_time_parameters(self, **kwargs):
        """
        Configure time stepping parameters.
        
        Sets up the time step, final time, and number of time steps for
        the simulation, and prepares time-dependent boundary conditions.
        
        Parameters
        ----------
        **kwargs : dict Must include 'dt' and 'TFin'
        """
        self.dt = kwargs.get("dt")
        self.Tfin = kwargs.get("TFin")
        self.num_time_steps = int(self.Tfin / self.dt)
        self.load_steps = np.linspace(0, self.Tfin, self.num_time_steps + 1)
        self.pb.bc_class.set_time_dependant_BCs(self.load_steps)
        
    def setup_dirk_scheme(self, **kwargs):
        """
        Configure the DIRK time integration scheme.
        
        Sets up the DIRK scheme parameters based on the specified method,
        initializing the necessary data structures for multi-stage integration.
        
        Parameters
        ----------
        **kwargs : dict May include 'dirk_method' to specify the DIRK scheme
        """
        self.dirk_method = kwargs.get("dirk_method", "BDF1")  # Par défaut: Backward Euler = BDF1
        self.dirk_params = DIRKParameters(self.dirk_method)
        print(f"Initializing with temporal scheme: {self.dirk_params}")
        self.initialize_stages()
        
    def initialize_stages(self):
        """
        Initialize data structures for DIRK stages.
        
        Creates arrays for storing intermediate solutions and right-hand sides
        for each stage of the DIRK scheme, based on the number of stages and
        function spaces.
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
        Export simulation results at the specified time.
        
        Calls the appropriate export functions for VTK and CSV formats
        based on the problem's export configuration.
        
        Parameters
        ----------
        t : float Current simulation time
        """
        self.pb.query_output(t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)

    def run_simulation(self, **kwargs):
        """
        Run the complete time-dependent simulation.
        
        Executes the main time stepping loop, solving the DIRK stages
        at each time step and exporting results at the specified frequency.
        
        Parameters
        ----------
        **kwargs : dict May include 'compteur' to specify export frequency
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
        Handle periodic result export.
        
        Exports results based on the specified frequency counter.
        
        Parameters
        ----------
        compteur_output : int Export frequency (number of time steps between exports)
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
        Update the current time and calculate intermediate stage times.
        
        Parameters
        ----------
        j : int Current time step index
        """
        self.t = self.load_steps[j]
        
        # Calcul des temps intermédiaires pour les étapes DIRK
        self.stage_times = [self.t + self.dirk_params.c[s] * self.dt for s in range(self.num_stages)]
    
    def configure_stage_residual(self, stage):
        """
        Configure the residual for a specific DIRK stage.
        
        Sets up the dt factor and source terms for the current stage,
        and updates boundary conditions for the stage time.
        
        Parameters
        ----------
        stage : int Stage index
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
        Calculate the source terms for a specific DIRK stage.
        
        Implements the formula for computing the source terms s_h^{n,i}
        according to the DIRK scheme theory.
        
        Parameters
        ----------
        stage : int Stage index
            
        Returns
        -------
        list Source terms for each variable
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
        Solve a specific DIRK stage.
        
        Configures the system for the current stage, solves the nonlinear
        system, and stores the solution.
        
        Parameters
        ----------
        stage : int Stage index to solve
        """
        self.configure_stage_residual(stage)
        
        # Résolution du système non-linéaire
        self.solver.solve()
        
        # Stockage de la solution
        for i, u in enumerate(self.pb.U):
            self.stage_solutions[stage][i].x.array[:] = u.x.array
                
    def compute_final_solution(self):
        """
        Compute the final solution for the current time step.
        
        For BDF1, uses the solution directly from the single stage.
        For higher-order methods, applies the complete DIRK update formula.
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
        Update fields for the next time step.
        
        Copies the solutions from the current time step to the 'n' fields
        for the next iteration, and updates artificial viscosity if needed.
        """
        for x, x_n in zip(self.pb.U, self.pb.U_n):
            x_n.x.array[:] = x.x.array
        
        # Mise à jour de la viscosité artificielle basée sur le nouvel état
        # Désactivée par défaut, décommenter si nécessaire
        if self.pb.shock_stabilization:
            # raise ValueError("Actuellement buggué")
            self.pb.p_star_U.interpolate(self.pb.p_star_U_expr)