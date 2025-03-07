from __future__ import annotations
import logging
from petsc4py import PETSc
import ufl
import dolfinx
from time import time
from petsc4py.PETSc import ScatterMode, InsertMode

# Import le solveur de Newton par blocs existant
from .customblockedNewton import BlockedNewtonSolver
# Import les composants du solveur hybride
from .hybrid_solver import SolverStatistics

logger = logging.getLogger(__name__)

# Paramètres par défaut pour les solveurs
_direct_solver_default = {"solver": "mumps", "type": "lu", "blr": True}
_iterative_solver_default = {"solver": "gmres", "maximum_iterations": 200}
_hybrid_parameters_default = {"iteration_switch": 10, "user_switch": True}


class HybridBlockedNewtonSolver(BlockedNewtonSolver):
    """
    Extension du solveur de Newton par blocs qui utilise une stratégie de résolution hybride
    pour le système linéaire à chaque itération.
    
    Le solveur hybride alterne entre une résolution directe et itérative pour le système
    linéaire à chaque itération de Newton, en fonction des performances.
    """
    
    def __init__(self, F: list[ufl.form.Form], u: list[dolfinx.fem.Function],
                 J: list[list[ufl.form.Form]],
                 bcs: list[dolfinx.fem.DirichletBC] = [],
                 form_compiler_options: dict | None = None,
                 jit_options: dict | None = None,
                 petsc_options: dict | None = None,
                 entity_maps: dict | None = None,
                 hybrid_parameters: dict | None = None,
                 direct_solver: dict | None = None,
                 iterative_solver: dict | None = None,
                 log: bool = True,
                 timings: bool = False,
                 collect_stats: bool = True):
        """
        Initialise le solveur de Newton par blocs avec une stratégie de résolution hybride.
        
        Parameters
        ----------
        F: List of PDE residuals [F_0(u, v_0), F_1(u, v_1), ...]
        u: List of unknown functions u=[u_0, u_1, ...]
        J: UFL representation of the Jacobian matrix blocks
        bcs: List of Dirichlet boundary conditions
        form_compiler_options: Options pour la compilation FFCx
        jit_options: Options pour la compilation JIT CFFI
        petsc_options: Options pour le solveur PETSc
        entity_maps: Maps utilisées pour mapper les entités entre différents maillages
        hybrid_parameters: Paramètres pour le comportement hybride
        direct_solver: Configuration du solveur direct
        iterative_solver: Configuration du solveur itératif
        log: Activer le logging
        timings: Activer le suivi des temps d'exécution
        collect_stats: Activer la collecte de statistiques
        """
        # Initialiser le solveur de Newton par blocs
        super().__init__(F, u, J, bcs, form_compiler_options, jit_options, petsc_options, entity_maps)
        
        # Sauvegarder les paramètres du solveur hybride
        self.comm = u[0].function_space.mesh.comm
        self.hybrid_parameters = hybrid_parameters if hybrid_parameters is not None else _hybrid_parameters_default.copy()
        self.direct_solver = direct_solver if direct_solver is not None else _direct_solver_default.copy()
        self.iterative_solver = iterative_solver if iterative_solver is not None else _iterative_solver_default.copy()
        self.log = log
        self.timings = timings
        self.collect_stats = collect_stats
        
        # État du solveur hybride
        self.reuse_preconditioner = False
        self.is_direct_solve = True  # Forcer la première résolution à être directe
        self.tic = None
        
        # Configurer le solveur KSP pour la stratégie hybride
        self._configure_hybrid_ksp()
        
        # Configuration des callbacks pour la stratégie hybride
        # Sauvegarder les callbacks originaux
        self._original_update = None
        
        # Remplacer le callback de mise à jour
        self._original_update = self._update_function
        self.set_update(self._hybrid_update_function)
        
        # Statistiques
        if collect_stats:
            self.stats = SolverStatistics()
            self.residual_history = []
            self.krylov_solver.setMonitor(self._ksp_monitor)
    
    def _ksp_monitor(self, ksp, iteration, residual_norm):
        """
        Fonction de suivi pour KSP qui enregistre les normes résiduelles
        
        Parameters
        ----------
        ksp : PETSc.KSP L'objet KSP
        iteration : int Le numéro d'itération actuel
        residual_norm : float La norme du résidu
        """
        if hasattr(self, 'residual_history'):
            self.residual_history.append(residual_norm)
    
    def _configure_hybrid_ksp(self):
        """Configure le KSP pour utiliser la stratégie hybride"""
        ksp = self.krylov_solver
        pc = ksp.getPC()
        
        # Configurer pour une résolution directe initiale
        ksp.setType(PETSc.KSP.Type.PREONLY)
        
        # Options du préconditionneur (solveur direct)
        pc.setType(self.direct_solver["type"])
        
        # Définir le type de solveur direct
        opts = PETSc.Options()
        pc_prefix = pc.getOptionsPrefix() or ""
        opts[f"{pc_prefix}pc_factor_mat_solver_type"] = self.direct_solver["solver"]
        
        # Options spécifiques pour MUMPS avec compression BLR
        if self.direct_solver["solver"] == "mumps" and self.direct_solver.get("blr", False):
            opts[f"{pc_prefix}mat_mumps_icntl_35"] = 1  # Active BLR
            opts[f"{pc_prefix}mat_mumps_cntl_7"] = 1e-10  # Tolérance BLR
        
        # Paramètres généraux
        ksp.setErrorIfNotConverged(True)
        ksp.setNormType(PETSc.KSP.NormType.NATURAL)
        pc.setReusePreconditioner(True)
        
        # Appliquer les options
        ksp.setFromOptions()
    
    def print_log(self, message):
        """Affiche un message de log si le logging est activé"""
        if self.log and (self.comm.Get_rank() == 0):
            print(message)
    
    def print_timings(self, message):
        """Affiche le temps écoulé depuis le dernier tic"""
        if self.timings and self.tic is not None:
            elapsed = time() - self.tic
            if self.comm.Get_rank() == 0:
                print(f"{message} {elapsed}")
    
    def start_timer(self):
        """Démarre le timer"""
        self.tic = time()
    
    def _update_infos(self):
        """Affiche des informations sur la résolution"""
        if not hasattr(self, '_it'):
            return
            
        self.print_log(f"Converged in {self._it} iterations.")
        if self.reuse_preconditioner:
            self.print_log("Preconditioner will be reused on next solve.")
        else:
            self.print_log("Next solve will be a direct one with matrix factorization.")
    
    def preconditioner_choice(self):
        """
        Détermine si le préconditionneur doit être réutilisé en fonction
        du nombre d'itérations.
        """
        self._it = self.krylov_solver.getIterationNumber()
        self._converged = self.krylov_solver.getConvergedReason()
        
        if self._converged > 0:
            # Réutiliser si le nombre d'itérations est inférieur au seuil
            self.reuse_preconditioner = ((self._it < self.hybrid_parameters["iteration_switch"]) 
                                         and self.hybrid_parameters.get("user_switch", True))
        else:
            # Ne pas réutiliser si la résolution a échoué
            self.reuse_preconditioner = False
    
    def update_pc(self):
        """Configure la réutilisation du préconditionneur"""
        self.preconditioner_choice()
        pc = self.krylov_solver.getPC()
        pc.setReusePreconditioner(self.reuse_preconditioner)
        
        if self.log:
            self._update_infos()
        
        # Enregistrer la réutilisation ou la mise à jour du préconditionneur pour les statistiques
        if self.collect_stats:
            if self.reuse_preconditioner:
                self.stats.record_precond_reuse()
            else:
                self.stats.record_precond_update()
    
    def _hybrid_update_function(self, solver, dx, x):
        """
        Extension du callback de mise à jour de Newton pour gérer la stratégie hybride.
        Cette fonction est appelée après chaque itération de Newton.
        
        Parameters
        ----------
        solver : NewtonSolver
            Le solveur de Newton
        dx : PETSc.Vec
            La correction de Newton
        x : PETSc.Vec
            La solution actuelle
        """
        # Appeler d'abord la fonction d'origine pour mettre à jour la solution
        if self._original_update:
            self._original_update(solver, dx, x)
        else:
            # Si la fonction d'origine n'est pas disponible, utiliser le comportement par défaut de BlockedNewtonSolver
            offset_start = 0
            for ui in self._u:
                Vi = ui.function_space
                num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
                ui.x.petsc_vec.array_w[:num_sub_dofs] -= (
                    self.relaxation_parameter * dx.array_r[offset_start : offset_start + num_sub_dofs]
                )
                ui.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
                offset_start += num_sub_dofs
        
        # Post-traitement pour la stratégie hybride
        # Sauvegarder les statistiques du solveur KSP
        self._it = self.krylov_solver.getIterationNumber()
        self._converged = self.krylov_solver.getConvergedReason()
        
        # Si c'était une résolution directe, passer au solveur itératif pour la prochaine fois
        if self.is_direct_solve:
            self.print_log("First direct solve completed, switching to iterative solver for next solves")
            self.krylov_solver.setType(self.iterative_solver["solver"])
            self.krylov_solver.setFromOptions()
            self.is_direct_solve = False
        
        # Mettre à jour le préconditionneur pour la prochaine résolution
        self.update_pc()
        
        # Si le post-solve callback est défini, l'appeler
        if hasattr(self, '_post_solve_callback') and self._post_solve_callback is not None:
            self._post_solve_callback(self)
    
    def _pre_newton_iteration(self, x):
        """
        Méthode appelée avant chaque itération de Newton.
        Surcharge la méthode de la classe parente pour configurer le solveur hybride.
        """
        # Appeler la méthode parente
        super()._pre_newton_iteration(x)
        
        # Configurer le solveur KSP en fonction de l'état du solveur hybride
        ksp = self.krylov_solver
        current_solver_type = ksp.getType()
        
        # Si c'est une résolution directe, utiliser PREONLY
        if self.is_direct_solve:
            if current_solver_type != PETSc.KSP.Type.PREONLY:
                self.print_log("Switching to direct solver...")
                ksp.setType(PETSc.KSP.Type.PREONLY)
                ksp.setFromOptions()
        else:
            # Sinon, utiliser le solveur itératif
            if current_solver_type != self.iterative_solver["solver"]:
                self.print_log(f"Switching to iterative solver: {self.iterative_solver['solver']}...")
                ksp.setType(self.iterative_solver["solver"])
                if "maximum_iterations" in self.iterative_solver:
                    ksp.setTolerances(max_it=self.iterative_solver["maximum_iterations"])
                ksp.setFromOptions()
    
    def solve(self):
        """
        Résout le problème non-linéaire en utilisant la stratégie hybride.
        Surcharge la méthode de la classe parente pour ajouter la stratégie hybride.
        """
        self.start_timer()
        
        # Afficher le type de solveur au début pour debugging
        current_solver_type = self.krylov_solver.getType()
        self.print_log(f"Starting solve with solver type: {current_solver_type}")
        self.print_log(f"Direct solve: {self.is_direct_solve}, Preconditioner reuse: {self.reuse_preconditioner}")
        
        if self.collect_stats:
            self.residual_history = []
        
        try:
            # Tentative de résolution normale
            n, converged = super().solve()
            solve_time = time() - self.tic
            self.print_timings("Solve time:")
            
            # Enregistrer les statistiques
            if self.collect_stats and hasattr(self, '_it') and hasattr(self, '_converged'):
                self.stats.record_solve(self.is_direct_solve, self._it, solve_time, self._converged)
                self.print_log(f"Solve completed with {self._it} iterations, using {current_solver_type} solver")
                
            return n, converged
            
        except Exception as e:
            # En cas d'échec, forcer une résolution directe
            self.print_log(f"Error caught: {str(e)}")
            self.print_log("Switching to direct solver for next iteration...")
            
            # Configurer pour résolution directe
            self.krylov_solver.setType(PETSc.KSP.Type.PREONLY)
            self.krylov_solver.setFromOptions()
            self.is_direct_solve = True
            
            # Nouvelle tentative
            self.start_timer()
            try:
                n, converged = super().solve()
                solve_time = time() - self.tic
                self.print_timings("Direct solve time:")
                
                # Enregistrer les statistiques
                if self.collect_stats:
                    self._it = 1  # Une seule itération pour le solveur direct
                    self._converged = self.krylov_solver.getConvergedReason()
                    self.stats.record_solve(self.is_direct_solve, self._it, solve_time, self._converged)
                return n, converged
            except Exception as e2:
                # Échec même avec le solveur direct
                self.print_log(f"Direct solver also failed: {str(e2)}")
                if self.collect_stats:
                    self.stats.record_failure()
                raise
    
    def get_statistics(self):
        """
        Retourne l'objet de statistiques
        
        Returns
        -------
        SolverStatistics : L'objet contenant les statistiques du solveur
        """
        if not hasattr(self, 'collect_stats') or not self.collect_stats:
            self.print_log("Statistics collection is disabled.")
            return None
        return self.stats
    
    def print_statistics(self, detailed=False):
        """
        Affiche les statistiques de résolution
        
        Parameters
        ----------
        detailed : bool, optional
            Si True, affiche des statistiques détaillées
        """
        if not hasattr(self, 'collect_stats') or not self.collect_stats:
            self.print_log("Statistics collection is disabled.")
            return
        
        if self.comm.Get_rank() == 0:
            self.stats.print_summary(detailed)