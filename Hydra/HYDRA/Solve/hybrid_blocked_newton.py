from time import time
from petsc4py import PETSc
from dolfinx.log import LogLevel, set_log_level
from .customblockedNewton import BlockedNewtonSolver

class HybridBlockedNewtonSolver(BlockedNewtonSolver):
    """
    Solveur de Newton hybride qui alterne entre solveurs linéaires directs et itératifs.
    
    Cette implémentation commence par une résolution directe pour la robustesse,
    puis bascule vers un solveur itératif pour l'efficacité. En cas d'échec du
    solveur itératif, elle retombe sur un solveur direct.
    """
    def __init__(self, F, u, J, bcs=None, petsc_options=None, entity_maps=None,
                 hybrid_parameters=None, direct_solver=None, iterative_solver=None):
        """
        Initialise le solveur de Newton hybride.
        
        Parameters
        ----------
        F, u, J, bcs, petsc_options, entity_maps:
            Mêmes paramètres que pour BlockedNewtonSolver
            
        hybrid_parameters : dict, optional
            Paramètres contrôlant le comportement hybride:
            - iteration_switch: Nombre maximal d'itérations avant de reconstruire le préconditionneur
            - debug: Activer les sorties de debug
            
        direct_solver : dict, optional
            Configuration du solveur direct:
            - solver: Type de solveur (mumps, superlu_dist, etc.)
            - type: Type de factorisation (lu, cholesky, etc.)
            
        iterative_solver : dict, optional
            Configuration du solveur itératif:
            - solver: Type de solveur (gmres, cg, etc.)
            - maximum_iterations: Nombre maximal d'itérations
        """
        # Initialiser la classe parente
        super().__init__(F, u, J, bcs, petsc_options, entity_maps)
        
        # Paramètres par défaut
        self.hybrid_parameters = hybrid_parameters or {"iteration_switch": 10,"debug": False}
        
        self.direct_solver = direct_solver or {"solver": "mumps","type": "lu"}
        
        self.iterative_solver = iterative_solver or {"solver": "gmres","maximum_iterations": 200}
        
        # État de la stratégie hybride
        self.reuse_preconditioner = False
        self.is_direct_solve = True  # La première résolution est toujours directe
        self.ksp_iterations = 0
        self.debug = self.hybrid_parameters.get("debug", False)
        
        # Supprimer les logs excessifs
        set_log_level(LogLevel.WARNING)
        
        # Configurer KSP pour la résolution directe initialement
        self._configure_ksp_direct()
        
        # Remplacer le callback de mise à jour pour gérer la stratégie hybride
        self._original_update = self._update_function
        self.set_update(self._hybrid_update_function)

    def _log(self, message):
        """Affiche un message de debug si le mode debug est activé."""
        if self.debug and self._u[0].function_space.mesh.comm.rank == 0:
            print(f"[Hybrid] {message}")

    def _configure_ksp_direct(self):
        """Configure KSP pour une résolution directe."""
        ksp = self.krylov_solver
        pc = ksp.getPC()
        
        # Utiliser le type KSP PREONLY pour la résolution directe
        ksp.setType(PETSc.KSP.Type.PREONLY)
        
        # Définir le type de préconditionneur
        pc.setType(self.direct_solver["type"])
        
        # Configurer le solveur direct
        opts = PETSc.Options()
        prefix = pc.getOptionsPrefix() or ""
        
        # Définir le type de solveur
        opts[f"{prefix}pc_factor_mat_solver_type"] = self.direct_solver["solver"]
        
        # Configurer MUMPS si utilisé
        if self.direct_solver["solver"] == "mumps":
            opts[f"{prefix}mat_mumps_icntl_35"] = 1  # Activer la compression BLR
            opts[f"{prefix}mat_mumps_cntl_7"] = 1e-10  # Tolérance BLR
        
        # Appliquer les options
        ksp.setFromOptions()
        
    def _configure_ksp_iterative(self):
        """Configure KSP pour une résolution itérative."""
        ksp = self.krylov_solver
        pc = ksp.getPC()
        
        # Définir le type de solveur itératif
        ksp.setType(self.iterative_solver["solver"])
        
        # Définir le nombre maximum d'itérations
        if "maximum_iterations" in self.iterative_solver:
            ksp.setTolerances(max_it=self.iterative_solver["maximum_iterations"])
        
        # Configurer pour réutiliser le préconditionneur
        pc.setReusePreconditioner(self.reuse_preconditioner)
        
        # Appliquer les options
        ksp.setFromOptions()
        
    def _pre_newton_iteration(self, x):
        """Prépare l'itération de Newton."""
        # Appeler la méthode parente
        super()._pre_newton_iteration(x)
        
        # Configurer KSP selon la stratégie
        if self.is_direct_solve:
            self._log("Utilisation du solveur direct")
            self._configure_ksp_direct()
        else:
            self._log(f"Utilisation du solveur itératif avec réutilisation du préconditionneur: {self.reuse_preconditioner}")
            self._configure_ksp_iterative()
        
    def _hybrid_update_function(self, solver, dx, x):
        """Met à jour la solution et gère la stratégie hybride."""
        # Appeler la méthode de mise à jour originale
        self._original_update(solver, dx, x)
        
        # Obtenir les statistiques KSP
        self.ksp_iterations = self.krylov_solver.getIterationNumber()
        converged_reason = self.krylov_solver.getConvergedReason()
        
        # Stratégie pour la première résolution directe
        if self.is_direct_solve:
            self._log("Première résolution directe terminée, passage au solveur itératif")
            self.is_direct_solve = False
            self.reuse_preconditioner = True
            return
            
        # Mettre à jour la stratégie de réutilisation du préconditionneur en fonction des performances
        if converged_reason > 0:
            # Convergence réussie
            self._log(f"Résolution itérative convergée en {self.ksp_iterations} itérations")
            
            # Décider si on réutilise le préconditionneur
            max_it = self.hybrid_parameters.get("iteration_switch", 10)
            self.reuse_preconditioner = self.ksp_iterations < max_it
            
            if not self.reuse_preconditioner:
                self._log("Trop d'itérations, reconstruction du préconditionneur pour la prochaine fois")
        else:
            # Échec de convergence
            self._log(f"Échec de convergence de la résolution itérative, raison: {converged_reason}")
            self.reuse_preconditioner = False
            
    def solve(self):
        """
        Résout le problème non-linéaire avec la stratégie hybride.
        
        Returns
        -------
        tuple
            (nombre d'itérations, drapeau de convergence)
        """
        start_time = time()
        
        try:
            # Tentative de résolution standard
            n, converged = super().solve()
            
            solve_time = time() - start_time
            self._log(f"Résolution terminée en {n} itérations et {solve_time:.3f} secondes")
            
            return n, converged
            
        except Exception as e:
            # Si le solveur itératif échoue, essayer le solveur direct
            if not self.is_direct_solve:
                self._log(f"Échec du solveur itératif: {str(e)}")
                self._log("Passage au solveur direct")
                
                # Configurer pour une résolution directe
                self.is_direct_solve = True
                
                # Nouvelle tentative avec le solveur direct
                start_time = time()
                try:
                    n, converged = super().solve()
                    
                    solve_time = time() - start_time
                    self._log(f"Résolution directe terminée en {n} itérations et {solve_time:.3f} secondes")
                    
                    # Continuer à utiliser le solveur direct pour la prochaine étape
                    self.is_direct_solve = True
                    return n, converged
                    
                except Exception as e2:
                    self._log(f"Le solveur direct a également échoué: {str(e2)}")
                    raise
            else:
                # Si le solveur direct échoue à la première tentative, relancer l'erreur
                raise