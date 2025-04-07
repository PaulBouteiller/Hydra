from typing import Callable
from dolfinx.fem.petsc import (create_matrix_block, create_vector_block,
                             assemble_matrix_block, assemble_vector_block,
                             assemble_matrix_nest, assemble_vector_nest, 
                             apply_lifting_nest, set_bc_nest)
from dolfinx.fem import bcs_by_block, extract_function_spaces
from dolfinx.cpp.la.petsc import scatter_local_vectors
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx.log import LogLevel, set_log_level
from petsc4py.PETSc import ScatterMode, InsertMode, Options, KSP, PC

class BaseNewtonSolver(NewtonSolver):
    """Classe de base pour les solveurs de Newton utilisant différentes structures PETSc"""
    
    def __init__(self, F, u, J, petsc_options, bcs=None):
        """
        Initialise le solveur de Newton de base
        
        Args:
            F: list[ufl.form.Form] Liste des résidus des EDP [F_0(u, v_0), F_1(u, v_1), ...]
            u: list[dolfinx.fem.Function] Liste des fonctions inconnues u=[u_0, u_1, ...]
            J: list[list[ufl.form.Form]] Représentation UFL du Jacobien
            bcs: list[dolfinx.fem.DirichletBC] Liste des conditions aux limites de Dirichlet
            petsc_options: Options pour le solveur de Krylov PETSc
        """
        # Initialisation de la classe de base
        self.comm = u[0].function_space.mesh.comm
        super().__init__(self.comm)
        
        # Configuration des options PETSc
        prefix = self.krylov_solver.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        if petsc_options is not None:
            # Définir les options PETSc
            opts = Options()
            opts.prefixPush(prefix)
            for k, v in petsc_options.items():
                opts[k] = v
            opts.prefixPop()
            
            self.krylov_solver.rtol = petsc_options.get("relative_tolerance")
            self.krylov_solver.atol = petsc_options.get("absolute_tolerance")
            self.krylov_solver.setFromOptions()
        
        # Stockage des formes et des fonctions
        self._F = F
        self._a = J
        self._bcs = bcs if bcs is not None else []
        self._u = u
        self._pre_solve_callback = None
        self._post_solve_callback = None
        
        # Sauvegarde des options
        self.atol = petsc_options.get("absolute_tolerance")
        self.rtol = petsc_options.get("relative_tolerance")
        self.debug = petsc_options.get("debug", False)
        
        # À implémenter dans les sous-classes:
        # 1. Création des structures pour les vecteurs et matrices
        # 2. Configuration des méthodes d'assemblage et de mise à jour
        # 3. Configuration du solveur de debug si nécessaire
        
    def set_pre_solve_callback(self, callback: Callable[["BaseNewtonSolver"], None]):
        """Définit une fonction callback appelée avant chaque itération de Newton"""
        self._pre_solve_callback = callback
    
    def set_post_solve_callback(self, callback: Callable[["BaseNewtonSolver"], None]):
        """Définit une fonction callback appelée après chaque itération de Newton"""
        self._post_solve_callback = callback
    
    @property
    def L(self):
        """Forme linéaire compilée (forme résiduelle)"""
        return self._F
    
    @property
    def a(self):
        """Forme bilinéaire compilée (forme jacobienne)"""
        return self._a
    
    @property
    def u(self):
        """Fonctions solution"""
        return self._u
    
    def __del__(self):
        """Nettoie les ressources PETSc"""
        if hasattr(self, '_J'):
            self._J.destroy()
        if hasattr(self, '_b'):
            self._b.destroy()
        if hasattr(self, '_dx'):
            self._dx.destroy()
        if hasattr(self, '_x'):
            self._x.destroy()
    
    def solve(self):
        """Résout le problème non-linéaire"""
        # Créer des variables pour suivre les informations d'itération
        iteration_info = {"current": 0, "initial_residual": None}
        
        # Callback pour afficher l'itération et le résidu
        def custom_pre_callback(solver):
            solver._assemble_residual(solver._x, solver._b)
            residual_norm = solver._b.norm()
            if iteration_info["current"] == 0:
                iteration_info["initial_residual"] = residual_norm
            rel_residual = residual_norm / iteration_info["initial_residual"] if iteration_info["initial_residual"] else 0
            print(f"Newton iteration {iteration_info['current']}: r (abs) = {residual_norm} (tol = {solver.atol}), r (rel) = {rel_residual} (tol = {solver.rtol})")
            
            # Incrémenter le compteur d'itérations
            iteration_info["current"] += 1
        
        # Définir le callback
        self.set_pre_solve_callback(custom_pre_callback)
        set_log_level(LogLevel.WARNING)  # Supprimer les messages INFO
        
        if self.debug:
            n, converged = self.debug_solve()
        else:
            n, converged = super().solve(self._x)
            
            # Afficher un message de fin
            print(f"Newton solver finished in {n} iterations and {n} linear solver iterations.")
        
        return n, converged


class BlockedNewtonSolver(BaseNewtonSolver):
    """Solveur de Newton utilisant la structure 'block' de PETSc"""
    
    def __init__(self, F, u, J, petsc_options, bcs=None):
        """Initialisation du solveur de Newton avec structure 'block'"""
        super().__init__(F, u, J, petsc_options, bcs)
        
        # Créer les structures pour les vecteurs et matrices
        self._b = create_vector_block(self._F)
        self._J = create_matrix_block(self._a)
        self._dx = create_vector_block(self._F)
        self._x = create_vector_block(self._F)
        
        # Configurer les options du solveur
        prefix = self.krylov_solver.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        self._J.setOptionsPrefix(prefix)
        self._J.setFromOptions()
        
        # Configurer les méthodes d'assemblage et de mise à jour
        self.setJ(self._assemble_jacobian, self._J)
        self.setF(self._assemble_residual, self._b)
        self.set_form(self._pre_newton_iteration)
        self.set_update(self._update_function)
        
        # Configurer le solveur de debug si nécessaire
        if self.debug:
            self._setup_debug_solver()
    
    def _setup_debug_solver(self):
        """Configure le solveur de débogage pour le format 'block'"""
        self.linear_solver = KSP().create(self.comm)
        
        # Configurer pour utiliser un solveur direct
        self.linear_solver.setType(KSP.Type.PREONLY)
        self.linear_solver.getPC().setType(PC.Type.LU)
        opts = Options()
        prefix = f"projector_{id(self)}"
        self.linear_solver.setOptionsPrefix(prefix)
        option_prefix = self.linear_solver.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        self.linear_solver.setFromOptions()
        self.linear_solver.setOperators(self._J)
    
    def _pre_newton_iteration(self, x):
        """Fonction appelée avant le calcul du résidu ou du jacobien"""
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        
        # Scatter les vecteurs locaux
        scatter_local_vectors(x, [ui.x.petsc_vec.array_r for ui in self._u],
                             [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                              for ui in self._u])
        x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _assemble_residual(self, x, b):
        """Assemble le résidu F dans le vecteur b"""
        # Mettre à zéro le vecteur résidu
        with b.localForm() as b_local:
            b_local.set(0.0)
        
        # Assembler le vecteur résidu
        assemble_vector_block(b, self._F, self._a, bcs=self._bcs, x0=x, alpha=-1.0)
        b.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)
    
    def _assemble_jacobian(self, x, A):
        """Assemble la matrice jacobienne"""
        # Mettre à zéro la matrice
        A.zeroEntries()
        # Assembler la matrice jacobienne
        assemble_matrix_block(A, self._a, bcs=self._bcs)
        A.assemble()
    
    def _update_function(self, solver, dx, x):
        """Met à jour la solution"""
        # Mettre à jour la solution
        offset_start = 0
        for ui in self._u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            ui.x.petsc_vec.array_w[:num_sub_dofs] -= (
                self.relaxation_parameter * dx.array_r[offset_start:offset_start + num_sub_dofs]
            )
            ui.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            offset_start += num_sub_dofs
            
        if self._post_solve_callback is not None:
            self._post_solve_callback(self)
    
    def debug_solve(self, print_steps=True):
        """Version de debug du solveur de Newton pour le format 'block'"""
        # Paramètres d'itération
        max_it = int(1e3)
        tol = 1e-8
        n = 0  # nombre d'itérations du solveur de Newton
        converged = False
        
        # Initialiser avec _pre_newton_iteration
        self._pre_newton_iteration(self._x)
        
        # Obtenir la norme du résidu initial pour le critère de convergence relatif
        self._assemble_residual(self._x, self._b)
        initial_residual_norm = self._b.norm()
        if print_steps:
            print(f"Initial residual norm: {initial_residual_norm}")
        
        while n < max_it:
            # Assembler le Jacobien et le résidu
            self._assemble_jacobian(self._x, self._J)
            self._assemble_residual(self._x, self._b)
            
            # Résoudre le système linéaire J*dx = -b
            self.linear_solver.solve(self._b, self._dx)
            self._dx.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            # Mettre à jour la solution
            self._update_function(self, self._dx, self._x)
            
            # Mettre à jour le vecteur x
            scatter_local_vectors(self._x, [ui.x.petsc_vec.array_r for ui in self._u],
                                 [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                                 for ui in self._u])
            self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            # Calculer les normes pour le critère de convergence
            n += 1
            correction_norm = self._dx.norm()
            self._assemble_residual(self._x, self._b)
            error_norm = self._b.norm()
            rel_error_norm = error_norm / initial_residual_norm if initial_residual_norm > 0 else error_norm
            
            if print_steps:
                print(f"Iteration {n}: Correction norm {correction_norm}, "
                     f"Residual (abs) {error_norm}, Residual (rel) {rel_error_norm}")
            
            # Vérifier la convergence
            if correction_norm < tol or error_norm < tol:
                converged = True
                break
        
        return n, converged


class NestedNewtonSolver(BaseNewtonSolver):
    """Solveur de Newton utilisant la structure 'nest' de PETSc"""
    
    def __init__(self, F, u, J, petsc_options, bcs=None):
        """Initialisation du solveur de Newton avec structure 'nest'"""
        super().__init__(F, u, J, petsc_options, bcs)
        
        # Stocker les options PETSc pour y accéder dans d'autres méthodes
        self.petsc_options = petsc_options
        
        # Créer les structures pour les vecteurs et matrices nest
        self._b = assemble_vector_nest(self._F)
        self._J = assemble_matrix_nest(self._a, bcs=self._bcs)
        self._J.assemble()
        
        # Créer les vecteurs pour l'incrémentation et la solution
        self._dx = self._b.copy()
        self._x = self._b.copy()
        
        # Configurer les options du solveur
        prefix = self.krylov_solver.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        self._J.setOptionsPrefix(prefix)
        self._J.setFromOptions()
        
        # Configurer les méthodes d'assemblage et de mise à jour
        self.setJ(self._assemble_jacobian, self._J)
        self.setF(self._assemble_residual, self._b)
        self.set_form(self._pre_newton_iteration)
        self.set_update(self._update_function)
        
        # Configuration spécifique pour le format 'nest'
        self._setup_fieldsplit_preconditioner()
        
        # Configurer le solveur de debug si nécessaire
        if self.debug:
            self._setup_debug_solver()
    
    def _setup_fieldsplit_preconditioner(self):
        """Configure le préconditionneur par champs pour le format 'nest'"""
        # Vérifier si on demande un solveur direct pour toute la matrice
        use_direct_solver = self.petsc_options.get("use_direct_solver", False)
        
        if use_direct_solver:
            # Convertir la matrice nest en matrice monolithique pour solveur direct
            self.krylov_solver.setType("preonly")
            self.krylov_solver.getPC().setType("lu")
            
            # Spécifier le solveur LU à utiliser (MUMPS est souvent disponible)
            solver_package = self.petsc_options.get("direct_solver_package", "mumps")
            self.krylov_solver.getPC().setFactorSolverType(solver_package)
            
            # Convertir implicitement la matrice nest en matrice AIJ
            opts = Options()
            prefix = self.krylov_solver.getOptionsPrefix()
            opts[f"{prefix}pc_factor_mat_solver_type"] = solver_package
            
            # Pour les matrices symétriques, on peut utiliser:
            if self.petsc_options.get("symmetric", False):
                opts[f"{prefix}pc_factor_mat_ordering_type"] = "nd"
        else:
            # Configuration standard avec fieldsplit
            self.krylov_solver.setType("gmres")  # ou "minres" pour des systèmes symétriques
            self.krylov_solver.getPC().setType("fieldsplit")
            self.krylov_solver.getPC().setFieldSplitType(PC.CompositeType.ADDITIVE)
            
            # Récupérer les index sets pour chaque champ
            nest_IS = self._J.getNestISs()
            fields = []
            for i in range(len(self._u)):
                fields.append((f"field_{i}", nest_IS[0][i]))
            
            # Configurer le préconditionneur par champs
            self.krylov_solver.getPC().setFieldSplitIS(*fields)
            
            # Configuration de chaque sous-solveur (importante pour nest)
            field_split_ksp = self.krylov_solver.getPC().getFieldSplitSubKSP()
            for i, ksp_sub in enumerate(field_split_ksp):
                ksp_sub.setType("preonly")
                ksp_sub.getPC().setType("lu")  # ou "gamg" pour des systèmes plus grands
    
    def _setup_debug_solver(self):
        """Configure le solveur de débogage pour le format 'nest'"""
        self.nest_solver = KSP().create(self.comm)
        
        # Vérifier si un solveur direct est demandé pour le mode debug
        use_direct_solver = self.petsc_options.get("use_direct_solver", False)
        
        if use_direct_solver:
            # Configurer un solveur direct pour toute la matrice
            self.nest_solver.setType("preonly")
            self.nest_solver.getPC().setType("lu")
            solver_package = self.petsc_options.get("direct_solver_package", "mumps")
            
            # Configurer les options du solveur direct
            opts = Options()
            prefix = f"dbg_nest_{id(self)}_"
            self.nest_solver.setOptionsPrefix(prefix)
            opts[f"{prefix}pc_factor_mat_solver_type"] = solver_package
            
            if self.petsc_options.get("symmetric", False):
                opts[f"{prefix}pc_factor_mat_ordering_type"] = "nd"
        else:
            # Configuration standard avec fieldsplit
            self.nest_solver.setType("gmres")
            self.nest_solver.getPC().setType("fieldsplit")
            self.nest_solver.getPC().setFieldSplitType(PC.CompositeType.ADDITIVE)
            
            # Configurer les champs pour fieldsplit
            nest_IS = self._J.getNestISs()
            fields = []
            for i in range(len(self._u)):
                fields.append((f"field_{i}", nest_IS[0][i]))
            self.nest_solver.getPC().setFieldSplitIS(*fields)
            
            # Configurer chaque sous-solveur
            field_split_ksp = self.nest_solver.getPC().getFieldSplitSubKSP()
            for i, ksp_sub in enumerate(field_split_ksp):
                ksp_sub.setType("preonly")
                ksp_sub.getPC().setType("ilu")  # ILU pour le debug (plus rapide que LU complet)
        
        self.nest_solver.setTolerances(rtol=1e-10, atol=1e-12)
        self.nest_solver.setFromOptions()
    
    def _pre_newton_iteration(self, x):
        """Fonction appelée avant le calcul du résidu ou du jacobien"""
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        
        # Distribuer la solution précédente aux sous-vecteurs
        sub_vecs = x.getNestSubVecs()
        for i, ui in enumerate(self._u):
            sub_vecs[i].array[:] = ui.x.petsc_vec.array_r
            sub_vecs[i].ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _assemble_residual(self, x, b):
        """Assemble le résidu F dans le vecteur b"""
        # Mettre à zéro le vecteur résidu
        for b_sub in b.getNestSubVecs():
            with b_sub.localForm() as b_local:
                b_local.set(0.0)
        
        # Assembler le vecteur résidu
        assemble_vector_nest(b, self._F)
        
        # Appliquer les conditions de Dirichlet
        apply_lifting_nest(b, self._a, bcs=self._bcs, x0=x, alpha=-1.0)
        
        # Synchroniser les processus
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        
        # Appliquer les valeurs des conditions aux limites
        bcs0 = bcs_by_block(extract_function_spaces(self._F), self._bcs)
        set_bc_nest(b, bcs0)
    
    def _assemble_jacobian(self, x, A):
        """Assemble la matrice jacobienne"""
        # Obtenir les dimensions de la matrice nest
        nest_size = A.getNestSize()
        rows, cols = nest_size
        
        # Réinitialiser la matrice - seulement pour les sous-matrices non nulles
        for i in range(rows):
            for j in range(cols):
                sub_mat = A.getNestSubMatrix(i, j)
                if sub_mat is not None:
                    sub_mat.zeroEntries()
        
        # Assembler la matrice jacobienne par blocs
        assemble_matrix_nest(A, self._a, bcs=self._bcs)
        A.assemble()
    
    def _update_function(self, solver, dx, x):
        """Met à jour la solution"""
        # Mettre à jour les fonctions u avec l'incrément dx
        dx_sub_vecs = dx.getNestSubVecs()
        for i, ui in enumerate(self._u):
            ui_array = ui.x.petsc_vec.array_w
            dx_array = dx_sub_vecs[i].array_r
            
            # Mettre à jour les degrés de liberté locaux
            local_size = len(ui_array)
            if len(dx_array) < local_size:
                local_size = len(dx_array)
            
            ui_array[:local_size] -= self.relaxation_parameter * dx_array[:local_size]
            ui.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        if self._post_solve_callback is not None:
            self._post_solve_callback(self)
    
    def solve(self):
        """Résout le problème non-linéaire"""
        n, converged = super().solve()
        
        # Mettre à jour les fantômes après résolution
        for sub_vec in self._x.getNestSubVecs():
            sub_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        return n, converged
    
    def debug_solve(self, print_steps=True):
        """Version de debug du solveur de Newton pour le format 'nest'"""
        # Paramètres d'itération
        max_it = int(1e3)
        tol = 1e-8
        n = 0  # nombre d'itérations
        converged = False
        
        # Initialiser avec _pre_newton_iteration
        self._pre_newton_iteration(self._x)
        
        # Obtenir la norme du résidu initial
        self._assemble_residual(self._x, self._b)
        initial_residual_norm = self._b.norm()
        if print_steps:
            print(f"Initial residual norm: {initial_residual_norm}")
        
        while n < max_it:
            # Assembler le Jacobien et le résidu
            self._assemble_jacobian(self._x, self._J)
            self._assemble_residual(self._x, self._b)
            
            # Résoudre le système avec le solveur compatible nest
            self.nest_solver.setOperators(self._J)
            self.nest_solver.solve(self._b, self._dx)
            
            # Mettre à jour les fantômes
            for sub_vec in self._dx.getNestSubVecs():
                sub_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)

            # Mettre à jour la solution
            self._update_function(self, self._dx, self._x)
            
            # Mettre à jour le vecteur x
            for i, ui in enumerate(self._u):
                self._x.getNestSubVecs()[i].array[:] = ui.x.petsc_vec.array_r
                self._x.getNestSubVecs()[i].ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            # Calculer les normes pour le critère de convergence
            n += 1
            correction_norm = self._dx.norm()
            self._assemble_residual(self._x, self._b)
            error_norm = self._b.norm()
            rel_error_norm = error_norm / initial_residual_norm if initial_residual_norm > 0 else error_norm
            
            if print_steps:
                print(f"Iteration {n}: Correction norm {correction_norm}, "
                      f"Residual (abs) {error_norm}, Residual (rel) {rel_error_norm}")
            
            # Vérifier la convergence
            if correction_norm < tol or error_norm < tol:
                converged = True
                break
        
        return n, converged


def create_newton_solver(solver_type, F, u, J, petsc_options=None, bcs=None):
    """
    Crée un solveur de Newton du type spécifié
    
    Args:
        solver_type: str, 'block' ou 'nest'
        F: list[ufl.form.Form] Liste des résidus des EDP
        u: list[dolfinx.fem.Function] Liste des fonctions inconnues
        J: list[list[ufl.form.Form]] Représentation UFL du Jacobien
        petsc_options: dict, Options pour le solveur de Krylov PETSc
            Options spéciales pour 'nest' :
               - use_direct_solver: bool, Utiliser un solveur direct pour toute la matrice
               - direct_solver_package: str, Le package solveur à utiliser (default: 'mumps')
               - symmetric: bool, Indique si la matrice est symétrique
        bcs: list[dolfinx.fem.DirichletBC] Liste des conditions aux limites de Dirichlet
    
    Returns:
        Un solveur de Newton du type spécifié
    
    Exemple d'utilisation:
        # Créer un solveur avec structure 'block'
        solver = create_newton_solver('block', F, u, J, petsc_options, bcs)
        
        # Créer un solveur 'nest' avec solveur direct MUMPS
        petsc_options = {
            "relative_tolerance": 1e-10,
            "absolute_tolerance": 1e-12,
            "use_direct_solver": True,
            "direct_solver_package": "mumps"
        }
        solver = create_newton_solver('nest', F, u, J, petsc_options, bcs)
    """   
    if solver_type.lower() == 'block':
        return BlockedNewtonSolver(F, u, J, petsc_options, bcs)
    elif solver_type.lower() == 'nest':
        return NestedNewtonSolver(F, u, J, petsc_options, bcs)
    else:
        raise ValueError(f"Type de solveur inconnu: {solver_type}. Utiliser 'block' ou 'nest'.")