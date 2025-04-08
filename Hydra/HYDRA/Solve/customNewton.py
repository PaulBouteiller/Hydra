
"""
Custom Newton solvers for block-structured nonlinear systems
===========================================================

This module implements custom Newton-type solvers for nonlinear systems arising in HDG
discretizations of fluid dynamics problems. It provides specialized handling of block-structured
matrices and supports both monolithic and nested matrix formats in PETSc.

The implementation offers:
- Block and nested matrix assembly strategies
- Customizable Jacobian and residual evaluation
- Flexible linear solver configuration through PETSc options
- Debug capabilities for solver troubleshooting
- Comprehensive callbacks for monitoring convergence

The module defines an abstract base class for Newton solvers and concrete implementations
for different matrix storage formats (block and nested).

Classes:
--------
BlockMethods : Utility class for block matrix operations
    Provides methods for vector and matrix creation, assembly, and synchronization
    Specialized for PETSc's block matrix format

NestMethods : Utility class for nested matrix operations
    Provides methods for vector and matrix creation, assembly, and synchronization
    Specialized for PETSc's nested matrix format

BaseNewtonSolver : Abstract base class for Newton solvers
    Defines the common interface and functionality for all Newton solvers
    Implements monitoring and callback functionality
    Provides debug solving capabilities

BlockedNewtonSolver : Newton solver for block matrices
    Implements the Newton method using PETSc's block matrix format
    Provides efficient assembly and solution update methods

NestedNewtonSolver : Newton solver for nested matrices
    Implements the Newton method using PETSc's nested matrix format
    Provides field-split preconditioning capabilities

Functions:
----------
create_newton_solver : Factory function for Newton solver creation
    Creates an appropriate Newton solver based on specified type and configuration

Notes:
------
The module supports both block and nested matrix formats in PETSc, which offer different
performance characteristics depending on the problem structure and solver configuration.
Debug capabilities are provided for troubleshooting nonlinear convergence issues.
"""
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


class BlockMethods:
    """
    Utility methods for PETSc block matrix and vector operations.
    
    This class provides static methods to handle operations on PETSc's block
    matrix format, including creation, assembly, and synchronization of 
    matrices and vectors.
    """
    
    def create_vector(F):
        """
        Create a PETSc vector in block format from a UFL form.
        
        Parameters
        ----------
        F : list of UFL forms List of residual forms defining the structure
            
        Returns
        -------
        PETSc.Vec Block vector with appropriate structure
        """
        return create_vector_block(F)
    
    def create_matrix(a):
        """
        Create a PETSc matrix in block format from a UFL form.
        
        Parameters
        ----------
        a : list of lists of UFL forms
            Bilinear forms defining the block matrix structure
            
        Returns
        -------
        PETSc.Mat Block matrix with appropriate structure
        """
        return create_matrix_block(a)
    
    def assemble_residual(b, F, a, bcs, x, alpha=-1.0):
        """
        Assemble residual into a block vector.
        
        Assemble the residual forms into the provided block vector,
        applying boundary conditions and lifting.
        
        Parameters
        ----------
        b : PETSc.Vec Vector to assemble into (will be zeroed)
        F : list of UFL forms Residual forms to assemble
        a : list of lists of UFL forms Jacobian forms for boundary condition lifting
        bcs : list of DirichletBC Boundary conditions to apply
        x : PETSc.Vec Current solution vector
        alpha : float, optional Scaling factor for boundary condition lifting
        """
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector_block(b, F, a, bcs=bcs, x0=x, alpha=alpha)
        b.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)
    
    def assemble_jacobian(A, a, bcs):
        """
        Assemble Jacobian into a block matrix.
        
        Assemble the Jacobian forms into the provided block matrix,
        applying boundary conditions.
        
        Parameters
        ----------
        A : PETSc.Mat Matrix to assemble into (will be zeroed)
        a : list of lists of UFL forms Jacobian forms to assemble
        bcs : list of DirichletBC Boundary conditions to apply
        """
        A.zeroEntries()
        assemble_matrix_block(A, a, bcs=bcs)
        A.assemble()
    
    def sync_vector_to_x(x, u):
        """
        Synchronize FEniCSx functions to a PETSc block vector.
        
        Copy values from a list of FEniCSx functions to a PETSc block vector.
        
        Parameters
        ----------
        x : PETSc.Vec PETSc block vector to update
        u : list of Function List of FEniCSx functions containing the values to copy
        """
        scatter_local_vectors(x, [ui.x.petsc_vec.array_r for ui in u],
                         [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                          for ui in u])
        x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)


class NestMethods:
    """
    Utility methods for PETSc nested matrix and vector operations.
    
    This class provides static methods to handle operations on PETSc's nested
    matrix format, including creation, assembly, and synchronization of 
    matrices and vectors.
    """
    
    def create_vector(F):
        """
        Create a PETSc vector in nested format from a UFL form.
        
        Parameters
        ----------
        F : list of UFL forms List of residual forms defining the structure
            
        Returns
        -------
        PETSc.Vec Nested vector with appropriate structure
        """
        return assemble_vector_nest(F)
    
    def create_matrix(a, bcs=None):
        """
        Create a PETSc matrix in nested format from a UFL form.
        
        Parameters
        ----------
        a : list of lists of UFL forms Bilinear forms defining the nested matrix structure
        bcs : list of DirichletBC, optional Boundary conditions to consider in matrix structure
            
        Returns
        -------
        PETSc.Mat Nested matrix with appropriate structure
        """
        mat = assemble_matrix_nest(a, bcs=bcs)
        mat.assemble()
        return mat
    
    def assemble_residual(b, F, a, bcs, x, alpha=-1.0):
        """
        Assemble residual into a nested vector.
        
        Assemble the residual forms into the provided nested vector,
        applying boundary conditions and lifting.
        
        Parameters
        ----------
        b : PETSc.Vec Nested vector to assemble into (will be zeroed)
        F : list of UFL forms Residual forms to assemble
        a : list of lists of UFL forms Jacobian forms for boundary condition lifting
        bcs : list of DirichletBC Boundary conditions to apply
        x : PETSc.Vec
            Current solution vector
        alpha : float, optional
            Scaling factor for boundary condition lifting
        """
        for b_sub in b.getNestSubVecs():
            with b_sub.localForm() as b_local:
                b_local.set(0.0)
        
        assemble_vector_nest(b, F)
        apply_lifting_nest(b, a, bcs=bcs, x0=x, alpha=alpha)
        
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        
        bcs0 = bcs_by_block(extract_function_spaces(F), bcs)
        set_bc_nest(b, bcs0)
    
    def assemble_jacobian(A, a, bcs):
        """
        Assemble Jacobian into a nested matrix.
        
        Assemble the Jacobian forms into the provided nested matrix,
        applying boundary conditions.
        
        Parameters
        ----------
        A : PETSc.Mat Nested matrix to assemble into (will be zeroed)
        a : list of lists of UFL forms Jacobian forms to assemble
        bcs : list of DirichletBC Boundary conditions to apply
        """
        nest_size = A.getNestSize()
        rows, cols = nest_size
        
        for i in range(rows):
            for j in range(cols):
                sub_mat = A.getNestSubMatrix(i, j)
                if sub_mat is not None:
                    sub_mat.zeroEntries()
        
        assemble_matrix_nest(A, a, bcs=bcs)
        A.assemble()
    
    def sync_vector_to_x(x, u):
        """
        Synchronize FEniCSx functions to a PETSc nested vector.
        
        Copy values from a list of FEniCSx functions to a PETSc nested vector.
        
        Parameters
        ----------
        x : PETSc.Vec PETSc nested vector to update
        u : list of Function List of FEniCSx functions containing the values to copy
        """
        sub_vecs = x.getNestSubVecs()
        for i, ui in enumerate(u):
            sub_vecs[i].array[:] = ui.x.petsc_vec.array_r
            sub_vecs[i].ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)


class BaseNewtonSolver(NewtonSolver):
    """
    Base class for custom Newton solvers with block structure handling.
    
    This abstract class extends FEniCSx's NewtonSolver to provide enhanced
    capabilities for handling block-structured matrices and vectors arising
    in coupled multiphysics problems.
    """
    
    def __init__(self, F, u, J, petsc_options, debug, bcs=None):
        """
        Initialize the base Newton solver.
        
        Parameters
        ----------
        F : list of UFL forms List of residual forms [F_0(u, v_0), F_1(u, v_1), ...]
        u : list of Function List of solution functions u=[u_0, u_1, ...]
        J : list of lists of UFL forms UFL representation of the Jacobian
        petsc_options : dict Options for the Krylov solver
        debug : bool Enable debug mode for detailed solver output
        bcs : list of DirichletBC, optional List of Dirichlet boundary conditions
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
        self.debug = debug
        self.petsc_options = petsc_options
        
        # Ces attributs seront initialisés dans les sous-classes
        self._b = None
        self._J = None
        self._dx = None
        self._x = None
        
        # À implémenter dans les sous-classes:
        # 1. Création des structures pour les vecteurs et matrices
        # 2. Configuration des méthodes d'assemblage et de mise à jour
        # 3. Configuration du solveur de debug si nécessaire
    
    def set_pre_solve_callback(self, callback: Callable[["BaseNewtonSolver"], None]):
        """
        Set a callback function to be called before each Newton iteration.
        
        Parameters
        ----------
        callback : callable Function to be called with the solver instance as argument
        """
        self._pre_solve_callback = callback
    
    def set_post_solve_callback(self, callback: Callable[["BaseNewtonSolver"], None]):
        """
        Set a callback function to be called after each Newton iteration.
        
        Parameters
        ----------
        callback : callable Function to be called with the solver instance as argument
        """
        self._post_solve_callback = callback
    
    def _init_common(self):
        """
        Initialize common components for all solver types.
        """
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
        """
        Solve the nonlinear problem.
        
        Implements an enhanced Newton solution process with iteration monitoring
        and detailed output.
        
        Returns
        -------
        tuple (number of iterations, convergence flag)
        """
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
    
    def debug_solve(self, print_steps=True):
        """
        Implement a debug version of the Newton solver with detailed output.
        
        Provides a more transparent view of the Newton iteration process,
        useful for debugging convergence issues.
        
        Parameters
        ----------
        print_steps : bool, optional  Whether to print detailed information for each step
            
        Returns
        -------
        tuple (number of iterations, convergence flag)
        """
        # Paramètres d'itération
        max_it = int(1e3)
        tol = 1e-8
        n = 0
        converged = False
        
        # Initialiser avec _pre_newton_iteration
        self._pre_newton_iteration(self._x)
        
        # Obtenir la norme du résidu initial
        self._assemble_residual(self._x, self._b)
        initial_residual_norm = self._b.norm()
        if print_steps:
            print(f"Initial residual norm: {initial_residual_norm}")
        
        while n < max_it:
            # Assemblage et résolution
            self._assemble_jacobian(self._x, self._J)
            self._assemble_residual(self._x, self._b)
            
            # Résoudre le système linéaire (méthode spécifique à implémenter)
            self._solve_linear_system(self._b, self._dx)
            
            # Mettre à jour la solution
            self._update_function(self, self._dx, self._x)
            
            # Synchroniser le vecteur x (méthode spécifique)
            self._sync_solution_vector()
            
            # Calcul des normes et convergence
            n += 1
            correction_norm = self._dx.norm()
            self._assemble_residual(self._x, self._b)
            error_norm = self._b.norm()
            rel_error_norm = error_norm / initial_residual_norm if initial_residual_norm > 0 else error_norm
            
            if print_steps:
                print(f"Iteration {n}: Correction norm {correction_norm}, "
                      f"Residual (abs) {error_norm}, Residual (rel) {rel_error_norm}")
            
            if correction_norm < tol or error_norm < tol:
                converged = True
                break
        
        return n, converged


class BlockedNewtonSolver(BaseNewtonSolver):
    """
    Newton solver using PETSc's block matrix format.
    
    This implementation of the Newton solver uses PETSc's block matrix format,
    which stores the matrix as a monolithic block. This format is often more
    efficient for direct solvers.
    """
    
    def __init__(self, F, u, J, petsc_options, debug, bcs=None):
        """
        Initialize the block Newton solver.
        
        Parameters
        ----------
        see BaseNewtonSolver
        """
        super().__init__(F, u, J, petsc_options, debug, bcs)
        
        # Créer les structures pour les vecteurs et matrices
        self._b = BlockMethods.create_vector(self._F)
        self._J = BlockMethods.create_matrix(self._a)
        self._dx = BlockMethods.create_vector(self._F)
        self._x = BlockMethods.create_vector(self._F)
        
        # Initialisation commune
        self._init_common()
    
    def _setup_debug_solver(self):
        """
        Set up a debug solver for the block format.
        
        Configures a direct solver for debugging, typically using LU factorization
        with MUMPS for robust solution of the linear system.
        """
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
        """
        Perform pre-iteration setup.
        
        Called before calculating the residual or Jacobian to prepare
        the solution vector and execute any user-defined callbacks.
        
        Parameters
        ----------
        x : PETSc.Vec Current solution vector
        """
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        
        # Scatter les vecteurs locaux
        BlockMethods.sync_vector_to_x(x, self._u)
    
    def _assemble_residual(self, x, b):
        """
        Assemble the residual vector.
        
        Parameters
        ----------
        x : PETSc.Vec Current solution vector
        b : PETSc.Vec Residual vector to assemble into
        """
        BlockMethods.assemble_residual(b, self._F, self._a, self._bcs, x)
    
    def _assemble_jacobian(self, x, A):
        """
        Assemble the Jacobian matrix.
        
        Parameters
        ----------
        x : PETSc.Vec Current solution vector
        A : PETSc.Mat Jacobian matrix to assemble into
        """
        BlockMethods.assemble_jacobian(A, self._a, self._bcs)
    
    def _solve_linear_system(self, b, x):
        """
        Solve the linear system for the block format.
        
        Parameters
        ----------
        b : PETSc.Vec Right-hand side vector
        x : PETSc.Vec Solution vector to update
        """
        self.linear_solver.solve(b, x)
        x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _update_function(self, solver, dx, x):
        """
        Update the solution after a Newton step.
        
        Parameters
        ----------
        solver : BaseNewtonSolver The solver instance
        dx : PETSc.Vec Increment to apply to the solution
        x : PETSc.Vec Solution vector to update
        """
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
    
    def _sync_solution_vector(self):
        """
        Synchronize the solution vector for the block format.
        
        Copies values from the solution functions to the PETSc vector.
        """
        BlockMethods.sync_vector_to_x(self._x, self._u)

class NestedNewtonSolver(BaseNewtonSolver):
    """
    Newton solver using PETSc's nested matrix format.
    
    This implementation of the Newton solver uses PETSc's nested matrix format,
    which stores the matrix as a collection of submatrices. This format is often
    more suitable for field-split preconditioning and iterative solvers.
    """
    
    def __init__(self, F, u, J, petsc_options, debug, bcs=None):
        """
        Initialize the nested Newton solver.
        
        Parameters
        ----------
        see BaseNewtonSolver
        """
        super().__init__(F, u, J, petsc_options, debug, bcs)
        
        # Créer les structures pour les vecteurs et matrices nest
        self._b = NestMethods.create_vector(self._F)
        self._J = NestMethods.create_matrix(self._a, bcs=self._bcs)
        
        # Créer les vecteurs pour l'incrémentation et la solution
        self._dx = self._b.copy()
        self._x = self._b.copy()
        
        # Initialisation commune
        self._init_common()
        
        # Configuration spécifique pour le format 'nest'
        self._setup_fieldsplit_preconditioner()
    
    def _setup_fieldsplit_preconditioner(self):
        """
        Configure field-split preconditioner for the nested format.
        
        Sets up a field-split preconditioner that can efficiently handle
        the coupled system by treating each field separately.
        """
        self.krylov_solver.setType("gmres")
        self.krylov_solver.getPC().setType("fieldsplit")
        self.krylov_solver.getPC().setFieldSplitType(PC.CompositeType.ADDITIVE)
        
        # Récupérer les index sets pour chaque champ
        nest_IS = self._J.getNestISs()
        fields = []
        for i in range(len(self._u)):
            fields.append((f"field_{i}", nest_IS[0][i]))
        
        # Configurer le préconditionneur par champs
        self.krylov_solver.getPC().setFieldSplitIS(*fields)
        
        # Configuration de chaque sous-solveur
        field_split_ksp = self.krylov_solver.getPC().getFieldSplitSubKSP()
        for i, ksp_sub in enumerate(field_split_ksp):
            ksp_sub.setType("preonly")
            ksp_sub.getPC().setType("lu")
    
    def _setup_debug_solver(self):
        """
        Set up a debug solver for the nested format.
        
        Configures a field-split solver for debugging, using direct solvers
        for each field.
        """
        self.nest_solver = KSP().create(self.comm)
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
            ksp_sub.getPC().setType("ilu")
        
        self.nest_solver.setTolerances(rtol=1e-10, atol=1e-12)
        self.nest_solver.setFromOptions()
    
    def _pre_newton_iteration(self, x):
        """
        Perform pre-iteration setup for nested format.
        
        Called before calculating the residual or Jacobian to prepare
        the solution vector and execute any user-defined callbacks.
        
        Parameters
        ----------
        x : PETSc.Vec Current solution vector
        """
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        
        # Distribuer la solution précédente aux sous-vecteurs
        NestMethods.sync_vector_to_x(x, self._u)
    
    def _assemble_residual(self, x, b):
        """
        Assemble the residual vector for nested format.
        
        Parameters
        ----------
        x : PETSc.Vec Current solution vector
        b : PETSc.Vec Residual vector to assemble into
        """
        NestMethods.assemble_residual(b, self._F, self._a, self._bcs, x)
    
    def _assemble_jacobian(self, x, A):
        """
        Assemble the Jacobian matrix for nested format.
        
        Parameters
        ----------
        x : PETSc.Vec
            Current solution vector
        A : PETSc.Mat
            Jacobian matrix to assemble into
        """
        NestMethods.assemble_jacobian(A, self._a, self._bcs)
    
    def _solve_linear_system(self, b, x):
        """
        Solve the linear system for the nested format.
        
        Parameters
        ----------
        b : PETSc.Vec Right-hand side vector
        x : PETSc.Vec Solution vector to update
        """
        self.nest_solver.setOperators(self._J)
        self.nest_solver.solve(b, x)
        
        # Mettre à jour les fantômes
        for sub_vec in x.getNestSubVecs():
            sub_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _update_function(self, solver, dx, x):
        """
        Update the solution after a Newton step for nested format.
        
        Parameters
        ----------
        solver : BaseNewtonSolver The solver instance
        dx : PETSc.Vec Increment to apply to the solution
        x : PETSc.Vec Solution vector to update
        """
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
    
    def _sync_solution_vector(self):
        """
        Synchronize the solution vector for the nested format.
        
        Copies values from the solution functions to the PETSc vector.
        """
        NestMethods.sync_vector_to_x(self._x, self._u)
    
    def solve(self):
        """
        Solve the nonlinear problem with the nested format.
        
        Extends the base solve method with additional ghost updates
        specific to the nested format.
        
        Returns
        -------
        tuple (number of iterations, convergence flag)
        """
        n, converged = super().solve()
        
        # Mettre à jour les fantômes après résolution
        for sub_vec in self._x.getNestSubVecs():
            sub_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        return n, converged


def create_newton_solver(solver_type, F, u, J, petsc_options, debug, bcs=None):
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
    """   
    if solver_type.lower() == 'block':
        return BlockedNewtonSolver(F, u, J, petsc_options, debug, bcs)
    elif solver_type.lower() == 'nest':
        return NestedNewtonSolver(F, u, J, petsc_options, debug, bcs)
    else:
        raise ValueError(f"Type de solveur inconnu: {solver_type}. Utiliser 'block' ou 'nest'.")