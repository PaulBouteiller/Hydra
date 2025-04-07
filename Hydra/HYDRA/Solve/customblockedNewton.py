from typing import Callable
from petsc4py import PETSc
from dolfinx.fem.petsc import (create_matrix_block, create_vector_block,
                               assemble_matrix_block, assemble_vector_block)

from dolfinx.cpp.la.petsc import scatter_local_vectors
from petsc4py.PETSc import ScatterMode, InsertMode
from dolfinx.log import set_log_level, LogLevel
from dolfinx.cpp.nls.petsc import NewtonSolver

class BlockedNewtonSolver(NewtonSolver):
    def __init__(self, F, u, J, petsc_options, bcs = [],
                 entity_maps: dict | None = None):
        """Initialize solver for solving a non-linear problem using Newton's method.
        Args:
            F: list[ufl.form.Form] List of PDE residuals [F_0(u, v_0), F_1(u, v_1), ...]
            u: list[dolfinx.fem.Function] List of unknown functions u=[u_0, u_1, ...]
            J: list[list[ufl.form.Form]] UFL representation of the Jacobian
            bcs: list[dolfinx.fem.DirichletBC] List of Dirichlet boundary conditions
                Note:
                    If not provided, the Jacobian will be computed using the
                    assumption that the test functions come from a ``ufl.MixedFunctionSpace``
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all
                other option values.
            petsc_options:
                Options passed to the PETSc Krylov solver.
            entity_maps: Maps used to map entities between different meshes.
                Only needed if the forms have not been compiled a priori,
                and has coefficients, test, or trial functions that are defined on different meshes.
        """
        # Initialize base class
        comm = u[0].function_space.mesh.comm
        super().__init__(comm)

        # Set PETSc options for Krylov solver
        prefix = self.krylov_solver.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        if petsc_options is not None:
            # Set PETSc options
            opts = PETSc.Options()
            opts.prefixPush(prefix)
            for k, v in petsc_options.items():
                opts[k] = v
            opts.prefixPop()

            self.krylov_solver.rtol = petsc_options.get("relative_tolerance")
            self.krylov_solver.atol = petsc_options.get("absolute_tolerance")
            self.krylov_solver.setFromOptions()
        self._F = F
        self._a = J

        self._bcs = bcs
        self._u = u
        self._pre_solve_callback: Callable[["BlockedNewtonSolver"], None] | None = None
        self._post_solve_callback: Callable[["BlockedNewtonSolver"], None] | None = None

        # Create structures for holding arrays and matrix
        self._b = create_vector_block(self._F)
        self._J = create_matrix_block(self._a)
        self._dx = create_vector_block(self._F)
        self._x = create_vector_block(self._F)
        self._J.setOptionsPrefix(prefix)
        self._J.setFromOptions()

        self.setJ(self._assemble_jacobian, self._J)
        self.setF(self._assemble_residual, self._b)
        self.set_form(self._pre_newton_iteration)
        self.set_update(self._update_function)

        self.atol = petsc_options.get("absolute_tolerance")
        self.rtol = petsc_options.get("relative_tolerance")
        self.debug = petsc_options.get("debug", False)
        if self.debug:
            self.linear_solver = PETSc.KSP().create(comm)
            
            # On demande que le solveur soit direct
            self.linear_solver.setType(PETSc.KSP.Type.PREONLY)
            # Un préconditionneur de type LU est utilisé
            self.linear_solver.getPC().setType(PETSc.PC.Type.LU)
            opts = PETSc.Options()
            # prefix = f"solver_{id(self.solver)}"
            prefix = f"projector_{id(self)}"
            self.linear_solver.setOptionsPrefix(prefix)
            option_prefix = self.linear_solver.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            self.linear_solver.setFromOptions()
            #La matrice A utilisée pour résoudre le système Ax=b est self.A
            self.linear_solver.setOperators(self._J)

    def set_pre_solve_callback(self, callback: Callable[["BlockedNewtonSolver"], None]):
        """Set a callback function that is called before each Newton iteration."""
        self._pre_solve_callback = callback

    def set_post_solve_callback(self, callback: Callable[["BlockedNewtonSolver"], None]):
        """Set a callback function that is called after each Newton iteration."""
        self._post_solve_callback = callback

    @property
    def L(self):
        """Compiled linear form (the residual form)"""
        return self._F

    @property
    def a(self):
        """Compiled bilinear form (the Jacobian form)"""
        return self._a

    @property
    def u(self):
        return self._u

    def __del__(self):
        self._J.destroy()
        self._b.destroy()
        self._dx.destroy()
        self._x.destroy()

    def _pre_newton_iteration(self, x):
        """Function called before the residual or Jacobian is
        computed.
        Args: x: PETSc.Vec The vector containing the latest solution
        """
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        # Scatter previous solution `u=[u_0, ..., u_N]` to `x`; the
        # blocked version used for lifting
        scatter_local_vectors(x, [ui.x.petsc_vec.array_r for ui in self._u],
                              [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                               for ui in self._u])
        x.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)

    def _assemble_residual(self, x, b):
        """Assemble the residual F into the vector b.
        Args:
            x: PETSc.Vec The vector containing the latest solution
            b: PETSc.Vec Vector to assemble the residual into
        """
        # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector_block(b, self._F, self._a, bcs=self._bcs, x0=x, **{"alpha": -1.0})
        b.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)

    def _assemble_jacobian(self, x, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.
        Args: x: PETSc.Vec The vector containing the latest solution
        """
        # Assemble Jacobian
        A.zeroEntries()
        assemble_matrix_block(A, self._a, bcs=self._bcs)
        A.assemble()

    def _update_function(self, solver, dx: PETSc.Vec, x: PETSc.Vec):
        if self._post_solve_callback is not None:
            self._post_solve_callback(self)
        # Update solution
        offset_start = 0
        for ui in self._u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            ui.x.petsc_vec.array_w[:num_sub_dofs] -= (
                self.relaxation_parameter * dx.array_r[offset_start : offset_start + num_sub_dofs]
            )
            ui.x.petsc_vec.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
            offset_start += num_sub_dofs

    def solve(self):
        """Solve non-linear problem into function. Returns the number
        of iterations and if the solver converged."""
        # Créer des variables pour suivre les informations d'itération
        iteration_info = {"current": 0, "initial_residual": None}
        # Callback pour afficher l'itération et le résidu avant la résolution
        def custom_pre_callback(solver):
            solver._assemble_residual(solver._x, solver._b)
            residual_norm = solver._b.norm()
            if iteration_info["current"] == 0:
                iteration_info["initial_residual"] = residual_norm
            rel_residual = residual_norm / iteration_info["initial_residual"] if iteration_info["initial_residual"] else 0
            print(f"Newton iteration {iteration_info['current']}: r (abs) = {residual_norm} (tol = 1e-10), r (rel) = {rel_residual} (tol = 1e-09)")
            
            # Incrémenter le compteur d'itérations
            iteration_info["current"] += 1
        
        # Définir le nouveau callback
        self.set_pre_solve_callback(custom_pre_callback)
        set_log_level(LogLevel.WARNING)  # Supprimer les messages INFO
        if self.debug:
            n, converged = self.debug_solve()
        else:
            n, converged = super().solve(self._x)
            self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            # Afficher un message de fin
            print(f"Newton solver finished in {n} iterations and {n} linear solver iterations.")
        
        return n, converged
    
    def debug_solve(self, print_steps=True, print_solution=False):
        # Paramètres d'itération
        max_it = int(1e3)
        tol = 1e-8
        i = 0  # nombre d'itérations du solveur de Newton
        converged = False
        
        # Créer un vecteur pour la mise à jour
        dx = self._dx
        
        # Initialiser le vecteur x
        scatter_local_vectors(self._x, [ui.x.petsc_vec.array_r for ui in self._u],
                              [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                               for ui in self._u])
        self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        # Obtenir la norme du résidu initial pour le critère de convergence relatif
        self._assemble_residual(self._x, self._b)
        initial_residual_norm = self._b.norm()
        if print_steps:
            print(f"Initial residual norm: {initial_residual_norm}")
        
        while i < max_it:
            # Assembler le Jacobien et le résidu
            self._assemble_jacobian(self._x, self._J)
            self._assemble_residual(self._x, self._b)
            
            # Résoudre le système linéaire J*dx = -b
            self.linear_solver.solve(self._b, dx)
            dx.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            # Mettre à jour la solution
            self._update_function(self, self._dx, self._x)
            
            # Mettre à jour le vecteur x
            scatter_local_vectors(self._x, [ui.x.petsc_vec.array_r for ui in self._u],
                                  [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                                   for ui in self._u])
            self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            # Calculer les normes pour le critère de convergence
            i += 1
            correction_norm = dx.norm()
            self._assemble_residual(self._x, self._b)
            error_norm = self._b.norm()
            rel_error_norm = error_norm / initial_residual_norm if initial_residual_norm > 0 else error_norm
            
            if print_steps:
                print(f"Iteration {i}: Correction norm {correction_norm}, "
                      f"Residual (abs) {error_norm}, Residual (rel) {rel_error_norm}")
            
            # Vérifier la convergence
            if correction_norm < tol or error_norm < tol:
                converged = True
                break
            
            # Si le callback post-solve est défini, l'appeler
            if self._post_solve_callback is not None:
                self._post_solve_callback(self)
        
        if print_solution:
            if converged:
                print(f"Solution reached in {i} iterations. Final residual: {error_norm}")
            else:
                print(f"No solution found after {i} iterations. Final residual: {error_norm}")
        
        return i, converged