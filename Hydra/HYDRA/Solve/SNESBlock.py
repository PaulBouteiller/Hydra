#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 07:55:55 2025

@author: bouteillerp
"""
from petsc4py import PETSc
from dolfinx.fem import form
from dolfinx.fem.petsc import (create_matrix_block, create_vector_block,
                              assemble_matrix_block, assemble_vector_block)
from dolfinx.cpp.la.petsc import scatter_local_vectors
from petsc4py.PETSc import ScatterMode, InsertMode

class BlockedSNESSolver:
    def __init__(self, F, u, J, bcs=[], 
                 form_compiler_options=None, 
                 jit_options=None,
                 petsc_options=None,
                 entity_maps=None):
        """Initialize solver for solving a non-linear problem using SNES.
        
        Args:
            F: list[ufl.form.Form] List of PDE residuals [F_0(u, v_0), F_1(u, v_1), ...]
            u: list[dolfinx.fem.Function] List of unknown functions u=[u_0, u_1, ...]
            J: list[list[ufl.form.Form]] UFL representation of the Jacobian
            bcs: list[dolfinx.fem.DirichletBC] List of Dirichlet boundary conditions
            form_compiler_options: Options used in FFCx compilation of this form
            jit_options: Options used in CFFI JIT compilation of C code
            petsc_options: Options passed to the PETSc SNES solver
            entity_maps: Maps used to map entities between different meshes
        """
        print("Solve with SNES")
        # Get MPI communicator from first function's mesh
        self.comm = u[0].function_space.mesh.comm
        
        # Create SNES solver
        self.snes = PETSc.SNES().create(self.comm)
        
        # Compile forms
        self._F = form(F, form_compiler_options=form_compiler_options,
                      jit_options=jit_options, entity_maps=entity_maps)
        self._a = form(J, form_compiler_options=form_compiler_options,
                      jit_options=jit_options, entity_maps=entity_maps)
        
        self._bcs = bcs
        self._u = u
        
        # Create vectors and matrices
        self._b = create_vector_block(self._F)
        self._J = create_matrix_block(self._a)
        self._x = create_vector_block(self._F)
        
        # Set SNES options
        if petsc_options is not None:
            opts = PETSc.Options()
            for k, v in petsc_options.items():
                opts[k] = v
            self.snes.setFromOptions()
        
        # Configure SNES solver
        self.snes.setFunction(self._assemble_residual, self._b)
        self.snes.setJacobian(self._assemble_jacobian, self._J)
        
        # Set the initial guess (important to scatter the initial values)
        self._scatter_to_x()
        
        # Set callback for solution update
        self.snes.setMonitor(self._monitor)
        
        # Initialize relaxation parameter (equivalent to in BlockedNewtonSolver)
        self.relaxation_parameter = 1.0
        
        # Callbacks similar to BlockedNewtonSolver
        self._pre_solve_callback = None
        self._post_solve_callback = None
    
    def set_pre_solve_callback(self, callback):
        """Set a callback function that is called before solving."""
        self._pre_solve_callback = callback
        
    def set_post_solve_callback(self, callback):
        """Set a callback function that is called after solving."""
        self._post_solve_callback = callback
    
    def _scatter_to_x(self):
        """Scatter from function spaces to the solution vector x."""
        scatter_local_vectors(self._x, [ui.x.petsc_vec.array_r for ui in self._u],
                             [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                             for ui in self._u])
        self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _assemble_residual(self, snes, x, b):
        """Assemble the residual F into the vector b.
        
        Args:
            snes: PETSc.SNES The SNES solver
            x: PETSc.Vec The vector containing the latest solution
            b: PETSc.Vec Vector to assemble the residual into
        """
        # Clear the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        
        # Assemble the residual with boundary conditions
        assemble_vector_block(b, self._F, self._a, bcs=self._bcs, x0=x, **{"alpha": -1.0})
        b.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)
    
    def _assemble_jacobian(self, snes, x, J, P):
        """Assemble the Jacobian matrix.
        
        Args:
            snes: PETSc.SNES The SNES solver 
            x: PETSc.Vec The vector containing the latest solution
            J: PETSc.Mat Matrix to assemble the Jacobian into
            P: PETSc.Mat Matrix to assemble the preconditioner into (same as J here)
        """
        # Zero and then assemble the Jacobian matrix
        J.zeroEntries()
        assemble_matrix_block(J, self._a, bcs=self._bcs)
        J.assemble()
        
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
    
    def _monitor(self, snes, it, rnorm):
        """Monitor function for SNES.
        
        This function is called after each nonlinear iteration.
        At this point, we want to update our functions.
        """
        # Get current solution
        x = snes.getSolution()
        
        # Update the functions using the current solution
        offset_start = 0
        for ui in self._u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            
            # Apply relaxation parameter similar to BlockedNewtonSolver
            current_vals = ui.x.petsc_vec.array_r[:num_sub_dofs].copy()
            new_vals = x.array_r[offset_start:offset_start + num_sub_dofs]
            
            # Compute the increment (modified by relaxation)
            increment = (new_vals - current_vals) * self.relaxation_parameter
            
            # Update function with the relaxed increment
            ui.x.petsc_vec.array_w[:num_sub_dofs] = current_vals + increment
            ui.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            
            offset_start += num_sub_dofs

        if self._post_solve_callback:
            self._post_solve_callback(self)
    
    def solve(self):
        """Solve non-linear problem. Returns the number of iterations and if the solver converged."""
        if self._pre_solve_callback:
            self._pre_solve_callback(self)
        
        # Update the solution vector from the functions
        self._scatter_to_x()
        
        # Solve the nonlinear problem
        self.snes.solve(None, self._x)
        
        # Check convergence
        converged = self.snes.getConvergedReason() > 0
        iterations = self.snes.getIterationNumber()
        
        # Update the solution back to the DOLFINx functions 
        # (should already be done by the monitor, but added for safety)
        offset_start = 0
        for ui in self._u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            ui.x.petsc_vec.array_w[:num_sub_dofs] = self._x.array_r[offset_start:offset_start + num_sub_dofs]
            ui.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
            offset_start += num_sub_dofs
        
        return iterations, converged
    
    def __del__(self):
        """Clean up PETSc objects."""
        if hasattr(self, '_J'):
            self._J.destroy()
        if hasattr(self, '_b'):
            self._b.destroy()
        if hasattr(self, '_x'):
            self._x.destroy()