from petsc4py import PETSc
from dolfinx.fem.petsc import (create_matrix_block, create_vector_block,
                              assemble_matrix_block, assemble_vector_block)
from dolfinx.cpp.la.petsc import scatter_local_vectors
from petsc4py.PETSc import ScatterMode, InsertMode

class BlockedSNESSolver:
    def __init__(self, F, u, J, bcs=[], petsc_options=None):
        # Get MPI communicator from first function's mesh
        self.comm = u[0].function_space.mesh.comm
        self.snes = PETSc.SNES().create(self.comm)
        print("Available SNES types:", self.snes.getType())
        # self.snes.setType("newton")
        self.snes.setType("ngs")
        
        # Compile forms
        self._F = F
        self._a = J
        self._bcs = bcs
        self._u = u
        
        # Create vectors and matrices
        self._b = create_vector_block(self._F)
        self._J = create_matrix_block(self._a)
        self._x = create_vector_block(self._F)
        
        # Configure SNES solver sans utiliser setFromOptions
        self.snes.setFunction(self._assemble_residual, self._b)
        self.snes.setJacobian(self._assemble_jacobian, self._J)
        
        # Tolérance et itérations maximales
        self.snes.setTolerances(rtol=1e-4, atol=1e-4, stol=1e-4, max_it=100)
        
        # Configuration du solveur linéaire interne
        ksp = self.snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")        
        # Appliquer les options modifiées
        self.snes.setFromOptions()
        
        # Set the initial guess
        self._scatter_to_x()
        
        # Initialize relaxation parameter
        self.relaxation_parameter = 1.0
        
    def _scatter_to_x(self):
        """Scatter from function spaces to the solution vector x."""
        scatter_local_vectors(self._x, [ui.x.petsc_vec.array_r for ui in self._u],
                             [(ui.function_space.dofmap.index_map, ui.function_space.dofmap.index_map_bs)
                             for ui in self._u])
        self._x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
    
    def _assemble_residual(self, snes, x, b):
        """Assemble the residual F into the vector b."""
        with b.localForm() as b_local:
            b_local.set(0.0)        
        # Assemble the residual with boundary conditions
        assemble_vector_block(b, self._F, self._a, bcs=self._bcs, x0=x, **{"alpha": -1.0})
        b.ghostUpdate(InsertMode.INSERT_VALUES, ScatterMode.FORWARD)
    
    def _assemble_jacobian(self, snes, x, J, P):
        """Assemble the Jacobian matrix."""
        J.zeroEntries()
        assemble_matrix_block(J, self._a, bcs=self._bcs)
        J.assemble()
    
    def solve(self):
        """Solve non-linear problem. Returns the number of iterations and if the solver converged."""
        # Update the solution vector from the functions
        self._scatter_to_x()
        # Try to solve the nonlinear problem
        self.snes.solve(None, self._x)
        
        # Check convergence
        converged = self.snes.getConvergedReason() > 0
        iterations = self.snes.getIterationNumber()
        
        if self.comm.rank == 0:
            print(f"SNES solver: {iterations} iterations, converged = {converged} (reason: {self.snes.getConvergedReason()})")
        
        # Update the solution back to the DOLFINx functions
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
        self._J.destroy()
        self._b.destroy()
        self._x.destroy()