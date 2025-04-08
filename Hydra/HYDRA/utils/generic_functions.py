"""
Generic utility functions for numerical simulations
=================================================

This module provides a collection of general-purpose utility functions for
numerical simulations in the HYDRA framework. These functions handle common
tasks such as eigenvalue calculations, sequence operations, variable extraction,
and parallel data gathering.

The utilities are organized into several categories:
- Mathematical operations (max/min of sequences, eigenvalues)
- UFL expression manipulation and transformation
- Parallel computing utilities for MPI operations
- Mesh and visualization helpers

Functions:
----------
euler_eigenvalues_2D(v, n, c) : Calculate eigenvalues of Euler flux Jacobian
    Returns the characteristic speeds for the 2D Euler equations

max_abs_of_sequence(a) : Maximum absolute value in a sequence
    Computes max(|a₁|, |a₂|, ..., |aₙ|) for UFL expressions

max_of_sequence(a) : Maximum value in a sequence
    Computes max(a₁, a₂, ..., aₙ) for UFL expressions

map_ufl_operator_to_sequence(a, op) : Apply UFL operator to sequence
    Maps binary UFL operators across sequences of expressions

extract_primitive_variables(U) : Extract primitive variables from conservative form
    Converts [ρ, ρu, ρE] to [ρ, u, E] for fluid dynamics calculations

gather_function(u) : Gather function values in parallel
    Collects distributed function data onto the root process

gather_vector(local_vector, local_range, size) : Gather vector data in parallel
    General-purpose utility for collecting distributed vectors

gather_coordinate(V) : Gather mesh coordinates in parallel
    Collects distributed mesh coordinate data onto the root process

plot_mesh(mesh, tags) : Visualize mesh with markers
    Creates a PyVista-based visualization of the mesh with boundary markers
"""

from ufl import max_value as Max
from ufl.core.expr import Expr
from ufl import dot
from mpi4py import MPI
from numpy import asarray, int32, zeros

def euler_eigenvalues_2D(v, n, c):
    """
    Calculate eigenvalues of the Euler flux Jacobian.
    
    Computes the characteristic speeds (eigenvalues) of the Euler equations
    in the direction of the normal vector n.
    
    Parameters
    ----------
    v : UFL expression Velocity vector
    n : UFL expression Normal vector
    c : UFL expression Sound speed
        
    Returns
    -------
    list List of eigenvalues [λ₁, λ₂] for the Jacobian in direction n
    """
    # vitesse normale (v·n)
    v_n = dot(v, n)
    return [v_n - c, v_n + c]

def max_abs_of_sequence(a):
    """
    Calculate the maximum absolute value in a sequence.
    
    Computes max(|a₁|, |a₂|, |a₃|, ..., |aₙ|) for a sequence of UFL expressions.
    Notes
    -----
    This is required because (currently) ufl only allows two values in
    the constuctor of :py:meth:`ufl.Max`.
    Parameters
    ----------
    a Sequence of ufl expressions
    Returns
    -------
    Maximum of absolute values of elements in the sequence
    """
    if isinstance(a, Expr):
        return abs(a)

    assert isinstance(a, (list, tuple))

    a = list(map(abs, a))
    return max_of_sequence(a)

def max_of_sequence(a):
    """
    Calculate the maximum value in a sequence.
    
    Computes max(a₁, a₂, a₃, ..., aₙ) for a sequence of UFL expressions.
    Notes
    -----
    This is required because (currently) ufl only allows two values in
    the constuctor of :py:meth:`ufl.Max`.

    Returns
    -------
    Maximum value of elements in the sequence
    """
    return map_ufl_operator_to_sequence(a, Max)

def map_ufl_operator_to_sequence(a, op):
    """
    Utility function to map an operator to a sequence of N UFL expressions
    Notes
    -----
    This is required because (currently and commonly) ufl only allows
    two values in the constuctors of :py:meth:`ufl.MathOperator`. This is
    intended to be used with :py:meth:`ufl.Min` and :py:mesh`ufl.Max`.
    """
    if isinstance(a, Expr):
        return a

    assert isinstance(a, (list, tuple))

    alpha = op(a[0], a[1])
    for j in range(2, len(a)):
        alpha = op(alpha, a[j])
    return alpha

def extract_primitive_variables(U):
    """
    Extract primitive variables from conservative variables.
    
    Decomposes the vector of conservative variables U = [ρ, ρu, ρE]
    into primitive variables [ρ, u, E].
    
    Parameters
    ----------
    U : list of UFL expressions
        Conservative variables [ρ, ρu, ρE]
        
    Returns
    -------
    tuple
        (ρ, u, E) - Primitive variables:
        - ρ: Density
        - u: Velocity (vector)
        - E: Total specific energy
        
    Notes
    -----
    This transformation is necessary for computing pressure and other
    thermodynamic quantities that are naturally expressed in terms of
    primitive variables.
    """
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    return rho, u, E
    
def gather_function(u):
    """
    Gather function values from all processes to the root process.
    
    Collects distributed function values into a single array on process 0.
    
    Parameters
    ----------
    u : Function FEniCSx function to gather
        
    Returns
    -------
    ndarray Array containing all function values (on root process only)
    """
    dofmap = u.function_space.dofmap
    imap = dofmap.index_map
    local_range = asarray(imap.local_range, dtype = int32) * dofmap.index_map_bs
    size_global = imap.size_global * dofmap.index_map_bs
    return gather_vector(u.x.petsc_vec.array, local_range, size_global)
    
def gather_vector(local_vector, local_range, size):
    """
    Gather vector values from all processes to the root process.
    
    General-purpose utility for collecting distributed vectors.
    
    Parameters
    ----------
    local_vector : ndarray Local portion of the vector on this process
    local_range : ndarray Local range indices [start, end]
    size : int Global size of the vector
        
    Returns
    -------
    ndarray Array containing the complete vector (on root process only)
    """
    comm = MPI.COMM_WORLD
    ranges = comm.gather(local_range, root=0)
    data = comm.gather(local_vector, root=0)
    global_array = zeros(size)
    if comm.rank == 0:
        for r, d in zip(ranges, data):
            global_array[r[0]:r[1]] = d
        return global_array
    
def gather_coordinate(V):
    """
    Gathers mesh coordinates from all MPI processes into a single global array on rank 0.
    
    Parameters
    ----------
    V : FunctionSpace
        The function space whose mesh coordinates are to be gathered.
        Must be a vector-valued space for geometric coordinates.
    
    Returns
    -------
    numpy.ndarray or None
        On rank 0: Returns a (N, 3) array containing all mesh coordinates,
        where N is the global number of degrees of freedom.
        On other ranks: Returns None.
    
    Notes
    -----
    - Operates in parallel using MPI communication
    - Only rank 0 receives and returns the complete coordinate array
    - Assumes 3D coordinates (x, y, z)
    - Coordinates are ordered according to the global DOF numbering
    """
    
    comm = MPI.COMM_WORLD
    dofmap = V.dofmap
    imap = dofmap.index_map
    local_range = asarray(imap.local_range, dtype = int32) * dofmap.index_map_bs
    size_global = imap.size_global * dofmap.index_map_bs
    x = V.tabulate_dof_coordinates()[:imap.size_local]
    x_glob = comm.gather(x, root = 0)
    ranges = comm.gather(local_range, root=0)
    global_array = zeros((size_global,3))
    if comm.rank == 0:
        for r, d in zip(ranges, x_glob):
            global_array[r[0]:r[1], :] = d
        return global_array
    
def plot_mesh(mesh, tags):
    """
    Visualize a mesh with boundary markers.
    
    Creates a PyVista-based visualization of the mesh, highlighting
    boundary markers with different colors.
    
    Parameters
    ----------
    mesh : Mesh FEniCSx mesh to visualize
    tags : MeshTags Tags for mesh entities (e.g., boundaries)
    """
    from pyvista import Plotter, UnstructuredGrid
    from dolfinx.plot import vtk_mesh
    from numpy import full_like, bool_
    plotter = Plotter()
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    if tags is None:
        ugrid = UnstructuredGrid(*vtk_mesh(mesh))
    else:
        # Exclude indices marked zero
        exclude_entities = tags.find(0)
        marker = full_like(tags.values, True, dtype=bool_)
        marker[exclude_entities] = False
        ugrid = UnstructuredGrid(*vtk_mesh(mesh, tags.dim, tags.indices[marker]))
        print(tags.indices[marker], tags.values[marker])
        ugrid.cell_data["Marker"] = tags.values[marker]

    plotter.add_mesh(ugrid, show_edges=True, line_width=3)
    plotter.show_axes()
    plotter.show()