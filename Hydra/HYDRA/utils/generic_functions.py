from ufl import max_value as Max
from ufl.core.expr import Expr
from ufl import dot
from mpi4py import MPI
from numpy import asarray, int32, zeros

def euler_eigenvalues_2D(v, n, c):
    """
    v = velocity
    n = n = FacetNormal(msh)
    Retourne la liste [lambda1, lambda2,, lambda4] pour le Jacobien dans la direction n.
    """
    # vitesse normale (v·n)
    v_n = dot(v, n)
    return [v_n - c, v_n + c]

def max_abs_of_sequence(a):
    r"""Utility function to generate the maximum of the absolute values of
    elements in a sequence

    .. math:: \max(|a_1|, |a_2|, |a_3|, \ldots, |a_N|)
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
    r"""Utility function to generate the maximum of the absolute values of
    elements in a sequence
    .. math:: \max(a_1, a_2, a_3, \ldots, a_N)
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
    """Extrait les variables primitives à partir des variables conservatives.

    Cette fonction décompose le vecteur des variables conservatives U = [ρ, ρu, ρE]
    en variables primitives [ρ, u, E] où ρ est la densité, u est la vitesse,
    et E est l'énergie spécifique totale.
    
    Parameters
    ----------
    U : list[ufl.core.expr.Expr] Liste des variables conservatives [ρ, ρu, ρE].
    Returns
    -------
    tuple Triplet (ρ, u, E) des variables primitives, où:
        - ρ : Densité
        - u : Vitesse (vecteur)
        - E : Énergie spécifique totale  
    Notes
    -----
    Cette transformation est nécessaire pour calculer la pression et d'autres
    quantités thermodynamiques qui sont naturellement exprimées en termes des
    variables primitives.
    """
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    return rho, u, E
    
def gather_function(u):
    """
    Rassemble les inconnus dans un unique vecteur sur le processus 0
    Parameters
    ----------
    u : Function.
    Returns
    -------
    global_array : np.array contenant la concaténation des inconnues portées
                    par les différents processus
    """
    dofmap = u.function_space.dofmap
    imap = dofmap.index_map
    local_range = asarray(imap.local_range, dtype = int32) * dofmap.index_map_bs
    size_global = imap.size_global * dofmap.index_map_bs
    return gather_vector(u.x.petsc_vec.array, local_range, size_global)
    
def gather_vector(local_vector, local_range, size):
    """
    Rassemble les inconnus dans un unique vecteur sur le processus 0
    Parameters
    ----------
    u : Function.
    Returns
    -------
    global_array : np.array contenant la concaténation des inconnues portées
                    par les différents processus
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