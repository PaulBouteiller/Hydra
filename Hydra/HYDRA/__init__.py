from __future__ import division, print_function
from mpi4py import MPI
from dolfinx.mesh import (create_interval, create_unit_interval, 
                          create_unit_square, create_rectangle, 
                          CellType, create_box, locate_entities_boundary)
from dolfinx.fem import Constant, Expression
from ufl import (cofac, det, as_vector, conditional, lt, And, gt, Or, 
                 action, SpatialCoordinate)
from basix.ufl import element, quadrature_element
import numpy as np
from dolfinx import __version__

from .utils.default_parameters import *
from .utils.generic_functions import plot_mesh
from .ConstitutiveLaw.material import *
from .VariationalFormulation.Euler import CompressibleEuler
from .VariationalFormulation.NavierStokes import CompressibleNavierStokes

from .utils.MyExpression import MyConstant

from .Solve.Solve import Solve
print("Loading Hydra base on dolfinx version " + __version__)
