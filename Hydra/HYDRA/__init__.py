"""
HYDRA: Hybridizable Discontinuous Galerkin (HDG) solver for compressible fluid dynamics
====================================================================================

HYDRA is a Python package implementing high-order HDG methods for solving compressible
Euler and Navier-Stokes equations using the FEniCSx finite element framework.

The package provides:
- Advanced numerical methods for compressible flows
- Modular formulations for Euler and Navier-Stokes equations
- Multiple equation of state (EOS) models
- Various shock-capturing techniques
- Efficient linear and non-linear solvers
- Time integration using DIRK (Diagonally Implicit Runge-Kutta) schemes
- Export utilities for post-processing and visualization

The implementation is based on the hybridizable discontinuous Galerkin (HDG) method,
which combines the advantages of discontinuous Galerkin methods (local conservation,
stability for advection-dominated problems) with the efficiency of hybrid methods
(reduced global system size).

This package is built on top of the FEniCSx framework, which provides efficient
finite element implementations and seamless parallel computing through PETSc.

Main components:
- ConstitutiveLaw: Material models and equations of state
- VariationalFormulation: Weak formulations for various fluid dynamics problems
- Solve: Linear and nonlinear solvers with time integration
- Export: Data export for visualization and post-processing
- utils: Utility functions and parameter settings
"""

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

from .utils.generic_functions import plot_mesh
from .ConstitutiveLaw.material import Material
from .VariationalFormulation.Euler import CompressibleEuler
from .VariationalFormulation.NavierStokes import CompressibleNavierStokes

from .utils.MyExpression import MyConstant
# from .utils.holo_utils import *

from .Solve.Solve import Solve
# from .Solve.VF_informed_Solve import JAXFLUIDS_HYDRA_SOLVE
print("Loading Hydra base on dolfinx version " + __version__)
