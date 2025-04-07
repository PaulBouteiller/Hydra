# HYDRA 

## A High-Order Hybridized Discontinuous Galerkin Framework for Computational Fluid Dynamics

HYDRA is an computational fluid dynamics (CFD) framework built on DOLFINx, a component of the FEniCSx Project ecosystem. It specializes in solving compressible flow problems using hybridized discontinuous Galerkin (HDG) methods.

**Note**: HYDRA is a research framework in active development and is **not** ready for production use.
## Features

- **High-Order Accuracy**: Supports arbitrary-order polynomial approximations for improved accuracy
- **Robust Shock Capturing**: Multiple shock sensor implementations (Ducros, Fernandez) for handling discontinuities
- **Advanced Numerical Fluxes**: Various Riemann solvers including Rusanov, HLL, and HLLC
- **Time Integration**: Multiple DIRK (Diagonally Implicit Runge-Kutta) schemes of orders 1-5

## Getting Started

### Installation

1. Install DOLFINx following [official instructions](https://github.com/FEniCS/dolfinx)
2. Clone the HYDRA repository
3. Add the HYDRA directory to your PYTHONPATH

## Advanced Features

### Equations of State

HYDRA supports multiple equations of state:
- **Ideal Gas (GP)**: Perfect gas law
- **Compressible Hyperelastic (U1)**: Single coefficient hyperelastic model

### Boundary Conditions

Various boundary conditions are available:
- Wall boundary conditions
- Pressure boundary conditions
- Inflow/outflow conditions

### Time Integration Schemes

Available DIRK schemes:
- **BDF1**: First-order backward Euler
- **SDIRK2**: Second-order with γ = 1-1/√2
- **SDIRK3**: Third-order L-stable scheme
- **SDIRK4**: Fourth-order L-stable (Hairer & Wanner)
- **SDIRK5**: Fifth-order L-stable (Cooper & Sayfy)
- And many more...

## Documentation

Comprehensive documentation is available in the docstrings of each module. Key modules to explore:
- `Problem.py`: Base problem definition
- `Euler.py`: Compressible Euler equations
- `NavierStokes.py`: Compressible Navier-Stokes equations
- `Solve.py`: Time integration and nonlinear solvers
- `riemann_solvers.py`: Numerical flux implementations

## Contact

For questions, suggestions, or contributions, please contact paul.bouteiller@cea.fr.

---


