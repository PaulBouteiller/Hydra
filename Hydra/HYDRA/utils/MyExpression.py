"""
Created on Thu Jul 21 09:52:08 2022

@author: bouteillerp
"""
"""
Custom expressions for time-dependent boundary conditions and sources
===================================================================

This module provides specialized classes for defining time-dependent expressions
in the HYDRA simulation framework. These expressions are particularly useful for
representing boundary conditions, source terms, and other time-varying inputs
in fluid dynamics simulations.

The implementation includes:
- Step functions (constant value over specific time intervals)
- Ramp functions (linear increase over time)
- Smoothed step functions (continuous transitions)
- Hat functions (triangular profiles)
- User-defined tabulated functions

Each expression type supports automatic time-dependent evaluation and can be
used in FEniCSx variational formulations through the provided interface.

Classes:
--------
MyConstant : Factory class for creating time-dependent expressions
    Provides a unified interface for all expression types
    Selects the appropriate concrete implementation based on type

Creneau : Step function expression
    Implements a constant value over [0, t_crit] and zero elsewhere

SmoothCreneau : Smoothed step function
    Implements a smooth transition to and from a constant plateau value

Chapeau : Hat function expression
    Implements a triangular profile with peak at t_crit/2

Rampe : Ramp function expression
    Implements a linear increase with specified slope

UserDefined : User-specified tabulated expression
    Implements interpolation from user-provided time-value pairs

Tabulated_BCs : Factory class specifically for boundary conditions
    Creates expressions suitable for time-dependent boundary conditions

MyConstantExpression : Extended expression with function composition
    Combines an expression with a spatial function for space-time dependence

Functions:
----------
interpolation_lin(temps_originaux, valeurs_originales, nouveaux_temps)
    Linear interpolation utility for tabulated data
"""

from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType
from scipy import interpolate

def interpolation_lin(temps_originaux, valeurs_originales, nouveaux_temps):
    """
    Linearly interpolate values at new time points.
    
    Parameters
    ----------
    temps_originaux : array Original time points
    valeurs_originales : array Values at original time points
    nouveaux_temps : array New time points for interpolation
        
    Returns
    -------
    array Interpolated values at new time points
    """
    f = interpolate.interp1d(temps_originaux, valeurs_originales)
    return f(nouveaux_temps)

class MyConstant:
    """
    Factory class for creating time-dependent expressions.
    
    This class provides a unified interface for creating various types
    of time-dependent expressions, such as step functions, ramp functions,
    and user-defined functions.
    """
    
    def __init__(self, mesh, *args, **kwargs):
        """
        Initialize a time-dependent expression.
        
        Parameters
        ----------
        mesh : Mesh FEniCSx mesh
        *args : tuple  Arguments specific to the expression type
        **kwargs : dict Expression type and additional parameters:
            - Type: "Creneau", "SmoothCreneau", "Chapeau", "Rampe", or "UserDefined"
        """
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "SmoothCreneau":
            self.Expression = self.SmoothCreneau(mesh, *args)
        elif kwargs.get("Type") == "Chapeau":
            self.Expression = self.Chapeau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        else: 
            raise ValueError("Wrong definition")
        
    class Creneau:  
        """
        Step function (constant value over a time interval).
        
        Creates a function that is constant from t=0 to t=t_crit,
        then zero afterwards.
        """
        
        def __init__(self, mesh, t_crit, amplitude):
            """
            Initialize a step function.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            t_crit : float
                End time of the step
            amplitude : float
                Value during the step
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Create time-dependent value array.
            
            Precomputes values at all time steps for efficient lookup
            during time-stepping.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation
            """
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)
                    
    class SmoothCreneau:  
        """
        Smoothed step function with ramp-up and ramp-down.
        
        Creates a function that ramps up linearly, maintains a constant value,
        then ramps down linearly.
        """
        
        def __init__(self, mesh, t_load, t_plateau, amplitude):
            """
            Initialize a smoothed step function.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            t_load : float
                Ramp-up time
            t_plateau : float
                Duration of constant value
            amplitude : float
                Maximum value
            """
            self.t_load = t_load
            self.t_plateau = t_plateau 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Create time-dependent value array.
            
            Precomputes values at all time steps for efficient lookup
            during time-stepping.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation
            """
            self.value_array = []
            t_fin_plate = self.t_load + self.t_plateau
            t_fin = 2 * self.t_load + self.t_plateau
            for i in range(len(load_steps)):
                load_step = load_steps[i]
                if load_step<=self.t_load:
                    self.value_array.append(self.amplitude * load_step/self.t_load)
                elif load_step >=self.t_load and load_step <=t_fin_plate:
                    self.value_array.append(self.amplitude)
                elif load_step >= t_fin_plate and load_step <= t_fin:
                    self.value_array.append(-self.amplitude / self.t_load * (load_step - t_fin_plate) + self.amplitude)
                else:
                    self.value_array.append(0)

    class Chapeau:  
        """
        Hat function (triangular profile).
        
        Creates a function that increases linearly to a peak at t_crit/2,
        then decreases linearly to zero at t_crit.
        """
        
        def __init__(self, mesh, t_crit, amplitude):
            """
            Initialize a hat function.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            t_crit : float
                Total duration (peak at t_crit/2)
            amplitude : float
                Maximum value at peak
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Create time-dependent value array.
            
            Precomputes values at all time steps for efficient lookup
            during time-stepping.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation
            """
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit/2:
                    self.value_array.append(2 * self.amplitude * load_steps[i]/self.t_crit)
                elif load_steps[i] >=self.t_crit/2 and load_steps[i] <=self.t_crit:
                    self.value_array.append(2 * self.amplitude * (1 - load_steps[i]/self.t_crit))
                else:
                    self.value_array.append(0)
            
    class Rampe:  
        """
        Ramp function (linear increase).
        
        Creates a function that increases linearly with specified slope.
        """
        
        def __init__(self, mesh, pente):
            """
            Initialize a ramp function.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            pente : float
                Slope of the ramp
            """
            self.pente = pente
            self.constant = Constant(mesh, ScalarType(0))
            self.v_constant = Constant(mesh, ScalarType(0))
            self.a_constant = Constant(mesh, ScalarType(0))
        
        def set_time_dependant_array(self, load_steps):
            """
            Create time-dependent value, velocity, and acceleration arrays.
            
            Precomputes values at all time steps for efficient lookup
            during time-stepping.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation
            """
            self.value_array = []
            self.speed_array = []
            self.acceleration_array = []
            for i in range(len(load_steps)):
                self.value_array.append(load_steps[i] * self.pente)
                self.speed_array.append(self.pente)
                self.acceleration_array.append(0)
        
    class UserDefined:
        """
        User-defined tabulated function.
        
        Creates a function based on user-provided values at specific time points.
        """
        
        def __init__(self, mesh, value_array, speed_array = None):
            """
            Initialize a user-defined function.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            value_array : array
                Values at each time step
            speed_array : array, optional
                Velocity values at each time step
            """
            self.constant = Constant(mesh, ScalarType(0))
            self.value_array = value_array
            self.speed_array = speed_array
        
        def set_time_dependant_array(self, load_steps):
            """
            Configure time-dependent arrays.
            
            Does nothing for user-defined arrays since they are
            provided directly.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation (unused)
            """
            pass
        
class MyConstantExpression(MyConstant):
    """
    Extended expression that combines a time-dependent function with a spatial function.
    """
    
    def __init__(self, function, mesh, *args, **kwargs):
        """
        Initialize a combined space-time expression.
        
        Parameters
        ----------
        function : Function
            Spatial function to combine with time-dependent behavior
        mesh : Mesh
            FEniCSx mesh
        *args : tuple
            Arguments for the time-dependent part
        **kwargs : dict
            Arguments for the time-dependent part
        """
        self.function = function
        MyConstant.__init__(mesh, *args, **kwargs)
        
class Tabulated_BCs:
    """
    Factory class for creating tabulated boundary conditions.
    """
    
    def __init__(self, mesh, *args, **kwargs):
        """
        Initialize tabulated boundary conditions.
        
        Parameters
        ----------
        mesh : Mesh
            FEniCSx mesh
        *args : tuple
            Arguments specific to the boundary condition type
        **kwargs : dict
            Boundary condition type:
            - Type: "Creneau", "Rampe", or "UserDefined"
        """
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        
    class Creneau:  
        """
        Step function boundary condition.
        
        Creates a boundary condition that is constant from t=0 to t=t_crit,
        then zero afterwards.
        """
        
        def __init__(self, mesh, t_crit, amplitude):
            """
            Initialize a step function boundary condition.
            
            Parameters
            ----------
            mesh : Mesh
                FEniCSx mesh
            t_crit : float
                End time of the step
            amplitude : float
                Value during the step
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, amplitude)
            
        def set_time_dependant_array(self, load_steps):
            """
            Create time-dependent value array.
            
            Precomputes values at all time steps for efficient lookup
            during time-stepping.
            
            Parameters
            ----------
            load_steps : array
                Time points for evaluation
            """
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)
    f = interpolate.interp1d(temps_originaux, valeurs_originales)
    return f(nouveaux_temps)

class MyConstant:
    def __init__(self, mesh, *args, **kwargs):
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "SmoothCreneau":
            self.Expression = self.SmoothCreneau(mesh, *args)
        elif kwargs.get("Type") == "Chapeau":
            self.Expression = self.Chapeau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        else: 
            raise ValueError("Wrong definition")
        
    class Creneau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)
                    
    class SmoothCreneau:  
        def __init__(self, mesh, t_load, t_plateau, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_load = t_load
            self.t_plateau = t_plateau 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            t_fin_plate = self.t_load + self.t_plateau
            t_fin = 2 * self.t_load + self.t_plateau
            for i in range(len(load_steps)):
                load_step = load_steps[i]
                if load_step<=self.t_load:
                    self.value_array.append(self.amplitude * load_step/self.t_load)
                elif load_step >=self.t_load and load_step <=t_fin_plate:
                    self.value_array.append(self.amplitude)
                elif load_step >= t_fin_plate and load_step <= t_fin:
                    self.value_array.append(-self.amplitude / self.t_load * (load_step - t_fin_plate) + self.amplitude)
                else:
                    self.value_array.append(0)

    class Chapeau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un chapeau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit/2:
                    self.value_array.append(2 * self.amplitude * load_steps[i]/self.t_crit)
                elif load_steps[i] >=self.t_crit/2 and load_steps[i] <=self.t_crit:
                    self.value_array.append(2 * self.amplitude * (1 - load_steps[i]/self.t_crit))
                else:
                    self.value_array.append(0)
            
    class Rampe:  
        def __init__(self, mesh, pente):
            """
            Défini chargement rampe.

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t : Float, instant courant.
            amplitude : Float, pente du chargement.
            Le chargement sera donc T^{D} = pente * t
            """
            self.pente = pente
            self.constant = Constant(mesh, ScalarType(0))
            self.v_constant = Constant(mesh, ScalarType(0))
            self.a_constant = Constant(mesh, ScalarType(0))
        
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            self.speed_array = []
            self.acceleration_array = []
            for i in range(len(load_steps)):
                self.value_array.append(load_steps[i] * self.pente)
                self.speed_array.append(self.pente)
                self.acceleration_array.append(0)
        
    class UserDefined:
        def __init__(self, mesh, value_array, speed_array = None):
            """
            Défini un chargement uniforme qui sera appliqué au bord.
            Il s'agit d'un chargement de la forme T^{D} = f(t).
            L'utilisateur doit donner en argument une liste ou un numpy
            array, qui contient les valeurs de la contrainte au bord,
            et de même longueur que le nombre de pas de temps.

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            value_array : numpy.array ou list, list des valeurs de la contrainte
                                                à appliquer.
            """
            self.constant = Constant(mesh, ScalarType(0))
            self.value_array = value_array
            self.speed_array = speed_array
        
        def set_time_dependant_array(self, load_steps):
            pass
        
class MyConstantExpression(MyConstant):
    def __init__(self, function, mesh, *args, **kwargs):
        self.function = function
        MyConstant.__init__(mesh, *args, **kwargs)
        
class Tabulated_BCs:
    def __init__(self, mesh, *args, **kwargs):
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        
    class Creneau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, amplitude)
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)