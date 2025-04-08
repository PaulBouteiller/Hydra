"""
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""
"""
Material models for compressible fluid dynamics simulations
==========================================================

This module provides a comprehensive framework for defining and managing material properties
in compressible fluid dynamics simulations. It includes classes for different equation of state
(EOS) models and deviatoric behavior models.

The main components include:
- Material: Main class for defining material properties
- U_EOS: Hyperelastic equation of state (U1-U8 family)
- GP_EOS: Ideal gas equation of state (gamma-law)
- None_deviatoric: Default deviatoric behavior (purely hydrostatic)

The module supports various material models with different complexity levels, from simple
ideal gases to complex hyperelastic materials. Each material can be defined by combining
an equation of state with a deviatoric behavior model.

Classes:
--------
Material : Main class for material definition
    Manages material properties, equation of state, and deviatoric behavior
    Provides methods for selecting and configuring material models
    
U_EOS : Hyperelastic equation of state
    Implements the U-family of hyperelastic EOS (U1-U8)
    Provides methods for pressure and sound speed calculation
    
GP_EOS : Ideal gas equation of state
    Implements the gamma-law equation of state for ideal gases
    Provides methods for pressure and sound speed calculation
    
None_deviatoric : Default deviatoric behavior
    Implements pure hydrostatic behavior (no deviatoric stresses)

Notes:
------
The Material class allows for modular definition of material behavior by combining
different EOS models with different deviatoric behavior models.
Additional material models can be added by extending the appropriate classes.
"""


from math import sqrt

class Material:
    def __init__(self, rho_0, C_mass, eos_type, dev_type, dico_eos, dico_devia):
        """
        Initialize a material model for fluid dynamics simulations.
        Create a material with specified density, heat capacity, equation of state,
        and deviatoric behavior.
        
        Parameters
        ----------
        rho_0 : float or Expression Initial mass density (M.L^-3)
        C_mass : float or Function Mass heat capacity in J.K^-1.kg^-1 (= M.L^2.T^-2.K^-1)
        eos_type : str Type of equation of state model ('U1', 'U2', ..., 'GP', etc.)
        dev_type : str or None Type of deviatoric behavior model (None for purely hydrostatic behavior)
        dico_eos : dict Dictionary containing parameters required for the equation of state model
        dico_devia : dict Dictionary containing parameters required for the deviatoric model
        """
        self.rho_0 = rho_0
        self.C_mass = C_mass
        self.eos_type = eos_type      
        self.dev_type = dev_type
        self.eos = self.eos_selection(self.eos_type)(dico_eos)
        self.devia = self.deviatoric_selection(self.dev_type)(dico_devia)
        self.celerity = self.eos.celerity(rho_0)
        
        print("La capacité thermique vaut", self.C_mass)        
        print("La masse volumique vaut", self.rho_0)
        print("La célérité des ondes élastique est", self.celerity)
        
    def eos_selection(self, eos_type):
        """
        Select the appropriate equation of state class based on type.
        
        Maps the EOS type string to the corresponding EOS class implementation.
        
        Parameters
        ----------
        eos_type : str Type of equation of state model ('U1', 'U2', ..., 'GP', etc.)
            
        Returns
        -------
        class The EOS class corresponding to the specified type
            
        Raises
        ------
        ValueError If an unknown EOS type is specified
        """
        if eos_type in ["U1", "U2", "U3", "U4", "U5", "U7", "U8"]:
            return U_EOS
        elif eos_type == "GP":
            return GP_EOS
        else:
            raise ValueError("Equation d'état inconnue")
        
    def deviatoric_selection(self, dev_type):
        """
        Select the appropriate deviatoric behavior class based on type.
        
        Maps the deviatoric behavior type string to the corresponding class implementation.
        
        Parameters
        ----------
        dev_type : str or None Type of deviatoric behavior model (None for purely hydrostatic behavior)
            
        Returns
        -------
        class The deviatoric behavior class corresponding to the specified type
            
        Raises
        ------
        ValueError If an unknown deviatoric behavior type is specified
        """
        if dev_type == None:
            return None_deviatoric
        else:
            raise ValueError("Comportement déviatorique inconnu")        

class U_EOS:
    def __init__(self, dico):
        """
        Initialize a hyperelastic isotropic one-parameter equation of state.
        
        Creates a hyperelastic isotropic equation of state with a single
        compressibility parameter (kappa) and thermal expansion coefficient.
        
        Parameters
        ----------
        dico : dict
            Dictionary containing the EOS parameters:
            - kappa: Bulk modulus
            - alpha: Thermal expansion coefficient in K^-1
            
        Raises
        ------
        ValueError If the required parameters are not provided in the dictionary
        """
        try:
            self.kappa = dico["kappa"]
            self.alpha = dico["alpha"]
        except KeyError:
            raise ValueError("Le matériau n'est pas correctement défini")
            
        print("Le module de compressibilité du matériau est", self.kappa)
        print("Le coefficient d'expansion thermique vaut", self.alpha)
        
    def celerity(self, rho_0):
        """
        Calculate the elastic wave speed in the hyperelastic medium.
        
        Computes an estimate of the elastic wave speed (sound speed) in a
        hyperelastic medium based on the bulk modulus and initial density.
        
        Parameters
        ----------
        rho_0 : float Initial mass density
            
        Returns
        -------
        float Estimated sound speed (c = sqrt(kappa/rho_0))
        """
        return sqrt(self.kappa / rho_0)

class GP_EOS:
    def __init__(self, dico):
        """
        Initialize an ideal gas (gamma-law) equation of state.
        
        Parameters
        ----------
        dico : dict
            Dictionary containing the EOS parameters:
            - gamma: Polytropic coefficient (ratio of specific heats Cp/Cv)
            
        Raises
        ------
        ValueError If the required parameter (gamma) is not provided in the dictionary
        """
        try:
            self.gamma = dico["gamma"]
        except KeyError:
            raise ValueError("La loi d'état du gaz parfait n'est pas correctement définie")

        print("Le coefficient polytropique vaut", self.gamma)
        
    def celerity(self, rho_0):
        """
        Calculate the acoustic wave speed in the ideal gas.
        
        This method is implemented as a placeholder in the base class.
        The actual sound speed calculation for ideal gases depends on temperature
        and is typically computed as c = sqrt(gamma*p/rho) during simulation.
        
        Parameters
        ----------
        rho_0 : float Initial mass density
            
        Returns
        -------
        None This base implementation returns None and is expected to be
            overridden or implemented differently during simulation
        """
        return
        return

class None_deviatoric:
    def __init__(self, dico):
        """
        Défini un comportement déviatorique nul(simulation purement hydro)
        """
        pass