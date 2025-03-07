"""
Created on Thu Jun 23 10:47:33 2022

@author: bouteillerp
"""

from ufl import dot

class EOS:
    def __init__(self, kinematic, quadrature):
        """
        Classe contenant toutes les équations d'état disponible du code CHARON.
        Le type d'équation d'état est entièrement déterminé lors de la création
        d'un objet de la classe Material.

        Parameters
        ----------
        kinematic : Objet de la classe Kinematic
        """
        self.kin = kinematic
        self.quad = quadrature
    
    def set_eos(self, rho, u, E, mat):
        """
        Renvoie l'expression de la pression
        Parameters
        ----------
        J : Jacobien de la transformation.
        mat : Objet de la classe material.
        T : champ de température actuelle
        T0 : champ de température initiale

        Returns
        -------
        p : Pression du modèle
        """
        eos_param = mat.eos
        if mat.eos_type == "U1":
            p = -eos_param.kappa * (mat.rho_0/rho - 1)
        elif mat.eos_type == "GP":
            p = (eos_param.gamma - 1) * rho * (E - 1./2 * dot(u, u))
        else:
            raise ValueError("Unknwon eos")
        return p