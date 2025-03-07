"""
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""
from math import sqrt

class Material:
    def __init__(self, rho_0, C_mass, eos_type, dev_type, dico_eos, dico_devia, **kwargs):
        """
        Création du matériau à l'étude.
        Parameters
        ----------
        rho_0 : Float or Expression, masse volumique initiale M.L^{-3}
        C_mass : Function ou Float, capacité thermique massique en J.K^{-1}.kg^{-1} (= M.L^{2}.T^{-2}.K^{-1})
        eos_type : String, type d'équation d'état souhaitée.
        dev_type : String, type d'équation déviatorique souhaitée.
        dico_eos : Dictionnaire, dico contenant les paramètres nécessaires
                                    à la création du modèle d'équation d'état.
        dico_devia : Dictionnaire, dico contenant les paramètres nécessaires
                                    à la création du modèle de comportement déviatorique.
        **kwargs : Paramètres optionnels supplémentaires utilisé pour la détonique
        """
        self.rho_0 = rho_0
        self.C_mass = C_mass
        self.eos_type = eos_type      
        self.dev_type = dev_type
        self.eos = self.eos_selection(self.eos_type)(dico_eos)
        self.devia = self.deviatoric_selection(self.dev_type)(dico_devia)
        self.celerity = self.eos.celerity(rho_0)
       
        self.e_activation = kwargs.get("e_activation", None)
        self.kin_pref = kwargs.get("kin_pref", None)
        
        print("La capacité thermique vaut", self.C_mass)        
        print("La masse volumique vaut", self.rho_0)
        print("La célérité des ondes élastique est", self.celerity)
        
    def eos_selection(self, eos_type):
        """
        Retourne le nom de la classe associée au modèle d'EOS choisi.
        Parameters
        ----------
        dev_type : String, nom du modèle d'équation d'état.
        Raises
        ------
        ValueError, erreur si un comportement déviatorique inconnu est demandé.
        """
        if eos_type in ["U1", "U2", "U3", "U4", "U5", "U7", "U8"]:
            return U_EOS
        elif eos_type == "GP":
            return GP_EOS
        elif eos_type == "NewtonianFluid":
            return NewtonianFluid_EOS
        else:
            raise ValueError("Equation d'état inconnue")
        
    def deviatoric_selection(self, dev_type):
        """
        Retourne le nom de la classe associée au modèle déviatorique retenu.
        Parameters
        ----------
        dev_type : String, nom du modèle déviatorique parmi:
                            None, IsotropicHPP, NeoHook, MooneyRivlin.
        Raises
        ------
        ValueError, erreur si un comportement déviatorique inconnu est demandé.
        """
        if dev_type == None:
            return None_deviatoric
        else:
            raise ValueError("Comportement déviatorique inconnu")        

class U_EOS:
    def __init__(self, dico):
        """
        Défini un objet possédant une équation d'état hyper-élastique isotrope 
        à un coefficient.

        Parameters
        ----------
        kappa : Float, module de compressibilité
        alpha : Float, coefficient de dilatation thermique en K^{-1}
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
        Renvoie une estimation de la célérité des ondes élastique 
        dans un milieu hyper-élastique.       
        """
        return sqrt(self.kappa / rho_0)

class GP_EOS:
    def __init__(self, dico):
        """
        Défini les paramètres d'un gaz suivant la loi des gaz parfaits. 
        Parameters
        ----------
        gamma : Float, coefficient polytropique du gaz.
        e_max : fFoat, estimation de l'énergie interne massique maximale (ne sert que pour estimation CFL)
        """
        try:
            self.gamma = dico["gamma"]
            self.e_max = dico["e_max"]
        except KeyError:
            raise ValueError("La loi d'état du gaz parfait n'est pas correctement définie")

        print("Le coefficient polytropique vaut", self.gamma)
        print("Une estimation de la température maximale est", self.e_max)
        
    def celerity(self, rho_0):
        """
        Renvoie une estimation de la célérité des ondes accoustiques
        """
        return sqrt(self.gamma * (self.gamma - 1) * self.e_max)
    
class NewtonianFluid_EOS:
    def __init__(self, dico):
        """
        Défini un objet possédant des 
        caractéristiques mécanique d'un fluide Newtobien

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        chiT : TYPE
            DESCRIPTION.
        """
        try:
            self.k = dico["k"]
            self.alpha = dico["alpha"]
            self.chiT = dico["chiT"]
        except KeyError:
            raise ValueError("La loi d'état du fluide Newtonien n'est pas correctement définie")
        
        print("La viscosité de volumique est", self.k)
        print("Le coefficient de conductivité thermique vaut", self.alpha)
        print("La compréssibilité à température constante", self.chiT)
        
    def celerity(self, rho_0):
        return sqrt(1/(self.chiT * rho_0))

class None_deviatoric:
    def __init__(self, dico):
        """
        Défini un comportement déviatorique nul(simulation purement hydro)
        """
        pass