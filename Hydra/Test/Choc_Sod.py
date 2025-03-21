"""
Created on Fri Feb 28 2025
@author: bouteillerp
"""

from HYDRA import *
from pandas import read_csv
import matplotlib.pyplot as plt

###### Modèle matériau ######
gamma = 1.4  # rapport des chaleurs spécifiques (cas standard pour Sod)
rho0 = 1.0   # densité de référence
C = 1.0      # vitesse du son de référence



##### NON il y a un problème avec la capacité thermique massique ici ????



# Densités des deux régions
rho_gauche = 1.0
rho_droite = 0.125

# Pressions des deux régions
p_gauche = 1.0
p_droite = 0.1


        
# Calcul de l'énergie interne spécifique (e) à partir de l'équation d'état du gaz parfait e = p/(rho*(gamma-1))
e_gauche = p_gauche / (rho_gauche * (gamma - 1.0))
e_droite = p_droite / (rho_droite * (gamma - 1.0))

dico_eos = {"gamma": gamma, "e_max" : 2 * max(e_gauche, e_droite) }  # équation d'état de type gaz parfait
dico_devia = {}
Gaz = Material(rho0, 1, "GP", None, dico_eos, dico_devia)

# Paramètres de maillage et simulation
Nx = 125   # nombre de cellules
Largeur = 0.1
Longueur = Nx * Largeur

# Paramètres temporels
t_end = 2  # temps final classique pour Sod
dt = 2.5e-2    # pas de temps
num_time_steps = int(t_end/dt)

class SodShockTube(CompressibleEuler):
    def __init__(self, material):
        CompressibleEuler.__init__(self, material, dt)
          
    def define_mesh(self):
        msh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [Nx, 1], CellType.quadrilateral)
        return msh
    
    def prefix(self):
        return "SodShockTube"
            
    def set_boundary(self):
        # Marquer les frontières: gauche, droite, bas, haut
        self.mark_boundary([1, 2, 3, 4], ["x", "x", "y", "y"], [0, Longueur, 0, Largeur])
        
    def set_boundary_conditions(self):
        # Conditions de paroi sur toutes les frontières
        self.bc_class.wall_residual(1, "x")
        self.bc_class.wall_residual(2, "x")
        self.bc_class.wall_residual(3, "y")
        self.bc_class.wall_residual(4, "y")
        
    def set_initial_conditions(self):
        """
        Conditions initiales pour le tube à choc de Sod avec variables HDG:
        - Côté gauche (x < 0.5*Longueur): rho=1.0, p=1.0, u=0
        - Côté droit (x >= 0.5*Longueur): rho=0.125, p=0.1, u=0
        """
        # Position du diaphragme
        x_diaphragme = Longueur * 0.5
        
        # Calcul de l'énergie totale spécifique (E = e + 0.5*u^2) pour chaque région
        # Ici u=0, donc E = e
        E_gauche = e_gauche
        E_droite = e_droite
        
        # Expressions pour les discontinuités
        x = SpatialCoordinate(self.mesh)
        rho_expr = conditional(lt(x[0], x_diaphragme), rho_gauche, rho_droite)
        E_expr = conditional(lt(x[0], x_diaphragme), E_gauche, E_droite)
        
        # Initialisation de la densité
        rho_expression = Expression(rho_expr, self.V_rho.element.interpolation_points())
        self.rho.interpolate(rho_expression)
        self.rho_n.interpolate(rho_expression)
        
        # Initialisation de l'énergie totale
        E_expression = Expression(E_expr, self.V_E.element.interpolation_points())
        self.E.interpolate(E_expression)
        self.E_n.interpolate(E_expression)

        # Initialisation des variables aux interfaces pour HDG
        # Pour le maillage de facettes
        x_facet = SpatialCoordinate(self.facet_mesh)
        rho_facet_expr = conditional(lt(x_facet[0], x_diaphragme), rho_gauche, rho_droite)
        E_facet_expr = conditional(lt(x_facet[0], x_diaphragme), E_gauche, E_droite)
        
        rhobar_expression = Expression(rho_facet_expr, self.V_rhobar.element.interpolation_points())
        self.rhobar.interpolate(rhobar_expression)
        
        Ebar_expression = Expression(E_facet_expr, self.V_Ebar.element.interpolation_points())
        self.Ebar.interpolate(Ebar_expression)

    def set_output(self):
        self.t_output_list = []
        return {}
        # return {'rho': True}
        
    def query_output(self, t):
        self.t_output_list.append(t)
    
    def csv_output(self):
        # Variables à exporter 
        return {"rho": True, "Pressure":True}
    
    def final_output(self):
        df = read_csv("SodShockTube-results/rho.csv")
        resultat = [df[colonne].to_numpy() for colonne in df.columns]
        n_sortie = len(self.t_output_list)
        # pas_espace = np.linspace(0, Longueur, Nx)
        # for i, t in enumerate(self.t_output_list):
        mask = resultat[1]<=1e-10
        rho_array = resultat[-1][mask] 
        x_array = resultat[0][mask]/Longueur
        print(len(x_array))
        # print("masque", mask)
        plt.plot(x_array, rho_array, marker = "x")
        plt.xlim(0, 1)
        # plt.ylim(-1.1 * magnitude, 100)
        plt.xlabel(r"Position (mm)", size = 18)
        # plt.ylabel(r"Contrainte (MPa)", size = 18)
        plt.legend()
        plt.show()
    
pb = SodShockTube(Gaz)
# Solve(pb, TFin = t_end, dt = dt)
Solve(pb, dirk_method="SDIRK212", TFin=t_end, dt=dt)