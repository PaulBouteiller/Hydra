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

# Densités des deux régions
rho_gauche = 1.0
rho_droite = 0.125

# Pressions des deux régions
p_gauche = 1.0
p_droite = 0.1
        
# Calcul de l'énergie interne spécifique (e) à partir de l'équation d'état du gaz parfait e = p/(rho*(gamma-1))
rhoE_gauche = p_gauche / (gamma - 1.0)
rhoE_droite = p_droite / (gamma - 1.0)

dico_eos = {"gamma": gamma}  # équation d'état de type gaz parfait
dico_devia = {}
Gaz = Material(rho0, 1, "GP", None, dico_eos, dico_devia)

#Degré d'interpolation #~Volumes finis si degré 0.
degree = 1
    
Nx = int(500 / (degree + 1))

Longueur = 1
Largeur = 0.1 / Nx

# Paramètres temporels
t_end = 0.1  # temps final classique pour Sod
dt = 1e-3    # pas de temps
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
        self.mesh_manager.mark_boundary([1, 2, 3, 4], ["x", "x", "y", "y"], [0, Longueur, 0, Largeur])
        
    def set_boundary_conditions(self):
        # Conditions de paroi sur toutes les frontières
        self.bc_class.wall_residual(1, "x")
        self.bc_class.wall_residual(2, "x")
        self.bc_class.wall_residual(3, "y")
        self.bc_class.wall_residual(4, "y")
        
    def fem_parameters(self):
        self.deg = degree
        
    def set_initial_conditions(self):
        """
        Conditions initiales pour le tube à choc de Sod avec variables HDG:
        - Côté gauche (x < 0.5*Longueur): rho=1.0, p=1.0, u=0
        - Côté droit (x >= 0.5*Longueur): rho=0.125, p=0.1, u=0
        """
        # Position du diaphragme
        x_diaphragme = Longueur * 0.5
        x = SpatialCoordinate(self.mesh)
        
        # Initialisation de la densité
        rho_expr = conditional(lt(x[0], x_diaphragme), rho_gauche, rho_droite)
        rho_expression = Expression(rho_expr, self.V_rho.element.interpolation_points())
        self.rho.interpolate(rho_expression)
        self.rho_n.interpolate(rho_expression)
        
        # Initialisation de l'énergie totale
        rhoE_expr = conditional(lt(x[0], x_diaphragme), rhoE_gauche, rhoE_droite)
        rhoE_expression = Expression(rhoE_expr, self.V_rhoE.element.interpolation_points())
        self.rhoE.interpolate(rhoE_expression)
        self.rhoE_n.interpolate(rhoE_expression)

        # Initialisation des variables aux interfaces pour HDG
        x_facet = SpatialCoordinate(self.facet_mesh)
        rho_facet_expr = conditional(lt(x_facet[0], x_diaphragme), rho_gauche, rho_droite)
        rhoE_facet_expr = conditional(lt(x_facet[0], x_diaphragme), rhoE_gauche, rhoE_droite)
        
        rhobar_expression = Expression(rho_facet_expr, self.V_rhobar.element.interpolation_points())
        self.rhobar.interpolate(rhobar_expression)
        
        rhoEbar_expression = Expression(rhoE_facet_expr, self.V_rhoEbar.element.interpolation_points())
        self.rhoEbar.interpolate(rhoEbar_expression)

    def set_output(self):
        self.t_output_list = []
        return {}
        
    def query_output(self, t):
        self.t_output_list.append(t)
    
    def csv_output(self):
        # Variables à exporter 
        return {"rho": True, "Pressure":True}
    
pb = SodShockTube(Gaz)
Solve(pb, dirk_method="BDF1", TFin=t_end, dt=dt)
# Solve(pb, dirk_method="SDIRK212", TFin=t_end, dt=dt)
# Solve(pb, dirk_method="SDIRK5", TFin=t_end, dt=dt)

import sod_shock_analytic

# Paramètres de la simulation
dustFrac = 0.0  # fraction de poussière (0 = gaz pur)
npts = 1000  # nombre de points pour la discrétisation
left_state = (p_gauche, rho_gauche, 0)  # état gauche (pression, densité, vitesse)
right_state = (p_droite, rho_droite, 0.)  # état droit (pression, densité, vitesse)





rho_df = read_csv("SodShockTube-results/rho.csv")
rho_result = [rho_df[colonne].to_numpy() for colonne in rho_df.columns]
if degree !=0:
    y = rho_result[1]
    mask = y<=1e-10
    
    x_array = rho_result[0][mask]
else:
    rho_array = rho_result[-1]
    x_array = rho_result[0]

p_df = read_csv("SodShockTube-results/Pressure.csv")
p_result = [p_df[colonne].to_numpy() for colonne in p_df.columns]
if degree !=0:
    y = rho_result[1]
    mask_p = y<=1e-10
    xp_array = p_result[0][mask_p]
else:
    p_array = p_result[-1] 
    xp_array = p_result[0]   

if degree !=0:
    p_array = p_result[-1][mask_p] 
    rho_array = rho_result[-1][mask]

# Calculer la solution
positions, regions, values = sod_shock_analytic.solve(
    left_state=left_state, 
    right_state=right_state, 
    geometry=(0., 1., 0.5),  # frontières gauche, droite et position du choc
    t=t_end, 
    gamma=gamma, 
    npts=npts, 
    dustFrac=dustFrac
)

# Afficher les positions des différentes caractéristiques
print('Positions:')
for desc, vals in positions.items():
    print('{0:10} : {1}'.format(desc, vals))

# Afficher les valeurs dans les différentes régions
print('Regions:')
for region, vals in sorted(regions.items()):
    print('{0:10} : {1}'.format(region, vals))

# Tracer les résultats
f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(values['x'], values['p'], linewidth=1.5, color='b')
axarr[0].plot(xp_array, p_array, linewidth=1.5, color='g')
axarr[0].set_ylabel('pressure')
axarr[0].set_ylim(0, 1.1)

axarr[1].plot(values['x'], values['rho'], linewidth=1.5, color='r')
axarr[1].plot(x_array, rho_array, linewidth=1.5, color='g')
axarr[1].set_ylabel('density')
axarr[1].set_ylim(0, 1.1)

plt.suptitle('Shocktube results at t={0}\ndust fraction = {1}, gamma={2}'
              .format(t_end, dustFrac, gamma))
# plt.savefig(f"fig/Sod"+str(i)+".png", bbox_inches = 'tight')
# plt.close()


# rho_df = read_csv("SodShockTube-results/rho.csv")
# rho_result = [rho_df[colonne].to_numpy() for colonne in rho_df.columns]
# if degree !=0:
#     y = rho_result[1]
#     mask = y<=1e-10
    
#     x_array = rho_result[0][mask]
# else:
#     rho_array = rho_result[-1]
#     x_array = rho_result[0]

# p_df = read_csv("SodShockTube-results/Pressure.csv")
# p_result = [p_df[colonne].to_numpy() for colonne in p_df.columns]
# if degree !=0:
#     y = rho_result[1]
#     mask_p = y<=1e-10
#     xp_array = p_result[0][mask_p]
# else:
#     p_array = p_result[-1] 
#     xp_array = p_result[0]   

# t_list = np.linspace(0, t_end, 101)

# for i, t in enumerate(t_list):
#     if degree !=0:
#         p_array = p_result[i+2][mask_p] 
#         rho_array = rho_result[i+2][mask]

#     # Calculer la solution
#     positions, regions, values = sod_shock_analytic.solve(
#         left_state=left_state, 
#         right_state=right_state, 
#         geometry=(0., 1., 0.5),  # frontières gauche, droite et position du choc
#         t=t, 
#         gamma=gamma, 
#         npts=npts, 
#         dustFrac=dustFrac
#     )
    
#     # Afficher les positions des différentes caractéristiques
#     print('Positions:')
#     for desc, vals in positions.items():
#         print('{0:10} : {1}'.format(desc, vals))
    
#     # Afficher les valeurs dans les différentes régions
#     print('Regions:')
#     for region, vals in sorted(regions.items()):
#         print('{0:10} : {1}'.format(region, vals))
    
#     # Tracer les résultats
#     f, axarr = plt.subplots(2, sharex=True)
    
#     axarr[0].plot(values['x'], values['p'], linewidth=1.5, color='b')
#     axarr[0].plot(xp_array, p_array, linewidth=1.5, color='g')
#     axarr[0].set_ylabel('pressure')
#     axarr[0].set_ylim(0, 1.1)
    
#     axarr[1].plot(values['x'], values['rho'], linewidth=1.5, color='r')
#     axarr[1].plot(x_array, rho_array, linewidth=1.5, color='g')
#     axarr[1].set_ylabel('density')
#     axarr[1].set_ylim(0, 1.1)
    
#     plt.suptitle('Shocktube results at t={0}\ndust fraction = {1}, gamma={2}'
#                  .format(t_end, dustFrac, gamma))
#     plt.savefig(f"fig/Sod"+str(i)+".png", bbox_inches = 'tight')
#     plt.close()