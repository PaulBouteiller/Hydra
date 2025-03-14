"""
Created on Tue Dec 24 13:13:39 2024
@author: bouteillerp
"""

from HYDRA import *

###### Modèle matériau ######
kappa = 1e3
rho0 = 1e-3
C = 1
dico_eos = {"kappa" : kappa, "alpha" : 1}
dico_devia = {}
Acier = Material(rho0, C, "U1", None, dico_eos, dico_devia)

N = 20
Largeur = 1
ratio = 1
Longueur = ratio * Largeur


t_end = 6e-4

dt = 1e-6
num_time_steps = int(t_end/dt)

class Square(CompressibleEuler):
    def __init__(self, material):
        CompressibleEuler.__init__(self, material, dt, isotherm = True)
          
    def define_mesh(self):
        msh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [ratio * N, N], CellType.quadrilateral)
        return msh
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Vortex"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mark_boundary([1, 2, 3, 4], ["x", "x", "y", "y"], [0, Longueur, 0, Largeur])
        
    def set_boundary_conditions(self):
        self.bc_class.wall_residual(1, "x")
        self.bc_class.wall_residual(2, "x")
        self.bc_class.wall_residual(3, "y")
        self.bc_class.wall_residual(4, "y")
        
    def set_initial_conditions(self):
        blade_width = 0.2
        blade_length = 0.1
        blade_offset = 0.1
        center_x = Longueur/2
        center_y = Largeur/2
        # Paramètres pour les pales horizontales
        x_gauche_sup = center_x - blade_offset - blade_length/2
        x_droit_sup = center_x - blade_offset + blade_length/2
        y_bas_sup = center_y + blade_offset - blade_width/2
        y_haut_sup = center_y + blade_offset + blade_width/2

        x_gauche_inf = center_x + blade_offset - blade_length/2
        x_droit_inf = center_x + blade_offset + blade_length/2
        y_bas_inf = center_y - blade_offset - blade_width/2
        y_haut_inf = center_y - blade_offset + blade_width/2

        # Paramètres pour les pales verticales
        x_gauche_gauche = center_x - blade_offset - blade_width/2
        x_droit_gauche = center_x - blade_offset + blade_width/2
        y_bas_gauche = center_y - blade_offset - blade_length/2
        y_haut_gauche = center_y - blade_offset + blade_length/2

        x_gauche_droite = center_x + blade_offset - blade_width/2
        x_droit_droite = center_x + blade_offset + blade_width/2
        y_bas_droite = center_y + blade_offset - blade_length/2     
        y_haut_droite = center_y + blade_offset + blade_length/2    
        v_imp = 500

        def create_initial_conditions(x, v_imp):
            # Conditions pour les vitesses horizontales
            condition_sup_gauche = conditional(And(And(gt(x[1], y_bas_sup), lt(x[1], y_haut_sup)),
                                                  And(gt(x[0], x_gauche_sup), lt(x[0], x_droit_sup))), v_imp, 0)

            condition_inf_droit = conditional(And(And(gt(x[1], y_bas_inf), lt(x[1], y_haut_inf)),
                                                And(gt(x[0], x_gauche_inf), lt(x[0], x_droit_inf))), -v_imp, 0)

            # Conditions pour les vitesses verticales
            condition_gauche = conditional(And(And(gt(x[1], y_bas_gauche), lt(x[1], y_haut_gauche)),
                                              And(gt(x[0], x_gauche_gauche), lt(x[0], x_droit_gauche))), v_imp, 0)

            condition_droite = conditional(And(And(gt(x[1], y_bas_droite), lt(x[1], y_haut_droite)),
                                              And(gt(x[0], x_gauche_droite), lt(x[0], x_droit_droite))), -v_imp, 0)

            # Combinaison des conditions
            u_x_condition = condition_sup_gauche + condition_inf_droit
            u_y_condition = condition_gauche + condition_droite
            
            return u_x_condition, u_y_condition

        # Application pour le maillage principal
        u_x_condition, u_y_condition = create_initial_conditions(SpatialCoordinate(self.mesh), v_imp)

        # Création du vecteur vitesse
        u_init = as_vector([u_x_condition, u_y_condition])
        u_expr = Expression(u_init, self.V_v.element.interpolation_points())
        self.u.interpolate(u_expr)
        self.u_n.interpolate(u_expr)

        # Application pour le facet mesh
        u_x_condition, u_y_condition = create_initial_conditions(SpatialCoordinate(self.facet_mesh), v_imp)

        # Création du vecteur vitesse pour le facet mesh
        ubar_init = as_vector([u_x_condition, u_y_condition])
        ubar_expr = Expression(ubar_init, self.V_vbar.element.interpolation_points())
        self.ubar.interpolate(ubar_expr)
        
        
        # # Création du vecteur vitesse
        # u_init = as_vector([u_x_condition, u_y_condition])
        # u_expr = Expression(u_init, self.V_v.element.interpolation_points())
        # self.U_base[1].interpolate(u_expr)
        # self.Un_base[1].interpolate(u_expr)

        # # Application pour le facet mesh
        # u_x_condition, u_y_condition = create_initial_conditions(SpatialCoordinate(self.facet_mesh), v_imp)

        # # Création du vecteur vitesse pour le facet mesh
        # ubar_init = as_vector([u_x_condition, u_y_condition])
        # ubar_expr = Expression(ubar_init, self.V_vbar.element.interpolation_points())
        # self.Ubar_base[1].interpolate(ubar_expr)

    def set_output(self):
        return {"v" : True}
    
pb = Square(Acier)
Solve(pb, TFin = t_end, dt = dt)
# DIRKSolve(pb, dirk_method="SDIRK2", TFin=t_end, dt=dt)
