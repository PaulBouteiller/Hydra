"""
Propagation d'un choc
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

N = 3
Largeur = 1
ratio = 4
Longueur = ratio * Largeur


t_end = 8e-3
t_unload = 1e-3

dt = 2e-5
num_time_steps = int(t_end/dt)

class Square(CompressibleEuler):
    def __init__(self, material):
        CompressibleEuler.__init__(self, material, dt)
          
    def define_mesh(self):
        msh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [ratio * N, N], CellType.quadrilateral)

        return msh
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Choc"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mark_boundary([1, 2, 3, 4], ["x", "x", "y", "y"], [0, Longueur, 0, Largeur])
        
 
    def set_boundary_conditions(self):
        self.bc_class.wall_residual(2, "x")
        self.bc_class.wall_residual(3, "y")
        self.bc_class.wall_residual(4, "y")
        magnitude = -1
        chargement = MyConstant(self.mesh, t_unload, magnitude, Type = "Creneau")
        
        self.bc_class.pressure_residual(chargement, 1)

    def set_output(self):
        # return {"v" : True, "rho" : True}
        return {"p" : True}
    
    def csv_output(self):
        # return {"v" : True, "rho" : True}
        return {"v" : True}
    
pb = Square(Acier)
Solve(pb, TFin = t_end, dt = dt)
