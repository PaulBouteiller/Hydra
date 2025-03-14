"""
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""
from dolfinx.io import VTKFile
from os import remove, path
from mpi4py.MPI import COMM_WORLD
from .csv_export import OptimizedCSVExport

class ExportResults:
    def __init__(self, problem, name, dictionnaire, dictionnaire_csv):
        """
        Initialise l'export des résultats.

        Parameters
        ----------
        problem : Objet de la classe Problem, problème mécanique qui a été résolu.
        name : String, nom du dossier dans lequel sera stocké les résultats.
        dictionnaire : Dictionnaire, dictionnaire contenant le nom des variables
                        que l'on souhaite exporter au format XDMF (Paraview).
        dictionnaire_csv : Dictionnaire, dictionnaire contenant le nom des variables
                            que l'on souhaite exporter au format csv.
        """
        self.pb = problem
        self.name = name
        self.dico = dictionnaire
        self.file_name = self.save_dir(name) + "results.vtk"
        if path.isfile(self.file_name) and COMM_WORLD.rank == 0 :
            remove(self.file_name)
            print("File has been found and deleted.")
        file_results = VTKFile(self.pb.mesh.comm, self.file_name, "w")
        file_results.write_mesh(self.pb.mesh)
        self.file_results = VTKFile(self.pb.mesh.comm, self.file_name, "a")
        self.csv = OptimizedCSVExport(self.save_dir(name), name, problem, dictionnaire_csv)

    def save_dir(self, name):
        """
        Renvoie nom du dossier dans lequel sera stocké les résultats.
        Parameters
        ----------
        name : String, nom du dossier dans lequel sera stocké les résultats.
        """
        savedir = name + "-" + "results" + "/"
        return savedir

    def export_results(self, t):
        """
        Exporte les résultats au format XDMF.

        Parameters
        ----------
        t : Float, temps de la simulation auquel les résultats sont exportés.
        """
        if  self.dico == {}:
            return       
        if self.dico.get("v"):
            self.file_results.write_function(self.pb.u, t)
            
        if self.dico.get("rho"):
            self.file_results.write_function(self.pb.rho, t)
            
        if self.dico.get("E"):
            self.file_results.write_function(self.pb.E, t)
            
        if self.dico.get("p"):
            self.pb.p_func.interpolate(self.pb.p_expr)
            self.file_results.write_function(self.pb.p_func, t)
