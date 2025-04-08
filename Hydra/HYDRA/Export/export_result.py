"""
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""

"""
Simulation result export management
==================================

This module provides a framework for exporting simulation results from HYDRA.
It serves as the main interface for all export operations, supporting multiple output formats
including VTK and CSV.

The ExportResults class coordinates all export operations, delegating specific formats to
specialized exporters such as OptimizedCSVExport. It manages file creation, directory structure,
and proper cleanup of temporary files.

The module supports:
- VTK file export for 3D visualization in tools like ParaView
- CSV export for post-processing and data analysis

Classes:
--------
ExportResults : Main export manager
    Coordinates all export operations
    Manages file handles and directory structure
    Delegates specific formats to specialized exporters
    Provides a unified interface for the simulation loop

Methods:
--------
export_results(t) : Export results at a specific time
    Manages the export of all requested fields to the appropriate formats at time t

save_dir(name) : Get the save directory path
    Returns the standardized path for saving results
"""

from dolfinx.io import VTKFile
from os import remove, path
from mpi4py.MPI import COMM_WORLD
from .csv_export import OptimizedCSVExport

class ExportResults:
    def __init__(self, problem, name, dictionnaire, dictionnaire_csv):
        """
        Initialize the results export manager.
        
        Creates a manager for exporting simulation results to various formats,
        including VTK for visualization and CSV for data analysis.
        
        Parameters
        ----------
        problem : Problem Problem object containing the simulation data
        name : str Base name for the simulation results and output directory
        dictionnaire : dict Dictionary specifying which fields to export to VTK format
        dictionnaire_csv : dict Dictionary specifying which fields to export to CSV format
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
        Generate the directory path for saving results.
        
        Creates a standardized directory path based on the simulation name.
        
        Parameters
        ----------
        name : str Base name for the simulation
            
        Returns
        -------
        str Directory path for saving results
        """
        savedir = name + "-" + "results" + "/"
        return savedir

    def export_results(self, t):
        """
        Export results to VTK format at the specified time.
        
        Writes the current values of all enabled fields to VTK format
        for visualization in tools like ParaView.
        
        Parameters
        ----------
        t : float Current simulation time
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
