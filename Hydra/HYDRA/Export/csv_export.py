"""
Created on Tue Sep 10 11:28:37 2024

@author: bouteillerp
"""

"""
Optimized CSV export for simulation results
===========================================

This module provides a CSV export system for HYDRA simulation results.
It enables efficient export of simulation data such as density, velocity, energy, pressure,
and velocity gradients with minimal I/O overhead.

The CSV export is designed to work with the main simulation loop, exporting data at specified
time intervals and organizing outputs by field type and time step.

Classes:
--------
OptimizedCSVExport : Main export class
    Manages the export of simulation results to CSV files
    Handles parallel coordination and file writing
    Provides methods for field selection, data gathering, and output formatting
    Supports post-processing of exported data

Notes:
------
The module implements various optimizations for handling large datasets in parallel environments.
Export settings are configured through a dictionary specifying which fields to export and
how to select the data points (all points, boundary points, etc.).
"""

from ..utils.generic_functions import gather_coordinate, gather_function

from dolfinx.fem import Expression, Function, locate_dofs_topological
from mpi4py.MPI import COMM_WORLD
from pandas import DataFrame, concat
from csv import writer, reader, field_size_limit
from numpy import ndarray, array, char
from os import remove, rename

class OptimizedCSVExport:
    def __init__(self, save_dir, name, pb, dico_csv):
        """
        Initialize the optimized CSV export system.
        
        Creates an export system for writing simulation results to CSV files
        with optimized performance for large datasets and parallel simulations.
        
        Parameters
        ----------
        save_dir : str Directory path where CSV files will be saved
        name : str Base name for the simulation results
        pb : Problem Problem object containing the simulation data and mesh
        dico_csv : dict Dictionary specifying which fields to export and how to select data points
        """
        self.save_dir = save_dir
        self.name = name
        self.pb = pb
        self.dim = pb.mesh.topology.dim
        self.dico_csv = dico_csv
        self.file_handlers = {}
        self.csv_writers = {}
        self.coordinate_data = {}
        self.initialize_export_settings()
        self.initialize_csv_files()

    def csv_name(self, name):
        """
        Generate the full path for a CSV file.
        """
        return self.save_dir+ f"{name}.csv"

    def initialize_export_settings(self):
        """
        Initialize the export settings for all field types.
        
        Sets up export configurations for velocity, energy, pressure, density,
        and velocity gradient fields based on the user-specified dictionary.
        """
        self.setup_velocity_export()
        self.setup_boundary_velocity_export()
        self.setup_rhov_export()
        self.setup_boundary_rhov_export()
        self.setup_energy_export()
        self.setup_pressure_export()
        self.setup_density_export()
        self.setup_velocity_gradient_export()
    
            
    def setup_field_export(self, field_key, field_space_attr):
        """
        Configure field export settings generically.
        
        Parameters
        ----------
        field_key : str Key in the export dictionary
        field_space_attr : str Name of the function space attribute
        """
        if field_key in self.dico_csv:
            setattr(self, f"csv_export_{field_key}", True)
            field_space = getattr(self.pb, field_space_attr)
            dte = self.dofs_to_exp(field_space, self.dico_csv.get(field_key))
            setattr(self, f"{field_key}_dte", dte)
            
            # Coordinate data
            self.coordinate_data[field_key] = self.get_coordinate_data(field_space, dte)
            
            # Name list et composantes selon la dimension
            if self.dim == 1:
                name_list = [field_key]
            elif self.dim == 2:
                cte = [self.comp_to_export(dte, i) for i in range(2)]
                setattr(self, f"{field_key}_cte", cte)
                name_list = [f"{field_key}_{{x}}", f"{field_key}_{{y}}"]
            elif self.dim == 3:
                cte = [self.comp_to_export(dte, i) for i in range(3)]
                setattr(self, f"{field_key}_cte", cte)
                name_list = [f"{field_key}_{{x}}", f"{field_key}_{{y}}", f"{field_key}_{{z}}"]
            
            setattr(self, f"{field_key}_name_list", name_list)
        else:
            setattr(self, f"csv_export_{field_key}", False)

    def setup_velocity_export(self):
        """Configure the velocity field export settings."""
        self.setup_field_export("v", "V_v")
    
    def setup_boundary_velocity_export(self):
        """Configure the boundary velocity field export settings."""
        self.setup_field_export("vbar", "V_vbar")
        
    def setup_rhov_export(self):
        """Configure the velocity field export settings."""
        self.setup_field_export("rhov", "V_rhov")
    
    def setup_boundary_rhov_export(self):
        """Configure the boundary velocity field export settings."""
        self.setup_field_export("rhovbar", "V_rhovbar")

    def setup_energy_export(self):
        """
        Configure the energy field export settings.
        
        Initializes the necessary data structures and selections for exporting
        energy data if requested in the export dictionary.
        """
        if "E" in self.dico_csv:
            self.csv_export_T = True
            self.T_dte = self.dofs_to_exp(self.pb.V_T, self.dico_csv.get("T"))
            self.coordinate_data["T"] = self.get_coordinate_data(self.pb.V_T, self.T_dte)
        else:
            self.csv_export_T = False

    def setup_pressure_export(self):
        """
        Configure the pressure field export settings.
        
        Initializes the necessary data structures and selections for exporting
        pressure data if requested in the export dictionary.
        """
        if "Pressure" in self.dico_csv:
            self.csv_export_P = True
            V_p = self.pb.V_rho
            self.p_dte = self.dofs_to_exp(V_p, self.dico_csv.get("Pressure"))
            self.coordinate_data["Pressure"] = self.get_coordinate_data(V_p, self.p_dte)
        else:
            self.csv_export_P = False

    def setup_density_export(self):
        """
        Configure the density field export settings.
        
        Initializes the necessary data structures and selections for exporting
        density data if requested in the export dictionary.
        """
        if "rho" in self.dico_csv:
            self.csv_export_rho = True
            self.rho_dte = self.dofs_to_exp(self.pb.V_rho, self.dico_csv.get("rho"))
            self.coordinate_data["rho"] = self.get_coordinate_data(self.pb.V_rho, self.rho_dte)
        else:
            self.csv_export_rho = False
            
    ##################################### Ajout Alice ##############################################
   
    def setup_velocity_gradient_export(self):
        """
        Configure the velocity gradient export settings.
        
        Initializes the necessary data structures and selections for exporting
        velocity gradient components if requested in the export dictionary.
        """
        if "L" in self.dico_csv:
            self.csv_export_L = True
            self.L_dte = self.dofs_to_exp(self.pb.V_L, self.dico_csv.get("L"))
            self.coordinate_data["L"] = self.get_coordinate_data(self.pb.V_L, self.L_dte)
            if self.dim == 1:
                self.L_name_list = ["L"]
            elif self.dim == 2:
                self.L_cte = [self.comp_to_export(self.L_dte, i) for i in range(4)]
                self.L_name_list = ["du_{x}/d_{x}", "du_{x}/d_{y}", "du_{y}/d_{x}", "du_{y}/d_{y}"]
            elif self.dim == 3:
                self.L_cte = [self.comp_to_export(self.L_dte, i) for i in range(9)]
                self.L_name_list = ["du_{x}/d_{x}", "du_{x}/d_{y}", "du_{x}/d_{z}", "du_{y}/d_{x}", "du_{y}/d_{y}", "du_{y}/d_{z}", "du_{z}/d_{x}", "du_{z}/d_{y}", "du_{z}/d_{z}"]
        else:
            self.csv_export_L = False
            
    #################################################################################################

    def initialize_csv_files(self):
        """
        Initialize CSV files for all enabled export fields.
        
        Creates CSV files with appropriate headers for each field specified
        in the export dictionary. Files are created only on the root process
        in parallel simulations.
        """
        if COMM_WORLD.Get_rank() == 0:
            for field_name, export_info in self.dico_csv.items():
                if field_name == "c":
                    for i in range(len(self.pb.material)):
                        self.create_csv_file(f"Concentration{i}")
                else:
                    self.create_csv_file(field_name)

    def create_csv_file(self, field_name):
        """
        Create a CSV file for a specific field.
        
        Parameters
        ----------
        field_name : str Name of the field for which to create a CSV file
        """
        file_path = self.csv_name(field_name)
        self.file_handlers[field_name] = open(file_path, 'w', newline='')
        self.csv_writers[field_name] = writer(self.file_handlers[field_name])
        headers = ["Time", field_name]
        self.csv_writers[field_name].writerow(headers)

    def dofs_to_exp(self, V, keyword):
        """
        Convert export keyword to degrees of freedom to export.
        
        Interprets the export specification to determine which degrees of freedom
        to export for a given function space.
        
        Parameters
        ----------
        V : FunctionSpace Function space containing the degrees of freedom
        keyword : bool or list or ndarray
            Export specification:
            - True: Export all degrees of freedom
            - ["Boundary", tag]: Export degrees of freedom on the specified boundary
            - ndarray: Export specific degrees of freedom
            
        Returns
        -------
        str or ndarray Specification of degrees of freedom to export
        """
        if isinstance(keyword, bool) and keyword is True:
            return "all"
        elif isinstance(keyword, list) and keyword[0] == "Boundary":
            return locate_dofs_topological(V, self.pb.facet_tag.dim, self.pb.facet_tag.find(keyword[1]))
        elif isinstance(keyword, ndarray):
            return keyword

    def comp_to_export(self, keyword, component):
        """
        Convert export keyword to component-specific degrees of freedom.
        
        For vector or tensor fields, determines which degrees of freedom correspond
        to a specific component.
        
        Parameters
        ----------
        keyword : str or ndarray Export specification from dofs_to_exp
        component : int Component index (0, 1, 2 for x, y, z components)
            
        Returns
        -------
        str or ndarray Component-specific degrees of freedom to export
        """
        if isinstance(keyword, str):
            return keyword
        elif isinstance(keyword, ndarray):
            vec_dof_to_exp = keyword.copy()
            vec_dof_to_exp *= self.pb.dim
            vec_dof_to_exp += component
            return vec_dof_to_exp

    def csv_export(self, t):
        """
        Export data to CSV files at the specified time.
        
        Writes the current values of all enabled fields to their respective
        CSV files at the given simulation time.
        
        Parameters
        ----------
        t : float Current simulation time
        """
        if not self.dico_csv:
            return
        if self.csv_export_v:
            self.export_field(t, "v", self.pb.U_base[1], self.v_cte, subfield_name = self.v_name_list)
        if self.csv_export_vbar:
            self.export_field(t, "vbar", self.pb.Ubar_base[1], self.vbar_cte, subfield_name = self.vbar_name_list)
        if self.csv_export_rhov:
            self.export_field(t, "rhov", self.pb.U[1], self.rhov_cte, subfield_name = self.rhov_name_list)
        if self.csv_export_rhovbar:
            self.export_field(t, "rhovbar", self.pb.Ubar[1], self.rhovbar_cte, subfield_name = self.rhovbar_name_list)
        if self.csv_export_P:
            self.pb.p_func.interpolate(self.pb.p_expr)
            self.export_field(t, "Pressure", self.pb.p_func, self.p_dte)
        if self.csv_export_rho:
            self.export_field(t, "rho", self.pb.rho, self.rho_dte)
        ################################### Ajout Alice ##########################################     
            
        if self.csv_export_L:
            self.export_field(t, "L", self.pb.u_list[2], self.L_cte, subfield_name = self.L_name_list)
            
        ##########################################################################################
            
    def export_field(self, t, field_name, field, dofs_to_export, subfield_name = None):
        """
        Export a specific field to its CSV file.
        
        Gathers the field data, processes it, and writes it to the appropriate CSV file.
        
        Parameters
        ----------
        t : float Current simulation time
        field_name : str Name of the field to export
        field : Function Function containing the field data
        dofs_to_export : str or ndarray Specification of which degrees of freedom to export
        subfield_name : list of str, optional Names for individual components of vector/tensor fields
        """
        if isinstance(subfield_name, list):
            n_sub = len(subfield_name)
            for i in range(n_sub):
                data = self.gather_field_data(field, dofs_to_export[i], size = n_sub, comp = i)
                self.write_field_data(field_name, t, data)
        else:
            data = self.gather_field_data(field, dofs_to_export)
            self.write_field_data(field_name, t, data)        

    def get_coordinate_data(self, V, key):
        """
        Gather coordinate data for the specified degrees of freedom.
        
        Collects mesh coordinates for the degrees of freedom to be exported,
        organizing them into a dictionary for post-processing.
        
        Parameters
        ----------
        V : FunctionSpace Function space containing the degrees of freedom
        key : str or ndarray Specification of which degrees of freedom to include
            
        Returns
        -------
        dict or None  Dictionary of coordinate arrays on rank 0, None on other ranks
        """
        def mpi_gather(V):
            if self.pb.mpi_bool:
                return gather_coordinate(V)
            else:
                return V.tabulate_dof_coordinates()
        dof_coords = mpi_gather(V)
        def specific(array, coord, dof_to_exp):
            if isinstance(dof_to_exp, str):
                return array[:, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]

        if COMM_WORLD.rank == 0:
            if self.dim == 1:
                data = {"x": specific(dof_coords, 0, key)}
            elif self.dim == 2:
                data = {"x": specific(dof_coords, 0, key), "y": specific(dof_coords, 1, key)}
            elif self.dim == 3:
                data = {"x": specific(dof_coords, 0, key), "y": specific(dof_coords, 1, key), "z" : specific(dof_coords, 2, key)}
            return data

    def gather_field_data(self, field, dofs_to_export, size = None, comp = None):
        """
        Gather field data for export.
        
        Collects the field values from all processes for the specified degrees of freedom.
        
        Parameters
        ----------
        field : Function Function containing the field data
        dofs_to_export : str or ndarray Specification of which degrees of freedom to export
        size : int, optional Size of the vector/tensor field (for vector/tensor components)
        comp : int, optional Component index for vector/tensor fields
            
        Returns
        -------
        ndarray Field values for the specified degrees of freedom
        """
        if self.pb.mpi_bool:
            field_data = gather_function(field)
        else:
            field_data = field.x.petsc_vec.array

        if isinstance(dofs_to_export, str) and dofs_to_export == "all" and size is None:
            return field_data
        elif isinstance(dofs_to_export, str) and dofs_to_export == "all" and size is not None:
            return field_data[comp::size]
        elif isinstance(dofs_to_export, ndarray):
            return field_data[dofs_to_export]
        else:
            return field_data
            
    def write_field_data(self, field_name, t, data):
        """
        Write field data to CSV file.
        
        Formats the field data and writes it to the appropriate CSV file
        with the current simulation time.
        
        Parameters
        ----------
        field_name : str Name of the field being exported
        t : float Current simulation time
        data : ndarray Field values to export
        """
        if COMM_WORLD.Get_rank() == 0:
            # Convertir en tableau NumPy si ce n'est pas déjà le cas
            if not isinstance(data, ndarray):
                data = array(data)
            
            # Aplatir le tableau si c'est un tableau multidimensionnel
            data = data.flatten()
            
            # Convertir les données en chaîne de caractères avec une précision fixe
            formatted_data = [f"{t:.6e}"]  # Temps avec 6 décimales
            formatted_data.extend(char.mod('%.6e', data))  # Données en notation scientifique
            
            # Écrire les données en tant que chaîne unique
            self.csv_writers[field_name].writerow([','.join(formatted_data)])
            self.file_handlers[field_name].flush()

    def close_files(self):
        """
        Close all CSV files and perform post-processing.
        """
        if COMM_WORLD.Get_rank() == 0:
            for handler in self.file_handlers.values():
                handler.close()
            self.post_process_all_files()

    def post_process_all_files(self):
        """
        Post-process all exported CSV files.
        """
        for field_name in self.dico_csv.keys():
            if field_name in ["T", "Pressure", "rho"]:
                self.post_process_csv(field_name)
            elif field_name == "U":
                self.post_process_csv(field_name, subfield_name = self.u_name_list)
            elif field_name == "v":
                self.post_process_csv(field_name, subfield_name = self.v_name_list)
            elif field_name == "vbar":
                self.post_process_csv(field_name, subfield_name = self.vbar_name_list)
            elif field_name == "rhov":
                self.post_process_csv(field_name, subfield_name = self.rhov_name_list)
            elif field_name == "rhovbar":
                self.post_process_csv(field_name, subfield_name = self.rhovbar_name_list)


    def post_process_csv(self, field_name, subfield_name = None):
        """
        Post-process a specific CSV file.
        
        Reformats the exported CSV file to include coordinate data and improve
        readability, creating a more analysis-friendly format.
        
        Parameters
        ----------
        field_name : str Name of the field to post-process
        subfield_name : list of str, optional Names for individual components of vector/tensor fields
        """
        input_file = self.csv_name(field_name)
        temp_output_file = self.csv_name(f"{field_name}_processed")

        def parse_row(row):
            return array([float(val) for val in row.split(',')])
        field_size_limit(int(1e9))  # Augmenter à une valeur très élevée, par exemple 1 milliard
        with open(input_file, 'r') as f:
            csv_reader = reader(f)
            headers = next(csv_reader)# Lire la première ligne (en-têtes)
            data = [parse_row(row[0]) for row in csv_reader]# Lire le reste des données

        data = array(data)
        times = data[:, 0]
        values = data[:, 1:]

        coord = DataFrame(self.coordinate_data[field_name])
        if subfield_name is None:
            times_pd = [f"t={t}" for t in times]
        else:
            times_pd = [f"{subfield_name[compteur%len(subfield_name)]} t={t}" for compteur, t in enumerate(times)]
        datas = DataFrame({name: lst for name, lst in zip(times_pd, values)})
        result = concat([coord, datas], axis=1)

        result.to_csv(temp_output_file, index=False, float_format='%.6e')

        remove(input_file)
        rename(temp_output_file, input_file)