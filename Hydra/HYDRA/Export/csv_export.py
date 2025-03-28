"""
Created on Tue Sep 10 11:28:37 2024

@author: bouteillerp
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
        return self.save_dir+ f"{name}.csv"

    def initialize_export_settings(self):
        self.setup_velocity_export()
        self.setup_energy_export()
        self.setup_pressure_export()
        self.setup_density_export()
            
    def setup_velocity_export(self):
        if "v" in self.dico_csv:
            self.csv_export_v = True
            self.v_dte = self.dofs_to_exp(self.pb.V_v, self.dico_csv.get("v"))
            self.coordinate_data["v"] = self.get_coordinate_data(self.pb.V_v, self.v_dte)
            if self.dim == 1:
                self.v_name_list = ["v"]
            elif self.dim == 2:
                self.v_cte = [self.comp_to_export(self.v_dte, i) for i in range(2)]
                self.v_name_list = ["u_{x}", "u_{y}"]
            elif self.dim == 3:
                self.v_cte = [self.comp_to_export(self.v_dte, i) for i in range(3)]
                self.v_name_list = ["v_{x}", "v_{y}", "v_{z}"]
        else:
            self.csv_export_v = False            

    def setup_energy_export(self):
        if "E" in self.dico_csv:
            self.csv_export_T = True
            self.T_dte = self.dofs_to_exp(self.pb.V_T, self.dico_csv.get("T"))
            self.coordinate_data["T"] = self.get_coordinate_data(self.pb.V_T, self.T_dte)
        else:
            self.csv_export_T = False

    def setup_pressure_export(self):
        if "Pressure" in self.dico_csv:
            self.csv_export_P = True
            V_p = self.pb.V_rho
            self.p_dte = self.dofs_to_exp(V_p, self.dico_csv.get("Pressure"))
            self.coordinate_data["Pressure"] = self.get_coordinate_data(V_p, self.p_dte)
        else:
            self.csv_export_P = False

    def setup_density_export(self):
        if "rho" in self.dico_csv:
            self.csv_export_rho = True
            self.rho_dte = self.dofs_to_exp(self.pb.V_rho, self.dico_csv.get("rho"))
            self.coordinate_data["rho"] = self.get_coordinate_data(self.pb.V_rho, self.rho_dte)
        else:
            self.csv_export_rho = False

    def initialize_csv_files(self):
        if COMM_WORLD.Get_rank() == 0:
            for field_name, export_info in self.dico_csv.items():
                if field_name == "c":
                    for i in range(len(self.pb.material)):
                        self.create_csv_file(f"Concentration{i}")
                else:
                    self.create_csv_file(field_name)

    def create_csv_file(self, field_name):
        file_path = self.csv_name(field_name)
        self.file_handlers[field_name] = open(file_path, 'w', newline='')
        self.csv_writers[field_name] = writer(self.file_handlers[field_name])
        headers = ["Time", field_name]
        self.csv_writers[field_name].writerow(headers)

    def dofs_to_exp(self, V, keyword):
        if isinstance(keyword, bool) and keyword is True:
            return "all"
        elif isinstance(keyword, list) and keyword[0] == "Boundary":
            return locate_dofs_topological(V, self.pb.facet_tag.dim, self.pb.facet_tag.find(keyword[1]))
        elif isinstance(keyword, ndarray):
            return keyword

    def comp_to_export(self, keyword, component):
        if isinstance(keyword, str):
            return keyword
        elif isinstance(keyword, ndarray):
            vec_dof_to_exp = keyword.copy()
            vec_dof_to_exp *= self.pb.dim
            vec_dof_to_exp += component
            return vec_dof_to_exp

    def csv_export(self, t):
        if not self.dico_csv:
            return
        if self.csv_export_v:
            self.export_field(t, "v", self.pb.U_base[1], self.v_cte, subfield_name = self.v_name_list)
        if self.csv_export_P:
            self.pb.p_func.interpolate(self.pb.p_expr)
            self.export_field(t, "Pressure", self.pb.p_func, self.p_dte)
        if self.csv_export_rho:
            self.export_field(t, "rho", self.pb.rho, self.rho_dte)
            
    def export_field(self, t, field_name, field, dofs_to_export, subfield_name = None):
        if isinstance(subfield_name, list):
            n_sub = len(subfield_name)
            for i in range(n_sub):
                data = self.gather_field_data(field, dofs_to_export[i], size = n_sub, comp = i)
                self.write_field_data(field_name, t, data)
        else:
            data = self.gather_field_data(field, dofs_to_export)
            self.write_field_data(field_name, t, data)        

    def get_coordinate_data(self, V, key):
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

    def export_free_surface(self, t):
        if COMM_WORLD.Get_rank() == 0:
            self.time.append(t)
            self.free_surf_v.append(self.pb.v.x.array[self.free_surf_dof][0])
            row = [t, self.free_surf_v[-1]]
            self.csv_writers["FreeSurf_1D"].writerow(row)
            self.file_handlers["FreeSurf_1D"].flush()

    def close_files(self):
        if COMM_WORLD.Get_rank() == 0:
            for handler in self.file_handlers.values():
                handler.close()
            self.post_process_all_files()

    def post_process_all_files(self):
        for field_name in self.dico_csv.keys():
            if field_name in ["d", "T", "Pressure", "rho", "VonMises"]:
                self.post_process_csv(field_name)
            elif field_name == "U":
                self.post_process_csv(field_name, subfield_name = self.u_name_list)
            elif field_name == "v":
                self.post_process_csv(field_name, subfield_name = self.v_name_list)


    def post_process_csv(self, field_name, subfield_name=None):
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