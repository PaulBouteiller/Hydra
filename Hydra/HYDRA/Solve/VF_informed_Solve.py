"""
Created on Wed Jun 25 11:15:09 2025

@author: bouteillerp
"""
from dolfinx.fem import functionspace, Function
from HYDRA import Solve as HydraSolve
from jaxfluids.simulation_manager import SimulationManager
from jaxfluids.data_types.ml_buffers import CallablesSetup, ParametersSetup
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.information import (WallClockTimes)
import numpy as np
from HYDRA.utils.holo_utils import (map_indices, create_custom_quadrature_element, L2_proj, create_vector_mapping,
                                    create_averaging_mapper, project_cell_to_facet, reordering_mapper, project_cell_to_facet_vector)

class JAXFLUIDS_HYDRA_SOLVE:
    def __init__(self, HDGProblem, input_manager, dictionnaire_solve, **kwargs):
        self.hydra_solver = HydraSolve(HDGProblem, dictionnaire_solve, **kwargs)
        self.hydra_dim = self.hydra_solver.pb.tdim 
        self.jaxfluids_solver = SimulationManager(input_manager)
        
        ratio = 3
        quad_element = create_custom_quadrature_element(ratio, equal_weight= False)
        vec_quad_element = create_custom_quadrature_element(ratio, equal_weight= False, el_type = "Vector")
        
        V_rho_quad = functionspace(HDGProblem.mesh, quad_element)
        V_rhou_quad = functionspace(HDGProblem.mesh, vec_quad_element)
        
        hydra_coords = V_rho_quad.tabulate_dof_coordinates()
        jax_coords = self.set_coordinate_array(self.jaxfluids_solver.domain_information.get_local_cell_centers())
        
        #Mapping from Jax to Hydra
        self.jf_to_hydra_mapper, self.multi_index = map_indices(jax_coords, hydra_coords, self.jaxfluids_solver.domain_information.inactive_axes)
        self.jf_to_hydra_vector_mapper = create_vector_mapping(self.jf_to_hydra_mapper, self.multi_index, dim = self.hydra_dim)
        if self.multi_index.shape == (0,):
            raise ValueError("JAXFluids mesh and Quadrature points are not matching")
        
        # Mapping from cell dof to facet dof        
        self.a_cell_mapper = create_averaging_mapper(HDGProblem.V_rho.tabulate_dof_coordinates(), tol=1e-8)
        self.a_facet_mapper = create_averaging_mapper(HDGProblem.V_rhobar.tabulate_dof_coordinates(), tol=1e-8)
        self.reordering_unique_coords = reordering_mapper(self.a_cell_mapper['unique_coords'], 
                                                          self.a_facet_mapper['unique_coords'], 
                                                          rtol=1e-6, atol=1e-6)
        self.reordering_unique_coords_vector = np.concatenate([self.reordering_unique_coords + d * len(self.reordering_unique_coords) for d in range(self.hydra_dim)])

        #Projector deffinition
        self.U_quad = [Function(V_rho_quad), Function(V_rhou_quad), Function(V_rho_quad)]
        self.proj = [L2_proj(u, u_quad) for u, u_quad in 
                     zip([HDGProblem.rho, HDGProblem.rhov, HDGProblem.rhoe], self.U_quad)]

    def set_coordinate_array(self, cell_centers):
            x_coords = cell_centers[0].squeeze()
            y_coords = cell_centers[1].squeeze() 
            z_coords = cell_centers[2].squeeze()
            if y_coords.size == 1:
                return np.column_stack([
                    x_coords, 
                    np.full_like(x_coords, y_coords.item()), 
                    np.zeros_like(x_coords)
                ])
            else:
                # x_coords, y_coords, z_coords = sim_manager.domain_information.get_local_cell_centers()
                X, Y, Z = np.meshgrid(x_coords.flatten(), y_coords.flatten(), z_coords.flatten(), indexing='ij')
                # coords_fenics_format
                return np.column_stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())])
        
    def simulate(
            self,
            jxf_buffers: JaxFluidsBuffers,
            ml_parameters: ParametersSetup = ParametersSetup(),
            ml_callables: CallablesSetup = CallablesSetup(),
        ) -> int:
        """Performs a conventional CFD simulation.

        :param jxf_buffers: _description_
        :type jxf_buffers: JaxFluidsBuffers
        :param ml_parameters: _description_, defaults to ParametersSetup()
        :type ml_parameters: ParametersSetup, optional
        :param ml_callables: _description_, defaults to CallablesSetup()
        :type ml_callables: CallablesSetup, optional
        :return: _description_
        :rtype: int
        """

        self.jaxfluids_solver.initialize(jxf_buffers)
        return_value = self.advance(
            jxf_buffers,
            ml_parameters,
            ml_callables
        )

        return return_value
    
    def advance(
            self,
            jxf_buffers: JaxFluidsBuffers,
            ml_parameters: ParametersSetup,
            ml_callables: CallablesSetup,
        ) -> bool:
        """Advances the initial buffers in time.
        """

        # START LOOP
        start_loop = self.jaxfluids_solver.synchronize_and_clock(
            jxf_buffers.simulation_buffers.material_fields.primitives)

        time_control_variables = jxf_buffers.time_control_variables
        physical_simulation_time = time_control_variables.physical_simulation_time
        simulation_step = time_control_variables.simulation_step

        wall_clock_times = WallClockTimes()


        compteur = 0

        while (
            physical_simulation_time < time_control_variables.end_time 
            and simulation_step < time_control_variables.end_step
        ):

            control_flow_params = self.jaxfluids_solver.compute_control_flow_params(
                time_control_variables, jxf_buffers.step_information)

            # PERFORM INTEGRATION STEP
            jxf_buffers, callback_dict_step = self.jaxfluids_solver.do_integration_step(
                jxf_buffers,
                control_flow_params,
                ml_parameters,
                ml_callables
            )

            nhx, nhy, nhz = self.jaxfluids_solver.domain_information.domain_slices_conservatives
            conservatives = jxf_buffers.simulation_buffers.material_fields.conservatives
            

            if np.any(np.isclose(physical_simulation_time, self.hydra_solver.load_steps, atol=5e-5)):
                
                rho = np.array(conservatives[0, nhx, nhy, nhz].squeeze())
                rhoux = np.array(conservatives[1, nhx, nhy, nhz].squeeze())
                rhouy = np.array(conservatives[2, nhx, nhy, nhz].squeeze())
                rhouz = np.array(conservatives[3, nhx, nhy, nhz].squeeze())
                rhoue = np.array(conservatives[4, nhx, nhy, nhz].squeeze())
    
                self.U_quad[0].x.array[self.jf_to_hydra_mapper] = rho[self.multi_index]
                self.U_quad[1].x.array[self.jf_to_hydra_vector_mapper["vx"]]= rhoux[self.multi_index]
                self.U_quad[1].x.array[self.jf_to_hydra_vector_mapper["vy"]] = rhouy[self.multi_index]
                self.U_quad[2].x.array[self.jf_to_hydra_mapper] = rhoue[self.multi_index]
                
                #Projection de la solution Jax et mise à jour des variables de trace
                for x in self.proj: x.solve()
                self.hydra_solver.pb.rhobar.x.array[:] = project_cell_to_facet(self.hydra_solver.pb.rho.x.array, self.a_cell_mapper, self.a_facet_mapper, self.reordering_unique_coords)
                self.hydra_solver.pb.rhovbar.x.array[:] = project_cell_to_facet_vector(self.hydra_solver.pb.rhov.x.array, self.a_cell_mapper, self.a_facet_mapper, self.reordering_unique_coords_vector)
                self.hydra_solver.pb.rhoebar.x.array[:] = project_cell_to_facet(self.hydra_solver.pb.rhoe.x.array, self.a_cell_mapper, self.a_facet_mapper, self.reordering_unique_coords)
                self.hydra_solver._inloop_solve(physical_simulation_time, 1)
            else:
                compteur+=1
            
            # NOTE UNPACK JAX FLUIDS BUFFERS
            # print("Le compteur est à", compteur)
            simulation_buffers = jxf_buffers.simulation_buffers
            time_control_variables = jxf_buffers.time_control_variables

            # UNPACK FOR WHILE LOOP
            physical_simulation_time = time_control_variables.physical_simulation_time
            simulation_step = time_control_variables.simulation_step

        # UNPACK JAX FLUIDS BUFFERS
        simulation_buffers = jxf_buffers.simulation_buffers
        time_control_variables = jxf_buffers.time_control_variables
        forcing_parameters = jxf_buffers.forcing_parameters
        step_information = jxf_buffers.step_information

        # FINAL OUTPUT
        self.jaxfluids_solver.output_writer.write_output(
            simulation_buffers,
            time_control_variables,
            wall_clock_times,
            forcing_parameters,
            force_output=True,
            simulation_finish=True,
            flow_statistics=step_information.statistics
        )

        # LOG SIMULATION FINISH
        end_loop = self.jaxfluids_solver.synchronize_and_clock(
            simulation_buffers.material_fields.primitives)
        self.jaxfluids_solver.logger.log_sim_finish(end_loop - start_loop)
        
        
        #Hydra_Solve_finish
        # Finalisation
        self.hydra_solver.export.csv.close_files()

        return bool(physical_simulation_time >= time_control_variables.end_time)