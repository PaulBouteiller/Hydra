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
from HYDRA.utils.holo_utils import map_indices, create_custom_quadrature_element, init_L2_projection_on_V
import jax.numpy as jnp


class JAXFLUIDS_HYDRA_SOLVE:
    def __init__(self, HDGProblem, input_manager, dictionnaire_solve, **kwargs):
        self.hydra_solver = HydraSolve(HDGProblem, dictionnaire_solve, **kwargs)
        self.jaxfluids_solver = SimulationManager(input_manager)
        
        
        quad_element = create_custom_quadrature_element(HDGProblem.deg, equal_weight= True)
        vec_quad_element = create_custom_quadrature_element(HDGProblem.deg, equal_weight= True, el_type = "Vector")
        V_rho_quad = functionspace(HDGProblem.mesh, quad_element)
        V_rhou_quad = functionspace(HDGProblem.mesh, vec_quad_element)
        
        hydra_coords = V_rho_quad.tabulate_dof_coordinates()
        jax_coords = self.set_coordinate_array(input_manager)
        self.hydra_to_jaxfluids_array_mapper = map_indices(hydra_coords, jax_coords)
        self.jaxfluids_to_hydra_array_mapper = map_indices(hydra_coords, jax_coords)
        self.hydra_dim = self.hydra_solver.pb.dim 
        self.hydra_to_jaxfluids_array_vect_mapper = self.hydra_dim * self.hydra_to_jaxfluids_array_mapper
        self.jaxfluids_to_hydra_array_vect_mapper = self.hydra_dim * self.jaxfluids_to_hydra_array_mapper
        
        
        self.numpy_arange = np.arange(len(self.hydra_to_jaxfluids_array_mapper)) 
        self.rho_quad = Function(V_rho_quad)
        self.rhov_quad = Function(V_rhou_quad)
        self.rhoe_quad = Function(V_rho_quad)
        self.U_quad = []
        self.init_projection = [init_L2_projection_on_V(u, u_quad) 
                                for u, u_quad in zip([HDGProblem.rho, HDGProblem.rhov, HDGProblem.rhoe],
                                                     [self.rho_quad, self.rhov_quad, self.rhoe_quad])]
        
    def set_coordinate_array(self, input_manager):
            cell_centers = self.jaxfluids_solver.domain_information.get_local_cell_centers()
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
                return np.column_stack([
                    x_coords.flatten(), 
                    y_coords.flatten(), 
                    np.zeros(x_coords.size)
                ])
        
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

        while (
            physical_simulation_time < time_control_variables.end_time 
            and simulation_step < time_control_variables.end_step
        ):

            start_step = self.jaxfluids_solver.synchronize_and_clock(
                jxf_buffers.simulation_buffers.material_fields.primitives)

            control_flow_params = self.jaxfluids_solver.compute_control_flow_params(
                time_control_variables, jxf_buffers.step_information)

            # PERFORM INTEGRATION STEP
            jxf_buffers, callback_dict_step = self.jaxfluids_solver.do_integration_step(
                jxf_buffers,
                control_flow_params,
                ml_parameters,
                ml_callables
            )
            is_hydra_time = jnp.abs(self.hydra_solver.t - physical_simulation_time) < 1e-6
            if is_hydra_time:
                self.hydra_solver.t+=self.hydra_solver.dt
                nhx, nhy, nhz = self.jaxfluids_solver.domain_information.domain_slices_conservatives
                conservatives = jxf_buffers.simulation_buffers.material_fields.conservatives
                
                # Extraction et mise en forme
                rho = np.array(conservatives[0, nhx, nhy, nhz].squeeze())
                rhoux = np.array(conservatives[1, nhx, nhy, nhz].squeeze())
                rhouy = np.array(conservatives[2, nhx, nhy, nhz].squeeze())
                rhouz = np.array(conservatives[3, nhx, nhy, nhz].squeeze())
                rhoue = np.array(conservatives[4, nhx, nhy, nhz].squeeze())
                self.rho_quad.x.array[self.numpy_arange] = np.array(conservatives[0, nhx, nhy, nhz].squeeze())[self.jaxfluids_to_hydra_array_mapper]
                print(self.rho_quad.x.array)
                self.init_projection[0].solve()
                print(self.hydra_solver.pb.rho.x.array)
                
                
                print("coucou")
                # self.hydra_solver.solve()
                # self.update_fields()
            
            
            # NOTE UNPACK JAX FLUIDS BUFFERS
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

        return bool(physical_simulation_time >= time_control_variables.end_time)