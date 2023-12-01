import pytest
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Var,
    Param,
    value,
    assert_optimal_termination,
    units as pyunits,
)
import re
from pyomo.network import Port
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
import watertap_contrib.reflo.analysis.net_metering.PV_RO_surrogate as PV_RO

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap_contrib.reflo.costing import REFLOCosting

from idaes.core.util.testing import initialization_tester
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)
import idaes.logger as idaeslog

# Get default solver for testing
solver = get_solver()
solver.options['max_iter'] = 400

from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import idaes.logger as idaeslog

class TestPVRO:
    @pytest.fixture(scope="class")
    def system_frame(self):
        m = PV_RO.system_build(0.0438, 30, 0.5)
        return m
    
    @pytest.mark.component
    def test_scenario_1_solve(self, system_frame):
        '''This test checks the aggregate flow of electricity and the total electricity 
        cost of electricity for a system where PV electricity generation and demand are equal.'''
        m = system_frame
        
        # This manually sets the PV electricity generation equal to the treatment electricity demand
        m.fs.energy.pv_elec_constraint = Constraint(
        expr=m.fs.energy.pv.electricity
        == -1.0000001*m.fs.treatment.costing.aggregate_flow_electricity
        )

        results = PV_RO.solve(m, tee=True)
        assert_optimal_termination(results)

        # Double check that the RO system solved as expected for comparison
        assert pytest.approx(1410880,rel=1e-2) == value(m.fs.treatment.costing.total_capital_cost)
        assert pytest.approx(1000,rel=1e-2) == value(m.fs.treatment.ro.area)
        assert pytest.approx(707.9,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.inlet.pressure[0], to_units=pyunits.psi))
        assert pytest.approx(0.0438,rel=1e-2) == value(m.fs.treatment.ro.feed_side.properties_in[0].flow_vol)
        assert pytest.approx(30530,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg/pyunits.L))
        assert pytest.approx(0.151,rel=1e-2) == value(m.fs.treatment.ro.recovery_vol_phase[0, "Liq"])
        
        # # Now check the net electricity flow and cost
        assert pytest.approx(0,abs=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity)
        assert pytest.approx(0,abs=1e-2) == value(m.fs.sys_costing.total_electric_operating_cost)

    @pytest.mark.unit
    def test_scenario_2_solution(self, system_frame):
        '''This test checks the aggregate flow of electricity and the total electricity 
        cost of electricity for a system where electricity demand exceeds the PV generation.'''
        m = system_frame

        # This manually sets the PV electricity generation to 50% of the treatment electricity demand
        m.fs.energy.pv_elec_constraint = Constraint(
            expr=m.fs.energy.pv.electricity
            == -0.5*m.fs.treatment.costing.aggregate_flow_electricity
        )

        results = PV_RO.solve(m, tee=True)
        assert_optimal_termination(results)

        # Double check that the RO system solved as expected for comparison
        assert pytest.approx(1410880,rel=1e-2) == value(m.fs.treatment.costing.total_capital_cost)
        assert pytest.approx(1000,rel=1e-2) == value(m.fs.treatment.ro.area)
        assert pytest.approx(707.9,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.inlet.pressure[0], to_units=pyunits.psi))
        assert pytest.approx(0.0438,rel=1e-2) == value(m.fs.treatment.ro.feed_side.properties_in[0].flow_vol)
        assert pytest.approx(30530,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg/pyunits.L))
        assert pytest.approx(0.151,rel=1e-2) == value(m.fs.treatment.ro.recovery_vol_phase[0, "Liq"])

        assert pytest.approx(101.98,rel=1e-2) == value(m.fs.treatment.costing.aggregate_flow_electricity)
        assert pytest.approx(-50.99,rel=1e-2) == value(m.fs.energy.costing.aggregate_flow_electricity)

        assert pytest.approx(50.99,rel=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity_purchased)
        assert pytest.approx(0, abs=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity_sold)

        assert pytest.approx(50.99,rel=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity)
        assert pytest.approx(31290.64,rel=1e-2) == value(m.fs.sys_costing.total_electric_operating_cost)


    @pytest.mark.unit
    def test_scenario_3_solution(self, system_frame):
        '''This test checks the aggregate flow of electricity and the total electricity 
        cost of electricity for a system where the PV generation exceeds the electricity demand.'''
        m = system_frame

        # This manually sets the PV electricity generation to 150% of the treatment electricity demand
        m.fs.energy.pv_elec_constraint = Constraint(
        expr=m.fs.energy.pv.electricity
        == -1.5*m.fs.treatment.costing.aggregate_flow_electricity
        )

        results = PV_RO.solve(m)

        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)

        assert_optimal_termination(results)

        # Double check that the RO system solved as expected for comparison
        assert pytest.approx(1410880,rel=1e-2) == value(m.fs.treatment.costing.total_capital_cost)
        assert pytest.approx(1000,rel=1e-2) == value(m.fs.treatment.ro.area)
        assert pytest.approx(707.9,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.inlet.pressure[0], to_units=pyunits.psi))
        assert pytest.approx(0.0438,rel=1e-2) == value(m.fs.treatment.ro.feed_side.properties_in[0].flow_vol)
        assert pytest.approx(30530,rel=1e-2) == value(pyunits.convert(m.fs.treatment.ro.feed_side.properties_in[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg/pyunits.L))
        assert pytest.approx(0.151,rel=1e-2) == value(m.fs.treatment.ro.recovery_vol_phase[0, "Liq"])

        assert pytest.approx(101.98,rel=1e-2) == value(m.fs.treatment.costing.aggregate_flow_electricity)
        assert pytest.approx(-152.98,rel=1e-2) == value(m.fs.energy.costing.aggregate_flow_electricity)

        assert pytest.approx(0, abs=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity_purchased)
        assert pytest.approx(50.99,rel=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity_sold)

        assert pytest.approx(-50.99,rel=1e-2) == value(m.fs.sys_costing.aggregate_flow_electricity)
        assert pytest.approx(-22350.01,rel=1e-2) == value(m.fs.sys_costing.total_electric_operating_cost)
