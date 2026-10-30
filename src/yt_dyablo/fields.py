"""
Field definitions for Dyablo frontend.
"""
import numpy as np
import unyt

from yt._typing import KnownFieldsT
from yt.fields.field_info_container import FieldInfoContainer
from yt.fields.fluid_fields import setup_gradient_fields

# Unit definitions
rho_units = "code_density"
mom_units = "code_density * code_velocity"
en_units = "code_density * code_velocity**2"
vel_units = "code_velocity"
B_UNITS = "code_magnetic"
PSI_UNITS = "code_magnetic * code_length / code_time"
DIVB_UNITS = "code_magnetic / code_length"

class DyabloFieldInfo(FieldInfoContainer):
    """Field info container for Dyablo code."""

    known_other_fields: KnownFieldsT = (
        ("rho", (rho_units, ["density"], r"\rho")),
        ("rho_vx", (mom_units, ["density_times_velocity_x"], r"\rho v_x")),
        ("rho_vy", (mom_units, ["density_times_velocity_y"], r"\rho v_y")),
        ("rho_vz", (mom_units, ["density_times_velocity_z"], r"\rho v_z")),
        ("e_tot", (en_units, ["total_energy_density"], r"E_\mathrm{tot}")),
        ("metallicity", ("dimensionless", ["metallicity"], None)),
        ("Bx", (B_UNITS, ["magnetic_field_x"], "B_x")),
        ("By", (B_UNITS, ["magnetic_field_y"], "B_y")),
        ("Bz", (B_UNITS, ["magnetic_field_z"], "B_z")),
        ("psi", (PSI_UNITS, ["psi"], r"\psi")),
        ("divB", (DIVB_UNITS, ["divB"], r"\nabla\cdot\mathbf{B}"))
    )

    known_particle_fields: KnownFieldsT = (
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_identity", ("", ["particle_index"], None)),
        ("particle_vx", (vel_units, ["particle_velocity_x"], None)),
        ("particle_vy", (vel_units, ["particle_velocity_y"], None)),
        ("particle_vz", (vel_units, ["particle_velocity_z"], None)),
        ("particle_mass", ("code_mass", [], None)),
        ("particle_birth_time", ("code_time", [], None)),
    )

    def setup_fluid_fields(self):
        """Set up derived fluid fields from conserved quantities."""

        # To compute pressure
        gamma = self.ds.parameters.get("gamma", 5.0 / 3.0)
        available_fields = {name for ftype, name in self.field_list if ftype == "dyablo"}
        has_b = any(f"B{a}" in available_fields for a in "xyz")

        # Create velocity components from momentum
        def _velocity_x(data):
            return data["dyablo", "rho_vx"] / data["dyablo", "rho"]

        def _velocity_y(data):
            return data["dyablo", "rho_vy"] / data["dyablo", "rho"]

        def _velocity_z(data):
            return data["dyablo", "rho_vz"] / data["dyablo", "rho"]

        # Create velocity magnitude
        def _velocity_magnitude(data):
            return (
                data["gas", "velocity_x"] ** 2
                + data["gas", "velocity_y"] ** 2
                + data["gas", "velocity_z"] ** 2
            ) ** 0.5
        
        # Create temperature
        def _temperature_over_mu(data):
            rv = data["gas", "pressure"] / data["gas", "density"]
            return rv * unyt.mp / unyt.kb
        
        def _pressure(data):
            kinetic_energy = 0.5 * (
                data["gas", "velocity_x"] ** 2
                + data["gas", "velocity_y"] ** 2
                + data["gas", "velocity_z"] ** 2
            )
            internal_energy = (
                data["dyablo", "e_tot"] / data["dyablo", "rho"] - kinetic_energy
            )
            return (gamma - 1.0) * data["dyablo", "rho"] * internal_energy

        def _sound_speed(data):
            with np.errstate(invalid="ignore"):
                return np.sqrt(gamma * _pressure(data) / data["dyablo", "rho"])

        def _magnetic_field_magnitude(data):
            return (
                data["dyablo", "Bx"] ** 2
                + data["dyablo", "By"] ** 2
                + data["dyablo", "Bz"] ** 2
            ) ** 0.5

        def _alfven_speed(data):
            return (_magnetic_field_magnitude(data) / data["dyablo", "rho"])**0.5
        
        def _entropy_function(data):
            # K = P / rho^gamma
            with np.errstate(invalid="ignore", divide="ignore"):
                return _pressure(data) / data["dyablo", "rho"] ** gamma

        def _specific_entropy(data):
            # s = c_v ln(K), c_v = 1/(gamma - 1), close to a constant
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.log(_entropy_function(data)) / (gamma - 1.0)

        def _mach_number(data):
            return _velocity_magnitude(data) / _sound_speed(data)
        
        self.add_field(
            ("gas", "pressure"),
            sampling_type="cell",
            function=_pressure,
            units=self.ds.unit_system["pressure"],
        )

        self.add_field(
            ("gas", "velocity_x"),
            sampling_type="cell",
            function=_velocity_x,
            units=self.ds.unit_system["velocity"],
        )

        self.add_field(
            ("gas", "velocity_y"),
            sampling_type="cell",
            function=_velocity_y,
            units=self.ds.unit_system["velocity"],
        )

        self.add_field(
            ("gas", "velocity_z"),
            sampling_type="cell",
            function=_velocity_z,
            units=self.ds.unit_system["velocity"],
        )

        self.add_field(
            ("gas", "temperature_over_mu"),
            sampling_type="cell",
            function=_temperature_over_mu,
            units=self.ds.unit_system["temperature"],
        )

        self.add_field(
            ("gas", "velocity_magnitude"),
            sampling_type="cell",
            function=_velocity_magnitude,
            units=self.ds.unit_system["velocity"],
        )

        self.add_field(
            ("gas", "sound_speed"),
            sampling_type="cell",
            function=_sound_speed,
            units=self.ds.unit_system["velocity"],
        )

        self.add_field(
            ("gas", "entropy_function"),
            sampling_type="cell",
            function=_entropy_function,
            units="",
        )

        self.add_field(
            ("gas", "specific_entropy"),
            sampling_type="cell",
            function=_specific_entropy,
            units="",
        )

        self.add_field(
            ("gas", "mach_number"),
            sampling_type="cell",
            function=_mach_number,
            units="",
        )

        setup_gradient_fields(self, ("gas", "specific_entropy"), "")

    def setup_particle_fields(self, ptype):
        """Set up particle fields."""
        super().setup_particle_fields(ptype)
