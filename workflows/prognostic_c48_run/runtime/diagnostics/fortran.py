from typing import Dict, Mapping, Sequence
import dataclasses
import os

import fv3config

from .time import TimeConfig

# keys are output_names, values are tuple of (module_name, field_name)
MODULE_FIELD_NAME_TABLE = {
    "pk": ("dynamics", "pk"),
    "bk": ("dynamics", "bk"),
    "hyam": ("dynamics", "hyam"),
    "hybm": ("dynamics", "hybm"),
    "HGTsfc": ("dynamics", "zsurf"),
    "lon": ("dynamics", "grid_lont"),
    "lat": ("dynamics", "grid_latt"),
    "lonb": ("dynamics", "grid_lon"),
    "latb": ("dynamics", "grid_lat"),
    "area": ("dynamics", "area"),
    "UGRDlowest": ("dynamics", "us"),
    "UGRD850": ("dynamics", "u850"),
    "UGRD500": ("dynamics", "u500"),
    "UGRD200": ("dynamics", "u200"),
    "UGRD50": ("dynamics", "u50"),
    "VGRDlowest": ("dynamics", "vs"),
    "VGRD850": ("dynamics", "v850"),
    "VGRD500": ("dynamics", "v500"),
    "VGRD200": ("dynamics", "v200"),
    "VGRD50": ("dynamics", "v50"),
    "TMP500_300": ("dynamics", "tm"),
    "TMPlowest": ("dynamics", "tb"),
    "TMP850": ("dynamics", "t850"),
    "TMP500": ("dynamics", "t500"),
    "TMP200": ("dynamics", "t200"),
    "w850": ("dynamics", "w850"),
    "w500": ("dynamics", "w500"),
    "w200": ("dynamics", "w200"),
    "w50": ("dynamics", "w50"),
    "VORT850": ("dynamics", "vort850"),
    "VORT500": ("dynamics", "vort500"),
    "VORT200": ("dynamics", "vort200"),
    "h850": ("dynamics", "z850"),
    "h500": ("dynamics", "z500"),
    "h200": ("dynamics", "z200"),
    "RH1000": ("dynamics", "rh1000"),
    "RH925": ("dynamics", "rh925"),
    "RH850": ("dynamics", "rh850"),
    "RH700": ("dynamics", "rh700"),
    "RH500": ("dynamics", "rh500"),
    "q1000": ("dynamics", "q1000"),
    "q925": ("dynamics", "q925"),
    "q850": ("dynamics", "q850"),
    "q700": ("dynamics", "q700"),
    "q500": ("dynamics", "q500"),
    "PRMSL": ("dynamics", "slp"),
    "PRESsfc": ("dynamics", "ps"),
    "PWAT": ("dynamics", "tq"),
    "VIL": ("dynamics", "lw"),
    "iw": ("dynamics", "iw"),
    "kinetic_energy": ("dynamics", "ke"),
    "total_energy": ("dynamics", "te"),
    "ucomp": ("dynamics", "ucomp"),
    "vcomp": ("dynamics", "vcomp"),
    "temp": ("dynamics", "temp"),
    "delp": ("dynamics", "delp"),
    "sphum": ("dynamics", "sphum"),
    "nhpres": ("dynamics", "pfnh"),
    "w": ("dynamics", "w"),
    "delz": ("dynamics", "delz"),
    "ps": ("dynamics", "ps"),
    "reflectivity": ("dynamics", "reflectivity"),
    "liq_wat": ("dynamics", "liq_wat"),
    "ice_wat": ("dynamics", "ice_wat"),
    "rainwat": ("dynamics", "rainwat"),
    "snowwat": ("dynamics", "snowwat"),
    "graupel": ("dynamics", "graupel"),
    "uflx": ("gfs_phys", "dusfci"),
    "vflx": ("gfs_phys", "dvsfci"),
    "CPRATsfc": ("gfs_phys", "cnvprcpb_ave"),
    "PRATEsfc": ("gfs_phys", "totprcpb_ave"),
    "ICEsfc": ("gfs_phys", "toticeb_ave"),
    "SNOWsfc": ("gfs_phys", "totsnwb_ave"),
    "GRAUPELsfc": ("gfs_phys", "totgrpb_ave"),
    "DSWRFsfc": ("gfs_phys", "DSWRF"),
    "USWRFsfc": ("gfs_phys", "USWRF"),
    "DSWRFtoa": ("gfs_phys", "DSWRFtoa"),
    "USWRFtoa": ("gfs_phys", "USWRFtoa"),
    "ULWRFtoa": ("gfs_phys", "ULWRFtoa"),
    "ULWRFsfc": ("gfs_phys", "ULWRF"),
    "DLWRFsfc": ("gfs_phys", "DLWRF"),
    "LHTFLsfc": ("gfs_phys", "lhtfl_ave"),
    "SHTFLsfc": ("gfs_phys", "shtfl_ave"),
    "HPBLsfc": ("gfs_phys", "hpbl"),
    "ICECsfc": ("gfs_sfc", "fice"),
    "SLMSKsfc": ("gfs_sfc", "SLMSKsfc"),
    "SPFH2m": ("gfs_sfc", "q2m"),
    "TMP2m": ("gfs_sfc", "t2m"),
    "TMPsfc": ("gfs_sfc", "tsfc"),
    "DPT2m": ("gfs_phys", "dpt2m"),
    "UGRD10m": ("gfs_phys", "u10m"),
    "VGRD10m": ("gfs_phys", "v10m"),
    "TMAX2m": ("gfs_phys", "tmpmax2m"),
    "MAXWIND10m": ("gfs_phys", "wind10mmax"),
    "SOILM": ("gfs_phys", "soilm"),
    "SOILT1": ("gfs_sfc", "SOILT1"),
    "SOILT2": ("gfs_sfc", "SOILT2"),
    "SOILT3": ("gfs_sfc", "SOILT3"),
    "SOILT4": ("gfs_sfc", "SOILT4"),
    "t_dt_nudge": ("dynamics", "t_dt_nudge"),
    "q_dt_nudge": ("dynamics", "q_dt_nudge"),
    "u_dt_nudge": ("dynamics", "u_dt_nudge"),
    "v_dt_nudge": ("dynamics", "v_dt_nudge"),
    "delp_dt_nudge": ("dynamics", "delp_dt_nudge"),
    "ps_dt_nudge": ("dynamics", "ps_dt_nudge"),
}


@dataclasses.dataclass
class FortranFileConfig:
    """Configurations for Fortran diagnostics defined in diag_table to be converted to zarr

    Attributes:
        name: filename of the diagnostic. Must include .zarr suffix.
        chunks: mapping of dimension names to chunk sizes
        variables: sequence of variable names
        times: time configuration. Only kinds 'interval', 'interval-average' or 'every'
            are allowed.
    """

    name: str
    chunks: Mapping[str, int]
    variables: Sequence[str] = ()
    times: TimeConfig = dataclasses.field(default_factory=lambda: TimeConfig())

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def to_fv3config_diag_file_config(self) -> fv3config.DiagFileConfig:
        if self.times.kind in ["interval", "interval-average"]:
            frequency = self.times.frequency
            frequency_units = "seconds"
        elif self.times.kind == "every":
            frequency = 0
            frequency_units = "seconds"
        else:
            raise NotImplementedError(
                "Fortran diagnostics can only use a times 'kind' that is one of "
                "'interval', 'interval-average' or 'every'."
            )
        reduction_method = (
            "average" if self.times.kind == "interval-average" else "none"
        )
        field_configs = [
            self._field_config_from_variable(variable, reduction_method)
            for variable in self.variables
        ]
        name_without_ext = os.path.splitext(self.name)[0]
        return fv3config.DiagFileConfig(
            name_without_ext, frequency, frequency_units, field_configs
        )

    @staticmethod
    def _field_config_from_variable(
        output_name: str, reduction_method: str
    ) -> fv3config.DiagFieldConfig:
        module_name, field_name = MODULE_FIELD_NAME_TABLE[output_name]
        return fv3config.DiagFieldConfig(
            module_name, field_name, output_name, reduction_method=reduction_method
        )
