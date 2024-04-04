from typing import Sequence, Mapping, Hashable, Any, Optional
import dataclasses

LOOKUP_DATA_PATH = "gs://vcm-ml-intermediate/radiation/lookupdata/lookup.tar.gz"  # noqa
FORCING_DATA_PATH = "gs://vcm-ml-intermediate/radiation/forcing/data.tar.gz"  # noqa


@dataclasses.dataclass()
class GFSPhysicsControlConfig:
    """
    Static configuration for the ported version of the Fortran GFS_physics_control
    structure ('model' in the Fortran radiation code).

    Args:
        levs: Number of model levels.
        nfxr: Number of radiative fluxes calculated.
        ncld: Number of cloud species. In physics namelist.
        ncnd: Number of condensate species. In physics namelist.
        fhswr: Shortwave radiation timestep in seconds. In physics namelist.
        fhlwr: Longwave radiation timestep in seconds. In physics namelist.
        imp_physics: Choice of microphysics scheme: 11 for GFDL microphysics scheme
            (only ported option), 8 for Thompson microphysics scheme, 10 for
            Morrison-Gettelman microphysics scheme
        lgfdlmprad:
        uni_cld:
        effr_in:
        indcld:
        num_p3d:
        npdf3d:
        ncnvcld3d:
        lmfdeep2:
        lmfshal:
        sup:
        kdt:
        do_sfcperts:
        pertalb:
        do_only_clearsky_rad:
        swhtr: Whether to output SW heating rate. In physics namelist.
        lwhtr: Whether to output LW heating rate. In physics namelist.
        lprnt: Verbosity flag.
        lssav:
        levr: Number of model levels for radiation calculations. If not
            present defaults to `levs`. `levr != levs` not implemented.
    """

    levs: int = 79
    nfxr: int = 45
    ncld: int = 5
    ncnd: int = 5
    fhswr: float = 3600.0
    fhlwr: float = 3600.0
    imp_physics: int = 11
    lgfdlmprad: bool = False
    uni_cld: bool = False
    effr_in: bool = False
    indcld: int = -1
    num_p3d: int = 1
    npdf3d: int = 0
    ncnvcld3d: int = 0
    lmfdeep2: bool = True
    lmfshal: bool = True
    sup: float = 1.0
    kdt: int = 1
    do_sfcperts: bool = False
    pertalb: Sequence[Sequence[float]] = dataclasses.field(
        default_factory=lambda: [[-999.0], [-999.0], [-999.0], [-999.0], [-999.0]]
    )
    do_only_clearsky_rad: bool = False
    swhtr: bool = True
    lwhtr: bool = True
    lprnt: bool = False
    lssav: bool = True
    levr: Optional[int] = None

    def __post_init__(self):
        if self.levr is not None and self.levr != self.levs:
            raise ValueError(
                "Setting namelist `levr` different from `npz` not implemented in "
                f"radiation port. Got `npz`={self.levs} and `levr`={self.levr}."
            )
        else:
            self.levr = self.levs

    @classmethod
    def from_namelist(cls, namelist: Mapping[Hashable, Any]):

        PHYSICS_NAMELIST_TO_GFS_CONTROL = {
            "imp_physics": "imp_physics",
            "ncld": "ncld",
            "ncnd": "ncld",
            "fhswr": "fhswr",
            "fhlwr": "fhlwr",
            "swhtr": "swhtr",
            "lwhtr": "lwhtr",
            "levr": "levr",
        }
        CORE_NAMELIST_TO_GFS_CONTROL = {"npz": "levs"}

        return cls(
            **dict(
                **_namelist_to_config_args(
                    namelist["gfs_physics_nml"], PHYSICS_NAMELIST_TO_GFS_CONTROL
                ),
                **_namelist_to_config_args(
                    namelist["fv_core_nml"], CORE_NAMELIST_TO_GFS_CONTROL
                ),
            )
        )


@dataclasses.dataclass
class RadiationConfig:
    """
    A configuration class for the radiation wrapper. These namelist flags and
    other attributes control the wrapper behavior. The documentation here is largely
    cut and pasted from the Fortran radiation routines.
    
    Args:
        iemsflg: Surface emissivity control flag. In physics namelist as 'iems'.
        ioznflg: Ozone data source control flag.
        ictmflg: Data IC time/date control flag.
            yyyy#, external data ic time/date control flag
            -2: same as 0, but superimpose seasonal cycle from climatology data set.
            -1: use user provided external data for the forecast time, no extrapolation.
            0: use data at initial cond time, if not available, use latest, no \
            extrapolation.
            1: use data at the forecast time, if not available, use latest and \
            extrapolation.
            yyyy0: use yyyy data for the forecast time no further data extrapolation.
            yyyy1: use yyyy data for the fcst. if needed, do extrapolation to \
            match the fcst time.
        isolar: Solar constant cntrl. In physics namelist as 'isol'.
            0: use the old fixed solar constant in "physcon"
            10: use the new fixed solar constant in "physcon"
            1: use noaa ann-mean tsi tbl abs-scale with cycle apprx
            2: use noaa ann-mean tsi tbl tim-scale with cycle apprx
            3: use cmip5 ann-mean tsi tbl tim-scale with cycl apprx
            4: use cmip5 mon-mean tsi tbl tim-scale with cycl apprx
        ico2flg: CO2 data source control flag. In physics namelist as 'ico2'.
        iaerflg: Volcanic aerosols. In physics namelist as 'iaer'.
        ialbflg: Surface albedo control flag. In physics namelist as 'ialb'.
        icldflg:
        ivflip: Vertical index direction control flag for radiation calculations.
            0: Top of model to surface
            1: Surface to top of model
        iovrsw: Cloud overlapping control flag for shortwave radiation. In physics
            namelist as 'iovr_sw'.
            0: random overlapping clouds
            1: maximum/random overlapping clouds
            2: maximum overlap cloud (not implemented in port)
            3: decorrelation-length overlap clouds
        iovrlw: Cloud overlapping control flag for longwave radiation. In physics
            namelist as 'iovr_lw'.
            0: random overlapping clouds
            1: maximum/random overlapping clouds
            2: maximum overlap cloud (not implemented in port)
            3: decorrelation-length overlap clouds
        isubcsw: Sub-column cloud approx flag in SW radiation. In physics
            namelist as 'isubc_sw'.
            0: no sub-column cloud treatment, use grid-mean cloud quantities
            1: MCICA sub-column, prescribed random numbers
            2: MCICA sub-column, providing optional seed for random numbers
        isubclw: Sub-column cloud approx flag in LW radiation. In physics
            namelist as 'isubc_lw'.
            0: no sub-column cloud treatment, use grid-mean cloud quantities
            1: MCICA sub-column, prescribed random numbers
            2: MCICA sub-column, providing optional seed for random numbers
        lcrick: Control flag for eliminating CRICK.
        lcnorm: Control flag for in-cloud condensate. In namelist as `ccnorm`.
            False: Grid-mean condensate
            True: Normalize grid-mean condensate by cloud fraction
        lnoprec: Precip effect on radiation flag (ferrier microphysics).
        iswcliq: Optical property for liquid clouds for SW.
        gfs_physics_control: GFSPhysicsControl data class
    """

    iemsflg: int = 1
    ioznflg: int = 7
    ictmflg: int = 1
    isolar: int = 2
    ico2flg: int = 2
    iaerflg: int = 111
    ialbflg: int = 1
    icldflg: int = 1
    ivflip: int = 1
    iovrsw: int = 1
    iovrlw: int = 1
    isubcsw: int = 2
    isubclw: int = 2
    lcrick: bool = False
    lcnorm: bool = False
    lnoprec: bool = False
    iswcliq: int = 1
    gfs_physics_control_config: GFSPhysicsControlConfig = dataclasses.field(
        default_factory=lambda: GFSPhysicsControlConfig()
    )

    @classmethod
    def from_namelist(cls, namelist: Mapping[Hashable, Any]) -> "RadiationConfig":
        """Generate RadiationConfig from fv3gfs namelist to ensure common keys are
        identical. Remaining values from RadiationConfig defaults.
        """

        gfs_physics_control_config = GFSPhysicsControlConfig.from_namelist(namelist)

        PHYSICS_NAMELIST_TO_RAD_CONFIG = {
            "iems": "iemsflg",
            "isol": "isolar",
            "ico2": "ico2flg",
            "iaer": "iaerflg",
            "ialb": "ialbflg",
            "iovr_sw": "iovrsw",
            "iovr_lw": "iovrlw",
            "isubc_sw": "isubcsw",
            "isubc_lw": "isubclw",
            "ccnorm": "lcnorm",
        }

        return cls(
            **dict(
                **_namelist_to_config_args(
                    namelist["gfs_physics_nml"], PHYSICS_NAMELIST_TO_RAD_CONFIG
                ),
                gfs_physics_control_config=gfs_physics_control_config,
            )
        )


def _namelist_to_config_args(
    namelist: Mapping[Hashable, Any], arg_mapping: Mapping[str, str]
) -> Mapping[str, Any]:
    config_args = {}
    for namelist_entry, config_arg in arg_mapping.items():
        if namelist_entry in namelist:
            config_args[config_arg] = namelist[namelist_entry]
    return config_args
