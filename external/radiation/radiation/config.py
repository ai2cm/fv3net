from typing import Sequence, Mapping, Hashable, Any
import dataclasses

LOOKUP_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz"  # noqa: E501
FORCING_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/data.tar.gz"  # noqa: 501


@dataclasses.dataclass
class RadiationConfig:
    """A configuration class for the radiation wrapper. These namelist flags and
    other attributes control the wrapper behavior.
    
    Args:
        imp_physics: Surface emissivity control flag. In physics namelist.
        iemsflg: In physics namelist as 'iems'.
        ioznflg: Ozone data source control flag.
        ictmflg: Data IC time/date control flag.
        isolar: Solar constant control flag. In physics namelist as 'isol'.
        ico2flg: CO2 data source control flag. In physics namelist as 'ico2'.
        iaerflg: Volcanic aerosols. In physics namelist as 'iaer'.
        ialbflg: Surface albedo control flag. In physics namelist as 'ialb'.
        icldflg
        ivflip: Vertical index direction control flag.
        iovrsw: Cloud overlapping control flag for SW.
        iovrlw: Cloud overlapping control flag for LW.
        isubcsw: Sub-column cloud approx flag in SW radiation. In physics
            namelist as 'isubc_sw'.
        isubclw: Sub-column cloud approx flag in LW radiation. In physics
            namelist as 'isubc_lw'.
        lcrick: Control flag for eliminating CRICK.
        lcnorm: Control flag for in-cld condensate.
        lnoprec: Precip effect on radiation flag (ferrier microphysics).
        iswcliq: Optical property for liquid clouds for SW.
        fhswr: Shortwave radiation timestep in seconds. In physics namelist.
        fhlwr: Longwave radiation timestep in seconds. In physics namelist.
        lsswr
        lslwr
        nfxr
        ncld: In physics namelist.
        ncnd: In physics namelist.
        lgfdlmprad
        uni_cld
        effr_in
        indcld
        num_p3d
        npdf3d
        ncnvcld3d
        lmfdeep2
        lmfshal
        sup
        kdt
        do_sfcperts
        pertalb
        do_only_clearsky_rad
        swhtr: In physics namelist.
        solcon
        lprnt
        lwhtr: In physics namelist.
        lssav
    """

    imp_physics: int = 11
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
    fhswr: float = 3600.0
    fhlwr: float = 3600.0
    lsswr: bool = True
    lslwr: bool = True
    nfxr: int = 45
    ncld: int = 5
    ncnd: int = 5
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
    solcon: float = 1320.8872136343873
    lprnt: bool = False
    lwhtr: bool = True
    lssav: bool = True

    @classmethod
    def from_physics_namelist(
        cls, physics_namelist: Mapping[Hashable, Any]
    ) -> "RadiationConfig":
        """Generate RadiationConfig from fv3gfs physics namelist to ensure common keys are
        identical. Remaining values from RadiationConfig defaults.
        """

        return cls(
            imp_physics=physics_namelist["imp_physics"],
            iemsflg=physics_namelist["iems"],
            isolar=physics_namelist["isol"],
            ico2flg=physics_namelist["ico2"],
            iaerflg=physics_namelist["iaer"],
            ialbflg=physics_namelist["ialb"],
            isubcsw=physics_namelist["isubc_sw"],
            isubclw=physics_namelist["isubc_lw"],
            ncld=physics_namelist["ncld"],
            ncnd=physics_namelist["ncld"],
            fhswr=physics_namelist["fhswr"],
            fhlwr=physics_namelist["fhlwr"],
            swhtr=physics_namelist["swhtr"],
            lwhtr=physics_namelist["lwhtr"],
        )
