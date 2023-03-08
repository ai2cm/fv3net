class DeprecationError(Exception):
    def __init__(
        self, message,
    ):
        message = (
            f"{message} \n Last working commit for 1D reservoir code is "
            "5f7d280e7654a345c5ba8c18c61367ffadfa6207"
        )
        super().__init__(message)


deprecated_names = [
    "HybridReservoirComputingModel",
    "DomainPredictor",
    "ReservoirOnlyDomainPredictor",
    "HybridDomainPredictor",
    "ImperfectModel",
    "Reservoir1DTrainingConfig",
    "SubdomainConfig",
]


def __getattr__(name):
    if name in deprecated_names:
        raise DeprecationError(f"module {__name__!r} has no attribute {name!r}")
