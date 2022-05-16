import doctest
from fv3fit.keras._models.shared import spectral_normalization

def test_spectral_normalization():
    doctest.testmod(spectral_normalization, raise_on_error=True)
