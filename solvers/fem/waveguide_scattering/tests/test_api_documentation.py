"""The user reference exposes workflows, not internal reader/assembly records."""
from pathlib import Path
import fem_waveguide_scattering as package


def test_reference_covers_selected_user_api():
    reference = (Path(__file__).parents[1] / 'API_REFERENCE.rst').read_text(encoding='utf-8')
    for name in package.__all__:
        assert f'``{name}``' in reference
    assert 'H5ResultData' not in reference
    assert '2.5D full-vector' in reference
    assert 'exp(+i*omega*t)' in reference
