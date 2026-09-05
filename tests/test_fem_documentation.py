"""Curated API and user documentation release checks."""
import importlib
import inspect
import io
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((ROOT / 'doc/public_api.json').read_text())


@pytest.mark.parametrize('package', INVENTORY)
def test_curated_public_api_and_signatures(package):
    spec = INVENTORY[package]
    module = importlib.import_module(package)
    assert set(module.__all__) == set(spec['exports'])
    reference = (ROOT / spec['documentation'] / 'API_REFERENCE.rst').read_text(encoding='utf-8')
    for name in spec['exports']:
        assert hasattr(module, name)
        assert f'``{name}``' in reference
    for name, methods in spec['solvers'].items():
        cls = getattr(module, name)
        assert all(p.kind == p.KEYWORD_ONLY for p in inspect.signature(cls).parameters.values())
        for method in methods:
            target = cls if method == '__init__' else getattr(cls, method)
            label = name if method == '__init__' else f'{name}.{method}'
            assert f'``{label}``' in reference
            for parameter in inspect.signature(target).parameters.values():
                if parameter.name not in ('self', 'cls'):
                    assert f'``{parameter.name}``' in reference
    assert not {'assemble_periodic_system_2d', 'linearized_pencil', 'build_node_prolongation',
                'H5ResultData', 'solve_qep_candidates', 'resolve_frequency'} & set(module.__all__)


@pytest.mark.parametrize('package', INVENTORY)
def test_rst_is_valid(package):
    from docutils.core import publish_doctree
    for filename in ('guide.rst', 'API_REFERENCE.rst'):
        path = ROOT / INVENTORY[package]['documentation'] / filename
        messages = io.StringIO()
        publish_doctree(path.read_text(encoding='utf-8'), source_path=str(path),
                        settings_overrides={'warning_stream': messages, 'halt_level': 6,
                                            'report_level': 2, 'syntax_highlight': 'none'})
        assert not messages.getvalue(), messages.getvalue()


def test_release_environment_is_project_named():
    assert (ROOT / 'environment.yml').read_text().startswith('name: cem\n')
    for directory in ('scripts', 'doc', 'solvers', 'libraries', 'examples', 'apps'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in ('.rst', '.ps1', '.yml', '.yaml'):
                assert 'RF_Engineering_env' not in path.read_text(encoding='utf-8'), path
