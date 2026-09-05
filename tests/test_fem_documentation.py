"""Public API coverage, argument tables, and the fixed-mesh example contract."""

import ast
import importlib
import inspect
import io
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('Electrostatic_Solver', 'FEM_Mode_Solver', 'FEM_Periodic_Solver', 'WaveFEM', 'TransmissionLineCalculator')


@pytest.mark.parametrize('package', PACKAGES)
def test_two_documents_and_valid_rst_argument_tables(package):
    docutils = pytest.importorskip('docutils.core')
    nodes = pytest.importorskip('docutils.nodes')
    directory = ROOT / package
    assert {p.name for p in directory.glob('*.rst')} == {'README.rst', 'API_REFERENCE.rst'}
    assert 'API_REFERENCE.rst' in (directory / 'README.rst').read_text(encoding='utf-8')
    for filename in ('README.rst', 'API_REFERENCE.rst'):
        path = directory / filename
        messages = io.StringIO()
        tree = docutils.publish_doctree(path.read_text(encoding='utf-8'), source_path=str(path),
                                        settings_overrides={'warning_stream': messages, 'halt_level': 6,
                                                            'report_level': 2, 'syntax_highlight': 'none'})
        assert not messages.getvalue(), messages.getvalue()
        if filename == 'API_REFERENCE.rst':
            tables = [table for table in tree.findall(nodes.table)
                      if 'Required / optional' in table.astext()]
            assert tables
            for table in tables:
                for row in table.findall(nodes.row):
                    assert len(list(row.findall(nodes.entry))) == 4


@pytest.mark.parametrize('package,directory', [
    ('Electrostatic_Solver', 'Electrostatic_Solver'), ('FEM_Mode_Solver', 'FEM_Mode_Solver'),
    ('FEM_Periodic_Solver', 'FEM_Periodic_Solver'), ('wavefem', 'WaveFEM'),
])
def test_reference_covers_current_public_api(package, directory):
    """Validate the maintained references directly, without documentation generators."""
    module = importlib.import_module(package)
    reference = (ROOT / directory / 'API_REFERENCE.rst').read_text(encoding='utf-8')
    entries = {}
    current_class = None
    for name, section in re.findall(r'^``([^`\n]+)``\n[~^]+\n(.*?)(?=^``[^`\n]+``\n[~^]+\n|\Z)',
                                    reference, re.MULTILINE | re.DOTALL):
        entries[name] = section
        if not name.startswith(f'{package}.'):
            entries[f'{package}.{name}'] = section
        if '.' not in name:
            if inspect.isclass(getattr(module, name, None)):
                current_class = name
            elif current_class:
                entries[f'{package}.{current_class}.{name}'] = section
    modules = [module]
    for path in Path(module.__file__).parent.glob('*.py'):
        if path.stem.startswith('_'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
               and target.id == '__all__' for target in node.targets) for node in tree.body):
            modules.append(importlib.import_module(f'{package}.{path.stem}'))

    def check(name, obj):
        # Re-exports may be documented under the package's shorter public name.
        candidates = (name, f'{package}.{name.split(".", 2)[-1]}')
        section = next((entries[key] for key in candidates if key in entries), None)
        assert section is not None, f'Missing API entry: {name}'
        assert 'Required / optional' in section, f'Missing input table: {name}'
        if isinstance(obj, property):
            obj = obj.fget
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            return
        for argument, parameter in signature.parameters.items():
            # Forwarding wrappers document the expanded backend options.
            if (argument not in ('self', 'cls') and parameter.kind not in
                    (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)):
                assert f'``{argument}``' in section, f'Missing argument: {name}.{argument}'

    seen = set()
    for exported_module in modules:
        for name in exported_module.__all__:
            obj = getattr(exported_module, name)
            if id(obj) in seen or not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            seen.add(id(obj))
            qualified = f'{exported_module.__name__}.{name}'
            check(qualified, obj)
            if inspect.isclass(obj) and not issubclass(obj, BaseException):
                for member, value in inspect.getmembers(obj):
                    if not member.startswith('_') and (inspect.isfunction(value)
                                                       or inspect.ismethod(value)
                                                       or isinstance(value, property)):
                        check(f'{qualified}.{member}', value)


@pytest.mark.parametrize('package', PACKAGES[:4])
def test_example_solves_explicitly_disable_refinement(package):
    for path in (ROOT / package / 'examples').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if name in ('solve', 'run', 'solve_modes', 'SolverOptions'):
                control = next((kw.value for kw in node.keywords if kw.arg == 'max_refinements'), None)
                assert isinstance(control, ast.Constant) and control.value == 0, (path, node.lineno)
            if name == 'sweep_frequencies':
                options = next(kw.value for kw in node.keywords if kw.arg == 'mode_options')
                assert isinstance(options, ast.Dict)
                assert any(isinstance(k, ast.Constant) and k.value == 'max_refinements'
                           and isinstance(v, ast.Constant) and v.value == 0
                           for k, v in zip(options.keys, options.values))


def test_runnable_tutorial_blocks_disable_refinement():
    docutils = pytest.importorskip('docutils.core')
    nodes = pytest.importorskip('docutils.nodes')
    for package in PACKAGES[:4]:
        for filename in ('README.rst', 'API_REFERENCE.rst'):
            path = ROOT / package / filename
            tree = docutils.publish_doctree(path.read_text(encoding='utf-8'),
                                            settings_overrides={'syntax_highlight': 'none'})
            for block in tree.findall(nodes.literal_block):
                if 'python' not in block.get('classes', []):
                    continue
                try:
                    code = ast.parse(block.astext())
                except SyntaxError:
                    # API signatures use annotated call notation, not runnable code.
                    continue
                for node in ast.walk(code):
                    if not isinstance(node, ast.Call):
                        continue
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                    if name in ('solve', 'run', 'solve_modes', 'SolverOptions'):
                        control = next((kw.value for kw in node.keywords if kw.arg == 'max_refinements'), None)
                        assert isinstance(control, ast.Constant) and control.value == 0, (path, block.astext())


def test_native_reference_covers_public_types_functions_and_members():
    directory = ROOT / 'TransmissionLineCalculator'
    reference = (directory / 'API_REFERENCE.rst').read_text(encoding='utf-8')
    header = (directory / 'native/model.hpp').read_text(encoding='utf-8')
    assert all(f'tl::{name}' in reference for name in re.findall(r'(?:struct|enum class) (\w+)', header))
    for line in header.splitlines():
        if line.startswith('    ') and line.rstrip().endswith(';'):
            match = re.search(r'\s(\w+)(?:\{.*\})?;$', line.strip())
            if match:
                assert f'``{match[1]}``' in reference, match[1]
    assert 'tl::defaultParameters' in reference and 'tl::solve' in reference
    example = (directory / 'examples/line_comparison.cpp').read_text(encoding='utf-8')
    assert 'parameters.maxRefinements = 0;' in example
