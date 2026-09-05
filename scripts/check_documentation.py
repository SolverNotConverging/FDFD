"""Validate maintained RST documents and relative reference links."""
import io
from pathlib import Path
import re
from docutils.core import publish_doctree

ROOT=Path(__file__).resolve().parents[1]


def main():
    errors=[]
    files=[]
    for directory in ('solvers','libraries','apps','docs'):
        files.extend((ROOT/directory).rglob('*.rst'))
    for path in files:
        if any(part in ('build','dist') or part.endswith('.egg-info') for part in path.parts):continue
        source=path.read_text(encoding='utf-8')
        warnings=io.StringIO()
        publish_doctree(source,source_path=str(path),settings_overrides={
            'warning_stream':warnings,'halt_level':6,'report_level':2,'syntax_highlight':'none'})
        if warnings.getvalue():errors.append(warnings.getvalue())
        for link in re.findall(r'`[^`]+? <([^>]+)>`_',source):
            if '://' in link or link.startswith(('mailto:','#')):continue
            if not (path.parent/link.split('#')[0]).exists():errors.append(f'{path}: broken local link {link}')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Validated {len(files)} maintained RST documents and relative links.')


if __name__=='__main__':main()
