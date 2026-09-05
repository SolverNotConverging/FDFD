"""Failed writes must preserve the user's previous result archive."""
import h5py
import pytest

from cem_common import PersistenceError
from cem_common.persistence import atomic_h5, write_value


def test_failed_write_preserves_archive_and_removes_temporary_file(tmp_path):
    path=tmp_path/'result.h5'
    with h5py.File(path,'w') as archive:
        archive.create_dataset('important',data=[1,2,3])
    original=path.read_bytes()
    with pytest.raises(PersistenceError):
        with atomic_h5(path) as archive:
            write_value(archive,'unsupported',lambda:None)
    assert path.read_bytes()==original
    assert list(tmp_path.iterdir())==[path]
