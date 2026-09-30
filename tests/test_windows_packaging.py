"""Provenance checks for the MSVC/vcpkg release packager."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "windows_packager", Path(__file__).resolve().parents[1] / "scripts/package_native_windows.py"
)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_status_includes_feature_dependencies(tmp_path):
    status = tmp_path / "installed/vcpkg/status"
    status.parent.mkdir(parents=True)
    status.write_text(
        "Package: qtbase\nFeature: widgets\nArchitecture: x64-windows\n"
        "Depends: freetype\nStatus: install ok installed\n\n"
        "Package: qtbase\nVersion: 6.11.0\nArchitecture: x64-windows\n"
        "Depends: zlib\nStatus: install ok installed\n\n"
        "Package: other\nVersion: 1\nArchitecture: x86-windows\n"
        "Status: install ok installed\n"
    )
    packages = packager.package_database(tmp_path)
    assert set(packages) == {"qtbase"}
    assert packages["qtbase"]["Version"] == "6.11.0"
    assert set(packages["qtbase"]["Depends"].split(", ")) == {"freetype", "zlib"}


def test_recipe_must_match_installed_binary(tmp_path):
    recipe = tmp_path / "portfile.cmake"
    recipe.write_bytes(b"original recipe")
    sbom = {"files": [{"SPDXID": "SPDXRef-port-file-0", "fileName": "./portfile.cmake",
                       "checksums": [{"algorithm": "SHA256", "checksumValue":
                                      hashlib.sha256(recipe.read_bytes()).hexdigest()}]}]}
    packager.verify_recipe(tmp_path, sbom)
    recipe.write_bytes(b"new incompatible recipe")
    with pytest.raises(RuntimeError, match="Recipe changed"):
        packager.verify_recipe(tmp_path, sbom)


def test_finish_preserves_only_hash_verified_sources(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "output", vcpkg_root=tmp_path / "vcpkg")
    bundle = args.output / packager.BUNDLE_NAME
    licenses = bundle / "licenses/example"
    licenses.mkdir(parents=True)
    downloads = args.vcpkg_root / "downloads"
    downloads.mkdir(parents=True)
    source = downloads / "source.tar.gz"
    source.write_bytes(b"exact upstream source archive")
    checksum = hashlib.sha512(source.read_bytes()).hexdigest()
    (licenses / "vcpkg.spdx.json").write_text(json.dumps({"packages": [{
        "SPDXID": "SPDXRef-resource-0", "name": "example", "downloadLocation": "https://example.com/source",
        "checksums": [{"algorithm": "SHA512", "checksumValue": checksum}]
    }]}))
    (bundle / "build-manifest.json").write_text(json.dumps({"packages": [{"name": "example"}]}))
    monkeypatch.setattr(packager, "qualify", lambda bundle: [])
    packager.finish(args)
    assert (args.output / "dependency-sources/source.tar.gz").read_bytes() == source.read_bytes()
    assert checksum in (bundle / "SOURCE_INDEX.md").read_text()
    source.write_bytes(b"tampered source")
    with pytest.raises(RuntimeError, match="Missing cached source archive"):
        packager.finish(args)
