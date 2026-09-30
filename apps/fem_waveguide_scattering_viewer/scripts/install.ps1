param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'FEMWaveguideScatteringViewer'),
    [string]$VcpkgRoot = 'C:\opt\vcpkg'
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$buildRoot = Join-Path $sourceRoot 'build/msvc-install'
. (Join-Path $PSScriptRoot '../../../scripts/setup_msvc_windows.ps1') -VcpkgRoot $VcpkgRoot
$cmake = (Get-Command cmake.exe).Source
$ctest = (Get-Command ctest.exe).Source

& $cmake --fresh -S $sourceRoot -B $buildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_TOOLCHAIN_FILE=$env:CMAKE_TOOLCHAIN_FILE" -DVCPKG_TARGET_TRIPLET=x64-windows
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& $cmake --build $buildRoot --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native viewer build failed.' }
& $cmake --install $buildRoot --config Release --prefix $destinationRoot
if ($LASTEXITCODE -ne 0) { throw 'Native viewer installation failed.' }

$viewer = Join-Path $destinationRoot 'bin\fem-waveguide-scattering-viewer.exe'
Write-Host "FEM Waveguide Scattering Viewer installed in $destinationRoot"
Write-Host "Run: $viewer [result.h5]"
