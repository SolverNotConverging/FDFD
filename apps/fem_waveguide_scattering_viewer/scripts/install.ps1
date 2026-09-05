param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'FEMWaveguideScatteringViewer'),
    [string]$MsysPrefix = 'C:\msys64\mingw64'
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$buildRoot = Join-Path $sourceRoot 'build'
$mingwBin = Join-Path $MsysPrefix 'bin'
$cmake = Join-Path $mingwBin 'cmake.exe'
$ninja = Join-Path $mingwBin 'ninja.exe'
$windeployqt = Join-Path $mingwBin 'windeployqt.exe'
$msysRoot = Split-Path -Parent $MsysPrefix
$ldd = Join-Path $msysRoot 'usr\bin\ldd.exe'

foreach ($required in ($cmake, $ninja, $windeployqt, $ldd)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required tool was not found: $required"
    }
}

$env:Path = "$mingwBin;$env:Path"
& $cmake --fresh -S $sourceRoot -B $buildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$MsysPrefix"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& $cmake --build $buildRoot --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native viewer build failed.' }
& $cmake --install $buildRoot --prefix $destinationRoot
if ($LASTEXITCODE -ne 0) { throw 'Native viewer installation failed.' }

$viewer = Join-Path $destinationRoot 'bin\fem-waveguide-scattering-viewer.exe'
& $windeployqt --release --no-translations --no-system-d3d-compiler $viewer
if ($LASTEXITCODE -ne 0) { throw 'Qt runtime deployment failed.' }

$dependencyLines = & $ldd $viewer
foreach ($line in $dependencyLines) {
    if ($line -match '=> /mingw64/bin/([^ ]+)') {
        $name = $Matches[1]
        $source = Join-Path $mingwBin $name
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination (Join-Path (Split-Path $viewer) $name) -Force
        }
    }
}

Write-Host "FEM Waveguide Scattering Viewer installed in $destinationRoot"
Write-Host "Run: $viewer [result.h5]"
