param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'TransmissionLineCalculator'),
    [string]$MsysPrefix = 'C:\msys64\mingw64',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$buildRoot = Join-Path $sourceRoot 'build'
$mingwBin = Join-Path $MsysPrefix 'bin'
$cmake = Join-Path $mingwBin 'cmake.exe'
$ctest = Join-Path $mingwBin 'ctest.exe'
$ninja = Join-Path $mingwBin 'ninja.exe'
$windeployqt = Join-Path $mingwBin 'windeployqt.exe'
$msysRoot = Split-Path -Parent $MsysPrefix
$ldd = Join-Path $msysRoot 'usr\bin\ldd.exe'
$gmshConfig = Join-Path $MsysPrefix 'lib\cmake\gmsh\gmshConfig.cmake'
$gmshHeader = Join-Path $MsysPrefix 'include\gmsh.h'
$gmshImportLibrary = Join-Path $MsysPrefix 'lib\libgmsh.dll.a'
$gmshRuntime = Join-Path $mingwBin 'libgmsh.dll'

$requiredTools = @(
    $cmake,
    $ninja,
    $windeployqt,
    $ldd,
    $gmshConfig,
    $gmshHeader,
    $gmshImportLibrary,
    $gmshRuntime
)
if (-not $SkipTests) {
    $requiredTools += $ctest
}
foreach ($required in $requiredTools) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required tool was not found: $required"
    }
}

$env:Path = "$mingwBin;$env:Path"
& $cmake --fresh -S $sourceRoot -B $buildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_TESTING=ON `
    "-DCMAKE_PREFIX_PATH=$MsysPrefix"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }

& $cmake --build $buildRoot --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native calculator build failed.' }

if (-not $SkipTests) {
    & $ctest --test-dir $buildRoot --output-on-failure -C Release
    if ($LASTEXITCODE -ne 0) { throw 'Native calculator tests failed.' }
}

& $cmake --install $buildRoot --prefix $destinationRoot
if ($LASTEXITCODE -ne 0) { throw 'Native calculator installation failed.' }

$binRoot = Join-Path $destinationRoot 'bin'
$calculator = Join-Path $binRoot 'transmission-line-calculator.exe'
$cli = Join-Path $binRoot 'transmission-line-calculator-cli.exe'
foreach ($executable in @($calculator, $cli)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Installed executable was not found: $executable"
    }
}

& $windeployqt --release --no-translations --no-system-d3d-compiler $calculator
if ($LASTEXITCODE -ne 0) { throw 'Qt runtime deployment failed.' }

# Gmsh is a direct native dependency of the statically linked tl-core. Copy it
# explicitly as well as discovering dependencies below, so deployment remains
# robust if the ldd output format changes.
Copy-Item -LiteralPath $gmshRuntime -Destination (Join-Path $binRoot 'libgmsh.dll') -Force

$runtimeNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
$missingDependencies = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($executable in @($calculator, $cli)) {
    $dependencyLines = & $ldd $executable
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency inspection failed for $executable"
    }
    foreach ($line in $dependencyLines) {
        if ($line -match '^\s*([^ ]+) => not found') {
            [void]$missingDependencies.Add($Matches[1])
        } elseif ($line -match '=> /mingw64/bin/([^ ]+)') {
            [void]$runtimeNames.Add($Matches[1])
        }
    }
}

if ($missingDependencies.Count -gt 0) {
    $missingList = ($missingDependencies | Sort-Object) -join ', '
    throw "MinGW runtime dependencies are missing: $missingList"
}

foreach ($name in $runtimeNames) {
    $source = Join-Path $mingwBin $name
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $binRoot $name) -Force
    }
}

Write-Host "Transmission Line Calculator installed in $destinationRoot"
Write-Host "GUI: $calculator"
Write-Host "CLI: $cli --help"
