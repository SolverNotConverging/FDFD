param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'TransmissionLineCalculator'),
    [string]$VcpkgRoot = 'C:\opt\vcpkg',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$buildRoot = Join-Path $sourceRoot 'build/msvc-install'
. (Join-Path $PSScriptRoot '../../../scripts/setup_msvc_windows.ps1') -VcpkgRoot $VcpkgRoot
$cmake = (Get-Command cmake.exe).Source
$ctest = (Get-Command ctest.exe).Source

& $cmake --fresh -S $sourceRoot -B $buildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_TESTING=ON `
    "-DCMAKE_TOOLCHAIN_FILE=$env:CMAKE_TOOLCHAIN_FILE" -DVCPKG_TARGET_TRIPLET=x64-windows
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }

& $cmake --build $buildRoot --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native calculator build failed.' }

if (-not $SkipTests) {
    & $ctest --test-dir $buildRoot --output-on-failure -C Release
    if ($LASTEXITCODE -ne 0) { throw 'Native calculator tests failed.' }
}

& $cmake --install $buildRoot --config Release --prefix $destinationRoot
if ($LASTEXITCODE -ne 0) { throw 'Native calculator installation failed.' }

$binRoot = Join-Path $destinationRoot 'bin'
$calculator = Join-Path $binRoot 'transmission-line-calculator.exe'
$cli = Join-Path $binRoot 'transmission-line-calculator-cli.exe'
foreach ($executable in @($calculator, $cli)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Installed executable was not found: $executable"
    }
}

# Verify the installed CLI with only Windows and installed DLLs on PATH.
$originalPath = $env:Path
$windowsRoot = [System.IO.Path]::GetFullPath($env:SystemRoot)
$sanitizedPathEntries = @(
    $binRoot,
    (Join-Path $windowsRoot 'System32'),
    $windowsRoot,
    (Join-Path $windowsRoot 'System32\Wbem'),
    (Join-Path $windowsRoot 'System32\WindowsPowerShell\v1.0')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
try {
    $env:Path = ($sanitizedPathEntries | Select-Object -Unique) -join ';'
    & $cli --smoke-test
    if ($LASTEXITCODE -ne 0) {
        throw 'Staged interactive TUI smoke test failed.'
    }
} finally {
    $env:Path = $originalPath
}

Write-Host "Transmission Line Calculator installed in $destinationRoot"
Write-Host "GUI: $calculator"
Write-Host "Interactive TUI: $cli"
