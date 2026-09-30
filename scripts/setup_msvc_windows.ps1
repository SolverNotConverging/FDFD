<#
.SYNOPSIS
Select x64 MSVC, Ninja, and C:\opt\vcpkg for this shell.
.DESCRIPTION
Dot-source before uv or CMake commands. Imports the Visual Studio developer
environment automatically. Does not modify persistent user settings.
#>
[CmdletBinding()]
param([string]$VcpkgRoot = 'C:\opt\vcpkg')

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'This helper requires Windows.' }
$VcpkgRoot = [IO.Path]::GetFullPath($VcpkgRoot)
if (-not (Test-Path -LiteralPath "$VcpkgRoot/scripts/buildsystems/vcpkg.cmake")) {
    throw "vcpkg is missing at $VcpkgRoot. Install it before building."
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsInstallation = & $vswhere -latest -prerelease -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstallation) { throw 'Install Visual Studio C++ build tools and the Windows SDK.' }
& "$vsInstallation/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'MSVC initialization failed.' }
# Remove inherited MSYS/MinGW paths in this process only.
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and $_ -notmatch '(?i)msys|mingw' }) -join ';'
$vsCmake = Join-Path $vsInstallation 'Common7/IDE/CommonExtensions/Microsoft/CMake'
$env:PATH = "$vsCmake/CMake/bin;$vsCmake/Ninja;$env:PATH"
$env:CC = 'cl'
$env:CXX = 'cl'
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_GENERATOR_PLATFORM = $null
$env:CMAKE_GENERATOR_TOOLSET = $null
$env:CMAKE_PREFIX_PATH = $null
$env:CMAKE_TOOLCHAIN_FILE = "$VcpkgRoot/scripts/buildsystems/vcpkg.cmake"
$env:VCPKG_ROOT = $VcpkgRoot
$env:VCPKG_DEFAULT_TRIPLET = 'x64-windows'
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=`"$($env:CMAKE_TOOLCHAIN_FILE.Replace('\', '/'))`" -DVCPKG_TARGET_TRIPLET=x64-windows"
$env:PATH = "$VcpkgRoot/installed/x64-windows/bin;$env:PATH"
foreach ($tool in @('cmake.exe', 'ninja.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Install the Visual Studio CMake tools component ($tool missing)." }
}
Write-Host "FDFD: x64 MSVC + Ninja; vcpkg: $VcpkgRoot"
