<#
.SYNOPSIS
Select the MSYS2 MinGW64 toolchain used by this checkout.

.DESCRIPTION
Dot-source this file before uv commands so the current shell uses Ninja, GCC,
and the matching MSYS2 dependency prefix:

    . .\scripts\setup_mingw_windows.ps1
    uv sync --reinstall-package fdfd

Pass -PersistUser to give newly started applications (including PyCharm) the
same environment. Restart those applications after persisting the settings.
Any generated CMake cache using another generator/compiler is archived next to
the build directory rather than deleted.
#>
[CmdletBinding()]
param(
    [string]$MsysRoot = 'C:\msys64',
    [switch]$PersistUser
)

$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'setup_mingw_windows.ps1 is only supported on Windows.'
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$mingwPrefix = [System.IO.Path]::GetFullPath((Join-Path $MsysRoot 'mingw64'))
$mingwBin = Join-Path $mingwPrefix 'bin'
$msysBin = Join-Path $MsysRoot 'usr\bin'
$required = @(
    (Join-Path $mingwBin 'gcc.exe'),
    (Join-Path $mingwBin 'g++.exe'),
    (Join-Path $mingwBin 'cmake.exe'),
    (Join-Path $mingwBin 'ninja.exe'),
    (Join-Path $mingwBin 'libgmsh.dll'),
    (Join-Path $mingwPrefix 'lib\libgmsh.dll.a')
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required MinGW64 toolchain file is missing: $path"
    }
}

function Add-PathEntry {
    param([string]$Value, [string]$Entry)
    $entries = @($Value -split ';' | Where-Object { $_ })
    if ($entries -notcontains $Entry) {
        return (@($Entry) + $entries) -join ';'
    }
    return $entries -join ';'
}

$env:PATH = Add-PathEntry -Value $env:PATH -Entry $msysBin
$env:PATH = Add-PathEntry -Value $env:PATH -Entry $mingwBin
$env:CMAKE_GENERATOR = 'Ninja'
$env:CC = (Join-Path $mingwBin 'gcc.exe')
$env:CXX = (Join-Path $mingwBin 'g++.exe')
$env:CMAKE_PREFIX_PATH = Add-PathEntry -Value $env:CMAKE_PREFIX_PATH -Entry $mingwPrefix

# A generator cannot be changed in-place. Preserve stale build products for
# diagnosis while freeing scikit-build-core's configured build directory.
$buildRoot = Join-Path $repositoryRoot 'build'
if (Test-Path -LiteralPath $buildRoot -PathType Container) {
    Get-ChildItem -LiteralPath $buildRoot -Directory | ForEach-Object {
        $cache = Join-Path $_.FullName 'CMakeCache.txt'
        if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
            return
        }
        $contents = Get-Content -LiteralPath $cache
        $isNinja = $contents -contains 'CMAKE_GENERATOR:INTERNAL=Ninja'
        $cacheGcc = (Join-Path $mingwBin 'gcc.exe').Replace('\', '/')
        $usesGcc = ($contents | Select-String -SimpleMatch $cacheGcc)
        if (-not ($isNinja -and $usesGcc)) {
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $destination = Join-Path $repositoryRoot ("build-stale-{0}-{1}" -f $_.Name, $stamp)
            Move-Item -LiteralPath $_.FullName -Destination $destination
            Write-Host "Archived stale CMake cache at $destination"
        }
    }
}

if ($PersistUser) {
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $userPath = Add-PathEntry -Value $userPath -Entry $msysBin
    $userPath = Add-PathEntry -Value $userPath -Entry $mingwBin
    $userPrefix = [Environment]::GetEnvironmentVariable('CMAKE_PREFIX_PATH', 'User')
    $userPrefix = Add-PathEntry -Value $userPrefix -Entry $mingwPrefix
    [Environment]::SetEnvironmentVariable('PATH', $userPath, 'User')
    [Environment]::SetEnvironmentVariable('CMAKE_PREFIX_PATH', $userPrefix, 'User')
    [Environment]::SetEnvironmentVariable('CMAKE_GENERATOR', 'Ninja', 'User')
    [Environment]::SetEnvironmentVariable('CC', $env:CC, 'User')
    [Environment]::SetEnvironmentVariable('CXX', $env:CXX, 'User')
    Write-Host 'Saved MinGW64 settings for newly started applications. Restart PyCharm before rebuilding.'
}

Write-Host "FDFD toolchain: $(& (Join-Path $mingwBin 'g++.exe') --version | Select-Object -First 1)"
Write-Host "CMake generator: $env:CMAKE_GENERATOR"
Write-Host "Dependency prefix: $env:CMAKE_PREFIX_PATH"
