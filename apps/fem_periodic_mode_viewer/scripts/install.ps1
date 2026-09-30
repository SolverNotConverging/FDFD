param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'FEMPeriodicModeViewer'),
    [string]$VcpkgRoot = 'C:\opt\vcpkg',
    [switch]$WithoutVtk
)

$ErrorActionPreference = 'Stop'

function Assert-NoReparsePointInAncestry {
    param([string]$Path, [string]$Label)
    $current = [System.IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrEmpty($current)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "$Label path traverses a junction or symbolic link: $current"
            }
        }
        $parent = [System.IO.Directory]::GetParent($current)
        if ($null -eq $parent) { break }
        $current = $parent.FullName
    }
}

$sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
Assert-NoReparsePointInAncestry -Path $sourceRoot -Label 'Source'
Assert-NoReparsePointInAncestry -Path $destinationRoot -Label 'Destination'
if ([System.IO.Path]::GetFileName($destinationRoot) -ne 'FEMPeriodicModeViewer') {
    throw "Destination directory must be named FEMPeriodicModeViewer: $destinationRoot"
}
$separator = [System.IO.Path]::DirectorySeparatorChar
$sourcePrefix = $sourceRoot + $separator
$destinationPrefix = $destinationRoot + $separator
$comparison = [System.StringComparison]::OrdinalIgnoreCase
if (
    $sourceRoot.Equals($destinationRoot, $comparison) -or
    $destinationRoot.StartsWith($sourcePrefix, $comparison) -or
    $sourceRoot.StartsWith($destinationPrefix, $comparison)
) {
    throw "Destination must not be the viewer source tree, its parent, or its child: $destinationRoot"
}
$destinationParent = Split-Path -Parent $destinationRoot
if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
    [void](New-Item -ItemType Directory -Path $destinationParent)
}
$stagingRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $destinationParent ".FEMPeriodicModeViewer-stage-$PID")
)
$backupRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $destinationParent ".FEMPeriodicModeViewer-backup-$PID")
)
$buildRoot = Join-Path $sourceRoot 'build/msvc-install'
. (Join-Path $PSScriptRoot '../../../scripts/setup_msvc_windows.ps1') -VcpkgRoot $VcpkgRoot
$cmake = (Get-Command cmake.exe).Source
$ctest = (Get-Command ctest.exe).Source

$parentPrefix = $destinationParent.TrimEnd('\') + '\'
foreach ($temporaryRoot in ($stagingRoot, $backupRoot)) {
    if (-not $temporaryRoot.StartsWith(
        $parentPrefix, [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Temporary install path escaped the destination parent: $temporaryRoot"
    }
    $leaf = [System.IO.Path]::GetFileName($temporaryRoot)
    if (-not $leaf.StartsWith('.FEMPeriodicModeViewer-', [System.StringComparison]::Ordinal)) {
        throw "Unexpected temporary install path: $temporaryRoot"
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        throw "Temporary install path already exists: $temporaryRoot"
    }
}

$vtkSetting = if ($WithoutVtk) { 'OFF' } else { 'ON' }
$movedExistingInstall = $false
try {
    & $cmake --fresh -S $sourceRoot -B $buildRoot -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$env:CMAKE_TOOLCHAIN_FILE" -DVCPKG_TARGET_TRIPLET=x64-windows `
    "-DFEM_PERIODIC_MODE_VIEWER_WITH_VTK=$vtkSetting"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
    & $cmake --build $buildRoot --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw 'Native viewer build failed.' }
    & $cmake --install $buildRoot --config Release --prefix $stagingRoot
    if ($LASTEXITCODE -ne 0) { throw 'Native viewer installation failed.' }

    if (Test-Path -LiteralPath $destinationRoot) {
        Move-Item -LiteralPath $destinationRoot -Destination $backupRoot
        $movedExistingInstall = $true
    }
    Move-Item -LiteralPath $stagingRoot -Destination $destinationRoot
    if ($movedExistingInstall) {
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
        $movedExistingInstall = $false
    }
} catch {
    if ($movedExistingInstall -and -not (Test-Path -LiteralPath $destinationRoot)) {
        Move-Item -LiteralPath $backupRoot -Destination $destinationRoot
        $movedExistingInstall = $false
    }
    throw
} finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}

Write-Host "FEM Periodic Mode Viewer installed in $destinationRoot"
Write-Host "VTK 3D support: $vtkSetting"
Write-Host "Run: $(Join-Path $destinationRoot 'bin\fem-periodic-mode-viewer.exe') [result.h5]"
