param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'FEMPeriodicViewer')
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
$target = [System.IO.Path]::GetFullPath($Destination)
Assert-NoReparsePointInAncestry -Path $sourceRoot -Label 'Source'
Assert-NoReparsePointInAncestry -Path $target -Label 'Destination'
if ([System.IO.Path]::GetFileName($target) -ne 'FEMPeriodicViewer') {
    throw "Refusing to remove a directory not named FEMPeriodicViewer: $target"
}
$separator = [System.IO.Path]::DirectorySeparatorChar
$sourcePrefix = $sourceRoot + $separator
$targetPrefix = $target + $separator
$comparison = [System.StringComparison]::OrdinalIgnoreCase
if (
    $sourceRoot.Equals($target, $comparison) -or
    $target.StartsWith($sourcePrefix, $comparison) -or
    $sourceRoot.StartsWith($targetPrefix, $comparison)
) {
    throw "Refusing to remove the viewer source tree, its parent, or its child: $target"
}
if (Test-Path -LiteralPath $target -PathType Container) {
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "Removed $target"
} else {
    Write-Host "FEM Periodic Mode Viewer is not installed at $target"
}
