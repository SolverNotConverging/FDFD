param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'FEMWaveguideScatteringViewer')
)

$ErrorActionPreference = 'Stop'
$target = [System.IO.Path]::GetFullPath($Destination)
if ([System.IO.Path]::GetFileName($target) -ne 'FEMWaveguideScatteringViewer') {
    throw "Refusing to remove a directory not named FEMWaveguideScatteringViewer: $target"
}
if (Test-Path -LiteralPath $target -PathType Container) {
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "Removed $target"
} else {
    Write-Host "FEM Waveguide Scattering Viewer is not installed at $target"
}
