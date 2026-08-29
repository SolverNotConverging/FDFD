param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'WaveFEMViewer')
)

$ErrorActionPreference = 'Stop'
$target = [System.IO.Path]::GetFullPath($Destination)
if ([System.IO.Path]::GetFileName($target) -ne 'WaveFEMViewer') {
    throw "Refusing to remove a directory not named WaveFEMViewer: $target"
}
if (Test-Path -LiteralPath $target -PathType Container) {
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "Removed $target"
} else {
    Write-Host "WaveFEM Viewer is not installed at $target"
}
