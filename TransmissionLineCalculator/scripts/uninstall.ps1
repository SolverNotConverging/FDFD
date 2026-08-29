param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'TransmissionLineCalculator')
)

$ErrorActionPreference = 'Stop'
$target = [System.IO.Path]::GetFullPath($Destination)
if ([System.IO.Path]::GetFileName($target) -ne 'TransmissionLineCalculator') {
    throw "Refusing to remove a directory not named TransmissionLineCalculator: $target"
}

if (Test-Path -LiteralPath $target -PathType Container) {
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "Removed $target"
} else {
    Write-Host "Transmission Line Calculator is not installed at $target"
}

