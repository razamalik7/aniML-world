#Requires -Version 5.1
<#
.SYNOPSIS
    Re-pin numpy and h5py inside Isaac Sim Kit Python (Windows).

.DESCRIPTION
    Isaac Lab expects numpy<2. A plain pip install h5py can pull NumPy 2.x and break Lab.
    h5py wheels sometimes conflict with Kit at runtime; numpy==1.26.4 + h5py==3.10.0 --no-deps
    is a known-stable combo for many Kit 4.x / Sim 5.x installs.

    Set ISAACLAB_ROOT if your Isaac Lab clone is not C:\isaac-lab\IsaacLab.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at: $($script:IsaacLabBat). Set env ISAACLAB_ROOT to your Isaac Lab root."
}

Push-Location $script:IsaacLabRoot
try {
    Write-Host "Using ISAACLAB_ROOT=$($script:IsaacLabRoot)"
    & .\isaaclab.bat -p -m pip install "numpy==$($script:KitNumpyVersion)"
    & .\isaaclab.bat -p -m pip install --no-deps "h5py==$($script:KitH5pyVersion)"
    Write-Host "Verifying imports..."
    & .\isaaclab.bat -p -c "import numpy, h5py; print('numpy', numpy.__version__, 'h5py', h5py.__version__)"
}
finally {
    Pop-Location
}
