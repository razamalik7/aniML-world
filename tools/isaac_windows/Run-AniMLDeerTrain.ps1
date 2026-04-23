#Requires -Version 5.1
<#
.SYNOPSIS
    Train aniML deer locomotion (ANYmal-C rough stand-in) via Isaac Lab RSL-RL.

.DESCRIPTION
    Sets PYTHONPATH to this repo and runs training/scripts/train.py with Kit Python.
    Set ANIML_ROOT if this repo is not next to the default path logic.

.PARAMETER ExtraArgs
    Extra arguments for rsl_rl train.py (e.g. --headless, Hydra overrides).
#>
[CmdletBinding()]
param(
    [string[]] $ExtraArgs = @("--headless")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"

if (-not $env:ANIML_ROOT) {
    $env:ANIML_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$animl = $env:ANIML_ROOT.TrimEnd('\', '/')
$trainScript = Join-Path $animl "training\scripts\train.py"

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at $($script:IsaacLabBat). Set ISAACLAB_ROOT."
}
if (-not (Test-Path -LiteralPath $trainScript)) {
    Write-Error "train.py not found at $trainScript. Set ANIML_ROOT to aniML_world repo root."
}

$env:PYTHONPATH = $animl
Push-Location $script:IsaacLabRoot
try {
    Write-Host "PYTHONPATH=$animl"
    Write-Host "Running: .\isaaclab.bat -p $trainScript --task AniML-Deer-Locomotion-Rough-v0 $($ExtraArgs -join ' ')"
    & .\isaaclab.bat -p $trainScript --task AniML-Deer-Locomotion-Rough-v0 @ExtraArgs
}
finally {
    Pop-Location
}
