#Requires -Version 5.1
<#
.SYNOPSIS
    Train AMP_PPO on the deer USD using the ithappy walk clip as motion prior.

.DESCRIPTION
    Uses training/scripts/train_amp.py which builds AMPOnPolicyRunner directly
    (Isaac Lab's default rsl_rl train.py doesn't know about AMP_PPO).

    Dependencies (installed into Isaac Lab's Python by the night run):
      - amp-rsl-rl  1.2.0
      - rsl-rl-lib  3.0.0  (amp-rsl-rl needs legacy ActorCritic import)
      - tensorboard

    Motion dataset must exist at:
      <repo>/assets/motions/deer_walk.npy

.PARAMETER NumEnvs
    Parallel envs. Default 1024 (safe on RTX 5070 Laptop, 12GB).
    Use 64 for a smoke test.

.PARAMETER MaxIters
    Override runner's max_iterations (default from config: 3000).

.PARAMETER Watch
    Launch with viewport (drops --headless). Forces NumEnvs to 32 if still default.
#>
[CmdletBinding()]
param(
    [int] $NumEnvs = 1024,
    [int] $MaxIters = -1,
    [switch] $Watch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"

if (-not $env:ANIML_ROOT) {
    $env:ANIML_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$animl = $env:ANIML_ROOT.TrimEnd('\', '/')
$trainScript = Join-Path $animl "training\scripts\train_amp.py"

# --- Resolve deer USD ---
if (-not $env:ANIML_DEER_USD -or -not (Test-Path -LiteralPath $env:ANIML_DEER_USD)) {
    $candidates = @(
        (Join-Path $animl "deer_world.usd"),
        (Join-Path $animl "assets\deer\deer.usd"),
        (Join-Path $animl "deer1.usdc"),
        (Join-Path $animl "deer1.usd"),
        (Join-Path $animl "deer.usdc")
    )
    $found = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($found) {
        $env:ANIML_DEER_USD = (Resolve-Path -LiteralPath $found).Path
        Write-Host "Using deer USD: $($env:ANIML_DEER_USD)"
    } else {
        Write-Error "Set ANIML_DEER_USD or place deer_world.usd under $animl"
    }
}

# Enable contact sensors — the walk env termination uses them.
if (-not $env:ANIML_DEER_ACTIVATE_CONTACT_SENSORS) {
    $env:ANIML_DEER_ACTIVATE_CONTACT_SENSORS = '1'
}

$extra = @()
if (-not $Watch) {
    $extra += "--headless"
} elseif ($NumEnvs -eq 1024) {
    $NumEnvs = 32
}

$extra += @("--task", "AniML-Deer-USD-Walk-AMP-v0", "--num_envs", "$NumEnvs")
if ($MaxIters -gt 0) {
    $extra += @("--max_iterations", "$MaxIters")
}

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at $($script:IsaacLabBat). Set ISAACLAB_ROOT."
}

$env:PYTHONPATH = $animl
Push-Location $script:IsaacLabRoot
try {
    Write-Host "Task=AniML-Deer-USD-Walk-AMP-v0  NumEnvs=$NumEnvs  Args: $($extra -join ' ')"
    & .\isaaclab.bat -p $trainScript @extra
}
finally {
    Pop-Location
}
