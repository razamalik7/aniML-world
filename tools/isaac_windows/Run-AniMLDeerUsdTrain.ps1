#Requires -Version 5.1
<#
.SYNOPSIS
    Train PPO on YOUR deer/quadruped USD (rough or flat MDP).

.DESCRIPTION
    Resolves deer USD in order:
      1) $env:ANIML_DEER_USD if set and file exists
      2) assets\deer\deer.usd
      3) deer1.usdc at repo root (common export name)
      4) deer1.usd, deer.usdc at repo root

    Optional env (PowerShell before running):
      $env:ANIML_DEER_ACTIVATE_CONTACT_SENSORS = '1'  # after USD has PhysX rigid bodies (spawn adds contact report API)
      $env:ANIML_DEER_IMU_LINK = 'pelvis'     # prim under Robot for height scanner + command frame
      $env:ANIML_DEER_BODY_BASE = 'base'      # trunk link for illegal contact / mass events
      $env:ANIML_DEER_BODY_FOOT = 'regex...'  # override foot contact regex if rewards look wrong
      $env:ANIML_DEER_BODY_THIGH = 'regex...'

.PARAMETER Flat
    Use task AniML-Deer-USD-Flat-v0 (plane, no height scan) — often shows a gait sooner; still real RL.

.PARAMETER Task
    Override full task id if needed.

.PARAMETER Watch
    Open Isaac with a visible viewport (drops --headless) and default --num_envs 32 for a clear view.
    Override with -ExtraArgs @("--num_envs","64") etc. Headless training keeps env cfg default (e.g. 1536).

.PARAMETER ExtraArgs
    Extra args for rsl_rl train.py. Default is --headless unless -Watch is set.
#>
[CmdletBinding()]
param(
    [switch] $Flat,
    [switch] $Watch,
    [string] $Task = "",
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

if (-not $env:ANIML_DEER_USD -or -not (Test-Path -LiteralPath $env:ANIML_DEER_USD)) {
    $candidates = @(
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
        Write-Error "Set ANIML_DEER_USD or place deer1.usdc (or assets\deer\deer.usd) under $animl"
    }
}

if (-not $Task) {
    $Task = if ($Flat) { "AniML-Deer-USD-Flat-v0" } else { "AniML-Deer-USD-Rough-v0" }
}

if ($Watch) {
    $ea = [System.Collections.ArrayList]@($ExtraArgs | Where-Object { $_ -ne "--headless" })
    $hasNumEnvs = $false
    for ($i = 0; $i -lt $ea.Count; $i++) {
        if ($ea[$i] -eq "--num_envs") { $hasNumEnvs = $true; break }
    }
    if (-not $hasNumEnvs) {
        [void]$ea.Insert(0, "32")
        [void]$ea.Insert(0, "--num_envs")
    }
    $ExtraArgs = @($ea)
    Write-Host "Watch mode: viewport on, --num_envs default 32. Args: $($ExtraArgs -join ' ')"
}

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at $($script:IsaacLabBat). Set ISAACLAB_ROOT."
}

$env:PYTHONPATH = $animl
Push-Location $script:IsaacLabRoot
try {
    Write-Host "Task=$Task  PYTHONPATH=$animl"
    & .\isaaclab.bat -p $trainScript --task $Task @ExtraArgs
}
finally {
    Pop-Location
}
