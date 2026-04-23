#Requires -Version 5.1
<#
.SYNOPSIS
    Copy a USD stage and append skeleton-driven capsule rigid bodies (+ optional joints).

.DESCRIPTION
    Wraps tools\autogen_skel_physics_proxies.py with isaaclab.bat (needs pxr / UsdPhysics).

    Example:
      .\Run-AutogenSkelPhysics.ps1 `
        -InputUsd "C:\Users\you\Downloads\aniML_world\deer_rl_workspace.usd" `
        -OutputUsd "C:\Users\you\Downloads\aniML_world\deer_rl_with_proxies.usd" `
        -RigRoot "/World/deer1/Deer_001_rig" `
        -Joints

    Dry-run (no file written):
      .\Run-AutogenSkelPhysics.ps1 -InputUsd "...\deer.usd" -RigRoot "/World/deer1/Deer_001_rig" -DryRun

.PARAMETER ExtraArgs
    Extra args passed through to the Python script after '--'.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $InputUsd,
    [string] $OutputUsd = "",
    [Parameter(Mandatory)][string] $RigRoot,
    [string] $SkeletonPath = "",
    [switch] $Joints,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"
if (-not $env:ANIML_ROOT) {
    $env:ANIML_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$animl = $env:ANIML_ROOT.TrimEnd('\', '/')
$py = Join-Path $animl "tools\autogen_skel_physics_proxies.py"

if (-not (Test-Path -LiteralPath $py)) {
    Write-Error "Missing $py"
}
if (-not (Test-Path -LiteralPath $InputUsd)) {
    Write-Error "Input USD not found: $InputUsd"
}
if (-not $DryRun -and -not $OutputUsd) {
    Write-Error "Pass -OutputUsd or use -DryRun"
}

$pyArgs = @(
    "--input", (Resolve-Path -LiteralPath $InputUsd).Path,
    "--rig-root", $RigRoot
)
if ($OutputUsd) {
    $pyArgs += @("--output", $OutputUsd)
}
if ($SkeletonPath) {
    $pyArgs += @("--skeleton", $SkeletonPath)
}
if ($Joints) { $pyArgs += "--joints" }
if ($DryRun) { $pyArgs += "--dry-run" }
$pyArgs += $ExtraArgs

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at $($script:IsaacLabBat). Set ISAACLAB_ROOT."
}

Push-Location $script:IsaacLabRoot
try {
    Write-Host "isaaclab.bat -p $py $($pyArgs -join ' ')"
    & .\isaaclab.bat -p $py @pyArgs
}
finally {
    Pop-Location
}
