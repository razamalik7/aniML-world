#Requires -Version 5.1
<#
.SYNOPSIS
    Short RSL-RL Cartpole run (smoke test). Headless by default.

.PARAMETER Task
    Gym registry id, default Isaac-Cartpole-v0 (override if your Lab version differs).

.PARAMETER ExtraArgs
    Additional arguments for train.py (Hydra overrides, etc.), e.g. agent.max_iterations=2

.EXAMPLE
    .\Run-CartpoleSmoke.ps1
.EXAMPLE
    .\Run-CartpoleSmoke.ps1 -Task Isaac-Cartpole-Direct-v0 -ExtraArgs @("agent.max_iterations=3")
#>
[CmdletBinding()]
param(
    [string] $Task = $null,
    [string[]] $ExtraArgs = @("agent.max_iterations=2", "--headless")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"

if (-not $Task) { $Task = $script:DefaultCartpoleTask }

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at: $($script:IsaacLabBat). Set env ISAACLAB_ROOT."
}

Push-Location $script:IsaacLabRoot
try {
    $train = $script:RlTrainScript
    Write-Host "Running: .\isaaclab.bat -p $train --task $Task $($ExtraArgs -join ' ')"
    & .\isaaclab.bat -p $train --task $Task @ExtraArgs
}
finally {
    Pop-Location
}
