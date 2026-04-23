#Requires -Version 5.1
<#
.SYNOPSIS
    Run Isaac Lab empty-scene tutorial (quick sanity check).

    Extra args are passed through to the Python script after create_empty.py, e.g. --headless if supported.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\config.ps1"

if (-not (Test-Path -LiteralPath $script:IsaacLabBat)) {
    Write-Error "isaaclab.bat not found at: $($script:IsaacLabBat). Set env ISAACLAB_ROOT."
}

Push-Location $script:IsaacLabRoot
try {
    $py = $script:VerifyScript
    Write-Host "Running: .\isaaclab.bat -p $py $args"
    & .\isaaclab.bat -p $py @args
}
finally {
    Pop-Location
}
