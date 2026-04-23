#Requires -Version 5.1
<#
.SYNOPSIS
    Run the automated Blender rig-to-URDF pipeline for any species.

.DESCRIPTION
    Opens a .blend file (headless), reads a species JSON config, generates
    per-link convex-hull collision from vertex weights, and exports a URDF
    + STL meshes folder that Isaac Sim can import directly.

    Requires Blender 4.2 LTS (or compatible) on PATH or via -BlenderExe.

.EXAMPLE
    # Deer (first time)
    .\Run-BlenderToUrdf.ps1 `
        -BlendFile "C:\path\deer.blend" `
        -SpeciesConfig "C:\path\aniML_world\configs\species\deer.json" `
        -OutputDir "C:\path\aniML_world\assets\deer_urdf"

    # Dry run (prints plan, no files written)
    .\Run-BlenderToUrdf.ps1 `
        -BlendFile "C:\path\deer.blend" `
        -SpeciesConfig "C:\path\aniML_world\configs\species\deer.json" `
        -OutputDir "." -DryRun

.PARAMETER BlenderExe
    Path to blender.exe. If omitted, tries common locations then PATH.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $BlendFile,
    [Parameter(Mandatory)][string] $SpeciesConfig,
    [Parameter(Mandatory)][string] $OutputDir,
    [string] $BlenderExe = "",
    [switch] $DryRun,
    [switch] $NoVisualMeshes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $env:ANIML_ROOT) {
    $env:ANIML_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$animl = $env:ANIML_ROOT.TrimEnd('\', '/')
$script = Join-Path $animl "tools\blender_urdf_pipeline\rig_to_urdf.py"

if (-not (Test-Path -LiteralPath $script)) {
    Write-Error "Pipeline script not found: $script"
}
if (-not (Test-Path -LiteralPath $BlendFile)) {
    Write-Error "Blend file not found: $BlendFile"
}
if (-not (Test-Path -LiteralPath $SpeciesConfig)) {
    Write-Error "Species config not found: $SpeciesConfig"
}

# --- Locate Blender ---
if ($BlenderExe -and (Test-Path -LiteralPath $BlenderExe)) {
    $blender = $BlenderExe
} else {
    $candidates = @(
        "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        "C:\Program Files\Blender Foundation\Blender\blender.exe",
        "$env:LOCALAPPDATA\Blender Foundation\Blender 4.2\blender.exe"
    )
    $blender = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $blender) {
        $blender = Get-Command blender -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    }
    if (-not $blender) {
        Write-Error "Blender not found. Pass -BlenderExe or add Blender to PATH."
    }
}

Write-Host "Blender: $blender"
Write-Host "Blend:   $BlendFile"
Write-Host "Config:  $SpeciesConfig"
Write-Host "Output:  $OutputDir"

$pyArgs = @(
    "--config", (Resolve-Path -LiteralPath $SpeciesConfig).Path,
    "--output-dir", $OutputDir
)
if ($DryRun) { $pyArgs += "--dry-run" }
if ($NoVisualMeshes) { $pyArgs += "--no-visual-meshes" }

& $blender --background (Resolve-Path -LiteralPath $BlendFile).Path `
    --python $script `
    -- @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Blender exited with code $LASTEXITCODE"
}
