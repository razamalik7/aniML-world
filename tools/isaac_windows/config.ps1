# Dot-source from other scripts: . "$PSScriptRoot\config.ps1"
# Override in your profile or before running: $env:ISAACLAB_ROOT = "D:\path\to\IsaacLab"

if (-not $env:ISAACLAB_ROOT) {
    $env:ISAACLAB_ROOT = "C:\isaac-lab\IsaacLab"
}
# aniML_world repo root (parent of /tools); Run-AniMLDeerTrain.ps1 sets if unset
if (-not $env:ANIML_ROOT) {
    $env:ANIML_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$script:IsaacLabRoot = $env:ISAACLAB_ROOT.TrimEnd('\', '/')
$script:IsaacLabBat = Join-Path $script:IsaacLabRoot "isaaclab.bat"

# Pins that avoid NumPy 2.x and Kit DLL issues on Windows (see conversation / Isaac Lab docs).
$script:KitNumpyVersion = "1.26.4"
$script:KitH5pyVersion = "3.10.0"

$script:VerifyScript = "scripts\tutorials\00_sim\create_empty.py"
$script:RlTrainScript = "scripts\reinforcement_learning\rsl_rl\train.py"
$script:DefaultCartpoleTask = "Isaac-Cartpole-v0"
