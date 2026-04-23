# Deer_001_walk.anim — Phase 0a Probe Report

Source: `C:\Users\nars7\Downloads\Unity Projects\aniML-world\Assets\ithappy\Animals_FREE\Animations\Other_Animations\Deer_001_walk.anim`

- Rotation curves: 34
- Position curves: 34
- Max keyframes per curve: 14
- Min keyframes per curve: 2

## Q1: Root motion

**VERDICT: IN-PLACE (will need to synthesize forward velocity)**

### `Deer_001_rig`


### `Deer_001_rig/Root`



## Q2: Shoulder non-X rotation amplitude

**VERDICT: SIGNIFICANT non-X shoulder motion (max 46.2°) -- extend URDF or bake into root**

Shoulder bones:
- `Deer_001_rig/Root/spine.003/shoulder.R` keys=11 x_amp=48.97° y_amp=6.46° z_amp=8.47°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R` keys=2 x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R` keys=11 x_amp=28.70° y_amp=4.79° z_amp=17.59°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R/foot.R` keys=11 x_amp=32.12° y_amp=4.53° z_amp=6.02°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R/foot.R/toe.R` keys=8 x_amp=34.09° y_amp=4.40° z_amp=10.08°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R` keys=9 x_amp=21.42° y_amp=45.68° z_amp=46.17°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R` keys=12 x_amp=15.20° y_amp=42.94° z_amp=34.94°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R` keys=11 x_amp=21.91° y_amp=6.13° z_amp=2.08°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R` keys=13 x_amp=55.33° y_amp=6.54° z_amp=23.29°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R/front_toe.R` keys=10 x_amp=51.72° y_amp=6.94° z_amp=5.78°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L` keys=8 x_amp=30.84° y_amp=6.22° z_amp=6.22°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L` keys=2 x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L` keys=9 x_amp=18.83° y_amp=3.34° z_amp=1.69°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L` keys=14 x_amp=58.45° y_amp=7.73° z_amp=21.16°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L/front_toe.L` keys=12 x_amp=46.07° y_amp=4.46° z_amp=5.01°
- `Deer_001_rig/Root/spine.003/shoulder.L` keys=10 x_amp=38.16° y_amp=5.62° z_amp=2.99°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L` keys=2 x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L` keys=12 x_amp=21.14° y_amp=4.55° z_amp=14.46°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L/foot.L` keys=10 x_amp=30.86° y_amp=4.35° z_amp=5.74°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L/foot.L/toe.L` keys=8 x_amp=30.90° y_amp=4.11° z_amp=9.34°

Thigh bones (sanity check — should swing on X):
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R` x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R` x_amp=28.70° y_amp=4.79° z_amp=17.59°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R/foot.R` x_amp=32.12° y_amp=4.53° z_amp=6.02°
- `Deer_001_rig/Root/spine.003/shoulder.R/thigh.R/shin.R/foot.R/toe.R` x_amp=34.09° y_amp=4.40° z_amp=10.08°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R` x_amp=15.20° y_amp=42.94° z_amp=34.94°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R` x_amp=21.91° y_amp=6.13° z_amp=2.08°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R` x_amp=55.33° y_amp=6.54° z_amp=23.29°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.R/front_thigh.R/front_shin.R/front_foot.R/front_toe.R` x_amp=51.72° y_amp=6.94° z_amp=5.78°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L` x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L` x_amp=18.83° y_amp=3.34° z_amp=1.69°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L` x_amp=58.45° y_amp=7.73° z_amp=21.16°
- `Deer_001_rig/Root/spine.003/spine.004/spine.005/front_shoulder.L/front_thigh.L/front_shin.L/front_foot.L/front_toe.L` x_amp=46.07° y_amp=4.46° z_amp=5.01°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L` x_amp=0.00° y_amp=0.00° z_amp=0.00°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L` x_amp=21.14° y_amp=4.55° z_amp=14.46°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L/foot.L` x_amp=30.86° y_amp=4.35° z_amp=5.74°
- `Deer_001_rig/Root/spine.003/shoulder.L/thigh.L/shin.L/foot.L/toe.L` x_amp=30.90° y_amp=4.11° z_amp=9.34°

## Decisions

- [ ] Root handling: IN-PLACE → synthesize velocity / HAS ROOT MOTION → use as-is
- [ ] Shoulder handling: ignore / extend URDF / bake into root