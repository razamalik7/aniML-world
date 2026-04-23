from pathlib import Path
from datetime import datetime

fmt = "%Y-%m-%d %H:%M"

config = Path(r'C:\Users\nars7\Downloads\aniML_world\configuration')
print('configuration/ directory:')
for f in sorted(config.iterdir()):
    mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime(fmt)
    print(f'  {f.name:<40}  {f.stat().st_size:8d} bytes  {mtime}')

print()
print('stand_v2 used USD:')
with open(r'C:\Users\nars7\Downloads\aniML_world\logs\rsl_rl\animl_deer_stand_v2\2026-04-10_17-51-46\params\env.yaml') as f:
    for line in f:
        if 'usd_path' in line and 'deer' in line.lower():
            print(' ', line.strip())
            break

f_s = Path(r'C:\Users\nars7\Downloads\aniML_world\deer_standing_robot.usd')
mtime_s = datetime.fromtimestamp(f_s.stat().st_mtime).strftime(fmt)
print(f'\ndeer_standing_robot.usd modified: {mtime_s}  ({f_s.stat().st_size} bytes)')

# What assets/ folder looks like for the deer standing URDF
assets = Path(r'C:\Users\nars7\Downloads\aniML_world\assets\deer_urdf')
print('\nassets/deer_urdf/ (immediate children):')
for f in sorted(assets.iterdir()):
    mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime(fmt)
    size = f.stat().st_size if f.is_file() else 0
    print(f'  {f.name:<40}  {"FILE" if f.is_file() else "DIR "}  {size:8d}  {mtime}')
