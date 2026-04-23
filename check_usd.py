from pathlib import Path
import os

raw = os.environ.get('ANIML_DEER_USD', '').strip()
print('ANIML_DEER_USD env var:', repr(raw))
if raw:
    p = Path(raw)
    print('  file exists:', p.is_file())

repo = Path(r'C:\Users\nars7\Downloads\aniML_world')
print('\nFallback search order:')
for rel in (
    Path('deer_world.usd'),
    Path('deer_fixed_robot.usd'),
    Path('deer_standing_robot.usd'),
    Path('deer_robot_clean.usd'),
    Path('deer_robot.usd'),
    Path('assets') / 'deer' / 'deer.usd',
    Path('deer1.usdc'),
    Path('deer1.usd'),
    Path('deer.usdc'),
):
    candidate = (repo / rel).resolve()
    exists = candidate.is_file()
    mark = 'FOUND' if exists else '     '
    print(f'  [{mark}] {rel}')
    if exists:
        print(f'         -> {candidate}')
        import os as os2
        stat = candidate.stat()
        from datetime import datetime
        mtime = datetime.fromtimestamp(stat.st_mtime)
        print(f'         -> size={stat.st_size} bytes, modified={mtime}')
        print('  ^^ THIS IS WHAT GETS USED ^^')
        break

print()
print('All USD files at repo root:')
for f in sorted(repo.glob('*.usd')):
    stat = f.stat()
    from datetime import datetime
    mtime = datetime.fromtimestamp(stat.st_mtime)
    print(f'  {f.name:40s}  {stat.st_size:8d} bytes  {mtime.strftime("%Y-%m-%d %H:%M")}')
for f in sorted(repo.glob('*.usda')):
    print(f'  {f.name:40s}  {f.stat().st_size:8d} bytes  (usda)')
