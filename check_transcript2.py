import json
path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
print(f'Total lines: {len(lines)}')
for i, line in enumerate(lines[:5]):
    try:
        ev = json.loads(line)
        print(f'Line {i}: keys={list(ev.keys())}')
        role = ev.get('role', '?')
        content = ev.get('content', '')
        print(f'  role={role}, content type={type(content).__name__}')
        if isinstance(content, str):
            print(f'  content[:300]: {content[:300]}')
        elif isinstance(content, list):
            print(f'  content is list of {len(content)}')
            for item in content[:2]:
                print(f'    item: {str(item)[:200]}')
    except Exception as ex:
        print(f'Line {i}: parse error {ex}')
    print()

# Check last event
print('=== LAST EVENT ===')
try:
    ev = json.loads(lines[-1])
    print(f'keys={list(ev.keys())}')
    print(f'role={ev.get("role")}')
    c = ev.get('content','')
    if isinstance(c, str):
        print(c[:500])
    elif isinstance(c, list):
        for item in c[:3]:
            print(str(item)[:300])
except Exception as ex:
    print(f'parse error: {ex}')
