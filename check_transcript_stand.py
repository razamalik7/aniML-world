import json
path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
events = []
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except:
            pass

# Search for the good standing run discussion
keywords = ['stand_v2', 'animl_deer_stand', '2026-04-10', 'baseline', 'wrong num', 'wrong format',
            'warm start', 'warmstart', 'model_499', 'model_866', '866', 'stand v2',
            'good standing', 'best standing', 'april 10', 'april10']

hits = []
for i, ev in enumerate(events):
    role = ev.get('role','')
    msg = ev.get('message', {})
    if isinstance(msg, dict):
        content = msg.get('content', [])
        if isinstance(content, list):
            text = ' '.join(str(c.get('text','') if isinstance(c,dict) else c) for c in content)
        else:
            text = str(content)
    else:
        text = str(msg)
    
    for kw in keywords:
        if kw.lower() in text.lower():
            hits.append((i, role, kw, text[:600]))
            break

print(f'Found {len(hits)} hits:\n')
for i, role, kw, text in hits:
    print(f'--- event {i} role={role} kw={kw!r} ---')
    for line in text.split('\n'):
        if kw.lower() in line.lower():
            print(f'  {line[:200]}')
    print()
