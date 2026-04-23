import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = r'C:\Users\nars7\.cursor\projects\c-Users-nars7-Downloads-aniML-world\agent-transcripts\5db78368-0548-4bc1-aec5-1279d2072d44\5db78368-0548-4bc1-aec5-1279d2072d44.jsonl'
events = []
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        try:
            events.append(json.loads(line))
        except:
            pass

# Focus on the key events around when deer_world.usd was created and the orientation settled
# Events 2470-2560 seem to be the critical window
for i in range(2470, 2570):
    if i >= len(events):
        break
    ev = events[i]
    role = ev.get('role','')
    msg = ev.get('message', {})
    if isinstance(msg, dict):
        content = msg.get('content', [])
        if isinstance(content, list):
            text_parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get('type') == 'text':
                        text_parts.append(c.get('text',''))
                    elif c.get('type') == 'tool_use':
                        name = c.get('name','')
                        inp = c.get('input', {})
                        if name == 'StrReplace':
                            text_parts.append(f'[TOOL: StrReplace {inp.get("path","")[-50:]}]')
                            text_parts.append(f'  old: {str(inp.get("old_string",""))[:150]}')
                            text_parts.append(f'  new: {str(inp.get("new_string",""))[:150]}')
                        elif name == 'Shell':
                            cmd = inp.get('command','')[:200]
                            text_parts.append(f'[TOOL: Shell] {cmd}')
                        elif name == 'Write':
                            text_parts.append(f'[TOOL: Write {inp.get("path","")[-50:]}]')
                        else:
                            text_parts.append(f'[TOOL: {name}]')
            text = '\n'.join(text_parts)
        else:
            text = str(content)
    else:
        text = str(msg)
    
    if text.strip():
        print(f'--- EVENT {i} [{role.upper()}] ---')
        for line in text.split('\n')[:8]:
            print(f'  {line[:200]}')
        print()
