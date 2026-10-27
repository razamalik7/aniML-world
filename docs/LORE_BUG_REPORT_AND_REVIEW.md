# Lore (`git-lore-cli`) — Bug Report & Developer Experience Review

**Date:** 2026-10-18  
**Package:** `git-lore-cli` v0.1.1  
**Tested on:** Windows 10/11, Node.js v24.19.0, Git 2.47+  
**Target Repository:** `aniML-world` (`C:\Users\nars7\Projects\aniML-world`)

---

## 1. The Bug: Windows Path-Slug Mismatch in Claude Session Collector

### Summary
When running `lore status` or `lore sync` on Windows, `git-lore-cli` failed to collect any Claude Code sessions (0 sessions discovered, despite 168+ MB of transcripts present in `~/.claude/projects/`).

### Root Cause
In `dist/collectors/claude.js` (and `src/collectors/claude.ts`):

```javascript
// Claude Code encodes project directory names by replacing ':', '\', '/', ' ' with '-'
const encodedRepoSlug = repoRoot.replace(/[:\\/ ]+/g, '-').toLowerCase();
```

1. **Delimiter collapsing:**
   On Windows, repository root paths begin with a drive letter and backslash, e.g. `C:\Users\nars7\Projects\aniML-world`.
   - Claude Code replaces each separator character with `-` **without collapsing** consecutive separators (`replace(/[:\\/ ]/g, '-')`). Thus, `C:\` contains both `:` and `\`, producing **two dashes** (`C--`).
   - Claude Code's on-disk project directory was:
     `~/.claude/projects/C--Users-nars7-Projects-aniML-world`
   - However, `git-lore-cli` used `+` in `/[:\\/ ]+/g`, which collapsed `:\` into a **single dash** (`c-users-nars7-projects-animl-world`).
   - Consequently, `dirSlug === encodedRepoSlug || dirSlug.startsWith(encodedRepoSlug)` always evaluated to `false`.

2. **Claude Worktree Directories:**
   Claude Code creates worktrees named like:
   `C--Users-nars7-Projects-aniML-world--claude-worktrees-jolly-clarke-e00da9`
   These were also missed due to the initial prefix mismatch.

3. **File Path Secondary Verification:**
   The fallback check:
   ```javascript
   if (!matchedCwd && normalizeFsPath(filePath).indexOf(encodedRepoSlug) === -1) {
       continue;
   }
   ```
   also failed because `normalizeFsPath(filePath)` contained `c--users-...` while `encodedRepoSlug` looked for `c-users-...`.

### The Fix
Update `collectClaudeCodeSessions` in `dist/collectors/claude.js` to support both single-dash and double-dash encodings:

```javascript
// Support both single-dash and multi-dash slugs (Windows C:\ -> C--)
const slug1 = repoRoot.replace(/[:\\/ ]+/g, '-').toLowerCase();
const slug2 = repoRoot.replace(/[:\\/ ]/g, '-').toLowerCase();
const isMatchingDir = (d) =>
    d === slug1 || d.startsWith(slug1) || d === slug2 || d.startsWith(slug2);

const projectDirs = fs.readdirSync(claudeProjectsDir, { withFileTypes: true });
const candidateFiles = [];
for (const dir of projectDirs) {
    if (!dir.isDirectory()) continue;
    const dirSlug = dir.name.toLowerCase();
    if (isMatchingDir(dirSlug)) {
        const fullDir = path.join(claudeProjectsDir, dir.name);
        for (const f of fs.readdirSync(fullDir)) {
            if (f.endsWith('.jsonl')) {
                candidateFiles.push(path.join(fullDir, f));
            }
        }
    }
}
```
And in the file-path safety check:
```javascript
const normFile = normalizeFsPath(filePath);
if (!matchedCwd && !normFile.includes(slug1) && !normFile.includes(slug2)) {
    continue;
}
```

### Result After Fix
Once patched, `lore` instantly collected all 6 Claude sessions (168.66 MB raw JSONL) and compressed them into 1.62 MB with zero errors.

---

## 2. Feature Suggestion: Native Claude Project Memory Ingestion

Claude Code stores persistent project takeaways in `~/.claude/projects/<slug>/memory/*.md` (with YAML frontmatter like `name`, `description`, `type`).

Currently, `git-lore-cli` only parses `.jsonl` transcripts. Adding an automatic importer for `~/.claude/projects/<slug>/memory/*.md` directly into `takeaways.json` (as `category: decision/gotcha/architecture`) would allow Lore to automatically absorb curated takeaways written by Claude without requiring manual `lore note` commands.

---

## 3. Developer Experience Review of `lore`

### What Works Incredibly Well (Highlights)

1. **Git Ref Architecture (`refs/ai/lore`) is Brilliant:**
   Syncing memory via hidden Git refs is an elegant architectural choice.
   - It doesn't pollute the git working tree, staging area, or git status.
   - It doesn't create merge conflicts on feature branches.
   - It travels anywhere `git push` and `git pull` go without needing external cloud databases or API tokens.

2. **Aggressive Token Compression (99% Reduction):**
   Raw AI transcripts are massive. In this repo, **168.66 MB** of raw Claude JSONL was reduced to **1.62 MB** of clean user/assistant turns. The living digest injected into `.agents/rules/lore-context.md` remains compact (~950–2,500 tokens), preventing context window blowup while preserving high-signal context.

3. **Local BM25 Search (`search.db` via SQLite):**
   Having full-text BM25 search run locally over past transcripts at zero token cost is very practical. Queries like `lore search "stylized water"` returned pinpoint relevant turns from months-old chats in under 15ms.

4. **Multi-Agent Interoperability:**
   Lore seamlessly connects Claude Code, Antigravity, and Cursor. A session started in Claude Code on one branch was immediately readable and searchable from Antigravity.

5. **SSH Cryptography:**
   Encrypting blobs with standard SSH public keys (`ssh-ed25519` + AES-256-GCM) means teams can push `refs/ai/lore` to public or private remotes safely without exposing proprietary reasoning or credentials.

### Areas for Improvement

1. **Windows Path Normalization in Collectors:**
   Ensure all collectors (`claude.ts`, `cursor.ts`, `codex.ts`) account for Windows drive letters (`C:\`), backslashes, and case-insensitivity.
2. **Auto-Import Claude Memory:**
   Automatically harvest `~/.claude/projects/<slug>/memory/*.md` as takeaways during `lore sync`.
3. **CLI Feedback on Big Repos:**
   When parsing 100+ MB of JSONL files on the first run, `lore status` takes ~700ms. A lightweight spinner or progress indicator makes the CLI feel more responsive.
4. **Git Liveness Warning:**
   If `refs/ai/lore` has never been pushed to remote, `lore status` could display a quick hint: `Tip: run 'lore sync' to push encrypted context to origin`.

---

### Conclusion
`lore` solves one of the most frustrating problems in modern AI coding: context fragmentation across tools, devices, and sessions. Once the Windows path-slug bug was resolved, it performed flawlessly.
