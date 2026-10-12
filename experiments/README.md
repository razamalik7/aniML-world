# Experiments

One launch script per training run, named by experiment tag. Each script is the complete
configuration for that run: environment variables (reward weights, command ranges, AMP
dataset mix, ASE settings) plus the training command. `run_map.txt` maps each tag to its
`logs/rsl_rl/...` run directory.

Write-ups of what each series tested and found are in `docs/` (start with
`docs/LEG_NEGLECT_findings.md` and `docs/ASE_NOTES.md`).
