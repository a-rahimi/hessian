# Newton experiment results

One block per completed experiment, appended in completion order. The Interpreter agent writes each block by parsing `experiments/runs/<id>/stdout.log` and comparing against all earlier blocks in this file.

**Schema per block:** YAML frontmatter (id, run_name, commit_hash, descent_score, ranks among prior runs, hypothesis_supported boolean) followed by a one-paragraph narrative interpretation.

**Ranking metric:** `probe_loss` at the final step (lower = better). Secondary: descent slope = (probe_loss[14] - probe_loss[0]) / 14.

---
