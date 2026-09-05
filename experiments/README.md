# Experiments

Every experiment in this repository has the same shape, and there is no second
shape. If you are adding one, copy an existing directory rather than inventing a
layout, and if you find yourself writing a runner or a log parser, stop, because
`harness.py` already is one and having two is how the old `run.py` copies drifted
apart from each other.

```
experiments/
  harness.py              the only runner and the only parser
  <name>/
    experiments.py        CONFIGS: the exact CLI args of every run
    run.py                three lines that hand CONFIGS to the harness
    plot.py               reads results/*.csv, returns a figure
    results/
      <config>.csv        one row per training step
      <figure>.png
    README.ipynb          optional: the writeup, rendering the figure inline
```

The rule that matters is that **a run's output is a CSV**. Anything that is only
in a log file, or only in prose, is a result nobody can plot, re-scale, or
compare against a later run. `experiments/archive/` is what that costs: 53 of the
55 runs there kept only their `cmd.sh`, so the numbers survive as narrative in
`results.md` and the runs themselves cannot be replotted.

## Running one

```bash
python experiments/streaming/run.py              # every config
python experiments/streaming/run.py tr_gelu      # one config by name
python experiments/streaming/plot.py             # redraw from the CSVs
```

`run.py` launches `src/train_newton.py` once per config, parses its per-step
stderr, and writes `results/<config>.csv`.

## What lands in the CSV

`train_newton.py` logs each step as a line of `key=value` pairs, and the harness
takes every pair it can read rather than a hand-picked few. So a CSV holds
whatever that run logged — `probe_loss`, `rho`, `trust_radius`, `lambda_star`,
`eig_min`/`eig_max`, and the rest — and adding a diagnostic to the training loop
makes it available to every experiment at once. Terse or non-ASCII log keys are
renamed on the way in (`t` becomes `wall_clock_s`, `|g|` becomes `grad_norm`,
`ρ` becomes `rho`); see `COLUMN_NAMES` in `harness.py`.

Older CSVs are narrower than newer ones because the run predates a column, so
`harness.load_csv` returns only the columns a file actually has, and a `plot.py`
that wants an uncommon one should check for it.

## The experiments

- `memorization/` — one fixed 32-example batch reused for the whole run, so it
  measures optimization on a fixed objective. Trust-region, Gauss-Newton and SGD
  crossed with tanh, gelu and relu.
- `streaming/` — the same model drawing a fresh minibatch every step, so it
  measures generalization. The same nine runs.
- `curvature-batch/` — whether estimating the trust region's curvature on a batch
  of its own helps, with the gradient and the accept/reject ratio left on the
  original batch.
- `archive/` — runs from the earlier Newton sweeps, interpreted in `results.md`
  and summarized in `summary-so-far.md`. Only two of them kept any data.

`queue.md` holds the hypothesis and flags for each planned or completed run, and
`results.md` holds the written interpretation.
