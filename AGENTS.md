# AGENTS.md

This repository uses git for fine-grained experiment tracking. Follow these rules for all future work.

## 1. Core Principle

- One commit should represent one clearly scoped experimental change.
- Do not mix unrelated changes in the same commit.
- Every method change must be independently comparable against its parent commit.
- If a change cannot be explained in one sentence, split it into multiple commits.

## 2. Commit Granularity

Prefer commits at this level:

- Data flow or split logic change
- One feature group change
- One target transformation change
- One model family change
- One fusion or postprocess change
- One evaluation or logging change

Avoid these in a single commit:

- Feature engineering + model tuning + logging refactor
- Multiple competing methods in one patch
- Experimental code plus unrelated documentation cleanup

## 3. Branch and Commit Workflow

- Keep `main` or the default branch clean and reproducible.
- Use short-lived branches only when a change is large enough to need isolation.
- Merge back only after the branch has a working run and clear result summary.
- Commit in the order: code change, run experiment, record result, then commit result summary if needed.

## 4. Naming Rules

Use commit messages that make comparison easy.

Suggested format:

```text
feat: add regime feature gating
fix: prevent target leakage in common component
exp: compare tree vs ridge on interval residual
docs: record V3.1 experiment results
refactor: isolate feature builder for scale experiments
```

For experiment commits, include the method name or experiment id in the message.

## 5. Experiment Tracking

Each experiment commit should preserve the following information:

- `experiment_id`
- `date`
- `feature_set`
- `target_type`
- `model_type`
- `postprocess_type`
- `split_config`
- `seed`
- `train_corr`
- `val_corr`
- `train_mse`
- `val_mse`
- `train_r2`
- `val_r2`
- `daily_corr_mean`
- `daily_corr_std`
- `runtime_sec`
- short `notes`

Record results in the experiment log markdown or CSV before or together with the commit.

## 6. Comparison Rule

When comparing methods:

- Change only one major variable at a time.
- Keep the same split, seed, and evaluation metric set.
- If two methods differ in more than one place, they are not directly comparable.
- Preserve the parent commit hash or experiment id in the log.

## 7. Preferred Version Control Pattern

For a new idea, use this sequence:

1. Create a minimal implementation commit.
2. Run the smallest valid experiment.
3. Commit the result table or result note.
4. If the idea is promising, extend it in a second commit.
5. If the idea fails, keep the failure commit and record why it failed.

This preserves the history of both successful and unsuccessful methods.

## 8. Reproducibility Rule

Before a commit is considered valid:

- The code must run from `python meow.py`.
- The split must be explicit.
- The seed must be fixed.
- The result must be reproducible from the recorded configuration.
- Any optional dependency must have a fallback or a clear note that it is optional.

## 9. Data and Artifact Rules

- Do not commit raw data, archives, large binary dumps, or generated artifacts unless explicitly required.
- Keep only source code, configuration, and concise result summaries under version control.
- If a plot, table, or CSV is important for comparison, keep the smallest reproducible version.

## 10. Documentation Rule

For each meaningful commit, update one of:

- `实验记录.md`
- `目前成果与计划实验思路汇报.md`
- a dedicated experiment summary file

The documentation should state:

- what changed
- why it changed
- what improved or regressed
- what will be compared next

## 11. Recommended Future Workflow

For this project, the default loop is:

1. Read the current experiment goal.
2. Change only one method.
3. Run the smallest meaningful validation.
4. Save metrics and notes.
5. Commit immediately with a focused message.
6. Compare with the parent commit.

## 12. Non-Negotiable Rules

- Do not batch several competing methods into one commit.
- Do not overwrite previous experiment history just to make the repo look clean.
- Do not skip logging because the result seems obvious.
- Do not use a commit unless the change can be compared against its parent.

