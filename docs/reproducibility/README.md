# Paper evidence

The latest manuscript is `writings/iceeict2027-overleaf/main.tex`, imported from the author-designated Overleaf ZIP. Import commit: `8579d0b07ea768a438a4eb946e2660ae65627281`. The archive SHA-256 is `9e9534611bffc2a233d688185de0d3b73e5102622a3838f1709e95edd6b3f159`. Earlier manuscript folders are retained for history and are not authoritative.

The evidence combines the full-interval baseline, the seed-0 windowed run, and fourteen ablation runs. Fourteen final checkpoints omitted by the earlier ignore rules have been recovered. `checksums.json` records the saved configurations, summaries, model weights, loss histories, stage records and aggregate outputs without changing their values.

Run `python scripts/ablation_report.py` to audit completed histories and recompute the reported medians, ranges, update counts and exact one-sided Mann-Whitney values. Full-interval baseline results are a separate reference and are not pooled with windowed seeds. The newer stability study uses different initial states and horizons and is separate from this manuscript.

## Historical source limit

`runs/paper_ablations/environment.jsonl` records training at commit `ed43703233264b5568ec4598dedc1cf84ac0f452` with `git_dirty=true`. The dirty diff was not preserved. The numerical settings and final checkpoints are saved, and the integration includes the audited result-branch code at `babc47aab9f7cdbe8fde832cdd1e670eab1f9210`. This does not establish byte-exact recovery of the historical training source.

## Completed training source recovery

The recorded `ed43703233264b5568ec4598dedc1cf84ac0f452` commit also contains the completed shared-network implementation, corrected causal single-network settings, Colab driver and regression tests. These existing files have now been recovered in this revision and the public repository. All fourteen saved ablation configurations match the recovered driver's non-path settings, including defaults for fields absent in older records. The obsolete standalone sequential prototype is excluded from the public repository. No replacement scientific method or new full experiment was introduced. The missing dirty diff still prevents certification of the exact historical working tree.

## Conference status

The formatting draft uses A4 IEEE conference typography and an anonymous author block. The user requested preservation of all prose and exact figure geometry, so the six-page limit and small printed figure labels remain unresolved. Font embedding is repaired without changing figure content. Do not submit the draft as a fully compliant final paper. The named public reproduction repository must not be linked from the anonymous submission.

The official initial-submission requirements are at https://iceeict.mist.ac.bd/authors/paper-submission/ and the final extended deadline is 25 October 2026 at https://iceeict.mist.ac.bd/authors/important-dates/. The official date page gives no exact cutoff time.

Final author review/approval, similarity screening and authenticated CMT fields remain pending. The confirmed LLM-use categories are formatting, typesetting, language editing or proofreading; no approval claim has been fabricated.
