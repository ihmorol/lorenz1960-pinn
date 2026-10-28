## Summary

- Deliver the final ICEEICT manuscript as a separate, anonymous six-page A4 project, with its PDF, text export, complete evidence archive and clean upload ZIP.
- Correct source-grounded protocol descriptions, invariant/statistical definitions, reference metadata and unsupported wording while preserving the authors' voice and verified numerical results.
- Regenerate all three affected scientific figures with readable embedded Times New Roman labels and archived data, resolving the previous figure-typography finding.
- Include exact text changes, before/after figure evidence, all 24 requirement verdicts and an independent review bound to the actual PDF.
- Add a dependency-free CI check that verifies the frozen artifacts and both archives against the recorded audit.

Refs: #39. This PR targets `main` from `dev/iceeict2027-submission`. It contains the final manuscript and its verification tools. Older training-code changes and the large historical experiment import remain on `dev/iceeict2027-revision`; they are not included in this PR.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Refactor / chore
- [x] Docs / config

Current published branch head: [`e46436a8772268f4c54abadc70570e3d2250c1a6`](https://github.com/ihmorol/lorenz1960-pinn/commit/e46436a8772268f4c54abadc70570e3d2250c1a6), with passing push CI. At the last API check, this PR still reported its preceding head; reviewers should verify the PR head has synchronized before approving.

## Restored method diagram

The authors rejected the compact six-box diagram because it obscured their original plan. This revision restores the original nine-stage branching structure and uses the available page space. The diagram grows from 225 to 543 PDF points high. Only its float changes from `[t]` to `[!t]` in the manuscript source; no prose or results are removed. The PDF remains six pages.

Review `review/method-original-vs-restored.png`, `review/method-restoration.md` and page 3 of the PDF. The independent audit checked the restored network blocks, decision labels, return paths and printed font sizes. Numerical figures are byte-identical to the preceding PR revision.

## Detailed changes

### Manuscript and scientific reporting

The supplied Overleaf paper was preserved before revision. Earlier manuscript folders and the original ZIPs remain on the historical revision branch. The authoritative deliverable in this PR is `writings/iceeict2027-final/main.tex`, with its compiled `main.pdf`; other paper versions should not be substituted during review.

The writing uses local corrections rather than a broad stylistic rewrite:

1. Keep the motivation technical: finding a workable training structure for the studied Lorenz-1960 initial-value problem. Do not expand into untested applications or claim a globally optimized architecture.
2. Describe DOP853 as the numerical reference. Remove the unsupported independent RK4 cross-check claim.
3. Correct the reported seed-0 snapshot interval to 2,000. Distinguish Adam updates, L-BFGS closure evaluations, stage caps and completed high-error runs.
4. State the actual run groups: candidate seeds 0-4; no-causal/no-warm/shared-network seeds 1-3; causal-single and full-interval seed 0. The paper describes its 16 completed runs.
5. Define both conserved quantities using the producing evaluator's orthonormal null-space basis of the coefficient vector. Define drift as the maximum absolute change divided by the initial invariant magnitude; do not infer a second measured invariant from the older one-invariant helper.
6. Describe the exploratory exact one-sided Mann-Whitney comparison: no-causal seeds 1-3 against candidate seeds 0-4, alternative `greater`, U=11 and p=0.196 after rounding. Preserve the small-group/shared-seed caveat and avoid treating nonsignificance as equivalence.
7. Replace unsupported necessity, measured-forgetting and universal-success claims with observed final-error and training-cost comparisons under the saved configurations. The positive conclusion leads with demonstrated accuracy and matched-seed warm-start findings.
8. Correct reference names/titles/metadata, including Revanth Mattey, Franz M. Rohrhofer and Clemens Goessnitzer, and explain the retention constraint that distinguishes bc-PINN from the saved shared-network control. All 13 cited records were checked in the preceding reference audit.
9. Correct local grammar, acronym definitions, citation spacing and captions. Keep the existing non-intellectual LLM-use acknowledgment and author responsibility statement.
10. Correct one overstatement found during the final figure pass: the code certifies updated causal weights before accepting a stage transition, rather than unconditionally certifying them after every Adam update. The figure caption now uses window indices k=0,...,26 consistently with the equations and algorithm.

The existing nine numbered equations, algorithm contents, tables, bibliography entries and IEEEtran class/style are unchanged relative to the preceding reviewed six-page manuscript. No verified result value was changed. The previous factual and reference corrections are retained.

### Figure regeneration

The latest author instruction explicitly allows regeneration and geometry changes from existing results. It supersedes the earlier exact-geometry fallback that left the figure-font finding unresolved.

| Figure | Final change | Data/protocol evidence |
| --- | --- | --- |
| Full-interval result | Separate solution/reference, signed component errors and residual-loss history into readable panels; enlarge labels and legend. | All 13,265 evaluation times and 46,251 saved loss records retained. Recomputed combined RMSE 2.2370605122285716 agrees with the archived summary. |
| Windowed solution/reference | Enlarge the three state comparisons and signed-error panels; distinguish prediction/reference line styles and state the 10^-4 error scale. | All 13,265 evaluation times retained. Recomputed combined RMSE 0.00006966965305660092 agrees with the archived summary. Largest error time 5.02117 supports the existing prose. |
| Training pipeline | Restore the original detailed nine-stage structure with network subblocks, three decision diamonds, formulas and stage/window feedback paths. Use 543 pt height at the existing column width and consistent zero-based window indices. | Preserve 27 separate 4x60 networks, hard initial conditions, causal stages and detached weights, verified threshold/cap, unweighted L-BFGS, endpoint/parameter handoff and DOP853 evaluation. |

Every regenerated figure uses embedded Times New Roman text, minimum 8.5 pt after placement in the actual manuscript PDF. All are vector PDFs with PNG previews. The original pipeline source was recovered; the compact numeric generators are transparently reconstructed from archived checkpoint inference and saved data, not described as recovered historical source. No smoothing, invented points, new training or new experiment was introduced.

### Deliverables and review order

| Artifact | Purpose |
| --- | --- |
| `writings/iceeict2027-manuscript.zip` | Clean 11-file upload project. `main.tex` is at its root; includes bibliography/class files, three vector figures, anonymous PDF and text export. No audit reports, run archives or figure-generation datasets. |
| `writings/iceeict2027-final/main.pdf` | Frozen anonymous submission candidate: six A4 pages, 2,725,485 bytes. |
| `writings/iceeict2027-final/main.tex` | Authoritative editable manuscript. |
| `writings/iceeict2027-final.zip` | Complete project and review/evidence package, including generators, plotting arrays and audit records. |
| `writings/iceeict2027-final/review/manuscript.diff` | Exact final text/caption changes from the preceding reviewed manuscript. |
| `writings/iceeict2027-final/review/figure-comparison.png` | Before/after visual comparison of all three affected figures. |
| `writings/iceeict2027-final/review/changes.md` | Final change log, preserved elements and PDF hash. |
| `writings/iceeict2027-final/review/requirements-verdict.md` | All 24 submission requirements, evidence and author/system-dependent statuses. |
| `writings/iceeict2027-final/review/independent-review.md` | Independent technical file/figure review and its explicit limits. |
| `writings/iceeict2027-final/figure-sources/` | Reproducible numeric/pipeline generators, self-contained plotting arrays, saved loss history and source/checkpoint hashes. |
| `docs/reproducibility/iceeict-pr-checks.json` | Fresh upload-package rebuild and positive/negative artifact-check evidence. |

The full earlier scientific/writing audit and broader source diff are preserved on the historical branch: [claim-to-evidence mapping](https://github.com/ihmorol/lorenz1960-pinn/blob/00faf052925cc24e6a621c18a33c1117f6313f59/docs/reproducibility/iceeict-revision-20261018/claim-to-evidence.md), [reference audit](https://github.com/ihmorol/lorenz1960-pinn/blob/00faf052925cc24e6a621c18a33c1117f6313f59/docs/reproducibility/iceeict-revision-20261018/reference-audit.md), [broader manuscript diff](https://github.com/ihmorol/lorenz1960-pinn/blob/00faf052925cc24e6a621c18a33c1117f6313f59/docs/reproducibility/iceeict-revision-20261018/source-changes.diff). The final directory in this PR is authoritative over older fallback versions linked in issue #39.

## Testing

### Fresh checks

- Extracted the clean upload ZIP into a new folder and ran `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from that folder. It produced six pages. Text and all rendered page pixels match the reviewed PDF; PDF byte identity after recompilation is not required because creation metadata can change.
- Independently inspected every final PDF page and all three figures. No clipped labels, overlapping content, missing glyphs, overfull boxes or unresolved references/citations were found.
- Verified all 27 font programs are embedded, zero Type 3 fonts, A4 pages, PDF size below 3,000,000 bytes, and minimum placed figure text of 8.5 pt in Times New Roman.
- Independently recomputed both plotted RMSEs; verified NPZ/CSV hashes, identical reference/time grids, full evaluation-point retention and saved-loss record count.
- Compared all nine numbered equations, the algorithm, inline tables, separate ablation table, bibliography and IEEEtran files with the preceding manuscript. Their existing contents are preserved.
- Ran `python scripts/verify_manuscript_artifacts.py` successfully. It checks all 50 manifest-listed frozen files, archive membership/CRC/hashes, checkout content, audit/PDF binding and the clean 11-file upload ZIP. Text checkout comparison handles Git's CRLF/LF conversion; frozen ZIP hashes stay byte-exact.
- Ran a negative check in a temporary project copy: modifying the manuscript PDF is rejected by the verifier.
- Passing GitHub CI at this PR's source head: [Manuscript artifacts](https://github.com/ihmorol/lorenz1960-pinn/actions/runs/39131546729).

CI protects delivery integrity. It does not rebuild TeX, rerun model training, authenticate author attestations or repeat the scientific study. Runtime training code is unchanged by this focused PR; no new training results or claim of fresh training-test coverage is made.

### Independent review binding

- Reviewed paper source commit: `70618fe6ca7766abe0bff5ee6086b69708a5bb9d`.
- Exact final PDF SHA-256: `2716949a56776b846f80cdca9e41e599c68d627c1e1afd52cd9fc975ffd321c4`.
- Clean upload ZIP SHA-256: `ba12992dbeaf551dc27dba2530a5315d462c944d065d0bfdafeba5d0ede9024b`.
- Independent verdict: PASS for the bounded technical file/figure review; 0 critical, 0 major, 0 minor findings. Previous M7 is resolved. The source/PDF are frozen at the restored-diagram commit; the later audit/packaging commit is separately validated.

### Suggested parallel verification

1. **Writing and claims reviewer:** inspect `main.tex`, the broader/final text diffs and the claim-to-evidence record. Confirm the team voice, exact seed/protocol/statistical descriptions and scoped conclusions. Report unsupported statements with precise file/line evidence; do not introduce new results or rewrite the paper broadly.
2. **Figure/data reviewer:** inspect `figure-sources/` and the three vector PDFs. Recompute the saved-array RMSEs, check scales/legends/fonts, and compare against the preserved figures. If regenerating, work in a scratch copy so audited outputs remain frozen. NumPy, Matplotlib and an installed Times New Roman font are needed; pipeline verification also uses PyMuPDF. Font files are not redistributed.
3. **Build/layout reviewer:** extract the clean upload ZIP into an empty folder, build `main.tex`, inspect all six pages and compare text/rendering against the frozen PDF. Run the artifact verifier on the PR checkout. Check anonymity, embedding, page size, file size, figures, citations and references.
4. **Authors/submission coordinator:** review the exact final PDF, confirm author approval and responsibility, check CMT metadata/account/presenter obligations, and decide whether the revised PDF needs a fresh similarity report. Keep private confirmations outside the public PR.

Each reviewer should record the full PR head SHA, final PDF hash, checks actually run, Pass/Fail verdict, severity-ranked findings and remaining gaps. Reviewers can work independently; proposed edits must be reconciled into a new audited PDF before replacing this frozen candidate.

## Breaking Changes

None. No runtime API, model implementation, training configuration or archived experiment result is modified by this PR.

## Deployment Notes

None. This PR does not submit to CMT, merge itself or alter the separately published reproduction repository. For a LaTeX editor, import the clean manuscript ZIP and select `main.tex`. For conference submission, use the anonymous `main.pdf`; the 3 MB limit is checked on that PDF, not on either larger ZIP archive.

Final author approval, CMT fields/presenter/cutoff checks and refreshed screening remain outside the automated file-readiness verdict. The author previously reported 7% similarity and 0% on an AI detector; reports were not independently inspected and this revised PDF was not rescanned. Scientific scope remains one orbit, small seed groups and comparisons with differing settings. No acceptance, indexing or broader generalization is certified by this PR.

## Reviewer Checklist

- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] Tests present and passing
- [ ] No sensitive data exposed
- [ ] Docs updated if needed
- [ ] Final writing retains the authors' voice and supported claims
- [ ] Figure data, labels, font sizes and reference/error scales verified
- [ ] Fresh upload-project build remains six A4 pages and below 3 MB
- [ ] Review verdict binds to the exact source SHA and PDF hash
- [ ] Author/CMT/screening gates remain explicit
