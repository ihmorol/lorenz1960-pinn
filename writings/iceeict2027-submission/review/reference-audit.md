# ICEEICT Reference Audit

Date: 2026-10-18
Scope: references actually cited by `writings/iceeict2027-submission/main.tex`
Style: IEEE, numbered in order of first citation

## Result

- Cited keys: 13
- Cited keys missing from `references.bib`: 0
- Bibliography entries not cited by the manuscript: 2 (`hao2024pinnacle`, `mcgreivy2024`)
- Material metadata errors corrected: 2 entries (`mattey2022bcpinn`, `rohrhofer2023`)
- Exact-title and capitalization corrections: 2 entries (`matthews2024pinnde`, `wang2022ntk`)
- DOI metadata added without adding printed URL fields: 3 entries

The two uncited entries were retained because this revision is limited to cited references and their presence does not add items to the IEEE bibliography output. No reference was added to increase citation count, and no cited source was removed.

## Corrections applied

| Key | Previous problem | Correction | Primary evidence |
|---|---|---|---|
| `mattey2022bcpinn` | First author was incorrectly recorded as Rathan Kumar Mattey, and the entry used the arXiv preprint title rather than the final journal title. | Corrected the author to Revanth Mattey, used the final journal title, and added DOI `10.1016/j.cma.2021.114474`. | [Elsevier final article](https://www.sciencedirect.com/science/article/pii/S0045782521006939) |
| `rohrhofer2023` | First author was incorrectly recorded as Jonas R. Rohrhofer and Clemens Gößnitzer was omitted. | Corrected the four-author list to Franz M. Rohrhofer, Stefan Posch, Clemens Gößnitzer, and Bernhard C. Geiger. | [TMLR paper on OpenReview](https://openreview.net/pdf?id=56cTmVrg5w) |
| `matthews2024pinnde` | BibTeX case folding produced “Pinnde” and lowercased the exact title. | Protected `PinnDE` and `Physics-Informed Neural Networks` with braces. Authors and arXiv identifier were already correct. | [arXiv record](https://arxiv.org/abs/2408.10011) |
| `wang2022ntk` | The title expanded PINNs rather than reproducing the final title; DOI was absent. | Restored the exact title and added DOI `10.1016/j.jcp.2021.110768`. | [Elsevier article](https://www.sciencedirect.com/science/article/pii/S002199912100663X) |
| `wang2021gradient` | Verified metadata, but DOI was absent. | Added DOI `10.1137/20M1318043`. | [SIAM journal record](https://epubs.siam.org/doi/abs/10.1137/20M1318043) |

The selected `IEEEtran.bst` does not print DOI fields in the generated bibliography, so these DOI additions do not consume manuscript space. URL fields were not added because this style does print them and the paper has a strict six-page limit.

## Per-entry verification

| Key | Primary source | Metadata check | Claim-support check | Action |
|---|---|---|---|---|
| `lorenz1960` | [Wiley, Tellus](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.2153-3490.1960.tb01307.x) | Lorenz; title; *Tellus* 12(3), 243–254 (1960); DOI verified. | Supports deriving a three-equation nonlinear system from a reduced flow model. | No change. |
| `raissi2019` | [Elsevier, JCP](https://www.sciencedirect.com/science/article/pii/S0021999118307125) | Raissi, Perdikaris, Karniadakis; *JCP* 378, 686–707 (2019); DOI verified. | Supports residual-based PINNs and forward/inverse differential-equation problems. | No change. |
| `wang2024causal` | [Elsevier, CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524000690) | Wang, Sankaran, Perdikaris; *CMAME* 421, 116813 (2024); DOI verified. | Supports time-dependent residual weighting and the causal-training motivation. It does not by itself prove that the present architecture is necessary. | No change. |
| `krishnapriyan2021` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html) | Five authors; NeurIPS 34 (2021) verified. | Supports PINN failure modes and a sequence-to-sequence strategy. “Short time segments help” is defensible when described as their tested sequence-to-sequence approach. | No change. |
| `lagaris1998` | [IEEE Xplore](https://ieeexplore.ieee.org/document/712178) | Lagaris, Likas, Fotiadis; *IEEE TNN* 9(5), 987–1000 (1998); DOI verified. | Directly supports trial solutions that satisfy initial or boundary conditions by construction. | No change. |
| `matthews2024pinnde` | [arXiv paper](https://arxiv.org/pdf/2408.10011) | Matthews and Alex Bihlo; arXiv:2408.10011 verified. | Section 4.2 explicitly uses the Lorenz–1960 initial state `(0.5, 0.75, 1.0)`. It studies a shorter interval and does not support the manuscript's final time. | Protected exact title capitalization. |
| `penwarden2023` | [Elsevier, JCP](https://www.sciencedirect.com/science/article/pii/S0021999123005594) | Five authors; *JCP* 493, 112464 (2023); DOI verified. | Supports causal sweeping, temporal decomposition, transfer-based initialization, and window propagation. | No change. |
| `wang2021gradient` | [SIAM JSC](https://epubs.siam.org/doi/abs/10.1137/20M1318043) | Wang, Teng, Perdikaris; 43(5), A3055–A3081 (2021); DOI verified. | Supports gradient imbalance and stiffness-related training failure. | Added DOI metadata. |
| `wang2022ntk` | [Elsevier, JCP](https://www.sciencedirect.com/science/article/pii/S002199912100663X) | Wang, Yu, Perdikaris; 449, 110768 (2022); DOI verified. | Supports unequal convergence rates among PINN loss components from an NTK analysis. | Corrected exact title; added DOI. |
| `rohrhofer2023` | [TMLR/OpenReview](https://openreview.net/forum?id=56cTmVrg5w) | Four authors and TMLR publication verified. | Supports the claim that fixed points can play a key role in PINN optimization on dynamical systems. The manuscript should avoid implying that every wrong trajectory is specifically caused by a fixed point. | Corrected author list. |
| `rathore2024` | [PMLR proceedings](https://proceedings.mlr.press/v235/rathore24a.html) | Five authors; ICML 2024, PMLR 235, 42159–42191 verified. | Directly supports the comparison in which Adam followed by L-BFGS performs better than Adam alone on the studied PINN benchmarks. | No change to avoid lengthening the printed entry. |
| `mattey2022bcpinn` | [Elsevier, CMAME](https://www.sciencedirect.com/science/article/pii/S0045782521006939) | Revanth Mattey and Susanta Ghosh; 390, 114474 (2022); DOI verified. | Supports sequential retraining of one network while enforcing previously obtained segment solutions. This backward-compatibility term distinguishes bc-PINN from a sequential shared-network control that lacks retention. | Corrected author, final title, and DOI. |
| `virtanen2020` | [Nature Methods](https://www.nature.com/articles/s41592-019-0686-2) | Virtanen et al.; 17(3), 261–272 (2020); DOI verified. | Supports identifying SciPy as the software source. DOP853-specific tolerances and grid size remain properties of this study's recorded configuration, not claims attributed to the SciPy paper. | No change. |

## Related-work wording guidance

These are suggestions for the manuscript owner; this audit does not edit `main.tex`.

1. Keep the fixed-point statement bounded: “For dynamical systems, fixed points can influence PINN optimization and attract training toward an incorrect solution.” This matches Rohrhofer et al. without assigning every observed error to that mechanism.
2. Describe bc-PINN precisely: “Mattey and Ghosh retrain one network over successive time segments while constraining it to retain solutions from earlier segments.” This makes the difference from the paper's shared-network control explicit.
3. Describe Krishnapriyan et al. as a sequence-to-sequence strategy rather than a generic proof that all short-window training succeeds.
4. Keep the Rathore statement restricted to the benchmarks evaluated in that paper: Adam followed by L-BFGS outperformed Adam alone in their study.
5. Use Wang et al. (2024) to motivate causal weighting, but base claims about the present model's accuracy or necessity on this paper's own saved comparisons.

## Remaining limits

- This audit verifies existence, metadata, and the specific descriptions used in the manuscript. It does not certify that the 13 references exhaust all relevant literature.
- Retraction status was not independently checked against a complete Retraction Watch database export. No retraction or expression-of-concern notice appeared on the primary records inspected.
- The two uncited bibliography entries were not included in the 13-entry primary-source audit.
