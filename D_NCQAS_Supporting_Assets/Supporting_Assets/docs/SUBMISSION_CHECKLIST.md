# Submission checklist — revision 3

## Completed computational work

- Corrected PPO actor/entropy gradients and checked vectorized gradients numerically.
- Added optimizer diagnostics, probability bookkeeping checks, frozen controls and independent probes.
- Selected optimizer/CEM/evolution settings only on development seeds.
- Ran development budget scaling and a separate fixed final protocol.
- Used 32 independent primary seed pairs and four prospectively specified comparisons.
- Added 16-seed single-component secondary ablations.
- Compared matched molecular optimization caps and parameter-shift refinement.
- Retained solver statuses and independently checked every selected molecular gradient.
- Regenerated tables/figures and the article from the verified raw outputs.
- Included exact dependencies, source/protocol hashes, test logs, executed notebook, MIT and CFF metadata.

Consult outputs/analysis/metadata/integrity_report.json and RELEASE_REVIEW.md for
final counts and limitations. Do not present development runs as independent final evidence.

## Author and venue completion

1. Confirm the author list/order, affiliations, corresponding author, contributions and ORCIDs.
2. Confirm inherited-code rights and attribution; review the MIT release notice.
3. Select a venue appropriate to a controlled reproducibility/diagnostics study.
   This manuscript does not claim state-of-the-art superiority or a new fidelity-trap mechanism.
4. Apply the actual venue template, page limit, bibliography and anonymization rules.
5. Complete funding/conflict declarations and approve the AI-assistance disclosure.
6. Deposit the exact release in a chosen repository/archive and insert its real URL/DOI into CITATION.cff.
7. If advertising live Colab support, test it interactively; validation here executed actual cells in-process.
8. For a thesis, add institutional front matter and integrate the article with supervisor-approved chapters.

Do not add unsupported hardware, mitigation, coevolutionary VQE, global-optimality,
or universal-superiority claims. The article is anonymous and venue neutral.
