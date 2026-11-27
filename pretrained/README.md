# Frozen CIMA immune-age clocks

`downstream/immune_age/seed42/clocks/` contains the female and male RNA, ATAC
and Fusion clocks. Each directory includes the fitted model, its ordered list
of 500 features and its development-set feature medians.

The CIMA immune-age workflow trains the multimodal AE, Flow and endpoint-based
DATP from the configured inputs before applying the frozen clocks. Run it from
the repository root with `bash scripts/run_immune_age_cima.sh`.
