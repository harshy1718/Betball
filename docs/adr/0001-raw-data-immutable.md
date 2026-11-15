# 0001 Raw data is immutable and hash-tracked
Status: accepted. Raw CSVs are copied to `data/raw` under season-coded names with SHA-256 in `manifest.json`. Processed tables are rebuilt, never edited. Reason: every result must be traceable to exact input bytes.
