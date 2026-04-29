# Protein Triplet Dataset Builder

Step 1 of the twilight-zone homology ML pipeline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python build_triplet_dataset.py
```

All files are downloaded to `data/` on first run and re-used on subsequent runs.
The final output is `data/training_triplets.tsv`.

## Output columns

| Column | Description |
|---|---|
| `anchor_id` | Human UniProt accession |
| `anchor_seq` | Human canonical sequence |
| `positive_id` | Yeast UniProt accession sharing ≥1 GO term with anchor |
| `positive_seq` | Yeast sequence |
| `negative_id` | Yeast UniProt accession sharing 0 GO terms with anchor |
| `negative_seq` | Yeast sequence |
| `shared_go_terms` | Semicolon-separated GO IDs shared by anchor & positive |
