"""
Step 1: Data Acquisition & Ground Truth Curation
=================================================
Downloads Swiss-Prot FASTA and GOA GAF files for H. sapiens and S. cerevisiae,
cleans sequences, curates experimentally-evidenced GO annotations, and builds
a supervised triplet dataset (Anchor, Positive, Negative) for Triplet Margin
Loss training.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import requests
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

URLS: Dict[str, str] = {
    "human_fasta": (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?format=fasta&query=%28reviewed%3Atrue%29+AND+%28taxonomy_id%3A9606%29"
    ),
    "yeast_fasta": (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?format=fasta&query=%28reviewed%3Atrue%29+AND+%28taxonomy_id%3A559292%29"
    ),
    "human_gaf": "http://geneontology.org/gene-associations/goa_human.gaf.gz",
    # sgd.gaf uses SGD systematic IDs (S000000001), not UniProt accessions.
    # The EBI GOA yeast file uses UniProtKB accessions in column 1, matching
    # the keys produced by parse_and_clean_fasta from the Swiss-Prot FASTA.
    "yeast_gaf": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/YEAST/goa_yeast.gaf.gz",
}

FILENAMES: Dict[str, str] = {
    "human_fasta": "human_swissprot.fasta",
    "yeast_fasta": "yeast_swissprot.fasta",
    "human_gaf": "goa_human.gaf",
    "yeast_gaf": "goa_yeast.gaf",
}

# Sequences shorter than MIN_LEN or longer than MAX_LEN are discarded.
# MAX_LEN=1024 matches ESM-C's effective token budget.
MIN_LEN: int = 30
MAX_LEN: int = 1024

# Non-standard / ambiguous amino acid codes to reject.
NON_STANDARD_AA: Set[str] = set("BJOUXZ")

# Evidence codes considered experimentally derived.
# IEA (electronic), ND (no data), and all purely computational codes are excluded.
EXPERIMENTAL_EVIDENCE_CODES: Set[str] = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

# Triplet generation controls
NEGATIVES_PER_ANCHOR: int = 3   # negative yeast proteins sampled per (anchor, positive) pair
MAX_POSITIVES_PER_ANCHOR: int = 5  # cap positives per anchor to limit dataset explosion
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# 0. Data Downloading
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    """
    Stream-download *url* to *dest_path*, writing in chunks to avoid OOM.
    Retries up to 3 times on transient HTTP errors.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            log.info("  GET %s (attempt %d/3)", url, attempt)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = 100 * downloaded / total
                            print(f"\r    {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
                print()
            log.info("  Saved → %s", dest_path)
            return
        except requests.RequestException as exc:
            log.warning("  Download failed: %s", exc)
            if attempt < 3:
                wait = 5 * attempt
                log.info("  Retrying in %ds …", wait)
                time.sleep(wait)
            else:
                raise


def _decompress_gaf_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gaf.gz file to *out_path*."""
    log.info("  Decompressing %s → %s", gz_path, out_path)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        while chunk := f_in.read(1 << 20):
            f_out.write(chunk)


def download_all_data(force: bool = False) -> None:
    """
    Download all required files into DATA_DIR.

    Parameters
    ----------
    force:
        Re-download even if the file already exists on disk.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, url in URLS.items():
        final_path = DATA_DIR / FILENAMES[key]
        if final_path.exists() and not force:
            log.info("Already present, skipping: %s", final_path)
            continue

        log.info("Downloading %s …", key)

        if url.endswith(".gz"):
            gz_path = final_path.with_suffix(".gaf.gz")
            _stream_download(url, gz_path)
            _decompress_gaf_gz(gz_path, final_path)
            gz_path.unlink(missing_ok=True)  # remove compressed copy
        else:
            _stream_download(url, final_path)


# ---------------------------------------------------------------------------
# 1. Sequence Parsing & Cleaning
# ---------------------------------------------------------------------------

def _uniprot_id_from_header(header: str) -> Optional[str]:
    """
    Extract the UniProt accession from a Swiss-Prot FASTA header.

    Swiss-Prot headers look like:
        >sp|P12345|GENE_HUMAN Description …
    We return the accession (P12345).
    """
    parts = header.split("|")
    if len(parts) >= 2:
        return parts[1].strip()
    return None


def parse_and_clean_fasta(fasta_path: Path) -> Dict[str, str]:
    """
    Parse a Swiss-Prot FASTA file and apply length + standard-AA filters.

    Parameters
    ----------
    fasta_path:
        Path to the .fasta file.

    Returns
    -------
    dict mapping UniProt_ID → canonical sequence (str, uppercase).
    """
    total = kept = dropped_len = dropped_aa = dropped_id = 0
    proteins: Dict[str, str] = {}

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        total += 1
        uid = _uniprot_id_from_header(record.description)
        if uid is None:
            dropped_id += 1
            continue

        seq = str(record.seq).upper()

        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            dropped_len += 1
            continue

        if any(aa in seq for aa in NON_STANDARD_AA):
            dropped_aa += 1
            continue

        proteins[uid] = seq
        kept += 1

    log.info(
        "  %s: %d total  |  kept %d  |  dropped (length) %d  |"
        "  dropped (non-standard AA) %d  |  dropped (bad header) %d",
        fasta_path.name,
        total,
        kept,
        dropped_len,
        dropped_aa,
        dropped_id,
    )
    return proteins


# ---------------------------------------------------------------------------
# 2. GO Annotation Curation
# ---------------------------------------------------------------------------

def parse_gaf(gaf_path: Path) -> Dict[str, Set[str]]:
    """
    Parse a Gene Association File (GAF 2.x) and return experimentally
    evidenced GO annotations.

    Only annotations whose Evidence Code is in EXPERIMENTAL_EVIDENCE_CODES
    are retained.  IEA, ND, and all other computational codes are discarded.

    Parameters
    ----------
    gaf_path:
        Path to the uncompressed .gaf file.

    Returns
    -------
    dict mapping UniProt_ID → frozenset of GO term IDs (e.g. 'GO:0006915').
    """
    annotations: Dict[str, Set[str]] = {}
    total_lines = kept_lines = skipped_evidence = 0

    with open(gaf_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!") or not line:   # comment / blank
                continue

            total_lines += 1
            cols = line.split("\t")
            if len(cols) < 15:
                continue  # malformed row

            # GAF columns (0-indexed):
            #  1  → DB Object ID  (UniProt accession)
            #  4  → GO ID
            #  6  → Evidence Code
            #  3  → Qualifier (may contain 'NOT')
            qualifier = cols[3].upper()
            if "NOT" in qualifier:
                continue  # negative annotation — skip

            uid = cols[1].strip()
            go_term = cols[4].strip()
            evidence = cols[6].strip().upper()

            if evidence not in EXPERIMENTAL_EVIDENCE_CODES:
                skipped_evidence += 1
                continue

            annotations.setdefault(uid, set()).add(go_term)
            kept_lines += 1

    log.info(
        "  %s: %d data rows  |  kept %d  |  dropped (non-experimental evidence) %d",
        gaf_path.name,
        total_lines,
        kept_lines,
        skipped_evidence,
    )
    return annotations


# ---------------------------------------------------------------------------
# 3. Triplet Dataset Generation
# ---------------------------------------------------------------------------

def _build_go_index(
    yeast_go: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """
    Build an inverted index: GO_term → set of yeast UniProt IDs annotated with it.

    This lets us retrieve candidate positives for a given anchor's GO terms in O(|GO|)
    rather than scanning every yeast protein.
    """
    index: Dict[str, Set[str]] = {}
    for uid, terms in yeast_go.items():
        for term in terms:
            index.setdefault(term, set()).add(uid)
    return index


def generate_triplets(
    human_seqs: Dict[str, str],
    yeast_seqs: Dict[str, str],
    human_go: Dict[str, Set[str]],
    yeast_go: Dict[str, Set[str]],
    negatives_per_anchor: int = NEGATIVES_PER_ANCHOR,
    max_positives_per_anchor: int = MAX_POSITIVES_PER_ANCHOR,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a DataFrame of (anchor, positive, negative) triplets.

    Strategy
    --------
    For each human anchor protein with experimental GO annotations:
      1. Find all yeast proteins sharing ≥1 GO term  → candidate positives.
      2. For each positive, sample *negatives_per_anchor* yeast proteins that
         share **zero** GO terms with the anchor → negatives.
      3. Record shared GO terms for interpretability.

    Performance notes
    -----------------
    * An inverted GO→yeast_proteins index avoids an O(|human| × |yeast|) scan.
    * Yeast proteins without any experimental annotation are never candidates.

    Parameters
    ----------
    human_seqs / yeast_seqs:
        UniProt_ID → sequence dicts from Step 1.
    human_go / yeast_go:
        UniProt_ID → GO term set dicts from Step 2.
    negatives_per_anchor:
        Number of negatives to sample per (anchor, positive) pair.
    max_positives_per_anchor:
        Maximum positives paired with a single anchor (to balance dataset).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        anchor_id, anchor_seq, positive_id, positive_seq,
        negative_id, negative_seq, shared_go_terms
    """
    rng = random.Random(seed)

    # Restrict to proteins that have BOTH a sequence AND GO annotations.
    anchors = {uid for uid in human_seqs if uid in human_go}
    yeast_annotated = {uid for uid in yeast_seqs if uid in yeast_go}

    log.info(
        "Eligible anchors (human, seq+GO): %d  |  "
        "Eligible yeast (seq+GO): %d",
        len(anchors),
        len(yeast_annotated),
    )

    # Inverted index for fast positive retrieval.
    go_to_yeast = _build_go_index({uid: yeast_go[uid] for uid in yeast_annotated})

    # All yeast UniProt IDs as a pool for negative sampling.
    yeast_pool = list(yeast_annotated)

    rows = []
    n_anchors_used = 0

    for anchor_id in anchors:
        anchor_go = human_go[anchor_id]

        # --- Positives: yeast proteins sharing ≥1 GO term --------------------
        candidate_positives: Set[str] = set()
        for term in anchor_go:
            candidate_positives.update(go_to_yeast.get(term, set()))

        if not candidate_positives:
            continue  # no yeast homologue with shared GO → skip anchor

        # Cap positives to keep dataset balanced.
        sampled_positives = rng.sample(
            sorted(candidate_positives),
            min(len(candidate_positives), max_positives_per_anchor),
        )

        # --- Negatives: yeast proteins sharing ZERO GO terms -----------------
        # Build complement on-the-fly; the sets are small relative to yeast_pool.
        candidate_negatives = [
            uid for uid in yeast_pool if yeast_go[uid].isdisjoint(anchor_go)
        ]

        if len(candidate_negatives) < negatives_per_anchor:
            continue  # too few negatives — skip to avoid imbalance

        n_anchors_used += 1
        anchor_seq = human_seqs[anchor_id]

        for pos_id in sampled_positives:
            shared = anchor_go & yeast_go[pos_id]
            sampled_negs = rng.sample(candidate_negatives, negatives_per_anchor)

            for neg_id in sampled_negs:
                rows.append(
                    {
                        "anchor_id": anchor_id,
                        "anchor_seq": anchor_seq,
                        "positive_id": pos_id,
                        "positive_seq": yeast_seqs[pos_id],
                        "negative_id": neg_id,
                        "negative_seq": yeast_seqs[neg_id],
                        "shared_go_terms": ";".join(sorted(shared)),
                    }
                )

        if n_anchors_used % 100 == 0:
            log.info("  … processed %d anchors, %d triplets so far", n_anchors_used, len(rows))

    log.info(
        "Triplet generation complete: %d anchors used → %d triplets",
        n_anchors_used,
        len(rows),
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------

def save_triplets(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save the triplet DataFrame to a tab-separated file.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`generate_triplets`.
    out_path:
        Destination .tsv path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    log.info("Saved %d triplets → %s", len(df), out_path)

    # Quick sanity summary
    log.info(
        "Unique anchors: %d  |  unique positives: %d  |  unique negatives: %d",
        df["anchor_id"].nunique(),
        df["positive_id"].nunique(),
        df["negative_id"].nunique(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Protein Triplet Dataset Builder")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 0 — Download
    # ------------------------------------------------------------------
    log.info("\n[Step 0] Downloading data files …")
    download_all_data(force=False)

    # ------------------------------------------------------------------
    # Step 1 — Parse & clean sequences
    # ------------------------------------------------------------------
    log.info("\n[Step 1] Parsing and cleaning FASTA files …")
    human_seqs = parse_and_clean_fasta(DATA_DIR / FILENAMES["human_fasta"])
    yeast_seqs = parse_and_clean_fasta(DATA_DIR / FILENAMES["yeast_fasta"])

    log.info(
        "Clean sequences — Human: %d  |  Yeast: %d",
        len(human_seqs),
        len(yeast_seqs),
    )

    # ------------------------------------------------------------------
    # Step 2 — Curate GO annotations
    # ------------------------------------------------------------------
    log.info("\n[Step 2] Curating GO annotations (experimental evidence only) …")
    human_go = parse_gaf(DATA_DIR / FILENAMES["human_gaf"])
    yeast_go = parse_gaf(DATA_DIR / FILENAMES["yeast_gaf"])

    log.info(
        "Proteins with experimental GO — Human: %d  |  Yeast: %d",
        len(human_go),
        len(yeast_go),
    )

    # ------------------------------------------------------------------
    # Step 3 — Generate triplets
    # ------------------------------------------------------------------
    log.info("\n[Step 3] Generating triplets …")
    triplets_df = generate_triplets(human_seqs, yeast_seqs, human_go, yeast_go)

    if triplets_df.empty:
        log.warning("No triplets were generated — check your data and filters.")
        return

    # ------------------------------------------------------------------
    # Step 4 — Save
    # ------------------------------------------------------------------
    log.info("\n[Step 4] Saving output …")
    save_triplets(triplets_df, DATA_DIR / "training_triplets.tsv")

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
