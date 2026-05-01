import pandas as pd

cols = ["qseqid","sseqid","pident","length","evalue","bitscore"]
fwd = pd.read_csv("human_vs_yeast.tsv", sep="\t", names=cols)
rev = pd.read_csv("yeast_vs_human.tsv", sep="\t", names=cols)

fwd_best = fwd.sort_values("bitscore", ascending=False).groupby("qseqid").first().reset_index()
rev_best = rev.sort_values("bitscore", ascending=False).groupby("qseqid").first().reset_index()

rbh = fwd_best.merge(
    rev_best[["qseqid","sseqid"]].rename(columns={"qseqid":"sseqid","sseqid":"qseqid"}),
    on=["qseqid","sseqid"]
)

print(f"Total RBH pairs: {len(rbh)}")
rbh.to_csv("rbh_pairs.tsv", sep="\t", index=False)
