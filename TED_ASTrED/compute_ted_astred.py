"""
TED and ASTrED between source/pivot subtitles and German direct/indirect subtitles.

Settings (source/pivot -> German target):
  (1) ES    -> DE_Direkt (direct)     (2) ES    -> DE (indirect)
  (3) EN    -> DE_Direkt (direct)     (4) EN    -> DE (indirect)
  (5) EN_CC -> DE_Direkt (direct)     (6) EN_CC -> DE (indirect)

Input : Subtitles Data/conllu/<version>/<episode>.conllu  (Stanza UD parses, sentence-aligned)
Output: TED_ASTrED/sentence_scores.csv  (one row per sentence pair per setting)

Metrics (Vanroy et al. 2021, `astred` package, unit costs):
  - TED    : tree edit distance between UD dependency trees, nodes labelled by deprel
  - ASTrED : aligned syntactic tree edit distance (nodes labelled by `connected_repr`, i.e. TED over
             word-alignment-informed trees)
Deprel subtypes are stripped (nsubj:pass -> nsubj), as in astred's default `include_subtypes=False`.

Word alignment: awesome-align (the aligner astred uses), via astred's own `Aligner` with default settings:
mBERT (bert-base-multilingual-cased), layer 8, softmax extraction, threshold 0.001.
As a cross-check, alignments are also computed with an independent re-implementation of the same extraction
on HF transformers; agreement between both aligners is stored per sentence (`align_identical`).

Requirements: astred, apted, torch, transformers, pandas (+ scipy, seaborn, openpyxl for the report).
Run once:  python TED_ASTrED/setup_awesome_align.py   (installs awesome-align + legacy-format mBERT into .deps/)
Then:      python TED_ASTrED/compute_ted_astred.py
           python TED_ASTrED/make_figures.py
           python TED_ASTrED/build_excel.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

DEPS = Path(__file__).resolve().parent / ".deps"
if not (DEPS / "pylib" / "awesome_align").exists() and "MBERT_DIR" not in os.environ:
    sys.exit("awesome-align not set up: run  python TED_ASTrED/setup_awesome_align.py  first")
sys.path.insert(0, str(DEPS / "pylib"))  # must precede the astred import (it checks for awesome_align on import)

import logging

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from astred import AlignedSentences, Sentence, Word
from astred import Aligner as AwesomeAligner
from astred.tree import AstredConfig

logging.getLogger("astred").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
CONLLU = ROOT / "Subtitles Data" / "conllu"
OUT = Path(__file__).resolve().parent / "sentence_scores.csv"

EPISODES = ["S01E02", "S02E06", "S02E08", "S02E10", "S03E07"]
SETTINGS = [  # (id, source_version, source_label, target_version, target_label)
    (1, "es", "ES source", "DE_Direkt", "direct"),
    (2, "es", "ES source", "de", "indirect"),
    (3, "en", "EN pivot", "DE_Direkt", "direct"),
    (4, "en", "EN pivot", "de", "indirect"),
    (5, "EN_CC", "EN_CC pivot", "DE_Direkt", "direct"),
    (6, "EN_CC", "EN_CC pivot", "de", "indirect"),
]

MODEL = "bert-base-multilingual-cased"
ALIGN_LAYER = 8
THRESHOLD = 1e-3


# ---------------------------------------------------------------- CoNLL-U
def read_conllu(path):
    """Return list of sentences; each sentence = list of (form, head, deprel, upos). MWT ranges skipped."""
    sents, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        cur.append((cols[1], int(cols[6]), cols[7].split(":")[0], cols[3]))
    if cur:
        sents.append(cur)
    return sents


def is_empty(sent):
    return all(upos in ("PUNCT", "SYM") for _, _, _, upos in sent)


def to_sentence(sent):
    return Sentence([Word(id=i, text=f, head=h, deprel=d) for i, (f, h, d, _) in enumerate(sent, 1)])


# ---------------------------------------------------------------- Aligner
class Aligner:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, output_hidden_states=True).to(self.dev).eval()
        self.cache = {}

    @torch.no_grad()
    def embed(self, key, words):
        if key in self.cache:
            return self.cache[key]
        pieces = [self.tok.tokenize(w) or [self.tok.unk_token] for w in words]
        bpe2word = [i for i, p in enumerate(pieces) for _ in p]
        ids = self.tok.convert_tokens_to_ids([t for p in pieces for t in p])[:510]
        bpe2word = bpe2word[: len(ids)]
        ids = torch.tensor([[self.tok.cls_token_id] + ids + [self.tok.sep_token_id]], device=self.dev)
        hid = self.model(ids).hidden_states[ALIGN_LAYER][0, 1:-1]
        self.cache[key] = (hid, bpe2word)
        return hid, bpe2word

    def align(self, skey, swords, tkey, twords):
        s, s_map = self.embed(skey, swords)
        t, t_map = self.embed(tkey, twords)
        sim = s @ t.T
        m = (torch.softmax(sim, -1) > THRESHOLD) & (torch.softmax(sim, 0) > THRESHOLD)
        pairs = {(s_map[i], t_map[j]) for i, j in m.nonzero().tolist()}
        return sorted(pairs)


class PatchedAwesomeAligner(AwesomeAligner):
    """astred 0.9.7's Aligner targets an older awesome-align whose BertForMaskedLM took (config, cls_id, sep_id).
    awesome-align 0.1.7 hard-codes CLS_ID=101 / SEP_ID=102 instead. Only model loading is changed here;
    tokenisation and alignment (`align`) are astred's unchanged code."""

    def __post_init__(self):
        from awesome_align import modeling
        from awesome_align.configuration_bert import BertConfig
        from awesome_align.tokenization_bert import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_or_path)
        assert (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id) == (modeling.CLS_ID, modeling.SEP_ID)
        assert not self.tokenizer.basic_tokenizer.do_lower_case, "mBERT-cased must not be lowercased"
        self.config = BertConfig.from_pretrained(self.model_name_or_path)
        self.model = modeling.BertForMaskedLM.from_pretrained(self.model_name_or_path, config=self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()


# ---------------------------------------------------------------- Main
def main():
    data = {(v, ep): read_conllu(CONLLU / v / f"{ep}.conllu")
            for v in {"es", "en", "EN_CC", "de", "DE_Direkt"} for ep in EPISODES}
    for ep in EPISODES:
        n = {len(data[(v, ep)]) for v in ("es", "en", "EN_CC", "de", "DE_Direkt")}
        assert len(n) == 1, f"sentence counts differ in {ep}: {n}"

    aligner = Aligner()  # re-implementation, used only as cross-check
    awesome = PatchedAwesomeAligner(model_name_or_path=os.environ.get("MBERT_DIR", str(DEPS / "mbert")))
    ted_cfg = AstredConfig(attr="deprel")
    rows = []
    for sid, src_v, src_lbl, tgt_v, tgt_lbl in SETTINGS:
        for ep in EPISODES:
            for i, (s, t) in enumerate(zip(data[(src_v, ep)], data[(tgt_v, ep)]), 1):
                row = dict(setting=sid, source=src_lbl, target=f"DE {tgt_lbl}", direction=tgt_lbl,
                           episode=ep, sent_id=i,
                           src_text=" ".join(w[0] for w in s), tgt_text=" ".join(w[0] for w in t),
                           src_len=len(s), tgt_len=len(t))
                if is_empty(s) or is_empty(t):
                    row["excluded"] = True
                    rows.append(row)
                    continue
                sw, tw = [w[0] for w in s], [w[0] for w in t]
                al = sorted(set(map(tuple, awesome.align(" ".join(sw), " ".join(tw)))))
                al_check = aligner.align((src_v, ep, i), sw, (tgt_v, ep, i), tw)
                astred = AlignedSentences(to_sentence(s), to_sentence(t), word_aligns=al)
                ted = AlignedSentences(to_sentence(s), to_sentence(t), word_aligns=al, ted_config=ted_cfg)
                row.update(excluded=False, TED=ted.ted, ASTrED=astred.ted,
                           word_cross=astred.word_cross, seq_cross=astred.seq_cross,
                           sacr_cross=astred.sacr_cross, n_aligns=len(al),
                           aligns=" ".join(f"{a}-{b}" for a, b in al),
                           align_identical=(al == al_check))
                rows.append(row)
            print(f"setting {sid} {src_v}->{tgt_v} {ep} done", flush=True)

    df = pd.DataFrame(rows)
    nodes = df["src_len"] + df["tgt_len"]
    df["TED_norm"] = df["TED"] / nodes
    df["ASTrED_norm"] = df["ASTrED"] / nodes
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"saved {OUT} ({len(df)} rows, {int(df['excluded'].sum())} excluded)")
    v = df[~df["excluded"]]
    print(f"awesome-align vs re-implementation identical: {v['align_identical'].mean():.4%} of {len(v)} pairs")


if __name__ == "__main__":
    main()
