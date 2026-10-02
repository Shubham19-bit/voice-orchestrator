"""mini-Jev: a tiny, domain-specific decision model with Jev-shaped answers.

Four heads, each a TF-IDF (word + character n-grams) -> logistic regression:
    eot      noul    P(patient finished their turn)
    tool     choice  which backend tool (5 options)
    urgency  score   0 routine .. 2 urgent (expected value over levels)
    overlap  choice  backchannel / interruption / other

Answers use Jev's response shapes (noul / choice+probabilities+confidence /
score+legend+probabilities+confidence) so it can stand in for Jev anywhere.
Inference is ~1 ms on a laptop CPU. No GPU, no network.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "minijev.pkl"
URGENCY_LEGEND = {"0": "routine", "1": "follow up within days", "2": "urgent"}


def with_end_markers(text: str) -> str:
    """End-of-turn depends mostly on how the text ENDS — make that explicit as tokens."""
    w = text.lower().split()
    if not w:
        return "EMPTY"
    marks = [f"END1_{w[-1]}", f"LEN_{min(len(w), 12)}"]
    if len(w) >= 2:
        marks.append(f"END2_{w[-2]}_{w[-1]}")
    if len(w) >= 3:
        marks.append(f"END3_{w[-3]}_{w[-2]}_{w[-1]}")
    marks.append(f"START_{w[0]}")
    return text.lower() + " " + " ".join(marks)


def _features(end_markers: bool = False) -> FeatureUnion:
    pre = with_end_markers if end_markers else str.lower
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=pre, token_pattern=r"[^\s]+", ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)),
    ])


def _head(end_markers: bool = False, C: float = 4.0, balanced: bool = False) -> Pipeline:
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None)
    return Pipeline([("f", _features(end_markers)), ("lr", lr)])


def _confidence(probs: list[float]) -> float:
    n = len(probs)
    return max(0.0, (n * max(probs) - 1) / (n - 1)) if n > 1 else 1.0


class MiniJev:
    def __init__(self) -> None:
        # Our synthetic data has ~2 "unfinished" examples per finished one (we cut each sentence
        # twice). Real pauses are closer to 50/50, so re-weight the classes to remove that bias —
        # otherwise the model is systematically under-confident that the patient is done.
        self.eot = _head(end_markers=True, balanced=True)
        self.tool = _head()
        self.urgency = _head()
        self.overlap = _head()
        self.meta: dict = {}

    # ------------------------------------------------------------------ training
    def fit(self, data: dict) -> "MiniJev":
        self.eot.fit(*data["eot"])
        self.tool.fit(*data["tool"])
        self.urgency.fit(*data["urgency"])
        self.overlap.fit(*data["overlap"])
        return self

    # ------------------------------------------------------------------ Jev-shaped answers
    def noul_eot(self, text: str) -> dict:
        p = self.eot.predict_proba([text or ""])[0]
        classes = list(self.eot.classes_)
        return {"type": "noul", "noul": float(p[classes.index(1)])}

    def _choice(self, head: Pipeline, text: str) -> dict:
        p = head.predict_proba([text or ""])[0]
        classes = [str(c) for c in head.classes_]
        probs = {c: float(v) for c, v in zip(classes, p)}
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "probabilities": probs, "confidence": _confidence(list(p))}

    def choice_tool(self, text: str) -> dict:
        return self._choice(self.tool, text)

    def choice_overlap(self, text: str) -> dict:
        return self._choice(self.overlap, text)

    def score_urgency(self, text: str) -> dict:
        p = self.urgency.predict_proba([text or ""])[0]
        levels = [int(c) for c in self.urgency.classes_]
        score = sum(lv * float(pv) for lv, pv in zip(levels, p))
        return {"type": "score", "score": score, "legend": URGENCY_LEGEND,
                "probabilities": {str(lv): float(pv) for lv, pv in zip(levels, p)},
                "confidence": _confidence(list(p))}

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = DEFAULT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path = DEFAULT_PATH) -> "MiniJev":
        with open(path, "rb") as f:
            return pickle.load(f)
