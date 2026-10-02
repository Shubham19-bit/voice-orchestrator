"""PHI redaction — keep protected health information out of logs, traces and
third-party decision calls, while keeping them debuggable.

Strategy ("redact at source"):
  * Every string that enters a trace span or log line goes through `redact()`.
  * Identifiers are replaced with typed placeholders: [NAME], [PHONE], [DOB] ...
  * Known patient identifiers (from the session's patient profile) are also replaced,
    and the patient gets a stable salted hash (`pseudonym()`) so you can still
    correlate all traces of one patient without ever storing who they are.

This is a regex baseline — good enough to demo the principle, NOT a compliance
guarantee. A production system would add an NER model (e.g. Presidio) and review.
"""
from __future__ import annotations

import hashlib
import os
import re

_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("MRN", re.compile(r"\b(?:mrn|medical record(?: number)?|patient id|member id)[\s:#-]*[a-z0-9-]{4,}\b", re.I)),
    ("DOB", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b")),
    ("DOB", re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b", re.I)),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+(?:[a-z]+\s){1,3}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct)\b", re.I)),
    ("NAME", re.compile(r"(?<=\bmy name is )[a-z]+(?: [a-z]+)?", re.I)),
    ("NAME", re.compile(r"(?<=\bthis is )[A-Z][a-z]+(?: [A-Z][a-z]+)?")),
    ("NAME", re.compile(r"\b(?:dr|doctor)\.? [A-Z][a-z]+\b")),
    ("ID", re.compile(r"\b\d{6,}\b")),  # any long digit run left over
]


class Redactor:
    def __init__(self, known_identifiers: list[str] | None = None, salt: str | None = None):
        self.salt = salt or os.environ.get("PHI_HASH_SALT", "dev-only-salt")
        self._known: list[re.Pattern] = []
        for ident in known_identifiers or []:
            for part in {ident, *ident.split()}:
                if len(part) >= 3:
                    self._known.append(re.compile(rf"\b{re.escape(part)}\b", re.I))

    def redact(self, text: str) -> str:
        if not text:
            return text
        out = text
        for pat in self._known:
            out = pat.sub("[NAME]", out)
        for label, pat in _PATTERNS:
            out = pat.sub(f"[{label}]", out)
        return out

    def pseudonym(self, identifier: str) -> str:
        """Stable, non-reversible id for correlating traces of one patient."""
        return "pt_" + hashlib.sha256((self.salt + identifier.lower()).encode()).hexdigest()[:10]


_default = Redactor()


def redact(text: str) -> str:
    return _default.redact(text)
