"""Authenticated role tags ("seals") for LLM text streams.

Defense against role-confusion prompt injection (arXiv:2603.12277):
untrusted text can *look* like a privileged role, but it can never carry
a valid seal, because seals are HMACs keyed with a per-session secret
that never appears in the text stream.

Each segment of the conversation is wrapped in a seal:

    ⟦seal role=user seq=3 mac=9f2c...⟧ hello ⟦/seal seq=3⟧

where

    mac = HMAC-SHA256(key, role | seq | prev_mac | SHA256(content))

The MACs form a hash chain: every seal commits to the seal before it,
so forging a tag, editing content, reordering, or splicing segments
between conversations all break verification.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import unicodedata
from dataclasses import dataclass, field

# Delimiters chosen from outside the ASCII range a keyboard produces,
# but that is NOT the security boundary — the MAC is. Even if a user
# types these exact characters, their segment cannot carry a valid MAC.
SEAL_OPEN = "⟦"   # ⟦
SEAL_CLOSE = "⟧"  # ⟧

MAC_BYTES = 16  # truncated HMAC-SHA256, 128-bit — plenty for a PoC
GENESIS = "genesis"

ROLES = ("system", "user", "assistant", "cot", "tool")


@dataclass
class Segment:
    role: str
    seq: int
    content: str
    content_hash: str
    prev_mac: str
    mac: str

    def header(self) -> str:
        return f"{SEAL_OPEN}seal role={self.role} seq={self.seq} mac={self.mac}{SEAL_CLOSE}"

    def footer(self) -> str:
        return f"{SEAL_OPEN}/seal seq={self.seq}{SEAL_CLOSE}"

    def render(self) -> str:
        return f"{self.header()}{self.content}{self.footer()}"


def new_session_key() -> bytes:
    return secrets.token_bytes(32)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _mac(key: bytes, role: str, seq: int, prev_mac: str, content_hash: str) -> str:
    msg = f"{role}|{seq}|{prev_mac}|{content_hash}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[: MAC_BYTES * 2]


def seal(key: bytes, role: str, seq: int, prev_mac: str, content: str) -> Segment:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    ch = _content_hash(content)
    return Segment(
        role=role,
        seq=seq,
        content=content,
        content_hash=ch,
        prev_mac=prev_mac,
        mac=_mac(key, role, seq, prev_mac, ch),
    )


@dataclass
class VerifyResult:
    seq: int
    role: str
    mac: str
    ok: bool
    reason: str = "verified"


def verify_chain(key: bytes, segments: list[Segment]) -> list[VerifyResult]:
    """Recompute the full MAC chain. Any forged, edited, reordered or
    spliced segment (and everything after it) fails."""
    results: list[VerifyResult] = []
    prev = GENESIS
    for i, seg in enumerate(segments):
        expected = _mac(key, seg.role, i, prev, _content_hash(seg.content))
        ok = seg.seq == i and hmac.compare_digest(expected, seg.mac) and seg.prev_mac == prev
        reason = "verified"
        if seg.seq != i:
            reason = f"sequence mismatch (claimed {seg.seq}, position {i})"
        elif seg.prev_mac != prev:
            reason = "chain break: prev_mac does not match prior segment"
        elif not hmac.compare_digest(expected, seg.mac):
            reason = "MAC invalid: content or role tag was forged or tampered"
        results.append(VerifyResult(seq=i, role=seg.role, mac=seg.mac, ok=ok, reason=reason))
        prev = seg.mac  # chain continues over the *claimed* mac
    return results


# ---------------------------------------------------------------------------
# Sanitization of untrusted input
# ---------------------------------------------------------------------------
# Untrusted content is neutralized before sealing, so tag-lookalikes can
# never be parsed as structure. This is defense-in-depth: even without
# sanitization, a forged tag carries no valid MAC.

# Patterns that imitate role structure across common chat templates.
TAG_PATTERNS: list[tuple[str, str]] = [
    (rf"[{SEAL_OPEN}{SEAL_CLOSE}]", "seal delimiter lookalike"),
    (r"<\|[^<>\n]{1,64}\|>", "chatml-style special token"),
    (r"(?i)</?\s*(system|user|assistant|developer|tool|think|thinking|cot|reasoning|instructions?)\b[^<>]{0,64}>", "xml-style role tag"),
    (r"(?i)\[/?\s*(INST|SYSTEM|SYS)\s*\]", "llama-style role tag"),
    (r"<<\s*/?\s*SYS\s*>>", "llama-style system tag"),
]

# Plain-text role declarations hijack perception too (paper §5.2). They are
# natural language, so we flag rather than rewrite them.
DECLARATION_PATTERNS: list[tuple[str, str]] = [
    (r"(?im)^\s*(system|assistant|developer|tool( output)?)\s*:", "plain-text role declaration"),
    (r"(?i)\bthe following (text|message) is from the (system|assistant|developer|admin)\b", "claimed role provenance"),
]

_ESCAPE_MAP = str.maketrans({
    "<": "‹",        # ‹
    ">": "›",        # ›
    "|": "¦",        # ¦
    "[": "⁅",        # ⁅
    "]": "⁆",        # ⁆
    SEAL_OPEN: "⁅",
    SEAL_CLOSE: "⁆",
})


@dataclass
class SanitizeSpan:
    original: str
    replacement: str
    reason: str
    action: str  # "escaped" or "flagged"


@dataclass
class SanitizeResult:
    clean: str
    spans: list[SanitizeSpan] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return any(s.action == "escaped" for s in self.spans)


def sanitize(text: str) -> SanitizeResult:
    """Neutralize tag-lookalikes in untrusted text.

    NFKC-normalize first so homoglyph tricks (fullwidth ＜, etc.) collapse
    into the ASCII forms the patterns match.
    """
    normalized = unicodedata.normalize("NFKC", text)
    spans: list[SanitizeSpan] = []

    def escape(match: re.Match, reason: str) -> str:
        original = match.group(0)
        replacement = original.translate(_ESCAPE_MAP)
        spans.append(SanitizeSpan(original, replacement, reason, "escaped"))
        return replacement

    clean = normalized
    for pattern, reason in TAG_PATTERNS:
        clean = re.sub(pattern, lambda m, r=reason: escape(m, r), clean)

    for pattern, reason in DECLARATION_PATTERNS:
        for m in re.finditer(pattern, clean):
            spans.append(SanitizeSpan(m.group(0), m.group(0), reason, "flagged"))

    return SanitizeResult(clean=clean, spans=spans)
