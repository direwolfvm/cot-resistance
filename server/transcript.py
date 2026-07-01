"""Sealed conversation transcript: an append-only chain of sealed segments."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import sealer
from .sealer import Segment, VerifyResult


@dataclass
class Transcript:
    key: bytes = field(default_factory=sealer.new_session_key)
    segments: list[Segment] = field(default_factory=list)

    @property
    def head_mac(self) -> str:
        return self.segments[-1].mac if self.segments else sealer.GENESIS

    def append(self, role: str, content: str) -> Segment:
        seg = sealer.seal(self.key, role, len(self.segments), self.head_mac, content)
        self.segments.append(seg)
        return seg

    def verify(self) -> list[VerifyResult]:
        return sealer.verify_chain(self.key, self.segments)

    def render(self) -> str:
        """The authenticated text stream, as the model consumes it."""
        return "\n".join(seg.render() for seg in self.segments)

    def render_unsealed(self) -> str:
        """Legacy-style stream with unauthenticated tags (defense OFF).

        This is what a conventional chat template produces: role tags are
        plain text, indistinguishable from tags an attacker types.
        """
        return "\n".join(f"<{seg.role}>{seg.content}</{seg.role}>" for seg in self.segments)
