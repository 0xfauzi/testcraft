from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional


@dataclass
class StreamConfig:
    temperature: float = 0.2
    chunk_size: int = 128


def stream_text(full_text: str, *, on_chunk: Optional[Callable[[str], None]] = None, config: Optional[StreamConfig] = None) -> Iterator[str]:
    cfg = config or StreamConfig()
    for i in range(0, len(full_text), cfg.chunk_size):
        chunk = full_text[i : i + cfg.chunk_size]
        if on_chunk:
            on_chunk(chunk)
        yield chunk


