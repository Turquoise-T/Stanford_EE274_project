"""
Native (table-header) LZ77+tANS codec using `tans_lz77_coder.py` streams.

This exercises:
- per-stream table_log selection (inside LZ77TANSStreamsEncoder)
- match_length tANS gating (inside LZ77TANSStreamsEncoder)
- frequency-table reuse across blocks (stateful encoder/decoder instances)
- optional literal modeling (`literal_model="class4"`)
"""

from typing import List, Optional

from scl.compressors.lz77 import (
    DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
    DEFAULT_MIN_MATCH_LEN,
    LZ77Decoder,
    LZ77Encoder,
)
from scl.compressors.tans_lz77_coder import LZ77TANSStreamsDecoder, LZ77TANSStreamsEncoder


class LZ77EncoderTANSNative(LZ77Encoder):
    def __init__(
        self,
        min_match_length: int = DEFAULT_MIN_MATCH_LEN,
        max_num_matches_considered: int = DEFAULT_MAX_NUM_MATCHES_CONSIDERED,
        initial_window: List[int] = None,
        *,
        table_log: int = 10,
        match_length_tans_threshold: int = 50,
        literal_model: str = "flat",
        reuse_tables: bool = True,
        reuse_min_samples: int = 256,
        reuse_max_rel_l1: float = 0.05,
    ):
        super().__init__(
            min_match_length=min_match_length,
            max_num_matches_considered=max_num_matches_considered,
            initial_window=initial_window,
        )
        self.streams_encoder = LZ77TANSStreamsEncoder(
            table_log=table_log,
            match_length_tans_threshold=match_length_tans_threshold,
            literal_model=literal_model,
            reuse_tables=reuse_tables,
            reuse_min_samples=reuse_min_samples,
            reuse_max_rel_l1=reuse_max_rel_l1,
        )


class LZ77DecoderTANSNative(LZ77Decoder):
    def __init__(self, initial_window: Optional[List[int]] = None, *, table_log: int = 10):
        super().__init__(initial_window=initial_window)
        # `table_log` is kept for API symmetry; per-stream logs are in the block header.
        self.streams_decoder = LZ77TANSStreamsDecoder(table_log=table_log)


