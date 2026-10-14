# -*- coding: utf-8 -*-
"""
vector_core/compression.py

High-performance, memory-aligned, SIMD-accelerated compression and encoding primitives
for the Vector Core Engine. Implements Dictionary Encoding, Run-Length Encoding (RLE),
and Roaring-style Dual-Container Bitmaps with zero-copy NumPy memory maps and rigorous
boundary verification.
"""

from __future__ import annotations

import mmap
import os
import struct
from typing import Tuple, Union, Optional, List

import numpy as np
import numba

# Ensure precise NumPy types
INT32_DTYPE = np.int32
INT64_DTYPE = np.int64
UINT8_DTYPE = np.uint8
UINT16_DTYPE = np.uint16
UINT32_DTYPE = np.uint32
UINT64_DTYPE = np.uint64
FLOAT32_DTYPE = np.float32
FLOAT64_DTYPE = np.float64

ALIGNMENT_BYTES = 64
DEFAULT_VECTOR_SIZE = 2048


class CompressionError(Exception):
    """Base exception class for all compression and encoding failures."""
    pass


class InvalidBufferError(CompressionError):
    """Raised when memory alignment or buffer bounds checks fail."""
    pass


class BitPackingError(CompressionError):
    """Raised when bit-packing or unpacking overflows target buffers."""
    pass


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def _pack_bits_32(values: np.ndarray, out: np.ndarray, bit_width: int) -> None:
    """
    JIT-compiled kernel for bit-packing 32-bit unsigned integers into a packed byte stream.
    """
    bit_mask = (1 << bit_width) - 1
    out_idx = 0
    bit_offset = 0
    
    for i in range(values.size):
        val = values[i] & bit_mask
        curr_byte_idx = bit_offset >> 3
        curr_bit_shift = bit_offset & 7
        
        # Write value across byte boundaries
        out[curr_byte_idx] |= np.uint8(val << curr_bit_shift)
        
        # Handle spillover into subsequent bytes
        bits_written = 8 - curr_bit_shift
        if bits_written < bit_width:
            remaining_bits = bit_width - bits_written
            out[curr_byte_idx + 1] |= np.uint8(val >> bits_written)
            if bits_written + 8 < bit_width:
                out[curr_byte_idx + 2] |= np.uint8(val >> (bits_written + 8))
                
        bit_offset += bit_width


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def _unpack_bits_32(packed: np.ndarray, out: np.ndarray, bit_width: int, count: int) -> None:
    """
    JIT-compiled kernel for unpacking bit-packed byte streams back into 32-bit integers.
    """
    bit_mask = (1 << bit_width) - 1
    bit_offset = 0
    
    for i in range(count):
        curr_byte_idx = bit_offset >> 3
        curr_bit_shift = bit_offset & 7
        
        # Read raw bytes safely
        val = np.uint32(packed[curr_byte_idx]) >> curr_bit_shift
        bits_read = 8 - curr_bit_shift
        
        if bits_read < bit_width and (curr_byte_idx + 1) < packed.size:
            val |= np.uint32(packed[curr_byte_idx + 1]) << bits_read
            bits_read += 8
            if bits_read < bit_width and (curr_byte_idx + 2) < packed.size:
                val |= np.uint32(packed[curr_byte_idx + 2]) << bits_read
                
        out[i] = val & bit_mask
        bit_offset += bit_width


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def _rle_encode_kernel(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled RLE encoder producing decoupled runs and values arrays.
    """
    n = values.size
    if n == 0:
        return np.empty(0, dtype=UINT32_DTYPE), np.empty(0, dtype=values.dtype)
        
    # Temporary upper bound buffers
    run_counts = np.empty(n, dtype=UINT32_DTYPE)
    run_values = np.empty(n, dtype=values.dtype)
    
    unique_count = 0
    current_val = values[0]
    current_run = 1
    
    for i in range(1, n):
        if values[i] == current_val:
            current_run += 1
        else:
            run_counts[unique_count] = current_run
            run_values[unique_count] = current_val
            unique_count += 1
            current_val = values[i]
            current_run = 1
            
    run_counts[unique_count] = current_run
    run_values[unique_count] = current_val
    unique_count += 1
    
    return run_counts[:unique_count], run_values[:unique_count]


@numba.njit(fastmath=True, cache=True, boundscheck=False)
def _rle_decode_kernel(run_counts: np.ndarray, run_values: np.ndarray, out: np.ndarray) -> None:
    """
    JIT-compiled RLE decoder restoring contiguous arrays from runs and values.
    """
    out_idx = 0
    for i in range(run_counts.size):
        count = run_counts[i]
        val = run_values[i]
        for j in range(count):
            out[out_idx] = val
            out_idx += 1


class AlignedBuffer:
    """
    Manages 64-byte aligned memory blocks using NumPy zero-copy memory mapping
    to maintain strict Arrow-compatible physical layouts.
    """
    def __init__(self, size_bytes: int, backing_file: Optional[str] = None) -> None:
        if size_bytes <= 0:
            raise InvalidBufferError("Buffer size must be greater than zero.")
            
        # Pad size to enforce 64-byte alignment multiples
        remainder = size_bytes % ALIGNMENT_BYTES
        if remainder != 0:
            size_bytes += (ALIGNMENT_BYTES - remainder)
            
        self.size_bytes = size_bytes
        self.backing_file = backing_file
        
        if backing_file:
            self.fd = os.open(backing_file, os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(self.fd, self.size_bytes)
            self._mmap = mmap.mmap(self.fd, self.size_bytes, access=mmap.ACCESS_WRITE)
            # Create a zero-copy numpy array view over the mmap buffer
            self.buffer = np.frombuffer(self._mmap, dtype=UINT8_DTYPE, count=self.size_bytes)
        else:
            self.fd = None
            # Allocate aligned memory block via bytearray + numpy view
            # Python's bytearray does not guarantee 64-byte alignment out-of-the-box,
            # so we over-allocate and compute aligned offsets.
            raw_data = bytearray(self.size_bytes + ALIGNMENT_BYTES)
            address = id(raw_data)
            # Force alignment via slicing
            offset = ALIGNMENT_BYTES - (address % ALIGNMENT_BYTES)
            if offset == ALIGNMENT_BYTES:
                offset = 0
            self._raw_memory = raw_data
            self.buffer = np.frombuffer(self._raw_memory, dtype=UINT8_DTYPE, offset=offset, count=self.size_bytes)
            
        # Verify alignment constraint
        if self.buffer.ctypes.data % ALIGNMENT_BYTES != 0:
            raise InvalidBufferError(f"Buffer address {self.buffer.ctypes.data} is not {ALIGNMENT_BYTES}-byte aligned.")

    def close(self) -> None:
        if self.backing_file:
            self._mmap.close()
            os.close(self.fd)
            
    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DictionaryEncoder:
    """
    Maintains distinct hash-to-integer symbol tables with fixed-width bit-packed code vectors.
    """
    def __init__(self) -> None:
        self.symbols: List[Any] = []
        self.symbol_to_id: dict = {}
        
    def encode(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes a raw columnar array into distinct dictionary symbols and integer code vectors.
        """
        if data.size == 0:
            return np.empty(0, dtype=UINT32_DTYPE), np.empty(0, dtype=data.dtype)
            
        codes = np.empty(data.size, dtype=UINT32_DTYPE)
        
        for i in range(data.size):
            val = data[i]
            # Handle NumPy types for hashing
            val_key = val.item() if hasattr(val, 'item') else val
            
            if val_key not in self.symbol_to_id:
                new_id = len(self.symbols)
                self.symbol_to_id[val_key] = new_id
                self.symbols.append(val_key)
                
            codes[i] = self.symbol_to_id[val_key]
            
        # Determine optimal bit-width for code vector packing
        max_code = len(self.symbols) - 1
        bit_width = max(1, max_code.bit_length())
        
        packed_bytes_count = (data.size * bit_width + 7) // 8
        packed_buffer = np.zeros(packed_bytes_count + ALIGNMENT_BYTES, dtype=UINT8_DTYPE)
        
        _pack_bits_32(codes, packed_buffer, bit_width)
        
        return packed_buffer[:packed_bytes_count], np.array(self.symbols)

    def decode(self, packed_buffer: np.ndarray, symbols: np.ndarray, total_count: int, bit_width: int) -> np.ndarray:
        """
        Decodes bit-packed code vectors back into original physical symbol values.
        """
        if total_count == 0:
            return np.empty(0, dtype=symbols.dtype)
            
        codes = np.empty(total_count, dtype=UINT32_DTYPE)
        _unpack_bits_32(packed_buffer, codes, bit_width, total_count)
        
        # Reconstruct original array via vectorized indexing
        return symbols[codes]


class RunLengthEncoder:
    """
    Executes decoupled run-length and value array encoding with entropy-based bit-packing thresholds.
    """
    def __init__(self, entropy_threshold: float = 0.8) -> None:
        self.entropy_threshold = entropy_threshold
        
    def _compute_entropy(self, counts: np.ndarray) -> float:
        """Calculates Shannon entropy of run lengths to drive bit-packing decisions."""
        total = np.sum(counts)
        if total == 0:
            return 0.0
        probabilities = counts.astype(FLOAT64_DTYPE) / total
        # Filter out zero probabilities to avoid log2(0)
        valid_probs = probabilities[probabilities > 0.0]
        entropy = -np.sum(valid_probs * np.log2(valid_probs))
        max_entropy = np.log2(counts.size) if counts.size > 1 else 1.0
        if max_entropy == 0:
            return 0.0
        return entropy / max_entropy

    def encode(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compresses input data into RLE run counts and run values.
        Returns packed counts, raw values, and bit-width metadata.
        """
        if data.size == 0:
            return np.empty(0, dtype=UINT8_DTYPE), np.empty(0, dtype=data.dtype), 0
            
        run_counts, run_values = _rle_encode_kernel(data)
        
        max_run = np.max(run_counts) if run_counts.size > 0 else 1
        bit_width = max(1, max_run.bit_length())
        
        # Evaluate entropy to decide whether to bit-pack run counts
        entropy_ratio = self._compute_entropy(run_counts)
        
        if entropy_ratio <= self.entropy_threshold:
            packed_bytes_count = (run_counts.size * bit_width + 7) // 8
            packed_counts = np.zeros(packed_bytes_count, dtype=UINT8_DTYPE)
            _pack_bits_32(run_counts, packed_counts, bit_width)
            # Use negative bit_width as convention for packed state, or store explicitly
            return packed_counts, run_values, bit_width
        else:
            # Fallback to raw uint32 run counts if entropy is too high for efficient packing
            return run_counts.view(UINT8_DTYPE), run_values, 0

    def decode(self, packed_counts: np.ndarray, run_values: np.ndarray, bit_width: int, total_elements: int) -> np.ndarray:
        """
        Restores raw arrays from RLE components.
        """
        if total_elements == 0:
            return np.empty(0, dtype=run_values.dtype)
            
        if bit_width > 0:
            run_count_elements = run_values.size
            counts = np.empty(run_count_elements, dtype=UINT32_DTYPE)
            _unpack_bits_32(packed_counts, counts, bit_width, run_count_elements)
        else:
            counts = packed_counts.view(UINT32_DTYPE)
            
        out = np.empty(total_elements, dtype=run_values.dtype)
        _rle_decode_kernel(counts, run_values, out)
        return out


class RoaringBitmapContainer:
    """
    Roaring-style dual-container bitmap index structuring dense ranges as bitsets
    and sparse cardinalities as sorted integer arrays.
    """
    def __init__(self, threshold: int = 4096) -> None:
        self.threshold = threshold
        self.cardinality = 0
        # Container type: 0 for array (sparse), 1 for bitset (dense)
        self.container_type = 0
        self._array = np.empty(0, dtype=UINT32_DTYPE)
        # Bitset sized for 65536 elements per container (standard Roaring chunk size)
        self._bitset = np.zeros(65536 // 64, dtype=UINT64_DTYPE)

    def add(self, value: int) -> None:
        if value < 0 or value >= 65536:
            raise ValueError("Container value out of 16-bit range [0, 65535].")
            
        if self.container_type == 0:
            # Sparse array mode
            idx = np.searchsorted(self._array, value)
            if idx < self._array.size and self._array[idx] == value:
                return  # Duplicate value
            self._array = np.insert(self._array, idx, value)
            self.cardinality += 1
            
            # Switch to dense bitset mode if cardinality exceeds threshold
            if self.cardinality > self.threshold:
                self._convert_to_bitset()
        else:
            # Dense bitset mode
            word_idx = value >> 6
            bit_idx = value & 63
            if (self._bitset[word_idx] & (1 << bit_idx)) == 0:
                self._bitset[word_idx] |= (1 << bit_idx)