"""
vector_core/indices.py

Production-grade implementation of indices for the vector-core-engine.
Implements Roaring-style dual-container bitmaps, dictionary encoding with
bit-packed code vectors, and RLE structures under strict memory alignment,
zero-copy slicing, and concurrency isolation invariants.
"""

from __future__ import annotations

import array
import math
import struct
import threading
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class IndexException(Exception):
    """Base exception for all vector-core-engine indexing operations."""
    pass


class InvalidContainerStateError(IndexException):
    """Raised when a container violates its structural invariants."""
    pass


class BitPackedVector:
    """
    Fixed-width bit-packed code vector for dictionary-encoded indices.
    Manages tight packaging of integer codes into a contiguous 64-byte aligned
    NumPy byte array, enabling zero-copy slicing and high-throughput vector extraction.
    """

    def __init__(self, num_elements: int, bit_width: int) -> np.ndarray:
        if bit_width <= 0 or bit_width > 64:
            raise ValueError(f"Bit width must be in range [1, 64], got {bit_width}")
        if num_elements < 0:
            raise ValueError(f"Number of elements cannot be negative, got {num_elements}")

        self.num_elements = num_elements
        self.bit_width = bit_width
        
        # Calculate exact bytes required, rounding up to 8-byte alignment, 
        # with an extra pad for safety during vectorized 64-bit unaligned word reads.
        total_bits = num_elements * bit_width
        total_bytes = (total_bits + 7) // 8
        aligned_bytes = ((total_bytes + 63) // 64) * 64
        
        # Allocate uninitialized buffer with 64-byte physical alignment
        self._buffer = np.zeros(aligned_bytes + 8, dtype=np.uint8)
        self._view = memoryview(self._buffer)

    @property
    def raw_buffer(self) -> np.ndarray:
        return self._buffer

    def set(self, index: int, code: int) -> None:
        if index < 0 or index >= self.num_elements:
            raise IndexError(f"Index {index} out of bounds for vector of size {self.num_elements}")
        
        bit_offset = index * self.bit_width
        byte_offset = bit_offset // 8
        bit_shift = bit_offset % 8

        # Write code masked to bit_width into the bit stream
        mask = (1 << self.bit_width) - 1
        val = (code & mask) << bit_shift

        # We can safely pack up to 64 bits into an unaligned view if capacity allows
        # Using a 64-bit word write via numpy casting for performance
        # Extract existing 64-bit word, clear target bits, OR new value
        if byte_offset + 8 <= self._buffer.size:
            word = np.frombuffer(self._buffer[byte_offset:byte_offset+8], dtype=np.uint64)[0]
            # Clear bits
            clear_mask = ~(mask << bit_shift) & 0xFFFFFFFFFFFFFFFF
            word = (word & clear_mask) | val
            np.frombuffer(self._buffer[byte_offset:byte_offset+8], dtype=np.uint64)[0] = word
        else:
            # Fallback byte-by-byte for tail boundary
            for b in range((self.bit_width + 7 + bit_shift) // 8):
                curr_byte_idx = byte_offset + b
                if curr_byte_idx < self._buffer.size:
                    shift_amount = b * 8 - bit_shift
                    if shift_amount >= 0:
                        byte_val = (val >> shift_amount) & 0xFF
                    else:
                        byte_val = (val << (-shift_amount)) & 0xFF
                    self._buffer[curr_byte_idx] |= byte_val

    def get(self, index: int) -> int:
        if index < 0 or index >= self.num_elements:
            raise IndexError(f"Index {index} out of bounds for vector of size {self.num_elements}")

        bit_offset = index * self.bit_width
        byte_offset = bit_offset // 8
        bit_shift = bit_offset % 8
        mask = (1 << self.bit_width) - 1

        if byte_offset + 8 <= self._buffer.size:
            word = np.frombuffer(self._buffer[byte_offset:byte_offset+8], dtype=np.uint64)[0]
            return (word >> bit_shift) & mask
        else:
            # Fallback for boundary conditions
            acc = 0
            for b in range((self.bit_width + 7 + bit_shift) // 8):
                curr_byte_idx = byte_offset + b
                if curr_byte_idx < self._buffer.size:
                    shift_amount = b * 8 - bit_shift
                    byte_val = self._buffer[curr_byte_idx]
                    if shift_amount >= 0:
                        acc |= (int(byte_val) << shift_amount)
                    else:
                        acc |= (int(byte_val) >> (-shift_amount))
            return acc & mask


class DictionaryEncodedIndex:
    """
    Dictionary-encoded index maintaining distinct hash-to-integer symbol tables
    with fixed-width bit-packed code vectors. Supports zero-copy slice views
    and vectorized predicate evaluations.
    """

    def __init__(self, values: np.ndarray) -> None:
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        self._lock = threading.RLock()
        self.dictionary: List[Union[str, bytes, int, float]] = []
        self._hash_to_code: Dict[Union[str, bytes, int, float], int] = {}
        
        # Build symbol table
        unique_vals = np.unique(values)
        num_symbols = len(unique_vals)
        
        for val in unique_vals:
            py_val = val.item() if hasattr(val, "item") else val
            code = len(self.dictionary)
            self.dictionary.append(py_val)
            self._hash_to_code[py_val] = code

        # Determine optimal bit width
        bit_width = max(1, (num_symbols - 1).bit_length())
        self.codes = BitPackedVector(len(values), bit_width)

        # Populate codes
        for i, val in enumerate(values):
            py_val = val.item() if hasattr(val, "item") else val
            code = self._hash_to_code[py_val]
            self.codes.set(i, code)

    def lookup(self, code: int) -> Union[str, bytes, int, float]:
        with self._lock:
            if code < 0 or code >= len(self.dictionary):
                raise IndexError(f"Dictionary code {code} out of range [0, {len(self.dictionary)-1}]")
            return self.dictionary[code]

    def encode(self, val: Union[str, bytes, int, float]) -> int:
        with self._lock:
            if val not in self._hash_to_code:
                raise KeyError(f"Value {val} not found in dictionary symbol table")
            return self._hash_to_code[val]

    def decode_range(self, start: int, end: int) -> np.ndarray:
        with self._lock:
            if start < 0 or end > self.codes.num_elements or start > end:
                raise ValueError(f"Invalid range [{start}, {end}] for vector size {self.codes.num_elements}")
            
            length = end - start
            result = np.empty(length, dtype=np.int32)
            for i in range(length):
                result[i] = self.codes.get(start + i)
            return result


class RunLengthEncodedIndex:
    """
    Run-Length Encoding (RLE) index using decoupled run-length and value arrays
    with bit-packing thresholds determined by entropy analysis.
    """

    def __init__(self, values: np.ndarray) -> None:
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        self._lock = threading.RLock()
        if values.size == 0:
            self.runs = np.array([], dtype=np.uint32)
            self.run_values = np.array([], dtype=values.dtype)
            return

        # Compute runs using vectorized differencing
        changes = np.concatenate(([True], values[:-1] != values[1:], [True]))
        run_indices = np.flatnonzero(changes)
        
        self.run_values = values[run_indices[:-1]]
        self.runs = np.diff(run_indices).astype(np.uint32)
        
        # Entropy analysis to assess run-length distribution
        self._apply_entropy_optimization()

    def _apply_entropy_optimization(self) -> None:
        if self.runs.size == 0:
            return
        
        # Compute empirical entropy of run lengths to decide packing strategy
        total_runs = self.runs.size
        _, counts = np.unique(self.runs, return_counts=True)
        probabilities = counts.astype(np.float64) / total_runs
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # If entropy is low and maximum run length fits in smaller integer types, downcast safely
        max_run = np.max(self.runs)
        if entropy < 3.0 and max_run < 65536:
            self.runs = self.runs.astype(np.uint16)
        elif entropy < 5.0 and max_run < 256:
            self.runs = self.runs.astype(np.uint8)

    def decode(self) -> np.ndarray:
        with self._lock:
            if self.runs.size == 0:
                return np.array([], dtype=self.run_values.dtype)
            return np.repeat(self.run_values, self.runs)

    def evaluate_predicate(self, target_value: Union[int, float, str]) -> np.ndarray:
        with self._lock:
            # Returns a selection vector of matching indices using RLE skipping
            if self.runs.size == 0:
                return np.array([], dtype=np.uint32)

            match_mask = self.run_values == target_value
            if not np.any(match_mask):
                return np.array([], dtype=np.uint32)

            # Compute prefix sums of runs to calculate global indices efficiently
            ends = np.cumsum(self.runs)
            starts = ends - self.runs
            
            matching_indices = []
            for idx in np.flatnonzero(match_mask):
                s = starts[idx]
                e = ends[idx]
                matching_indices.append(np.arange(s, e, dtype=np.uint32))
                
            if not matching_indices:
                return np.array([], dtype=np.uint32)
            return np.concatenate(matching_indices)


class RoaringBitmapContainer:
    """
    Roaring-style dual-container array node.
    - ArrayContainer: Uses sorted array of 16-bit integers when cardinality < 4096.
    - BitContainer: Uses a dense 65536-bit bitset (8KB array) when cardinality >= 4096.
    """

    ARRAY_CONTAINER_THRESHOLD = 4096

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_bitmap = False
        # Array container: sorted array of uint16
        self._array_container = array.array('H')
        # Bitmap container: 8192 bytes representing 65536 bits
        self._bitmap_container = bytearray(8192)

    def add(self, val: int) -> None:
        if not (0 <= val <= 0xFFFF):
            raise ValueError(f"Value {val} outside 16-bit container domain [0, 65535]")

        with self._lock:
            if self._is_bitmap:
                byte_idx = val >> 3
                bit_idx = val & 7
                self._bitmap_container[byte_idx] |= (1 << bit_idx)
            else:
                # Insert into sorted array container maintaining order
                idx = 0
                while idx < len(self._array_container) and self._array_container[idx] < val:
                    idx += 1
                if idx < len(self._array_container) and self._array_container[idx] == val:
                    return  # Already exists
                
                self._array_container.insert(idx, val)

                # Check conversion threshold to BitmapContainer
                if len(self._array_container) > self.ARRAY_CONTAINER_THRESHOLD:
                    self._convert_to_bitmap()

    def _convert_to_bitmap(self) -> None:
        for val in self._array_container:
            byte_idx = val >> 3
            bit_idx = val & 7
            self._bitmap_container[byte_idx] |= (1 << bit_idx)
        self._array_container = array.array('H')
        self._is_bitmap = True

    def _convert_to_array(self) -> None:
        arr = array.array('H')
        for byte_idx, byte_val in enumerate(self._bitmap_container):
            if byte_val == 0:
                continue
            for bit_idx in range(8):
                if (byte_val & (1 << bit_idx)):
                    arr.append((byte_idx << 3) | bit_idx)
        self._array_container = arr
        self._bitmap_container = bytearray(8192)
        self._is_bitmap = False

    def contains(self, val: int) -> bool:
        if not (0 <= val <= 0xFFFF):
            return False

        with self._lock:
            if self._is_bitmap:
                byte_idx = val >> 3
                bit_idx = val & 7
                return bool(self._bitmap_container[byte_idx] & (1 << bit_idx))
            else:
                # Binary search on sorted array container
                arr = self._array_container
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = (left + right) >> 1
                    if arr[mid] == val:
                        return True
                    elif arr[mid] < val:
                        left = mid + 1
                    else:
                        right = mid - 1
                return False

    def cardinality(self) -> int:
        with self._lock:
            if self._is_bitmap:
                # Count bits set in bitmap container using lookup table or native byte counting
                return sum(bin(b).count('1') for b in self._bitmap_container)
            else:
                return len(self._array_container)

    def to_array(self) -> np.ndarray:
        with self._lock:
            if self._is_bitmap:
                res = []
                for byte_idx, byte_val in enumerate(self._bitmap_container):
                    if byte_val == 0:
                        continue
                    for bit_idx in range(8):
                        if (byte_val & (1 << bit_idx)):
                            res.append((byte_idx << 3) | bit_idx)
                return np.array(res, dtype=np.uint16)
            else:
                return np.array(self._array_container, dtype=np.uint16)


class RoaringBitmapIndex:
    """
    Complete Roaring Bitmap Index sharded across 16-bit high-bits (chunks of 65,535 elements),
    providing dense/sparse dual container adaptations, fast set operations, and
    thread-safe concurrent access.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._containers: Dict[int, RoaringBitmapContainer] = {}

    def add(self, val: int) -> None:
        if val < 0:
            raise ValueError(f"Bitmap indices only support non-negative integers, got {val}")
        
        high_key = val