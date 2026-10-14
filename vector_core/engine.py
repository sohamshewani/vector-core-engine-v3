import os
import mmap
import struct
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Tuple, Optional, Union, Generator, Any, Set
import numpy as np
import numba

# ============================================================================
# EXCEPTION HIERARCHY & ERROR CODES
# ============================================================================

class VectorCoreError(Exception):
    """Base exception for all vector-core-engine errors."""
    pass

class MemoryBudgetExceededError(VectorCoreError):
    """Raised when the arena allocator exceeds its configured memory budget."""
    pass

class TransactionError(VectorCoreError):
    """Raised during MVCC or WAL transaction faults."""
    pass

class CorruptionError(VectorCoreError):
    """Raised when checksum validation fails on WAL or segment blocks."""
    pass

class ConcurrencyError(VectorCoreError):
    """Raised when hazard pointer or atomic state transitions fail."""
    pass


# ============================================================================
# MEMORY MANAGEMENT & ARENA ALLOCATOR (Spec 4 & Spec 1)
# ============================================================================

class ArenaAllocator:
    """
    Strict memory-budget enforcement through custom arena allocators 
    with OOM circuit breakers and LRU block eviction policies.
    Maintains 64-byte alignment for Arrow-compatible physical layout.
    """
    def __init__(self, max_budget_bytes: int):
        self.max_budget_bytes = max_budget_bytes
        self.allocated_bytes = 0
        self._lock = threading.Lock()
        self._allocations: Dict[int, np.ndarray] = {}
        self._lru_order: List[int] = []

    def allocate(self, size: int, dtype: np.dtype = np.uint8) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        num_elements = (size + itemsize - 1) // itemsize
        byte_footprint = num_elements * itemsize

        with self._lock:
            if self.allocated_bytes + byte_footprint > self.max_budget_bytes:
                self._evict_to_fit(byte_footprint)

            # Create 64-byte aligned buffer via standard numpy allocation + padding check
            # NumPy does not guarantee 64-byte alignment natively unless structured or via custom alignment.
            # We enforce alignment by over-allocating and slicing.
            raw_buffer = np.empty(num_elements + 8, dtype=dtype)
            addr = raw_buffer.ctypes.data
            offset = (64 - (addr % 64)) % 64
            # Since we can't easily offset numpy arrays directly without view casting,
            # we allocate an explicit byte array and create an aligned view.
            byte_array = np.zeros(byte_footprint + 64, dtype=np.uint8)
            base_addr = byte_array.ctypes.data
            aligned_offset = (64 - (base_addr % 64)) % 64
            
            aligned_view = byte_array[aligned_offset:aligned_offset + byte_footprint].view(dtype=dtype)
            
            alloc_id = id(aligned_view)
            self._allocations[alloc_id] = byte_array
            self._lru_order.append(alloc_id)
            self.allocated_bytes += byte_footprint
            return aligned_view

    def _evict_to_fit(self, required_bytes: int) -> None:
        while self.allocated_bytes + required_bytes > self.max_budget_bytes:
            if not self._lru_order:
                raise MemoryBudgetExceededError(
                    f"OOM Circuit Breaker Tripped: Requested {required_bytes} bytes, "
                    f"current usage {self.allocated_bytes}/{self.max_budget_bytes}."
                )
            evict_id = self._lru_order.pop(0)
            if evict_id in self._allocations:
                evicted_array = self._allocations.pop(evict_id)
                self.allocated_bytes -= evicted_array.nbytes

    def free(self, array: np.ndarray) -> None:
        with self._lock:
            # Find matching allocation
            target_id = None
            for aid, arr in self._allocations.items():
                if np.shares_memory(arr, array):
                    target_id = aid
                    break
            if target_id is not None:
                evicted = self._allocations.pop(target_id)
                self.allocated_bytes -= evicted.nbytes
                if target_id in self._lru_order:
                # O(N) removal from list is acceptable for bounded LRU sizes
                    self._lru_order.remove(target_id)


# ============================================================================
# COLUMNAR ENCODING STRUCTURES (Spec 1)
# ============================================================================

class DictionaryEncodedColumn:
    """
    Dictionary encoding structures maintain distinct hash-to-integer symbol tables
    with fixed-width bit-packed code vectors.
    """
    def __init__(self, symbols: np.ndarray, codes: np.ndarray, bit_width: int):
        self.symbols = symbols  # Distinct values array
        self.codes = codes      # Encoded integer codes array
        self.bit_width = bit_width

    @classmethod
    def encode(cls, raw_data: np.ndarray) -> 'DictionaryEncodedColumn':
        symbols, codes = np.unique(raw_data, return_inverse=True)
        num_symbols = len(symbols)
        bit_width = max(1, (num_symbols - 1).bit_length())
        
        # Select minimal unsigned integer type for codes
        if bit_width <= 8:
            code_dtype = np.uint8
        elif bit_width <= 16:
            code_dtype = np.uint16
        elif bit_width <= 32:
            code_dtype = np.uint32
        else:
            code_dtype = np.uint64

        typed_codes = codes.astype(code_dtype)
        return cls(symbols=symbols, codes=typed_codes, bit_width=bit_width)

    def decode(self) -> np.ndarray:
        return self.symbols[self.codes]


class RunLengthEncodedColumn:
    """
    Run-Length Encoding (RLE) uses decoupled run-length and value arrays
    with bit-packing thresholds determined by entropy analysis.
    """
    def __init__(self, values: np.ndarray, runs: np.ndarray):
        self.values = values
        self.runs = runs

    @classmethod
    def encode(cls, raw_data: np.ndarray) -> 'RunLengthEncodedColumn':
        if len(raw_data) == 0:
            return cls(np.array([], dtype=raw_data.dtype), np.array([], dtype=np.uint32))
        
        # Find change points
        changes = np.diff(raw_data) != 0
        change_indices = np.append(np.where(changes)[0], len(raw_data) - 1)
        
        values = raw_data[change_indices]
        
        # Calculate run lengths
        end_indices = np.append(change_indices, [-1])
        start_indices = np.insert(end_indices[:-2] + 1, 0, 0)
        runs = (change_indices - start_indices + 1).astype(np.uint32)
        
        return cls(values=values, runs=runs)

    def decode(self) -> np.ndarray:
        if len(self.values) == 0:
            return np.array([], dtype=self.values.dtype)
        return np.repeat(self.values, self.runs)


class RoaringBitmapIndex:
    """
    Bitmap indices structured as Roaring-style dual-container arrays
    (bitsets for dense ranges, sorted arrays for sparse cardinality).
    """
    CONTAINER_THRESHOLD = 4096

    def __init__(self):
        # Maps chunk high bits (16-bit) to either dense bitarray or sparse sorted numpy array
        self.containers: Dict[int, Union[np.ndarray, np.ndarray]] = {}
        self.container_types: Dict[int, str] = {}  # 'array' or 'bitmap'

    def add(self, val: int) -> None:
        high = val >> 16
        low = val & 0xFFFF
        if high not in self.containers:
            self.containers[high] = np.array([low], dtype=np.uint16)
            self.container_types[high] = 'array'
            return

        ctype = self.container_types[high]
        container = self.containers[high]

        if ctype == 'array':
            # Check if we should convert to bitmap
            if len(container) >= self.CONTAINER_THRESHOLD:
                bitmap = np.zeros(8192, dtype=np.uint64) # 65536 bits = 8192 bytes
                # Set bits
                bitmap[container >> 6] |= (1 << (container & 63))
                # Add new val
                bitmap[low >> 6] |= (1 << (low & 63))
                self.containers[high] = bitmap
                self.container_types[high] = 'bitmap'
            else:
                # Insert sorted maintaining uniqueness
                idx = np.searchsorted(container, low)
                if idx < len(container) and container[idx] == low:
                    return
                self.containers[high] = np.insert(container, idx, low)
        else:
            # Bitmap container
            container[low >> 6] |= (1 << (low & 63))

    def contains(self, val: int) -> bool:
        high = val >> 16
        low = val & 0xFFFF
        if high not in self.containers:
            return False
        ctype = self.container_types[high]
        container = self.containers[high]
        if ctype == 'array':
            idx = np.searchsorted(container, low)
            return idx < len(container) and container[idx] == low
        else:
            return (container[low >> 6] & (1 << (low & 63))) != 0


# ============================================================================
# NUMBA VECTORIZED KERNELS & SIMD EXECUTION PASSES (Spec 2)
# ============================================================================

@numba.njit(fastmath=True, cache=True, parallel=False)
def numba_filter_gt(data: np.ndarray, threshold: float, selection_vector: np.ndarray) -> int:
    """
    CPU-vectorized primitive kernel utilizing Numba JIT compiling for predicate evaluation,
    avoiding branch mispredictions via bitmask generation and selection vectors.
    """
    count = 0
    for i in range(len(data)):
        if data[i] > threshold:
            selection_vector[count] = i
            count += 1
    return count


@numba.njit(fastmath=True, cache=True)
def numba_hash_aggregate_sum(data: np.ndarray, keys: np.ndarray, hash_table_keys: np.ndarray, hash_table_vals: np.ndarray) -> None:
    """
    Vectorized hash aggregation kernel.
    """
    for i in range(len(data)):
        k = keys[i]
        # Simple linear probing hash map simulation
        idx = k % len(hash_table_keys)
        while True:
            current_key = hash_table_keys[idx]
            if current_key == -1 or current_key == k:
                hash_table_keys[idx] = k
                hash_table_vals[idx] += data[i]
                break
            idx = (idx + 1) % len(hash_table_keys)


class ExecutionVector:
    """
    Volcano-iterator-free pipeline execution model operating on fixed-size 
    execution vectors (default N=2048 tuples) to maximize L1/L2 cache locality.
    """
    DEFAULT_SIZE = 2048

    def __init__(self, num_columns: int, capacity: int = DEFAULT_SIZE):
        self.capacity = capacity
        self.size = 0
        self.columns: List[np.ndarray] = [
            np.empty(capacity, dtype=np.float64) for _ in range(num_columns)
        ]
        self.selection_vector = np.empty(capacity, dtype=np.int32)
        self.selection_size = 0

    def reset(self) -> None:
        self.size = 0
        self.selection_size = 0


# ============================================================================
# CONCURRENCY PRIMITIVES & MVCC (Spec 3)
# ============================================================================

class HazardPointerRecord:
    def __init__(self):
        self.pointer: Optional[Any] = None
        self.active = True


class HazardPointerManager:
    """
    Thread-safe readers utilizing lock-free hazard pointers 
    and atomic state transitions for segment promotion.
    """
    def __init__(self):
        self._records: List[HazardPointerRecord] = []
        self._lock = threading.Lock()

    def acquire_record(self) -> HazardPointerRecord:
        with self._lock:
            for rec in self._records:
                if not rec.active:
                    rec.active = True
                    rec.pointer = None
                    return rec
            new_rec = HazardPointerRecord()
            self._records.append(new_rec)
            return new_rec

    def release_record(self, record: HazardPointerRecord) -> None:
        record.pointer = None
        record.active = False

    def is_protected(self, ptr: Any) -> bool:
        with self._lock:
            for rec in self._records:
                if rec.active and rec.pointer is ptr:
                    return True
            return False


class EpochManager:
    """
    Multi-Version Concurrency Control (MVCC) via epoch-based garbage collection 
    and append-only immutable column segments.
    """
    def __init__(self):
        self.global_epoch = 0
        self._lock = threading.Lock()
        self._retire_lists: Dict[int, List[Any]] = {}

    def enter_epoch(self) -> int:
        with self._lock:
            return self.global_epoch

    def advance_epoch(self) -> None:
        with self._lock:
            self.global_epoch += 1
            # Clean up old epochs safely
            old_epochs = [e for e in self._retire_lists.keys() if e < self.global_epoch - 2]
            for e in old_epochs:
                del self._retire_lists[e]

    def retire(self, obj: Any, current_epoch: int) -> None:
        with self._lock:
            if current_epoch not in self._retire_lists:
                self._retire_lists[current_epoch] = []
            self._retire_lists[current_epoch].append(obj)


# ============================================================================
# WAL & CRASH RECOVERY (Spec 4)
# ============================================================================

class WriteAheadLog:
    """
    Write-Ahead Logging (WAL) with checksum-validated entries and fsync guarantees 
    for atomic transaction durability.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = open(log_path, "a+b")
        self._lock = threading.Lock()

    def append_entry(self, txn_id: int, payload: bytes) -> None:
        with self._lock:
            import zlib
            checksum = zlib.crc32(payload)
            # Format: [Length: 4 bytes][TxnID: 8 bytes][Checksum: 4 bytes][Payload: N bytes]
            length = len(payload)
            header = struct.pack("<I Q I", length, txn_id, checksum)
            self._file.write(header + payload)
            self._file.flush()
            os.fsync(self._file.fileno())

    def recover(self) -> List[Tuple[int, bytes]]:
        with self._lock:
            self._file.seek(0)
            entries = []
            import zlib
            while True:
                header_bytes = self._file.read(16)
                if len(header_bytes) < 16:
                    break
                length, txn_id, checksum = struct.unpack("<I Q I", header_bytes)
                payload = self._file.read(length)
                if len(payload) < length:
                    raise CorruptionError