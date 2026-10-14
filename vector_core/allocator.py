import os
import mmap
import math
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class AllocationException(Exception):
    """Raised when memory allocation fails or strict budget limits are breached."""
    pass

class OOMCircuitBreakerTripped(AllocationException):
    """Raised when the memory arena exceeds its hard limit and triggers the OOM circuit breaker."""
    pass

class ArenaBlock:
    """Represents a single contiguous 64-byte aligned memory-mapped allocation block."""
    def __init__(self, block_id: int, size_bytes: int, backing_file: Optional[str] = None):
        self.block_id = block_id
        self.size_bytes = size_bytes
        self.backing_file = backing_file
        self.pin_count = 0
        self.is_evicted = False
        self._lock = threading.Lock()
        
        # Ensure 64-byte alignment padding calculation
        alignment = 64
        actual_size = ((size_bytes + alignment - 1) // alignment) * alignment
        
        if backing_file is not None:
            self._fd = os.open(backing_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            os.ftruncate(self._fd, actual_size)
            self._mmap_buf = mmap.mmap(self._fd, actual_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        else:
            self._fd = None
            # Anonymous mmap for zero-copy volatile scratch spaces
            self._mmap_buf = mmap.mmap(-1, actual_size, flags=mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
            
        # Create zero-copy numpy array view backed directly by the memory-mapped buffer
        # Using uint8 as the primitive byte array layout
        self.array_view = np.frombuffer(self._mmap_buf, dtype=np.uint8, count=actual_size)
        
        # Verify alignment of the numpy buffer data pointer
        data_ptr = self.array_view.ctypes.data
        if data_ptr % 64 != 0:
            # Shift view to align precisely to 64-byte boundary
            offset = 64 - (data_ptr % 64)
            if offset < actual_size:
                self.array_view = self.array_view[offset:actual_size - (64 - offset)]

    def pin(self) -> None:
        with self._lock:
            if self.is_evicted:
                raise AllocationException(f"Attempted to pin evicted ArenaBlock {self.block_id}")
            self.pin_count += 1

    def unpin(self) -> None:
        with self._lock:
            if self.pin_count > 0:
                self.pin_count -= 1

    def evict(self) -> None:
        with self._lock:
            if self.pin_count > 0:
                raise AllocationException(f"Cannot evict ArenaBlock {self.block_id} with active pin_count={self.pin_count}")
            self.is_evicted = True
            try:
                self._mmap_buf.close()
            except Exception:
                pass
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self._mmap_buf.close()
        except Exception:
            pass
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass


class LRUBlockCache:
    """Least-Recently-Used cache tracker for unpinned memory blocks under memory pressure."""
    def __init__(self):
        self._cache: Dict[int, ArenaBlock] = {}
        self._order: List[int] = []
        self._lock = threading.Lock()

    def access(self, block_id: int) -> None:
        with self._lock:
            if block_id in self._cache:
                self._order.remove(block_id)
                self._order.append(block_id)

    def add(self, block: ArenaBlock) -> None:
        with self._lock:
            self._cache[block.block_id] = block
            if block.block_id in self._order:
                self._order.remove(block.block_id)
            self._order.append(block.block_id)

    def evict_lru(self) -> Optional[ArenaBlock]:
        with self._lock:
            for block_id in self._order:
                block = self._cache[block_id]
                if block.pin_count == 0 and not block.is_evicted:
                    self._order.remove(block_id)
                    del self._cache[block_id]
                    return block
            return None

    def remove(self, block_id: int) -> None:
        with self._lock:
            if block_id in self._cache:
                del self._cache[block_id]
            if block_id in self._order:
                self._order.remove(block_id)


class ArenaAllocator:
    """
    Custom arena allocator enforcing strict memory budgets, 64-byte alignment,
    OOM circuit breakers, and LRU block eviction policies.
    """
    def __init__(self, max_budget_bytes: int):
        self.max_budget_bytes = max_budget_bytes
        self.allocated_bytes = 0
        self._blocks: Dict[int, ArenaBlock] = {}
        self._lru_cache = LRUBlockCache()
        self._next_block_id = 0
        self._alloc_lock = threading.Lock()
        self._oom_tripped = False

    def allocate(self, size_bytes: int, backing_file: Optional[str] = None) -> ArenaBlock:
        with self._alloc_lock:
            if self._oom_tripped:
                raise OOMCircuitBreakerTripped("OOM Circuit Breaker is active. Allocations rejected.")

            alignment = 64
            actual_size = ((size_bytes + alignment - 1) // alignment) * alignment

            # Enforce budget and trigger eviction if necessary
            while self.allocated_bytes + actual_size > self.max_budget_bytes:
                evicted_block = self._lru_cache.evict_lru()
                if evicted_block is not None:
                    evicted_block.evict()
                    self.allocated_bytes -= evicted_block.array_view.nbytes
                    del self._blocks[evicted_block.block_id]
                else:
                    # Circuit breaker trip condition
                    self._oom_tripped = True
                    raise OOMCircuitBreakerTripped(
                        f"Memory budget exceeded: requested {actual_size} bytes, "
                        f"current allocated {self.allocated_bytes} bytes, limit {self.max_budget_bytes} bytes."
                    )

            block_id = self._next_block_id
            self._next_block_id += 1

            block = ArenaBlock(block_id=block_id, size_bytes=actual_size, backing_file=backing_file)
            self._blocks[block_id] = block
            self.allocated_bytes += block.array_view.nbytes
            self._lru_cache.add(block)
            return block

    def free(self, block: ArenaBlock) -> None:
        with self._alloc_lock:
            block.evict()
            if block.block_id in self._blocks:
                self.allocated_bytes -= block.array_view.nbytes
                del self._blocks[block.block_id]
            self._lru_cache.remove(block.block_id)

    def reset_circuit_breaker(self) -> None:
        with self._alloc_lock:
            self._oom_tripped = False

    def get_metrics(self) -> Dict[str, Any]:
        with self._alloc_lock:
            return {
                "max_budget_bytes": self.max_budget_bytes,
                "allocated_bytes": self.allocated_bytes,
                "active_blocks": len(self._blocks),
                "oom_tripped": self._oom_tripped
            }