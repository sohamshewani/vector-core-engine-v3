import os
import sys
import mmap
import threading
import logging
import struct
import zlib
from typing import List, Tuple, Optional, Dict, Any, Generator, Union
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_core.execution")

VECTOR_SIZE: int = 2048

class ExecutionError(Exception):
    """Base exception class for vector execution errors."""
    pass

class MemoryBudgetExceededError(ExecutionError):
    """Raised when memory allocation exceeds budget limits."""
    pass

class OOMCircuitBreaker:
    """Monitors memory usage and trips an OOM circuit breaker if limits are exceeded."""
    def __init__(self, max_budget_bytes: int) -> None:
        self.max_budget_bytes = max_budget_bytes
        self.current_allocated = 0
        self._lock = threading.Lock()

    def allocate(self, size: int) -> None:
        with self._lock:
            if self.current_allocated + size > self.max_budget_bytes:
                logger.critical(f"OOM Circuit Breaker Tripped! Requested: {size}, Allocated: {self.current_allocated}, Limit: {self.max_budget_bytes}")
                raise MemoryBudgetExceededError("Memory budget exceeded. OOM Circuit Breaker activated.")
            self.current_allocated += size

    def free(self, size: int) -> None:
        with self._lock:
            self.current_allocated = max(0, self.current_allocated - size)

class ColumnChunk:
    """Represents a contiguous 64-byte aligned, immutable columnar memory block (Arrow-compatible)."""
    def __init__(self, data: np.ndarray, alignment: int = 64) -> None:
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        self.raw_data = data
        self.alignment = alignment
        self.size_bytes = data.nbytes
        
        # Verify 64-byte alignment of the data buffer pointer
        ptr = data.ctypes.data
        if ptr % alignment != 0:
            # Reallocate with padding to achieve strict 64-byte alignment
            aligned_buffer = np.empty(data.nbytes + alignment, dtype=np.uint8)
            aligned_ptr = aligned_buffer.ctypes.data
            offset = (alignment - (aligned_ptr % alignment)) % alignment
            # Create a view aligned properly
            self._storage = aligned_buffer
            self.data = np.frombuffer(aligned_buffer[offset:offset+data.nbytes], dtype=data.dtype).reshape(data.shape)
            np.copyto(self.data, data)
        else:
            self._storage = data
            self.data = data

    def slice_zero_copy(self, start: int, end: int) -> 'ColumnChunk':
        sliced_array = self.data[start:end]
        sub_chunk = ColumnChunk.__new__(ColumnChunk)
        sub_chunk.raw_data = self.raw_data
        sub_chunk._storage = self._storage
        sub_chunk.alignment = self.alignment
        sub_chunk.data = sliced_array
        sub_chunk.size_bytes = sliced_array.nbytes
        return sub_chunk

class ExecutionVector:
    """Fixed-size execution vector holding data and an active selection bitmask."""
    def __init__(self, capacity: int = VECTOR_SIZE) -> None:
        self.capacity = capacity
        self.data: Dict[str, np.ndarray] = {}
        # Selection vector represented as a boolean numpy array for SIMD filtering
        self.selection_vector: np.ndarray = np.ones(capacity, dtype=bool)
        self.size: int = 0

    def add_column(self, name: str, array: np.ndarray) -> None:
        if len(array) > self.capacity:
            raise ValueError(f"Array length {len(array)} exceeds vector capacity {self.capacity}")
        self.data[name] = array
        self.size = len(array)
        if len(self.selection_vector) < self.size:
            self.selection_vector = np.ones(self.size, dtype=bool)

    def reset_selection(self) -> None:
        if self.size > 0:
            self.selection_vector[:self.size] = True

class Operator:
    """Abstract base class for vectorized pipeline execution operators."""
    def __init__(self) -> None:
        pass

    def open(self) -> None:
        raise NotImplementedError

    def next_vector(self) -> Optional[ExecutionVector]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

class ScanOperator(Operator):
    """Scans a columnar data source in fixed-size execution vectors."""
    def __init__(self, columns: Dict[str, ColumnChunk], batch_size: int = VECTOR_SIZE) -> None:
        super().__init__()
        self.columns = columns
        self.batch_size = batch_size
        self.total_rows = 0
        if columns:
            first_col = next(iter(columns.values()))
            self.total_rows = len(first_col.data)
        self.current_offset = 0

    def open(self) -> None:
        self.current_offset = 0

    def next_vector(self) -> Optional[ExecutionVector]:
        if self.current_offset >= self.total_rows:
            return None

        end_offset = min(self.current_offset + self.batch_size, self.total_rows)
        exec_vec = ExecutionVector(capacity=end_offset - self.current_offset)

        for col_name, chunk in self.columns.items():
            sliced = chunk.slice_zero_copy(self.current_offset, end_offset)
            exec_vec.add_column(col_name, sliced.data)

        self.current_offset = end_offset
        return exec_vec

    def close(self) -> None:
        self.current_offset = 0

class FilterOperator(Operator):
    """Applies vectorized predicate filtering using bitmask generation."""
    def __init__(self, child: Operator, column_name: str, op: str, value: Any) -> None:
        super().__init__()
        self.child = child
        self.column_name = column_name
        self.op = op
        self.value = value

    def open(self) -> None:
        self.child.open()

    def next_vector(self) -> Optional[ExecutionVector]:
        while True:
            vec = self.child.next_vector()
            if vec is None:
                return None

            col_data = vec.data.get(self.column_name)
            if col_data is None:
                continue

            # Vectorized primitive evaluation avoiding branch mispredictions
            if self.op == '=':
                mask = (col_data == self.value)
            elif self.op == '!=':
                mask = (col_data != self.value)
            elif self.op == '<':
                mask = (col_data < self.value)
            elif self.op == '<=':
                mask = (col_data <= self.value)
            elif self.op == '>':
                mask = (col_data > self.value)
            elif self.op == '>=':
                mask = (col_data >= self.value)
            else:
                raise ExecutionError(f"Unsupported operator: {self.op}")

            # Combine with existing selection vector using logical AND
            vec.selection_vector[:vec.size] &= mask[:vec.size]
            
            # If any rows survive, return the vector
            if np.any(vec.selection_vector[:vec.size]):
                return vec

    def close(self) -> None:
        self.child.close()

class HashAggregateOperator(Operator):
    """Performs vectorized hash aggregation without Volcano iterators in the inner loop."""
    def __init__(self, child: Operator, group_by_col: str, agg_col: str, agg_func: str = 'sum') -> None:
        super().__init__()
        self.child = child
        self.group_by_col = group_by_col
        self.agg_col = agg_col
        self.agg_func = agg_func.lower()
        self.aggregated_results: Dict[Any, float] = {}
        self._executed = False

    def open(self) -> None:
        self.child.open()
        self.aggregated_results.clear()
        self._executed = False

    def next_vector(self) -> Optional[ExecutionVector]:
        if self._executed:
            return None

        # Fully consume child operator into local hash aggregation map
        while True:
            vec = self.child.next_vector()
            if vec is None:
                break

            g_data = vec.data.get(self.group_by_col)
            a_data = vec.data.get(self.agg_col)
            if g_data is None or a_data is None:
                continue

            active_indices = np.where(vec.selection_vector[:vec.size])[0]
            if len(active_indices) == 0:
                continue

            groups = g_data[active_indices]
            values = a_data[active_indices]

            # CPU-vectorized hash aggregation accumulation
            for g, v in zip(groups, values):
                if g not in self.aggregated_results:
                    self.aggregated_results[g] = 0.0
                if self.agg_func == 'sum':
                    self.aggregated_results[g] += float(v)
                elif self.agg_func == 'count':
                    self.aggregated_results[g] += 1.0
                elif self.agg_func == 'max':
                    self.aggregated_results[g] = max(self.aggregated_results[g], float(v))
                elif self.agg_func == 'min':
                    self.aggregated_results[g] = min(self.aggregated_results[g], float(v))
                else:
                    raise ExecutionError(f"Unsupported aggregate function: {self.agg_func}")

        self._executed = True

        if not self.aggregated_results:
            return None

        # Convert aggregated map to an ExecutionVector output
        keys = list(self.aggregated_results.keys())
        vals = list(self.aggregated_results.values())
        
        out_vec = ExecutionVector(capacity=len(keys))
        out_vec.add_column(self.group_by_col, np.array(keys))
        out_vec.add_column(f"{self.agg_func}_{self.agg_col}", np.array(vals))
        return out_vec

    def close(self) -> None:
        self.child.close()

class TaskPartition:
    """Represents a chunk-level task partition for parallel query execution."""
    def __init__(self, operator: Operator, partition_id: int) -> None:
        self.operator = operator
        self.partition_id = partition_id

class ParallelQueryEngine:
    """Parallel query execution engine utilizing a work-stealing thread pool."""
    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _execute_partition(self, partition: TaskPartition) -> List[Dict[str, np.ndarray]]:
        results = []
        partition.operator.open()
        try:
            while True:
                vec = partition.operator.next_vector()
                if vec is None:
                    break
                active_mask = vec.selection_vector[:vec.size]
                # Materialize active rows based on selection vector
                materialized_batch = {}
                for k, arr in vec.data.items():
                    materialized_batch[k] = arr[:vec.size][active_mask]
                results.append(materialized_batch)
        finally:
            partition.operator.close()
        return results

    def execute_query(self, partitions: List[TaskPartition]) -> List[Dict[str, np.ndarray]]:
        futures: List[Future] = []
        for partition in partitions:
            futures.append(self.executor.submit(self._execute_partition, partition))

        all_results = []
        for future in futures:
            res = future.result()
            all_results.extend(res)
        return all_results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)