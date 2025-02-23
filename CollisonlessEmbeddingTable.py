import torch
from typing import Optional
from collections import defaultdict
import time


class CuckooHashTable:
    """Collisionless embedding table using Cuckoo Hashing."""

    def __init__(self, embedding_dim: int, max_tries: int = 20):
        self.embedding_dim = embedding_dim
        self.max_tries = max_tries
        self.hash_tables = [{}, {}]  # Two tables for cuckoo hashing
        self.last_access_time = {}   # For feature expiration
        self.frequency_count = defaultdict(int)  # For frequency filtering

    def _hash(self, key: int, table_id: int) -> int:
        """Hash function for each table."""
        if table_id == 0:
            return hash((key, 'table0'))
        return hash((key, 'table1'))

    def insert(self, key: int, embedding: torch.Tensor) -> bool:
        """Insert a key-embedding pair using cuckoo hashing."""
        current_key = key
        current_embedding = embedding

        for _ in range(self.max_tries):
            for table_id in range(2):
                h = self._hash(current_key, table_id)

                if h not in self.hash_tables[table_id]:
                    self.hash_tables[table_id][h] = (
                        current_key, current_embedding)
                    self.last_access_time[current_key] = time.time()
                    return True

                # Kick out the existing entry and try to reinsert it
                existing_key, existing_embedding = self.hash_tables[table_id][h]
                self.hash_tables[table_id][h] = (
                    current_key, current_embedding)

                current_key = existing_key
                current_embedding = existing_embedding

        # If we reach here, we need to rebuild the hash tables
        self._rebuild()
        return self.insert(key, embedding)

    def lookup(self, key: int) -> Optional[torch.Tensor]:
        """Look up an embedding by key."""
        for table_id in range(2):
            h = self._hash(key, table_id)
            if h in self.hash_tables[table_id]:
                stored_key, embedding = self.hash_tables[table_id][h]
                if stored_key == key:
                    self.last_access_time[key] = time.time()
                    self.frequency_count[key] += 1
                    return embedding
        return None

    def _rebuild(self):
        """Rebuild the hash tables when insertion fails."""
        old_tables = self.hash_tables
        self.hash_tables = [{}, {}]

        for table in old_tables:
            for _, (key, embedding) in table.items():
                self.insert(key, embedding)

    def clean_expired_features(self, max_age: float):
        """Remove features that haven't been accessed for max_age seconds."""
        current_time = time.time()
        expired_keys = []

        for key, last_access in self.last_access_time.items():
            if current_time - last_access > max_age:
                expired_keys.append(key)

        for key in expired_keys:
            self.remove(key)

    def remove(self, key: int):
        """Remove a key from the hash tables."""
        for table_id in range(2):
            h = self._hash(key, table_id)
            if h in self.hash_tables[table_id]:
                stored_key, _ = self.hash_tables[table_id][h]
                if stored_key == key:
                    del self.hash_tables[table_id][h]
                    if key in self.last_access_time:
                        del self.last_access_time[key]
                    if key in self.frequency_count:
                        del self.frequency_count[key]
                    return

    def filter_by_frequency(self, min_frequency: int):
        """Remove features that appear less than min_frequency times."""
        infrequent_keys = [
            k for k, v in self.frequency_count.items() if v < min_frequency]
        for key in infrequent_keys:
            self.remove(key)
