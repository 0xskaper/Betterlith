from collections import defaultdict
import torch.Tensor as tensor
import time


class CuckooHashTable:
    def __init__(self, embeddingDimension: int, maxTries: int = 20):
        self.embeddingDimension = embeddingDimension
        self.maxTries = maxTries
        self.hashTables = [{}, {}]
        self.lastAccessTime = {}
        self.frequencyCount = defaultdict(int)

    def _hash(self, key: int, tableId: int) -> int:
        if tableId == 0:
            return hash((key, 'table0'))
        return hash((key, 'table1'))

    def insert(self, key: int, embedding: tensor) -> bool:
        currentkey = key
        currentEmbedding = embedding

        for _ in range(self.maxTries):
            for tableId in range(2):
                hashValue = self._hash(currentkey, tableId)

                if hashValue not in self.hashTables[tableId]:
                    self.hashTables[tableId][hashValue] = (
                        currentkey, currentEmbedding)
                    self.lastAccessTime[currentkey] = time.Time()
                    return True

            existingKey, existingEmbedding = self.hashTables[tableId][hashValue]
            self.hashTables[tableId][hashValue] = (
                currentkey, currentEmbedding)

            currentkey = existingKey
            currentEmbedding = existingEmbedding
        self._rebuild()
        return self.insert(key, embedding)

    def _lookup(self, key: int) -> Optional[tensor]:
        for tableId in range(2):
            hashValue = self._hash(key, tableId)
            if hashValue in self.hashTables[tableId]:
                storedKey, embedding = self.hashTable[tableId][hashValue]
                if storedKey == key:
                    self.lastAccessTime[key] = time.Time()
                    self.frequencyCount[key] += 1
                    return embedding
        return None

    def _rebuild(self):
        oldTables = self.hashTables
        self.hashTables = [{}, {}]

        for table in oldTables:
            for _, (key, embedding) in table.items():
                self.insert(key, embedding)

    def cleanExpiredFeatures(self, maxAge: float):
        currentTime = time.Time()
        expiredKeys = []

        for key, lastAccess in self.lastAccessTime.items():
            if currentTime - lastAccess > maxAge:
                expiredKeys.append(key)

        for key in expiredKeys:
            self.remove(key)

    def filterByFrequency(self, minFrequency: int):
        infrequentKeys = [
            k for k, v in self.frequencyCount.items() if v < minFrequency]
        for key in infrequentKeys:
            self.remove(key)
