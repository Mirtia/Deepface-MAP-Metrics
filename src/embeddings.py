from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Embeddings:
    uid: str
    embeddings: Dict[str, List[float]]

    def __str__(self):
        return f"UID: {self.uid}\nEmbeddings: {self.embeddings}\n"
