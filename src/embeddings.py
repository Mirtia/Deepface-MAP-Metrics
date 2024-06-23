from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Embeddings:
    """
    A data structure to store embeddings for reuse in calculating dissimilarity scores.

    Attributes:
        uid (str): The unique identifier for the embeddings.
        embeddings (Dict[str, List[float]]): A dictionary of embeddings with keys as
                                             embedding types and values as embedding vectors.
    """

    uid: str
    embeddings: Dict[str, List[float]]

    def __str__(self):
        """
        Returns a string representation of the embeddings.
        """
        return f"UID: {self.uid}\nEmbeddings: {self.embeddings}\n"
