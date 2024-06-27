from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json
from deepface import DeepFace


@dataclass
class Embeddings:
    """Dataclass to store the embeddings of an image"""

    ID: str
    embeddings: Dict[str, List[float]]

    def __str__(self):
        """String representation of Embeddings"""
        return f"ID: {self.ID}\nEmbeddings: {self.embeddings}\n"


def save_embeddings(
    image_path: Path, model: str, detector: str, output_file: Path
) -> None:
    """Save embeddings to file

    Args:
        image_path (Path): The image path
        model (str): The model used to extract embeddings
        detector (str): The backend used for detection
        output_file (Path): The output file to save the embeddings
    """
    # We use deepface to calculate the embeddings
    embeddings = DeepFace.represent(
        img_path=str(image_path),
        model_name=model,
        detector_backend=detector,
        enforce_detection=False,
        align=True,
    )[0]["embedding"]

    with output_file.open("w") as df:
        json.dump(embeddings, df)

    print(
        f"Log: Saved embeddings for {image_path} using {model} and {detector} to {output_file}"
    )


def load_embeddings(embeddings_file: Path) -> List[float]:
    """Load embeddings from file

    Args:
        embeddings_file (Path): The embeddings path

    Returns:
        List[float]: The list of embeddings
    """
    with embeddings_file.open("r") as sf:
        embeddings = json.load(sf)
    return embeddings
