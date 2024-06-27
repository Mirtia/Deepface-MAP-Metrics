from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Embeddings:
    """Dataclass to store the embeddings of an image
    """    
    uid: str
    embeddings: Dict[str, List[float]]

    def __str__(self):
        """String representation of Embeddings"""
        return f"UID: {self.uid}\nEmbeddings: {self.embeddings}\n"


def save_embeddings(
    image_path: Path, model: str, detector: str, output_file: Path
) -> None:
    embeddings = DeepFace.represent(
        img_path=str(image_path),
        model_name=model,
        detector_backend=detector,
        enforce_detection=False,
        align=True,
    )[0]["embedding"]
    with output_file.open("w") as f:
        json.dump(embeddings, f)
    print(
        f"Log: Saved embeddings for {image_path} using {model}+{detector} to {output_file}"
    )


def load_embeddings(embeddings_file: Path) -> List[float]:
    with embeddings_file.open("r") as f:
        embeddings = json.load(f)
    return embeddings
