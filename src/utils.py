import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import cosine
from deepface import DeepFace


def save_embeddings(
    image_path: Path, model: str, detector: str, output_file: Path
) -> None:
    """
    Generates and saves the embeddings for a given image using the specified model and detector.

    Args:
        image_path (Path): The path to the image file.
        model (str): The name of the model to use for generating embeddings.
        detector (str): The detector to use for aligning the image.
        output_file (Path): The file path where the embeddings will be saved.
    """
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
    """
    Loads embeddings from a specified file.

    Args:
        embeddings_file (Path): The file path from which to load the embeddings.

    Returns:
        List[float]: A list of floating-point numbers representing the embeddings.
    """
    with embeddings_file.open("r") as f:
        embeddings = json.load(f)
    return embeddings


def calculate_cosine_similarity(
    embeddings1: List[float], embeddings2: List[float]
) -> float:
    """
    Calculates the cosine similarity between two sets of embeddings.

    Args:
        embeddings1 (List[float]): The first set of embeddings.
        embeddings2 (List[float]): The second set of embeddings.

    Returns:
        float: The cosine similarity score between the two sets of embeddings.
    """
    return cosine(embeddings1, embeddings2)


def find_corresponding_probes(
    morph_name: str, probe_files: List[str], bonafide_probes_dir: Path
) -> List[Path]:
    """
    Finds all corresponding probe images for a given morph file.

    Args:
        morph_name (str): The name of the morph file.
        probe_files (List[str]): A list of probe image filenames.
        bonafide_probes_dir (Path): The directory containing the probe images.

    Returns:
        List[Path]: A list of paths to the corresponding probe images.
    """
    subject1 = morph_name.split("_vs_")[0].split("_")[0].split("d")[0]
    matches = []
    for probe in probe_files:
        if probe.startswith(subject1):
            matches.append(bonafide_probes_dir / probe)
    print(f"Log: Found {len(matches)} corresponding probe images for {morph_name}")
    return matches


def get_valid_subjects(probe_files: List[str], min_count: int, database: str) -> set:
    """
    Returns a set of subject IDs that have at least MIN_COUNT probe images.

    Args:
        probe_files (List[str]): A list of probe image filenames.
        min_count (int): The minimum number of probe images required for a subject to be valid.
        database (str): The name of the database ("FRGC" or "FERET") to determine the filename structure.

    Returns:
        set: A set of subject IDs with sufficient probe images.
    """
    subject_probe_counts = {}
    for probe_file in probe_files:
        subject_id = (
            probe_file.split("_")[0]
            if database == "FERET"
            else probe_file.split(".")[0][:5]
        )
        if subject_id in subject_probe_counts:
            subject_probe_counts[subject_id] += 1
        else:
            subject_probe_counts[subject_id] = 1

    valid_subjects = {
        subject for subject, count in subject_probe_counts.items() if count >= min_count
    }
    return valid_subjects
