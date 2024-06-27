import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import cosine
from deepface import DeepFace

def calculate_cosine_similarity(
    embeddings_1: List[float], embeddings_2: List[float]
) -> float:
    """Calculates the cosine similarity between two given embeddings

    Args:
        embeddings_1 (List[float]): The first embeddings
        embeddings_2 (List[float]): The second embeddings

    Returns:
        float: The cosine similarity between the two embeddings
    """
    return cosine(embeddings_1, embeddings_2)


def find_corresponding_probes(
    morph_name: str, probe_files: List[str], bonafide_probes_dir: Path
) -> List[Path]:
    matches = []
    subject1 = morph_name.split("_vs_")[0].split("_")[0].split("d")[0]
    for probe in probe_files:
        if probe.startswith(subject1):
            matches.append(bonafide_probes_dir / probe)
    print(f"Log: Found {len(matches)} corresponding probe images for {morph_name}")
    return matches


def get_valid_subjects(probe_files: List[str], min_count: int, database: str) -> set:
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
