import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import cosine
from deepface import DeepFace


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


def calculate_cosine_similarity(
    embeddings1: List[float], embeddings2: List[float]
) -> float:
    return cosine(embeddings1, embeddings2)


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
