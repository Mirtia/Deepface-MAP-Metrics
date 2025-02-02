from pathlib import Path
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import cosine
import os


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
    """_summary_

    Args:
        morph_name (str): The morph name (it includes two subject IDS separated by '_vs_')
        probe_files (List[str]): The probe files
        bonafide_probes_dir (Path): The probe directory

    Returns:
        List[Path]: _description_
    """
    matches = []
    subject_id = morph_name.split("_vs_")[0].split("_")[0].split("d")[0]
    for probe in probe_files:
        if probe.startswith(subject_id):
            matches.append(bonafide_probes_dir / probe)
    print(f"Log: Found {len(matches)} corresponding probe images for {morph_name}")
    return matches


def get_valid_subjects(probe_files: List[str], min_count: int, database: str) -> set:
    """_summary_

    Args:
        probe_files (List[str]): The list of probe files
        min_count (int): The minimum number of probes required for a subject to be valid
        database (str): The database name

    Returns:
        set: The valid subject ids
    """
    subject_probe_counts = {}
    for probe_file in probe_files:
        # Extract the subject ID from the probe file name, depending on the database
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
        subject
        for subject, count in subject_probe_counts.items()
        if (count >= min_count)
    }
    return valid_subjects


def find_corresponding_files(subject_id: str, files: List, dir: Path) -> List:
    """Finds the corresponding files for a given subject ID in the given directory,
    concats path with the file name for each entry in the list

    Args:
        subject_id (str): The subject ID
        files (List): The list of files
        dir (Path): The directory path

    Returns:
        List: The corresponding files for a given subject ID in the given directory
    """
    return [os.path.join(dir, file) for file in files if subject_id in file]
