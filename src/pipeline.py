import os
from pathlib import Path
from typing import Any, Dict, List
from analysis_result import AnalysisResult
from utils import (
    save_embeddings,
    load_embeddings,
    find_corresponding_probes,
    get_valid_subjects,
)
from scipy.spatial.distance import cosine
from tqdm import tqdm


class DeepFacePipeline:
    """
    DeepFacePipeline class for managing face recognition processes such as analysis and dissimilarity score calculation.

    Attributes:
        DETECTORS (Dict[str, str]): A dictionary mapping detector names to tuples containing model and detector names.
        MIN_COUNT (int): Minimum count of probe images required for a valid subject.
        input_dir (Path): Input directory containing the datasets.
        databases (List[str]): List of available databases in the input directory.
        output_dir (Path): Output directory for saving results.
    """

    DETECTORS: Dict[str, str] = {
        "ArcFace+yunet": ("ArcFace", "yunet"),
        "Facenet512+retinaface": ("Facenet512", "retinaface"),
    }
    MIN_COUNT: int = 3

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.databases = os.listdir(input_dir)
        if "FERET" not in self.databases and "FRGC" not in self.databases:
            raise ValueError(
                "Error: Please provide a correct database: FERET or FRGC are required."
            )
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            print(f"Log: Creating output directory...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_subdir(self, database: str, suffix: str) -> Path:
        nested_dir = self.input_dir / database / suffix
        if not nested_dir.exists():
            raise ValueError(
                f"Error: {nested_dir} does not exist. Please provide a correct database structure."
            )
        return nested_dir

    def _analyze_FRGC(self) -> AnalysisResult:
        print("Log: Organizing files for FRGC...")
        bonafide_probes_dir = self.get_subdir("FRGC", "bonafide_probe")
        print(f"Log: Reading probe images from {bonafide_probes_dir} ...")
        bonafide_probes_list = os.listdir(bonafide_probes_dir)

        counts = self._get_counts(
            bonafide_probes_list, delimiter=".", identifier_length=5
        )
        counts_len = len(counts)
        print("Log: # Unique identifiers (before filtering): ", counts_len)

        filtered_counts = {k: v for k, v in counts.items() if v >= self.MIN_COUNT}
        filtered_counts_len = len(filtered_counts)
        print("Log: # Unique identifiers (after filtering): ", filtered_counts_len)

        filtered_out_percentage = round(
            (counts_len - filtered_counts_len) / counts_len, 2
        )
        print(
            f"Log: # Unique identifiers (filtered out percentage): {filtered_out_percentage}"
        )

        return AnalysisResult(
            database="FRGC",
            total_identifiers=counts_len,
            filtered_identifiers=filtered_counts_len,
            filtered_out_percentage=filtered_out_percentage,
        )

    def _analyze_FERET(self) -> AnalysisResult:
        print("Log: Organizing files for FERET...")
        bonafide_probes_dir = self.get_subdir("FERET", "bonafide_probe")
        print(f"Log: Reading probe images from {bonafide_probes_dir} ...")
        bonafide_probes_list = os.listdir(bonafide_probes_dir)

        counts = self._get_counts(
            bonafide_probes_list, delimiter="_", identifier_length=None
        )
        counts_len = len(counts)
        print("Log: # Unique identifiers (before filtering): ", counts_len)

        filtered_counts = {k: v for k, v in counts.items() if v >= self.MIN_COUNT}
        filtered_counts_len = len(filtered_counts)
        print("Log: # Unique identifiers (after filtering): ", filtered_counts_len)

        filtered_out_percentage = round(
            (counts_len - filtered_counts_len) / counts_len, 2
        )
        print(
            f"Log: # Unique identifiers (filtered out percentage): {filtered_out_percentage}"
        )

        return AnalysisResult(
            database="FERET",
            total_identifiers=counts_len,
            filtered_identifiers=filtered_counts_len,
            filtered_out_percentage=filtered_out_percentage,
        )

    def analyze(self, database: str) -> AnalysisResult:
        if database == "FRGC":
            return self._analyze_FRGC()
        elif database == "FERET":
            return self._analyze_FERET()
        else:
            raise ValueError(
                "Error: Please specify a valid database ('FRGC' or 'FERET')."
            )

    def _calculate_dissimilarity_scores_FRGC(self) -> None:
        print("Log: Calculating dissimilarity scores for FRGC...")
        morph_dirs = [
            "morphs_facefusion",
            "morphs_facemorpher",
            "morphs_opencv",
            "morphs_ubo",
        ]
        for morph_dir in morph_dirs:
            morph_path = self.get_subdir("FRGC", morph_dir)
            print(f"Log: Calculating dissimilarity scores for {morph_path}...")
            self.calculate_dissimilarity_scores(morph_path, "FRGC")

    def _calculate_dissimilarity_scores_FERET(self) -> None:
        # We don't care about FERET, as the analysis showed that there are not enough probe images
        raise NotImplementedError

    def _calculate_mated_scores_FRGC(self) -> None:
        print("Log: Calculating mated scores for FRGC...")
        bonafide_probes_dir = self.get_subdir("FRGC", "bonafide_probe")
        bonafide_reference_dir = self.get_subdir("FRGC", "bonafide_reference")

        # List all files in the bonafide probe and reference directories
        probe_files = os.listdir(bonafide_probes_dir)
        reference_files = os.listdir(bonafide_reference_dir)

        # Filter subjects with at least MIN_COUNT probe images
        valid_subjects = get_valid_subjects(probe_files, self.MIN_COUNT, "FRGC")

        # Initialize results per FRS
        results_per_frs: Dict[str, List[str]] = {
            frs: [] for frs in self.DETECTORS.keys()
        }

        # Supporting function to list corresponding files for a given subject
        def find_corresponding_files(subject_id, files, dir):
            return [os.path.join(dir, file) for file in files if subject_id in file]

        # For each valif subject id
        for subject_id in valid_subjects:
            # Find all the reference paths and probe paths for the subject
            reference_paths = find_corresponding_files(
                subject_id, reference_files, bonafide_reference_dir
            )
            probe_paths = find_corresponding_files(
                subject_id, probe_files, bonafide_probes_dir
            )

            # If no reference or probe paths are found, skip the subject
            if not probe_paths or not reference_paths:
                continue

            for reference_path in reference_paths:
                for probe_path in probe_paths:
                    for frs, (model, detector) in tqdm(
                        self.DETECTORS.items(),
                        desc=f"Processing detectors for {subject_id}",
                        total=len(self.DETECTORS),
                    ):
                        reference_embeddings_file = (
                            self.output_dir
                            / f"{os.path.splitext(os.path.basename(reference_path))[0]}_{model}_{detector}_embeddings.json"
                        )
                        probe_embeddings_file = (
                            self.output_dir
                            / f"{os.path.splitext(os.path.basename(probe_path))[0]}_{model}_{detector}_embeddings.json"
                        )

                        if not reference_embeddings_file.exists():
                            save_embeddings(
                                reference_path,
                                model,
                                detector,
                                reference_embeddings_file,
                            )
                        if not probe_embeddings_file.exists():
                            save_embeddings(
                                probe_path, model, detector, probe_embeddings_file
                            )

                        reference_embeddings = load_embeddings(
                            reference_embeddings_file
                        )
                        probe_embeddings = load_embeddings(probe_embeddings_file)

                        score = cosine(reference_embeddings, probe_embeddings)
                        print(f"{frs} - {subject_id}: {score:.6f}")

                        results_per_frs[frs].append(f"{subject_id}\t{score:.6f}")

        for frs, results in results_per_frs.items():
            output_file = self.output_dir / f"FRGC_{frs}_mated_scores.txt"
            with output_file.open("w") as f:
                print("Log: Writing results to ", output_file)
                print(f"Log: Number of results for {frs}: ", len(results))
                for result in results:
                    f.write(result + "\n")
                print(f"Log: Mated scores saved to {output_file}")

    def _calculate_non_mated_scores_FRGC(self) -> None:
        print("Log: Calculating non-mated scores for FRGC...")
        bonafide_probes_dir = self.get_subdir("FRGC", "bonafide_probe")
        bonafide_reference_dir = self.get_subdir("FRGC", "bonafide_reference")

        # List all files in the bonafide probe and reference directories
        probe_files = os.listdir(bonafide_probes_dir)
        reference_files = os.listdir(bonafide_reference_dir)

        # Filter subjects with at least MIN_COUNT probe images
        valid_subjects = get_valid_subjects(probe_files, self.MIN_COUNT, "FRGC")

        # Initialize results per FRS
        results_per_frs: Dict[str, List[str]] = {
            frs: [] for frs in self.DETECTORS.keys()
        }

        # Supporting function to list corresponding files for a given subject
        def find_corresponding_files(subject_id, files, dir):
            return [os.path.join(dir, file) for file in files if subject_id in file]

        # For each valid subject ID in the probe files
        for probe_subject_id in valid_subjects:
            probe_paths = find_corresponding_files(
                probe_subject_id, probe_files, bonafide_probes_dir
            )

            # For each valid subject ID in the reference files
            for reference_subject_id in valid_subjects:

                # Skip if the reference and probe subjects are the same
                if reference_subject_id == probe_subject_id:
                    continue

                reference_paths = find_corresponding_files(
                    reference_subject_id, reference_files, bonafide_reference_dir
                )

                for reference_path in reference_paths:
                    for probe_path in probe_paths:
                        for frs, (model, detector) in tqdm(
                            self.DETECTORS.items(),
                            desc=f"Processing detectors for {probe_subject_id} vs {reference_subject_id}",
                            total=len(self.DETECTORS),
                        ):
                            reference_embeddings_file = (
                                self.output_dir
                                / f"{os.path.splitext(os.path.basename(reference_path))[0]}_{model}_{detector}_embeddings.json"
                            )
                            probe_embeddings_file = (
                                self.output_dir
                                / f"{os.path.splitext(os.path.basename(probe_path))[0]}_{model}_{detector}_embeddings.json"
                            )

                            if not reference_embeddings_file.exists():
                                save_embeddings(
                                    reference_path,
                                    model,
                                    detector,
                                    reference_embeddings_file,
                                )
                            if not probe_embeddings_file.exists():
                                save_embeddings(
                                    probe_path, model, detector, probe_embeddings_file
                                )

                            reference_embeddings = load_embeddings(
                                reference_embeddings_file
                            )
                            probe_embeddings = load_embeddings(probe_embeddings_file)

                            score = cosine(reference_embeddings, probe_embeddings)
                            print(
                                f"{frs} - {probe_subject_id} vs {reference_subject_id}: {score:.6f}"
                            )

                            results_per_frs[frs].append(
                                f"{probe_subject_id}\t{reference_subject_id}\t{score:.6f}"
                            )

        for frs, results in results_per_frs.items():
            output_file = self.output_dir / f"FRGC_{frs}_non_mated_scores.txt"
            with output_file.open("w") as f:
                print("Log: Writing results to ", output_file)
                print(f"Log: Number of results for {frs}: ", len(results))
                for result in results:
                    f.write(result + "\n")
                print(f"Log: Non-mated scores saved to {output_file}")

    def dispatcher(self, command: str, *args: Any, **kwargs: Any) -> Any:
        commands = {
            "analyze_FRGC": self._analyze_FRGC,
            "analyze_FERET": self._analyze_FERET,
            "calculate_dissimilarity_scores_FRGC": self._calculate_dissimilarity_scores_FRGC,
            "calculate_dissimilarity_scores_FERET": self._calculate_dissimilarity_scores_FERET,
            "analyze": self.analyze,
            "calculate_dissimilarity_scores": self.calculate_dissimilarity_scores,
            "calculate_mated_scores_FRGC": self._calculate_mated_scores_FRGC,
            "calculate_non_mated_scores_FRGC": self._calculate_non_mated_scores_FRGC,
        }

        action = commands.get(command)

        if action:
            print(
                f"Log: Executing command '{command}' with args: {args}, kwargs: {kwargs}..."
            )
            return action(*args, **kwargs)
        else:
            print(f"Error: Command '{command}' not recognized.")
            return None

    def _get_counts(
        self,
        file_list: List[str],
        delimiter: str,
        identifier_length: int,
    ) -> Dict[str, int]:
        counts = {}
        for file in file_list:
            unique_identifier = file.split(delimiter)[0][:identifier_length]
            if unique_identifier in counts:
                counts[unique_identifier] += 1
            else:
                counts[unique_identifier] = 1
        return counts

    def call(self) -> None:
        if "FRGC" in self.databases:
            analysis_frgc: AnalysisResult = self.dispatcher("analyze", "FRGC")
            if analysis_frgc and analysis_frgc.filtered_out_percentage < 0.6:
                print("Log: Calculating dissimilarity scores for FRGC...")
                self.dispatcher("calculate_dissimilarity_scores_FRGC")
                print("Log: Calculating mated scores for FRGC...")
                self.dispatcher("calculate_mated_scores_FRGC")
                print("Log: Calculating non-mated scores for FRGC...")
                self.dispatcher("calculate_non_mated_scores_FRGC")
        # FERET is not suited in our case since our aanalysis showed that there are not
        # enough probe images to calculate the MAP metric
        # TODO: Expand in case of more databases
        if "FERET" in self.databases:
            analysis_feret: AnalysisResult = self.dispatcher("analyze", "FERET")
            if analysis_feret and analysis_feret.filtered_out_percentage < 0.6:
                print("Log: Calculating dissimilarity scores for FERET...")
                self.dispatcher("calculate_dissimilarity_scores_FERET")


        print("Log: Pipeline finished successfully.")
        # Proceed to computeMAP.py to calculate the MAP metric
        # Map input should look like this:
        # Folder that contains
        # The dissimilarity scores of each FR-Method (Output)
        # A json file that has the following information:
        # {
        #   "FRS_1": [
        #     THRESHOLD,
        #     false
        #   ],
        #   "FRS_2": [
        #     THRESHOLD,
        #     false
        #   ]}
        # The thresholds were calculated using the pyeer library
        # and they follow the FMR1000 condition (See more on README.md)
