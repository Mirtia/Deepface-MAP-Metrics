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
    A pipeline for managing the analysis and calculation of dissimilarity scores for facial recognition systems (FRS).

    Attributes:
        input_dir (Path): The input directory containing the databases.
        output_dir (Path): The directory where output files will be stored.
        databases (List[str]): A list of database names present in the input directory.
        DETECTORS (Dict[str, str]): A dictionary mapping FRS names to their model and detector combinations.
        MIN_COUNT (int): The minimum number of images required per subject for valid analysis.
    """

    # Models and detectors for each FRS
    DETECTORS: Dict[str, str] = {
        "ArcFace+yunet": ("ArcFace", "yunet"),
        "Facenet512+retinaface": ("Facenet512", "retinaface"),
    }
    # Minimum number of images required per subject
    MIN_COUNT: int = 3

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes the DeepFacePipeline with the given input and output directories.

        Args:
            input_dir (str): The directory containing the input databases.
            output_dir (str): The directory where the output will be saved.

        Raises:
            ValueError: If the input directory does not contain required databases (FERET or FRGC).
        """
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
        """
        Retrieves a nested directory path within the input directory for a given database and suffix.

        Args:
            database (str): The name of the database.
            suffix (str): The subdirectory name within the database directory.

        Returns:
            Path: The full path to the nested directory.

        Raises:
            ValueError: If the specified subdirectory does not exist.
        """
        nested_dir = self.input_dir / database / suffix
        if not nested_dir.exists():
            raise ValueError(
                f"Error: {nested_dir} does not exist. Please provide a correct database structure."
            )
        return nested_dir

    def _analyze_FRGC(self) -> AnalysisResult:
        """
        Analyzes the FRGC database to filter out identifiers with insufficient image counts.

        Returns:
            AnalysisResult: An AnalysisResult object containing the analysis metrics.
        """
        print("Log: Organizing files for FRGC...")
        bonafide_probes_dir = self.get_subdir("FRGC", "bonafide_probe")
        print(f"Log: Reading probe images from {bonafide_probes_dir} ...")
        bonafide_probes_list = os.listdir(bonafide_probes_dir)

        counts = self._get_counts(
            bonafide_probes_list, delimiter=".", identifier_length=5
        )
        counts_len = len(counts)
        print("Log: # Unique identifiers (before filtering): ", counts_len)

        filtered_counts = {k: v for k, v in counts.items() if v > self.MIN_COUNT}
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
        """
        Analyzes the FERET database to filter out identifiers with insufficient image counts.

        Returns:
            AnalysisResult: An AnalysisResult object containing the analysis metrics.
        """
        print("Log: Organizing files for FERET...")
        bonafide_probes_dir = self.get_subdir("FERET", "bonafide_probe")
        print(f"Log: Reading probe images from {bonafide_probes_dir} ...")
        bonafide_probes_list = os.listdir(bonafide_probes_dir)

        counts = self._get_counts(
            bonafide_probes_list, delimiter="_", identifier_length=None
        )
        counts_len = len(counts)
        print("Log: # Unique identifiers (before filtering): ", counts_len)

        filtered_counts = {k: v for k, v in counts.items() if v > self.MIN_COUNT}
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
        """
        Analyzes the specified database for image count sufficiency.

        Args:
            database (str): The name of the database to analyze ("FRGC" or "FERET").

        Returns:
            AnalysisResult: An AnalysisResult object containing the analysis metrics.

        Raises:
            ValueError: If an invalid database is specified.
        """
        if database == "FRGC":
            return self._analyze_FRGC()
        elif database == "FERET":
            return self._analyze_FERET()
        else:
            raise ValueError(
                "Error: Please specify a valid database ('FRGC' or 'FERET')."
            )

    def _calculate_dissimilarity_scores_FRGC(self) -> None:
        """
        Calculates the dissimilarity scores for the FRGC database morphs against bonafide probes.
        """
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
        """
        Calculates the dissimilarity scores for the FERET database morphs against bonafide probes.
        """
        print("Log: Calculating dissimilarity scores for FERET...")
        morph_dirs = [
            "morphs_facefusion",
            "morphs_facemorpher",
            "morphs_opencv",
            "morphs_ubo",
        ]
        for morph_dir in morph_dirs:
            morph_path = self.get_subdir("FERET", morph_dir)
            print(f"Log: Calculating dissimilarity scores for {morph_path}...")
            self.calculate_dissimilarity_scores(morph_path, "FERET")

    def calculate_dissimilarity_scores(self, morphs_dir: Path, database: str) -> None:
        """
        Calculates the dissimilarity scores between morph images and bonafide probes for the specified database.

        Args:
            morphs_dir (Path): The directory containing morph images.
            database (str): The name of the database ("FRGC" or "FERET").

            Raises:
                ValueError: If an invalid database is specified.
        """
        if database not in ["FRGC", "FERET"]:
            raise ValueError(
                "Error: Please specify a valid database ('FRGC' or 'FERET')."
            )

        morph_files = os.listdir(morphs_dir)
        bonafide_probes_dir = self.get_subdir(database, "bonafide_probe")
        probe_files = os.listdir(bonafide_probes_dir)

        valid_subjects = get_valid_subjects(probe_files, self.MIN_COUNT, database)
        print("Log: # Valid subjects: ", len(valid_subjects))
        print("Log: Valid subjects: ", valid_subjects)

        results_per_frs = {frs: [] for frs in self.DETECTORS.keys()}

        morph_id_counter = 1

        for morph_file in morph_files:
            morph_path = morphs_dir / morph_file
            subjects = morph_file.split("_vs_")
            subject_id1 = subjects[0].split("_")[0].split("d")[0]
            subject_id2 = subjects[1].split("_")[0].split("d")[0]

            if subject_id1 not in valid_subjects:
                print(
                    f"Log: Skipping {morph_file} due to insufficient probe images for {subject_id1}."
                )
                continue

            if subject_id2 and subject_id2 not in valid_subjects:
                print(
                    f"Log: Skipping {morph_file} due to insufficient probe images for {subject_id2}."
                )
                continue

            print(
                f"Log: Processing morph: {morph_file} for subjects: {subject_id1}, {subject_id2}"
            )

            morph_id = f"M{morph_id_counter:04d}"
            morph_id_counter += 1

            for subject_label, subject_id in [("S1", subject_id1), ("S2", subject_id2)]:
                if not subject_id:
                    continue

                probe_paths = find_corresponding_probes(
                    subject_id, probe_files, bonafide_probes_dir
                )

                if not probe_paths:
                    print(
                        f"Error: No corresponding probes found for {morph_file} ({subject_label})."
                    )
                    continue

                print(f"Log: Processing probes: {probe_paths} for {subject_label}")

                scores_per_frs_for_subject = {frs: [] for frs in self.DETECTORS.keys()}

                for probe_path in probe_paths:
                    print("Log: Processing probe: ", probe_path)
                    for frs, (model, detector) in tqdm(
                        self.DETECTORS.items(),
                        desc=f"Processing detectors for {subject_label}",
                        total=len(self.DETECTORS),
                    ):
                        # Save embeddings for probe and morph images if not already saved
                        probe_embeddings_file = (
                            self.output_dir
                            / f"{probe_path.stem}_{model}_{detector}_embeddings.json"
                        )
                        morph_embeddings_file = (
                            self.output_dir
                            / f"{morph_path.stem}_{model}_{detector}_embeddings.json"
                        )

                        if not probe_embeddings_file.exists():
                            save_embeddings(
                                probe_path, model, detector, probe_embeddings_file
                            )
                        else:
                            print(f"Log: Embeddings already saved for {probe_path}")
                        if not morph_embeddings_file.exists():
                            save_embeddings(
                                morph_path, model, detector, morph_embeddings_file
                            )
                        else:
                            print(f"Log: Embeddings already saved for {morph_path}")

                        probe_embeddings = load_embeddings(probe_embeddings_file)
                        morph_embeddings = load_embeddings(morph_embeddings_file)

                        score = cosine(probe_embeddings, morph_embeddings)
                        print(f"{frs} - {morph_file} ({subject_label}): {score:.6f}")

                        scores_per_frs_for_subject[frs].append(score)

                for frs, scores in scores_per_frs_for_subject.items():
                    formatted_scores = f"{morph_id}\t{subject_label}\t" + "\t".join(
                        f"{score:.6f}" for score in scores
                    )
                    results_per_frs[frs].append(formatted_scores)

        # Write all results to the respective output files
        for frs, results in results_per_frs.items():
            output_file = self.output_dir / f"{database}_{frs}_dissimilarity_scores.txt"
            with output_file.open("w") as f:
                print("Log: Writing results to ", output_file)
                print(f"Log: Number of results for {frs}: ", len(results))
                for result in results:
                    f.write(result + "\n")
                print(f"Log: Dissimilarity scores saved to {output_file}")

    def dispatcher(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """
        Dispatches a command to the corresponding method with optional arguments.

        Args:
            command (str): The name of the command to execute.
            *args (Any): Positional arguments to pass to the command.
            **kwargs (Any): Keyword arguments to pass to the command.

        Returns:
            Any: The result of the dispatched command, or None if the command is not recognized.
        """
        commands = {
            "analyze_FRGC": self._analyze_FRGC,
            "analyze_FERET": self._analyze_FERET,
            "calculate_dissimilarity_scores_FRGC": self._calculate_dissimilarity_scores_FRGC,
            "calculate_dissimilarity_scores_FERET": self._calculate_dissimilarity_scores_FERET,
            "analyze": self.analyze,
            "calculate_dissimilarity_scores": self.calculate_dissimilarity_scores,
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
        """
        Counts the occurrences of unique identifiers in the file list.

        Args:
            file_list (List[str]): A list of file names.
            delimiter (str): The delimiter used in the file name to separate identifiers.
            identifier_length (int): The length of the identifier substring to consider.

        Returns:
            Dict[str, int]: A dictionary mapping unique identifiers to their occurrence counts.
        """
        counts = {}
        for file in file_list:
            unique_identifier = file.split(delimiter)[0][:identifier_length]
            if unique_identifier in counts:
                counts[unique_identifier] += 1
            else:
                counts[unique_identifier] = 1
        return counts

    def call(self) -> None:
        """
        Calls the analysis and dissimilarity score calculations for all databases.
        """
        if "FRGC" in self.databases:
            analysis_frgc: AnalysisResult = self.dispatcher("analyze", "FRGC")
            if analysis_frgc and analysis_frgc.filtered_out_percentage < 0.6:
                print("Log: Calculating dissimilarity scores for FRGC...")
                self.dispatcher("calculate_dissimilarity_scores_FRGC")

        if "FERET" in self.databases:
            analysis_feret: AnalysisResult = self.dispatcher("analyze", "FERET")
            if analysis_feret and analysis_feret.filtered_out_percentage < 0.6:
                print("Log: Calculating dissimilarity scores for FERET...")
                self.dispatcher("calculate_dissimilarity_scores_FERET")
