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
    # The models and detectors chosen with this guideline https://github.com/serengil/deepface/tree/master/benchmarks
    DETECTORS: Dict[str, str] = {
        "ArcFace+yunet": ("ArcFace", "yunet"),
        "Facenet512+retinaface": ("Facenet512", "retinaface"),
    }
    # Minimum number of probe images per unique subject
    MIN_COUNT: int = 3

    def __init__(self, input_dir: str, output_dir: str):
        """_summary_

        Args:
            input_dir (str): The root directory of the input databases, see README.md for detailed structure
            output_dir (str): The output directory for the embeddings and mated/non-mated scores

        Raises:
            ValueError: Throws error when none of FRGC or FERET are provided
        """
        self.input_dir = Path(input_dir)
        self.databases = os.listdir(input_dir)
        # Not both of the databases are required, as the analysis showed that FERET is not suitable for the MIN_COUNT threshold = 3.
        # But, if someone wants to see the analysis for FERET, they can include the database in the root (Input) directory.
        # The code is constructed according to these databases (e.g. finding the subject ids by name).
        # Other databases may have other conventions so the code should be extended accordingly for those.
        if "FERET" not in self.databases and "FRGC" not in self.databases:
            raise ValueError(
                "Error: Please provide a correct database: at least FERET or FRGC are required."
            )
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            print(f"Log: Creating output directory...")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_subdir(self, database: str, suffix: str) -> Path:
        """Get nested directory from the database by concatenating the root directory name, the database name and the suffix.
        Asssumed structure is:
        Input
        ├── FERET
        │   ├── bonafide_probe
        │   ├── bonafide_reference
        │   ├── morphs_facefusion
        │   ├── morphs_facemorpher
        │   ├── morphs_opencv
        │   └── morphs_ubo
        ├── FRGC
        │   ├── bonafide_probe
        │   ├── bonafide_reference
        │   ├── morphs_facefusion
        │   ├── morphs_facemorpher
        │   ├── morphs_opencv
        │   └── morphs_ubo
        └── ...

        Args:
            database (str): The database name e.g. FRGC
            suffix (str): The nested directory name e.g. bonafide_reference

        Raises:
            ValueError: Throws an error when the nested directory does not exist

        Returns:
            Path: The concatenated path
        """
        nested_dir: Path = self.input_dir / database / suffix
        if not nested_dir.exists():
            raise ValueError(
                f"Error: {nested_dir} does not exist. Please provide a correct database structure."
            )
        return nested_dir

    def analyze(self, database: str) -> AnalysisResult:
        """Analyzes the FERET or FRGC database and checks the number of unique identifiers, if there are at least MIN_COUNT probe images for each id,
        and calculates the percentage of filtered out identifiers (those that do not have enough probe images).

        Raises:
            ValueError: Throws error if database is not supported

        Returns:
            AnalysisResult: Wrapper stucture for the analysis metrics
        """
        if database != "FERET" and database != "FRGC":
            raise ValueError(
                "Error: Please specify a valid database ('FRGC' or 'FERET')."
            )

        print("Log: Analyzing {database}...")

        bonafide_probes_dir = self._get_subdir(database, "bonafide_probe")
        print(f"Log: Reading probe images from {bonafide_probes_dir} ...")
        bonafide_probes_list = os.listdir(bonafide_probes_dir)

        counts = self._get_counts(
            delimiter=bonafide_probes_list, delimiter="_", id_length=None
        )
        counts_len = len(counts)
        print("Log: # Unique identifiers (before filtering): ", counts_len)

        filtered_counts = {
            key: value for key, value in counts.items() if (v >= self.MIN_COUNT)
        }
        filtered_counts_len = len(filtered_counts)
        print("Log: # Unique identifiers (after filtering): ", filtered_counts_len)

        filtered_out_percentage = round(
            (counts_len - filtered_counts_len) / counts_len, 2
        )
        print(
            f"Log: # Unique identifiers (filtered out percentage): {filtered_out_percentage}"
        )

        return AnalysisResult(
            database=database,
            total_identifiers=counts_len,
            filtered_identifiers=filtered_counts_len,
            filtered_out_percentage=filtered_out_percentage,
        )

    def _find_corresponding_files(subject_id: str, files: List, dir: Path) -> List:
        """Finds the corresponding files for a given subject ID in the given directory,
        concats path with the file name for each entry in the list

        Args:
            subject_id (str): subject ID
            files (List): list of files
            dir (Path): the directory path

        Returns:
            List: The corresponding files for a given subject ID in the given directory
        """
        return [os.path.join(dir, file) for file in files if subject_id in file]

    def calculate_dissimilarity_scores(self, database: str) -> None:
        """Calculates the dissimilarity scores for the FRGC or FERET databases

        Args:
            database (str): The database name ("FRGC" or "FERET")
        """
        # The nested morph directories for both cases
        morph_dirs = [
            "morphs_facefusion",
            "morphs_facemorpher",
            "morphs_opencv",
            "morphs_ubo",
        ]
        for morph_dir in morph_dirs:
            full_morph_path = self._get_subdir(database, morph_dir)
            print(f"Log: Calculating dissimilarity scores for {full_morph_path}...")
            self._calculate_dissimilarity_scores(full_morph_path, database)

    def _calculate_dissimilarity_scores(self, morphs_dir: Path, database: str) -> None:
        """Calculates the dissimilarity scores between morph images and bonafide probes for the given database

        Args:
            morphs_dir (Path): The directory containing morph images.
            database (str): The name of the database ("FRGC" or "FERET").
        """
        print(f"Log: Calculating dissimilarity scores for {database}...")
        morph_files = os.listdir(morphs_dir)
        bonafide_probes_dir = self._get_subdir(database, "bonafide_probe")
        probe_files = os.listdir(bonafide_probes_dir)

        # TODO: It keeps recalculating the valid subjects, which is not nice, maybe I'll add it as a field at the class
        valid_subjects = get_valid_subjects(probe_files, self.MIN_COUNT, database)

        results_per_frs = {frs: [] for frs in self.DETECTORS.keys()}

        morph_id_counter = 1

        for morph_file in morph_files:
            morph_path: Path = morphs_dir / morph_file
            subjects = morph_file.split("_vs_")
            subject_id_1 = subjects[0].split("_")[0].split("d")[0]
            subject_id_2 = subjects[1].split("_")[0].split("d")[0]

            # Iterates over all morph files
            # So we have to check if the subject IDs are valid
            if not (subject_id_2 and subject_id_2 in valid_subjects):
                print(f"Log: Skipping {morph_file} due to insufficient probe images.")
                continue

            print(
                f"Log: Processing morph: {morph_file} for subjects: {subject_id_1}, {subject_id_2}"
            )

            morph_id = f"M{morph_id_counter:04d}"
            morph_id_counter += 1

            for subject_label, subject_id in [
                ("S1", subject_id_1),
                ("S2", subject_id_2),
            ]:
                probe_paths = find_corresponding_probes(
                    subject_id, probe_files, bonafide_probes_dir
                )

                if not probe_paths:
                    # This code should never be reached
                    raise ValueError(
                        f"Error: No corresponding probes found for {morph_file} ({subject_label})."
                    )

                # print(f"Log: Processing probes: {probe_paths} for {subject_label}")

                scores_per_frs = {frs: [] for frs in self.DETECTORS.keys()}

                for probe_path in probe_paths:
                    # print("Log: Processing probe: ", probe_path)
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
                        # else:
                        # print(f"Log: Embeddings already saved for {probe_path}")
                        if not morph_embeddings_file.exists():
                            save_embeddings(
                                morph_path, model, detector, morph_embeddings_file
                            )
                        # else:
                        # print(f"Log: Embeddings already saved for {morph_path}")

                        probe_embeddings = load_embeddings(probe_embeddings_file)
                        morph_embeddings = load_embeddings(morph_embeddings_file)

                        score = cosine(probe_embeddings, morph_embeddings)
                        # .6 precision
                        print(f"{frs} - {morph_file} ({subject_label}): {score:.6f}")

                        scores_per_frs[frs].append(score)

                for frs, scores in scores_per_frs.items():
                    # Format the scores
                    formatted_scores = f"{morph_id}\t{subject_label}\t" + "\t".join(
                        f"{score:.6f}" for score in scores
                    )
                    # Two lists, one for each FRS
                    results_per_frs[frs].append(formatted_scores)

                for frs, results in results_per_frs.items():
                    output_file = self.output_dir / f".txt"
                    with output_file.open("w") as df:
                        print("Log: Writing results to ", output_file)
                        # print(f"Log: Number of results for {frs}: ", len(results))
                        for result in results:
                            df.write(result + "\n")
                        print(f"Log: Dissimilarity scores saved to {output_file}")

    def calculate_mated_scores(self, database: str) -> None:
        """Calculates the mated scores for the database

        Args:
            database (str): the name of the database

        """
        print(f"Log: Calculating mated scores for {database}...")
        bonafide_probes_dir = self._get_subdir(database, "bonafide_probe")
        bonafide_reference_dir = self._get_subdir(database, "bonafide_reference")

        # List all files in the bonafide probe and reference directories
        probe_files = os.listdir(bonafide_probes_dir)
        reference_files = os.listdir(bonafide_reference_dir)

        # Returns a set with at least MIN_COUNT probe images
        valid_subjects: set = get_valid_subjects(probe_files, self.MIN_COUNT, database)

        # Initialize results per FRS
        results_per_frs = {frs: [] for frs in self.DETECTORS.keys()}

        def find_corresponding_files(subject_id, files, dir):
            return [os.path.join(dir, file) for file in files if subject_id in file]

        # For each valid subject ID (The reason I kept filtering again and again
        # was because I didn't want to have a separate directory with the valid subjects)
        for subject_id in valid_subjects:
            # Find all the reference paths and probe paths for the subject
            reference_paths = find_corresponding_files(
                subject_id, reference_files, bonafide_reference_dir
            )
            probe_paths = find_corresponding_files(
                subject_id, probe_files, bonafide_probes_dir
            )

            # This should not happen if the subject is valid
            if not probe_paths or not reference_paths:
                raise ValueError(
                    f"Error: No probe or reference paths found for {subject_id}"
                )

            for reference_path in reference_paths:
                for probe_path in probe_paths:
                    # tqdm is kind of unnecessary here, but I like progress bars
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

                        # No need to recalculate the embeddings every time, we can just load them this way
                        # The embeddings are at the Output directory defined in the arguments
                        reference_embeddings = load_embeddings(
                            reference_embeddings_file
                        )
                        probe_embeddings = load_embeddings(probe_embeddings_file)

                        # We calculate the cosine similarity between the embeddings (TODO: See alignment)
                        score = cosine(reference_embeddings, probe_embeddings)
                        # print(f"{frs} - {subject_id}: {score:.6f}")

                        results_per_frs[frs].append(f"{subject_id}\t{score:.6f}")

        for frs, results in results_per_frs.items():
            output_file = self.output_dir / f"FRGC_{frs}_mated_scores.txt"
            with output_file.open("w") as df:
                print("Log: Writing results to ", output_file)
                print(f"Log: Number of results for {frs}: ", len(results))
                for result in results:
                    df.write(result + "\n")
                print(f"Log: Mated scores saved to {output_file}")

    def calculate_non_mated_scores(self, database: str) -> None:
        """Calculates the non-mated scores for the database

        Args:
            database (str): the name of the database
        """
        print("Log: Calculating non-mated scores for {database}...")
        bonafide_probes_dir = self._get_subdir(database, "bonafide_probe")
        bonafide_reference_dir = self._get_subdir(database, "bonafide_reference")

        # List all files in the bonafide probe and reference directories
        probe_files = os.listdir(bonafide_probes_dir)
        reference_files = os.listdir(bonafide_reference_dir)

        # Filter subjects with at least MIN_COUNT probe images
        valid_subjects: set = get_valid_subjects(probe_files, self.MIN_COUNT, database)

        # Initialize results per FRS
        results_per_frs = {frs: [] for frs in self.DETECTORS.keys()}

        for probe_subject_id in valid_subjects:
            probe_paths = find_corresponding_files(
                probe_subject_id, probe_files, bonafide_probes_dir
            )

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
                            # print(
                            #     f"{frs} - {probe_subject_id} vs {reference_subject_id}: {score:.6f}"
                            # )
                            # To use pyeer afterwards, we can awk the last column (for me it did not work to just have the file with multiple input columns)
                            # The subject ids are here for sanity check, same for the mated scores
                            results_per_frs[frs].append(
                                f"{probe_subject_id}\t{reference_subject_id}\t{score:.6f}"
                            )

        for frs, results in results_per_frs.items():
            output_file = self.output_dir / f"{database}_{frs}_non_mated_scores.txt"
            with output_file.open("w") as df:
                print("Log: Writing results to ", output_file)
                print(f"Log: Number of results for {frs}: ", len(results))
                for result in results:
                    df.write(result + "\n")
                print(f"Log: Non-mated scores saved to {output_file}")

    def _get_counts(
        self,
        file_list: List[str],
        delimiter: str,
        id_length: int,
    ) -> Dict[str, int]:
        """Gets the counts of unique identifiers in the file list (counts per subject ID)

        Args:
            file_list (List[str]): the list of files
            delimiter (str): the delimeter to help get the ID
            id_length (int): the expected length of the ID

        Returns:
            Dict[str, int]: A dictionary with the counts of unique identifiers (ID: count)
        """
        counts = {}
        for file in file_list:
            unique_identifier: str = file.split(delimiter)[0][:id_length]
            if unique_identifier in counts:
                counts[unique_identifier] += 1
            else:
                counts[unique_identifier] = 1
        return counts

    def call(self) -> None:
        """The call function that runs the complete pipeline"""
        if "FRGC" in self.databases:
            analysis_frgc: AnalysisResult = self.analyze("FRGC")
            if analysis_frgc and analysis_frgc.filtered_out_percentage < 0.6:
                self.calculate_dissimilarity_scores("FRGC")
                self.calculate_mated_scores("FRGC")
                self.calculate_non_mated_scores("FRGC")
        # FERET is not suited in our case since our analysis showed that there are not
        # enough probe images to calculate the MAP metric
        # Easily expand in case of more databases
        if "FERET" in self.databases:
            analysis_feret: AnalysisResult = self.analyze("FERET")
            if analysis_feret and analysis_feret.filtered_out_percentage < 0.6:
                # This is unreachable code at MIN_COUNT = 3
                self.calculate_dissimilarity_scores("FERET")

        print("Log: Pipeline finished...")
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
