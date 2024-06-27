import pandas as pd
from typing import List


class MRCalculator:
    """Scores (Match Rates) from U. Scherhag et al., Biometric systems under morphing attacks: Assess-
    ment of morphing techniques and vulnerability reporting
    It computes the MMPMR, FNMR, and RMMR scores
    where:
    - MMPMR: Mated Morph Presentation Match Rate
    - FNMR: False Non-Match Rate
    - RMMR: Relative Morph Match Rate
    """

    def __init__(
        self, score_mated_file: str, score_non_mated_file: str, threshold: float
    ) -> None:
        """Constructor of MRCalculator

        Args:
            score_mated_file (str): the path to the mated scores file
            score_non_mated_file (str): the path to the non-mated scores file
            threshold (float): the treshold of the FRS system
        """
        self.score_mated_file = score_mated_file
        self.score_non_mated_file = score_non_mated_file
        self.threshold = threshold

    def call(self) -> None:
        """The call function that computes the MMPMR, FNMR and RMMR scores"""
        MMPMR: float = self.calculate_MMPMR()
        FNMR: float = self.calculate_FNMR()
        RMMR: float = self.calculate_RMMR(MMPMR, FNMR)
        print(f"Log: MMPMR: {MMPMR:.3f}, FNMR: {FNMR:.3f}, RMMR: {RMMR:.3f}")

    def calculate_MMPMR(self) -> float:
        """Calculates MMPMR

        Returns:
            float: the MMPRMR score
        """
        mated_scores = self.read_scores(self.score_mated_file)
        mated_scores["match"] = mated_scores["score"] < self.threshold
        MMPRR: float = mated_scores["match"].mean()
        return MMPRR

    def calculate_FNMR(self) -> float:
        """Calculates FNMR

        Returns:
            float: the FNMR score
        """
        non_mated_scores = self.read_scores(self.score_non_mated_file)
        non_mated_scores["non_match"] = non_mated_scores["score"] >= self.threshold
        FNMR = non_mated_scores["non_match"].mean()
        return FNMR

    def calculate_RMMR(self, MMPMR: float, FNMR: float) -> float:
        """Calculates RMMR
        In the paper it is defined as followed:
        RMMR(τ) = 1 + (MMPMR(τ) - (1 - FNMR(τ))) = 1 + (MMPMR(τ) - TMR(τ))
        We use the first equality to compute the RMMR score
        Args:
            MMPMR (float): MMPMR score
            FNMR (float): FNMR score

        Returns:
            float: RMMR score
        """
        return 1 + (MMPMR - (1 - FNMR))

    def read_scores(self, file_path: str) -> pd.DataFrame:
        """Reads the scores from the provided file

        Args:
            file_path (str): The path of the file

        Returns:
            pd.DataFrame: The scores read from the file
        """
        try:
            scores = pd.read_csv(file_path, names=["score"], dtype={"score": float})
        except Exception as e:
            print(f"Error: Could not read scores from {file_path}: {e}")
        return scores
