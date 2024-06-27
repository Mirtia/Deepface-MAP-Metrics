import pandas as pd
from typing import List


class MRCalculator:
    """_summary_
    """    
    def __init__(
        self, score_mated_file: str, score_non_mated_file: str, threshold: float
    ) -> None:
        self.score_mated_file = score_mated_file
        self.score_non_mated_file = score_non_mated_file
        self.threshold = threshold

    def call(self) -> None:
        """The call function that computes the mmpr, fnmr and rmmr scores"""
        mmpmr: float = self.calculate_mmpmr()
        fnmr: float = self.calculate_fnmr()
        rmmr: float = self.calculate_rmmr(mmpmr, fnmr)
        print("Log: MMPMR, FNMR, and RMMR calculated")
        print(f"Log: MMPMR: {mmpmr:.3f}")
        print(f"Log: FNMR: {fnmr:.3f}")
        print(f"Log: RMMR: {rmmr:.3f}")

    def calculate_mmpmr(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """        
        mated_scores = self.read_scores(self.score_mated_file)
        mated_scores["match"] = mated_scores["score"] < self.threshold
        mmpmr: float = mated_scores["match"].mean()
        return mmpmr

    def calculate_fnmr(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """        
        non_mated_scores = self.read_scores(self.score_non_mated_file)
        non_mated_scores["non_match"] = non_mated_scores["score"] >= self.threshold
        fnmr: float = non_mated_scores["non_match"].mean()
        return fnmr

    def calculate_rmmr(self, mmpmr: float, fnmr: float) -> float:
        """_summary_

        Args:
            mmpmr (float): _description_
            fnmr (float): _description_

        Returns:
            float: _description_
        """        
        rmmr: float = 1 + (mmpmr - (1 - fnmr))
        return rmmr

    def read_scores(self, file_path: str) -> pd.DataFrame:
        """_summary_

        Args:
            file_path (str): _description_

        Returns:
            pd.DataFrame: _description_
        """        
        try:
            scores = pd.read_csv(file_path, names=["score"], dtype={"score": float})
        except Exception as e:
            print(f"Error: Could not read scores from {file_path}: {e}")
            raise
        return scores
