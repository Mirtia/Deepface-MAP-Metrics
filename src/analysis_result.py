"""
"""
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Dataclass to store the analysis results"""

    database: str
    total_identifiers: int
    filtered_identifiers: int
    filtered_out_percentage: float

    def __str__(self):
        """String representation of AnalysisResult"""
        return (
            f"Database: {self.database}\n"
            f"Total Identifiers: {self.total_identifiers}\n"
            f"Filtered Identifiers: {self.filtered_identifiers}\n"
            f"Filtered Out Percentage: {self.filtered_out_percentage:.2f}%\n"
        )
