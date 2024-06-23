from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """
    A data structure to store the analysis results.

    Attributes:
        database (str): The name of the database being analyzed.
        total_identifiers (int): The total number of unique identifiers in the database.
        filtered_identifiers (int): The number of identifiers remaining after filtering.
        filtered_out_percentage (float): The percentage of identifiers filtered out.
    """

    database: str
    total_identifiers: int
    filtered_identifiers: int
    filtered_out_percentage: float

    def __str__(self):
        """
        Returns a string representation of the analysis results.
        """
        return (
            f"Database: {self.database}\n"
            f"Total Identifiers: {self.total_identifiers}\n"
            f"Filtered Identifiers: {self.filtered_identifiers}\n"
            f"Filtered Out Percentage: {self.filtered_out_percentage:.2f}%\n"
        )
