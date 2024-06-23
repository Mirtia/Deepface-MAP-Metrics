import argparse
from pipeline import DeepFacePipeline

def parse() -> tuple:
    parser = argparse.ArgumentParser(
        prog="DissimilarityCalculator",
        description="Calculates the dissimilarity scores for a given database",
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Input directory of the database", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Ouput directory of disimilarity scores",
        required=True,
    )

    args: argparse.Namespace = parser.parse_args()

    print("Log: Parsing arguments...")
    input_dir: str = args.input
    output_dir: str = args.output
    print(f"Log: Input directory: {input_dir}\nOutput directory: {output_dir}")
    return input_dir, output_dir


def main() -> None:
    input_dir, output_dir = parse()
    pipeline = DeepFacePipeline(input_dir,output_dir)
    pipeline.call()

if __name__ == "__main__":
    main()
