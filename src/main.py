import argparse
from dataclasses import dataclass
from pipeline import DeepFacePipeline
from mr import MRCalculator


@dataclass
class Arguments:
    input: str = None
    output: str = None
    mode: str = None
    score_mated: str = None
    score_non_mated: str = None
    threshold: float = None


def parse() -> Arguments:
    parser = argparse.ArgumentParser(
        prog="BiometricPipeline",
        description="Calculates the dissimilarity scores, mated, and non-mated scores for a given database or calculates MMPMR and RMMR",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input directory of the database or non-mated scores file",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory of dissimilarity scores",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode to run: 'deepface' or 'mmpmr'",
        required=True,
        choices=["deepface", "mmpmr"],
    )
    parser.add_argument(
        "-g",
        "--mated",
        type=str,
        help="File containing mated (genuine) scores",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--non-mated",
        type=str,
        help="File containing non-mated (impostor) scores",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Threshold for similarity/dissimilarity scores",
        required=False,
    )

    args = parser.parse_args()

    print("Log: Parsing arguments...")
    print(
        f"Log: Input directory: {args.input}\nOutput directory: {args.output}\nMode: {args.mode}\nMated scores file: {args.mated}\nNon-mated scores file: {args.non_mated}\nThreshold: {args.threshold}"
    )
    return Arguments(
        input=args.input,
        output=args.output,
        mode=args.mode,
        score_mated=args.mated,
        score_non_mated=args.non_mated,
        threshold=args.threshold,
    )


def main() -> None:
    args: Arguments = parse()
    if args.mode == "deepface":
        if args.input and args.output:
            pipeline = DeepFacePipeline(args.input, args.output)
            pipeline.call()
        else:
            print(
                "Error: Input and output directories are required for DeepFacePipeline."
            )
    elif args.mode == "mmpmr":
        if args.score_mated and args.score_non_mated and args.threshold is not None:
            mr = MRCalculator(args.score_mated, args.score_non_mated, args.threshold)
            mr.call()
        else:
            print(
                "Error: Mated and non-mated score files and threshold are required for MMPRCalculator."
            )


if __name__ == "__main__":
    main()
