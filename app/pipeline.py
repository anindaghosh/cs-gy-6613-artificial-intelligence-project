import argparse

from loguru import logger

from app.featurizer import etl_featurize_step
from app.preprocessing.chunk import etl_chunk
from app.preprocessing.clean import etl_clean
from app.scraper import etl_scrape

STEPS = ["scrape", "clean", "chunk", "featurize"]


def main():
    parser = argparse.ArgumentParser(
        description="Run the ingestion pipeline. Each step is tracked as its own ClearML task."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=STEPS,
        help="steps to run, in order (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only featurize the first N chunks"
    )
    args = parser.parse_args()

    for step in STEPS:
        if step not in args.steps:
            continue

        logger.info(f"Running step: {step}")

        if step == "scrape":
            etl_scrape()
        elif step == "clean":
            etl_clean()
        elif step == "chunk":
            etl_chunk()
        elif step == "featurize":
            etl_featurize_step(limit=args.limit)


if __name__ == "__main__":
    main()
