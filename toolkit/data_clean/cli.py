"""数据清洗命令行入口。"""

from __future__ import annotations

import argparse

from toolkit.data_clean.config import DEFAULT_THRESHOLDS, DataCleanConfig
from toolkit.data_clean.paths import parse_cli_path
from toolkit.data_clean.pipeline import DataCleaner


def build_config_from_args(args: argparse.Namespace) -> DataCleanConfig:
    thresholds = {
        "dark_brightness_lt": args.dark_brightness_lt,
        "odd_aspect_ratio_lt": args.odd_aspect_ratio_lt,
        "low_information_entropy_lt": args.low_information_entropy_lt,
        "blurry_blurriness_lt": args.blurry_blurriness_lt,
        "odd_size_lt": args.odd_size_lt,
    }
    return DataCleanConfig(
        output_name=args.output_name,
        overwrite=not args.no_overwrite,
        skip_cleanvision=args.skip_cleanvision,
        require_cleanvision=args.require_cleanvision,
        thresholds=thresholds,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean an image dataset with labels.")
    parser.add_argument("--data-root", required=True, type=parse_cli_path, help="Dataset root directory.")
    parser.add_argument("--output-name", default="clean_output", help="Output directory name under data-root.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the output directory already exists.")
    parser.add_argument(
        "--skip-cleanvision",
        action="store_true",
        help="Only validate labels and export; skip CleanVision quality filters.",
    )
    parser.add_argument(
        "--require-cleanvision",
        action="store_true",
        help="Fail if cleanvision is not installed.",
    )
    parser.add_argument("--dark-brightness-lt", type=float, default=DEFAULT_THRESHOLDS["dark_brightness_lt"])
    parser.add_argument("--odd-aspect-ratio-lt", type=float, default=DEFAULT_THRESHOLDS["odd_aspect_ratio_lt"])
    parser.add_argument(
        "--low-information-entropy-lt",
        type=float,
        default=DEFAULT_THRESHOLDS["low_information_entropy_lt"],
    )
    parser.add_argument("--blurry-blurriness-lt", type=float, default=DEFAULT_THRESHOLDS["blurry_blurriness_lt"])
    parser.add_argument("--odd-size-lt", type=float, default=DEFAULT_THRESHOLDS["odd_size_lt"])
    args = parser.parse_args(argv)

    DataCleaner(build_config_from_args(args)).run(args.data_root)


if __name__ == "__main__":
    main()
