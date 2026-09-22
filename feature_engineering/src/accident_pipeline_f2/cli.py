from __future__ import annotations

import argparse

from .interface import build_f2_interface
from .pipeline import load_f2_config, run_align, run_calibrate, run_extract


def main() -> None:
    parser = argparse.ArgumentParser(description="F2_v1 alignment + six-scenario extraction")
    parser.add_argument("--config", default="configs/pipeline_f2_v1.local.yaml")
    parser.add_argument("command",
                        choices=["align", "calibrate", "extract", "interface", "run-all"])
    args = parser.parse_args()
    cfg = load_f2_config(args.config)
    if args.command == "align":
        run_align(cfg)
    elif args.command == "calibrate":
        run_calibrate(cfg)
    elif args.command == "extract":
        run_extract(cfg)
    elif args.command == "interface":
        build_f2_interface(cfg)
    else:
        run_align(cfg)
        run_calibrate(cfg)
        run_extract(cfg)
        build_f2_interface(cfg)


if __name__ == "__main__":
    main()
