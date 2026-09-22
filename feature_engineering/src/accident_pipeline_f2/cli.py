from __future__ import annotations

import argparse

from .interface import build_f2_interface
from .pipeline import load_f2_config, run_align, run_calibrate, run_extract
from .r2_pipeline import build_r2_interface, load_f2r2_config, run_r2_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="F2_v1 alignment + six-scenario extraction; "
                                                 "F2R2_v1 volatility/quality/dirless (FEAT-005)")
    parser.add_argument("--config", default="configs/pipeline_f2_v1.local.yaml")
    parser.add_argument("command",
                        choices=["align", "calibrate", "extract", "interface", "run-all",
                                 "r2scan", "interface-r2", "r2-all"])
    args = parser.parse_args()
    if args.command in ("r2scan", "interface-r2", "r2-all"):
        cfg = load_f2r2_config(args.config)
        if args.command == "r2scan":
            run_r2_scan(cfg)
        elif args.command == "interface-r2":
            build_r2_interface(cfg)
        else:
            run_r2_scan(cfg)
            build_r2_interface(cfg)
        return
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
