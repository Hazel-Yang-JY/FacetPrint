from __future__ import annotations

import argparse
import json
import logging

from .construction import construct
from .training import train
from .verification import verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetprint",
        description="FacetPrint quantization-robust black-box fingerprinting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    construct_parser = subparsers.add_parser("construct", help="construct a private fingerprint pair")
    construct_parser.add_argument("--config", required=True)
    construct_parser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train", help="embed and merge the fingerprint")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--max-steps", type=int)

    verify_parser = subparsers.add_parser("verify", help="verify a suspect or deployed checkpoint")
    verify_parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()
    if args.command == "construct":
        path = construct(args.config, force=args.force)
        print(json.dumps({"selection": str(path.resolve())}, ensure_ascii=False))
    elif args.command == "train":
        path = train(args.config, max_steps=args.max_steps)
        print(json.dumps({"fingerprint_model": str(path.resolve())}, ensure_ascii=False))
    elif args.command == "verify":
        print(json.dumps(verify(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
