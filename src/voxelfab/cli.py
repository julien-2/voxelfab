"""CLI `vf` : run | info | demo | bench.

    vf run deck.yaml [-o outdir] [--backend cpu]
    vf info deck.yaml
    vf demo [-o outdir]
    vf bench [n]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .deck import DEMO_DECK, DeckError, load_deck
from .ops import available_backends
from .run import DeckRunner, run_deck


def _bench(n: int):
    from .bench import bench
    bench(n)


def cmd_run(args):
    run_deck(Path(args.deck), Path(args.outdir or Path(args.deck).parent),
             backend=args.backend)


def cmd_demo(args):
    path = Path("demo_deck.yaml")
    path.write_text(DEMO_DECK)
    print(f"[demo] deck écrit : {path}")
    run_deck(path, Path(args.outdir or "."), backend=args.backend)


def cmd_info(args):
    deck = load_deck(Path(args.deck))
    steps = deck.get("steps", [])
    print(f"deck     : {args.deck}")
    print(f"version  : {deck.get('version')}")
    print(f"grid     : {deck.get('grid')}")
    print(f"matériaux: {', '.join(deck.get('materials', {}))}")
    if "layout" in deck:
        print(f"layout   : {deck['layout'].get('file')} "
              f"couches={list(deck['layout'].get('layers', {}))}")
    print(f"étapes   : {len(steps)}")
    for i, s in enumerate(steps, 1):
        print(f"  {i:2d}. {s.get('op'):10s} "
              f"{ {k: v for k, v in s.items() if k != 'op'} }")
    print(f"sorties  : {[o.get('type') for o in deck.get('outputs', [])]}")


def main():
    ap = argparse.ArgumentParser(
        prog="vf", description="moteur de fabrication virtuelle voxel "
                               "(decks YAML -> structures 3D + métrologie)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="exécuter un deck")
    p_run.add_argument("deck")
    p_run.add_argument("-o", "--outdir")
    p_run.add_argument("--backend", default="cpu",
                       choices=available_backends(),
                       help="backend d'exécution (défaut : cpu)")
    p_run.set_defaults(fn=cmd_run)

    p_info = sub.add_parser("info", help="valider et résumer un deck")
    p_info.add_argument("deck")
    p_info.set_defaults(fn=cmd_info)

    p_demo = sub.add_parser("demo", help="écrire + exécuter le deck de démo")
    p_demo.add_argument("-o", "--outdir")
    p_demo.add_argument("--backend", default="cpu",
                        choices=available_backends())
    p_demo.set_defaults(fn=cmd_demo)

    p_bench = sub.add_parser("bench",
                             help="micro-benchmark des étapes (grille n³)")
    p_bench.add_argument("n", nargs="?", type=int, default=256)
    p_bench.set_defaults(fn=lambda a: _bench(a.n))

    args = ap.parse_args()
    try:
        args.fn(args)
    except DeckError as e:
        print(f"ERREUR de deck : {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
