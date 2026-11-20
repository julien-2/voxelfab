"""CLI `vf` : run | info | demo | bench | xmf.

    vf run deck.yaml [-o outdir] [--backend cpu|gpu|auto]
    vf info deck.yaml
    vf demo [-o outdir]
    vf bench [n]
    vf xmf [structure.h5]
    vf history history.h5 [--step K] [-o outdir] [--format h5|png|vtk]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .deck import DEMO_DECK, DeckError, load_deck
from .ops import available_backends
from .run import DeckRunner, run_deck


def _bench(n: int, backend: str):
    from .bench import bench
    bench(n, backend)


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


def cmd_xmf(args):
    from .export import write_xmf
    write_xmf(args.file)


def cmd_history(args):
    """`vf history FICHIER` liste les étapes ; `--step K` exporte l'état K."""
    from .history import load_history, grid_at
    meta = load_history(args.file)
    print(f"history  : {args.file}")
    print(f"mode={meta.get('mode')}  grille={list(meta['grid'])}  "
          f"pitch={meta.get('pitch_nm')} nm  backend={meta.get('backend')}  "
          f"étapes={len(meta['steps'])}")
    for s in meta["steps"]:
        print(f"  étape {s['index']:3d}  {s['op']:<14s} Δ={s['n_changed']:>9d} vx"
              f"  sha={s['step_sha256'][:12]}…  {s['params']}")
    if args.step is not None:
        from .export import export_h5, export_vtk, render_views
        g = grid_at(args.file, args.step)
        outdir = Path(args.outdir or Path(args.file).parent)
        outdir.mkdir(parents=True, exist_ok=True)
        base = outdir / f"step_{args.step:04d}"
        if args.format == "h5":
            export_h5(g, filename=str(base) + ".h5",
                      provenance={"backend": str(meta.get("backend", "?"))})
        elif args.format == "png":
            render_views(g, filename=str(base) + ".png",
                         title=f"étape {args.step} — {args.file}")
        else:
            export_vtk(g, filename=str(base) + ".vtk")


def main():
    ap = argparse.ArgumentParser(
        prog="vf", description="moteur de fabrication virtuelle voxel "
                               "(decks YAML -> structures 3D + métrologie)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="exécuter un deck")
    p_run.add_argument("deck")
    p_run.add_argument("-o", "--outdir")
    p_run.add_argument("--backend", default="cpu",
                       choices=available_backends() + ["auto"],
                       help="backend d'exécution : cpu, gpu, ou auto "
                            "(gpu si un device CUDA est détecté, sinon cpu ; "
                            "défaut : cpu)")
    p_run.set_defaults(fn=cmd_run)

    p_info = sub.add_parser("info", help="valider et résumer un deck")
    p_info.add_argument("deck")
    p_info.set_defaults(fn=cmd_info)

    p_demo = sub.add_parser("demo", help="écrire + exécuter le deck de démo")
    p_demo.add_argument("-o", "--outdir")
    p_demo.add_argument("--backend", default="cpu",
                        choices=available_backends() + ["auto"],
                        help="cpu, gpu, ou auto (gpu si un device CUDA est "
                             "détecté, sinon cpu ; défaut : cpu)")
    p_demo.set_defaults(fn=cmd_demo)

    p_bench = sub.add_parser("bench",
                             help="micro-benchmark des étapes (grille n³)")
    p_bench.add_argument("n", nargs="?", type=int, default=256)
    _backends = available_backends()
    p_bench.add_argument("--backend",
                         default="all" if "gpu" in _backends else "cpu",
                         choices=_backends + ["all"],
                         help="backend à benchmarker ('all' = comparatif)")
    p_bench.set_defaults(fn=lambda a: _bench(a.n, a.backend))

    p_xmf = sub.add_parser(
        "xmf", help="pont Xdmf pour visualiser un structure.h5 "
                    "dans ParaView")
    p_xmf.add_argument("file", nargs="?", default="structure.h5",
                       help="fichier .h5 (défaut : structure.h5)")
    p_xmf.set_defaults(fn=cmd_xmf)

    p_hist = sub.add_parser("history",
                            help="lire un history.h5 : lister les étapes, "
                                 "ou exporter l'état à une étape (--step)")
    p_hist.add_argument("file", help="fichier history.h5")
    p_hist.add_argument("--step", type=int, default=None,
                        help="exporter l'état à l'étape K (0 = initial)")
    p_hist.add_argument("-o", "--outdir", default=None,
                        help="répertoire de sortie pour --step")
    p_hist.add_argument("--format", default="h5", choices=["h5", "png", "vtk"],
                        help="format d'export pour --step (défaut : h5)")
    p_hist.set_defaults(fn=cmd_history)

    args = ap.parse_args()
    try:
        args.fn(args)
    except DeckError as e:
        print(f"ERREUR de deck : {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
