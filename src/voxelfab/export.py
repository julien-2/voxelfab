"""Exports et rendus : HDF5 (structure), coupes 2D, vues 3D, VTK, STL.

Le HDF5 est le format de structure canonique (l'API fichier du produit) :
grille uint8 chunkée + gzip, table des matériaux, attributs de provenance.
h5py est importé paresseusement ; matplotlib, pyvista et
scikit-image/numpy-stl restent optionnels (extras [view] / stl).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .grid import Materials, VoxelGrid


# --------------------------------------------------------------------------
# Structure HDF5 (format canonique)
# --------------------------------------------------------------------------

#: clés de provenance écrites dans le fichier (sous-ensemble déterministe :
#: pas de timings ni dechemin local -> deux runs identiques donnent un
#: fichier identique)
_H5_PROV_KEYS = ("deck", "deck_sha256", "grid", "pitch_nm",
                 "voxelfab_version", "backend", "checkpoint_step")


def export_h5(grid: VoxelGrid, filename: str = "structure.h5",
              provenance: dict | None = None):
    """Exporte la structure au format HDF5.

    Contenu :
      /grid       uint8 (nx, ny, nz), chunké (128³ max) + gzip + shuffle
      /materials  table (id, name, color) des matériaux, air exclu
      attrs       pitch_nm, structure_sha256 (hash des octets de la grille)
                  + provenance déterministe (deck_sha256, version, backend…)

    La grille compressée est le stockage ; le hash de structure reste défini
    sur les octets bruts de la grille (numpy), pas sur le fichier.
    """
    try:
        import h5py
    except ImportError:
        print("  [export] h5py absent : export HDF5 ignoré "
              "(uv pip install h5py)")
        return
    str_dt = h5py.string_dtype("utf-8")
    mat_dt = np.dtype([("id", "u1"), ("name", str_dt), ("color", str_dt)])
    mats = np.array(
        [(mid, grid.materials.name(mid), grid.materials.color(mid))
         for mid in grid.materials.ids if mid != Materials.AIR],
        dtype=mat_dt)
    chunks = (min(grid.nx, 128), min(grid.ny, 128), min(grid.nz, 128))
    with h5py.File(filename, "w") as f:
        f.create_dataset("grid", data=grid.data, chunks=chunks,
                         compression="gzip", compression_opts=4,
                         shuffle=True)
        f.create_dataset("materials", data=mats)
        f.attrs["pitch_nm"] = grid.pitch
        f.attrs["structure_sha256"] = hashlib.sha256(
            grid.data.tobytes()).hexdigest()
        for k in _H5_PROV_KEYS:
            if provenance and k in provenance:
                f.attrs[f"prov_{k}"] = provenance[k]
    print(f"  [export] {filename} "
          f"({grid.nx * grid.ny * grid.nz} voxels, {len(mats)} matériaux)")


def load_h5(filename: str) -> VoxelGrid:
    """Recharge une structure HDF5 en VoxelGrid (viewer v0.3, round-trip).

    Les matériaux sont réenregistrés dans l'ordre croissant des ids —
    c'est ainsi qu'export_h5 les écrit, les ids sont donc stables.
    """
    import h5py
    with h5py.File(filename, "r") as f:
        nx, ny, nz = f["grid"].shape
        g = VoxelGrid(nx, ny, nz, pitch=float(f.attrs["pitch_nm"]))
        for row in f["materials"][()]:
            name = row["name"].decode("utf-8") \
                if isinstance(row["name"], bytes) else str(row["name"])
            color = row["color"].decode("utf-8") \
                if isinstance(row["color"], bytes) else str(row["color"])
            mid = g.add_material(name, color)
            if mid != int(row["id"]):
                raise ValueError(f"{filename} : ids de matériaux non "
                                 f"consécutifs (attendu {mid}, trouvé "
                                 f"{int(row['id'])}) — fichier non écrit "
                                 f"par export_h5 ?")
        g.data = np.asarray(f["grid"][()], dtype=np.uint8)
    return g


def write_xmf(filename: str = "structure.h5") -> str | None:
    """Écrit le pont Xdmf (.xmf) pour visualiser `filename` dans ParaView.

    ParaView n'ouvre pas le .h5 directement (HDF5 maison) : le .xmf ne
    fait que pointer sur /grid — le .h5 reste la source unique, et les
    coordonnées sont en nanomètres réels (pas = pitch_nm lu dans les
    attrs). Attention : Xdmf énumère les dimensions du plus rapide au
    plus lent (z, y, x), l'inverse de NumPy.

    Le .xmf référence le .h5 par son **nom de fichier** (pas le chemin
    complet) : le couple se déplace ensemble. Retourne le chemin du .xmf
    écrit, None si h5py est absent ou le fichier est introuvable.
    """
    try:
        import h5py
    except ImportError:
        print("  [export] h5py absent : pont Xdmf ignoré "
              "(uv pip install h5py)")
        return None
    h5 = Path(filename)
    if not h5.exists():
        print(f"  [export] {filename} introuvable : pont Xdmf ignoré")
        return None
    with h5py.File(h5, "r") as f:
        nx, ny, nz = f["grid"].shape
        pitch = float(f.attrs["pitch_nm"])
    xml = f"""<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="voxelfab">
      <Topology TopologyType="3DCoRectMesh"
                NumberOfElements="{nz + 1} {ny + 1} {nx + 1}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>
        <DataItem Format="XML" Dimensions="3">{pitch} {pitch} {pitch}</DataItem>
      </Geometry>
      <Attribute Name="material" AttributeType="Scalar" Center="Cell">
        <DataItem Format="HDF" Dimensions="{nz} {ny} {nx}"
                  NumberType="UInt" Precision="1">{h5.name}:/grid</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    out = h5.with_suffix(".xmf")
    out.write_text(xml, encoding="utf-8")
    print(f"  [export] {out} ({nx}x{ny}x{nz}, pas {pitch} nm — "
          f"ouvrir le .xmf dans ParaView, à côté du .h5)")
    return str(out)


# --------------------------------------------------------------------------
# Rendus matplotlib + exports optionnels
# --------------------------------------------------------------------------

def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : pas de figure.")
        return None
    return plt


def plot_cross_section(grid: VoxelGrid, y: int, title: str,
                       filename: str = "flow_cmos.png"):
    """Coupe x-z à y fixe, colorée par matériau."""
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.colors import to_rgb
    sec = grid.data[:, y, :].T                      # (nz, nx)
    rgb = np.zeros((*sec.shape, 3))
    for mid in grid.materials.ids:
        rgb[sec == mid] = to_rgb(grid.materials.color(mid))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(rgb, origin="lower", aspect="equal",
              extent=[0, grid.nx, 0, grid.nz])
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("z (voxels)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"  [plot] coupe sauvegardée : {filename}")
    plt.close(fig)


def render_views(grid: VoxelGrid, filename: str = "mos3d.png",
                 title: str = "MOSFET 3D"):
    """Rendu 3D économique : projection du dessus (matériau de surface,
    via le champ de hauteur) + coupes x-z et y-z. O(N) vectorisé,
    contrairement à Axes3D.voxels qui est inutilisable au-delà de ~50k voxels."""
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.colors import to_rgb
    rgb = np.zeros((256, 3))
    for mid in grid.materials.ids:
        rgb[mid] = to_rgb(grid.materials.color(mid))

    # vue de dessus : couleur du matériau au sommet de chaque colonne
    h = grid.height_map()
    top = grid.data[grid._xs, grid._ys, np.clip(h, 0, grid.nz - 1)]
    top_view = rgb[top].transpose(1, 0, 2)          # (ny, nx, 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].imshow(top_view, origin="lower",
                   extent=[0, grid.nx, 0, grid.ny])
    axes[0].set_title("vue de dessus (surface)")
    axes[0].set_xlabel("x (vx)"); axes[0].set_ylabel("y (vx)")

    sec = grid.data[:, grid.ny // 2, :].T            # coupe x-z
    axes[1].imshow(rgb[sec], origin="lower", aspect="equal",
                   extent=[0, grid.nx, 0, grid.nz])
    axes[1].set_title(f"coupe x-z (y={grid.ny // 2})")
    axes[1].set_xlabel("x (vx)"); axes[1].set_ylabel("z (vx)")

    sec = grid.data[grid.nx // 2, :, :].T            # coupe y-z
    axes[2].imshow(rgb[sec], origin="lower", aspect="equal",
                   extent=[0, grid.ny, 0, grid.nz])
    axes[2].set_title(f"coupe y-z (x={grid.nx // 2})")
    axes[2].set_xlabel("y (vx)"); axes[2].set_ylabel("z (vx)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"  [plot] vues 3D sauvegardées : {filename}")
    plt.close(fig)


def export_vtk(grid: VoxelGrid, filename: str = "mos3d.vtk"):
    """Export .vtk pour ParaView/PyVista (si pyvista est installé)."""
    try:
        import pyvista as pv
    except ImportError:
        print("  [export] pyvista absent : export VTK ignoré "
              "(uv pip install 'voxelfab[view]')")
        return
    img = pv.ImageData(dimensions=(grid.nx + 1, grid.ny + 1, grid.nz + 1))
    img.cell_data["material"] = grid.data.flatten(order="F").astype(np.int32)
    img.save(filename)
    print(f"  [export] {filename} ({grid.nx * grid.ny * grid.nz} cellules)")


def export_stl(grid: VoxelGrid, prefix: str = "mos3d"):
    """Export STL par matériau (marching cubes, si scikit-image installé)."""
    try:
        from skimage import measure
        from stl import mesh as stl_mesh
    except ImportError:
        print("  [export] scikit-image/numpy-stl absent : export STL ignoré")
        return
    for mid in grid.materials.ids:
        if mid == Materials.AIR:
            continue
        vol = grid.data == mid
        if not vol.any():
            continue
        v, f, _, _ = measure.marching_cubes(vol.astype(np.uint8), level=0.5)
        m = stl_mesh.Mesh(np.zeros(f.shape[0], dtype=stl_mesh.Mesh.dtype))
        m.vectors = v[f]
        out = f"{prefix}_{grid.materials.name(mid).split()[0]}.stl"
        m.save(out)
        print(f"  [export] {out} ({f.shape[0]} faces)")
