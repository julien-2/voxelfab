"""Exports et rendus : coupes 2D, vues 3D, VTK, STL.

matplotlib est importé paresseusement (le coeur du moteur n'en dépend pas) ;
pyvista et scikit-image/numpy-stl restent optionnels (extras [view] / stl).
"""

from __future__ import annotations

import numpy as np

from .grid import Materials, VoxelGrid


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
