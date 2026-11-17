# Tests du pont Xdmf (.xmf) : ParaView lit le .xmf qui pointe sur le .h5.
#
# Contrat vérifié :
#   - Topology NumberOfElements = nz+1 ny+1 nx+1   (Xdmf énumère du plus
#     rapide au plus lent, l'inverse de NumPy) ;
#   - Geometry ORIGIN_DXDYDZ : origine 0 et pas = pitch_nm (nm réels) ;
#   - Attribute Center="Cell" sur /grid, Dimensions = nz ny nx,
#     HDF + UInt + précision 1 octet ;
#   - référence **relative** au .h5 (le couple se déplace ensemble) ;
#   - fichier introuvable -> None (pas d'exception).

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import VoxelGrid                      # noqa: E402
from voxelfab.export import export_h5, write_xmf   # noqa: E402


def _tiny_grid():
    """Grille 3×2×4, pas 2 nm : 2 voxels de Si puis 1 voxel de SiO2."""
    g = VoxelGrid(3, 2, 4, pitch=2.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :2] = Si
    g.data[:, :, 2] = SiO2
    return g, Si, SiO2


def test_xmf_roundtrip(tmp_path):
    """export_h5 puis write_xmf : XML conforme (ordre z-y-x, nm, HDF)."""
    g, Si, SiO2 = _tiny_grid()
    assert (g.data[:, :, 2] == SiO2).all()      # sanity de la fixture
    h5_path = tmp_path / "structure.h5"
    export_h5(g, str(h5_path))
    out = write_xmf(str(h5_path))
    assert out == str(tmp_path / "structure.xmf")

    root = ET.parse(out).getroot()
    assert root.tag == "Xdmf" and root.attrib["Version"] == "2.0"
    grid = root.find("./Domain/Grid")
    assert grid is not None and grid.attrib["Name"] == "voxelfab"

    # Topology : nz+1 ny+1 nx+1 (du plus rapide au plus lent)
    topo = grid.find("Topology")
    assert topo is not None
    assert topo.attrib["TopologyType"] == "3DCoRectMesh"
    assert topo.attrib["NumberOfElements"] == "5 3 4"      # nz=4, ny=2, nx=3

    # Geometry : origine 0, pas = pitch_nm (nanomètres réels)
    geo = grid.find("Geometry")
    assert geo is not None and geo.attrib["GeometryType"] == "ORIGIN_DXDYDZ"
    items = geo.findall("DataItem")
    assert len(items) == 2
    assert items[0].text.split() == ["0", "0", "0"]
    assert [float(v) for v in items[1].text.split()] == [2.0, 2.0, 2.0]

    # Attribute : au centre des cellules, Dimensions = nz ny nx, HDF relatif
    attr = grid.find("Attribute")
    assert attr is not None
    assert attr.attrib["Name"] == "material"
    assert attr.attrib["Center"] == "Cell"
    di = attr.find("DataItem")
    assert di is not None
    assert di.attrib["Dimensions"] == "4 2 3"             # nz ny nx
    assert di.attrib["Format"] == "HDF"
    assert di.attrib["NumberType"] == "UInt"
    assert di.attrib["Precision"] == "1"
    assert di.text is not None and di.text.strip() == "structure.h5:/grid"


def test_xmf_missing_file(tmp_path):
    """Fichier h5 introuvable : retourne None sans lever."""
    assert write_xmf(str(tmp_path / "nulle_part.h5")) is None
