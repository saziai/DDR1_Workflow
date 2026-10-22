"""Shared configuration and helpers for the DDR1 workflow."""
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# ----------------------------------------------------------------
# Fingerprint k-NN regressor (Tanimoto similarity, binary arrays)
# ----------------------------------------------------------------
def tanimoto_kernel(X1, X2):
    """Tanimoto kernel for binary fingerprint arrays.
    Defined at module level (not in a notebook cell) so that SVR models using it
    can be saved with joblib and reloaded in a fresh session."""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    inter = X1 @ X2.T
    union = X1.sum(axis=1)[:, None] + X2.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class TanimotoKNN(BaseEstimator, RegressorMixin):
    """k-NN regressor using Tanimoto similarity on binary fingerprints.

    Matches RDKit's BulkTanimotoSimilarity on binary arrays. Operates
    directly on the numpy fingerprint matrix; no ExplicitBitVect needed.
    """

    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = (np.asarray(X) > 0).astype(np.float32)
        self.n_bits_ = self.X_.sum(axis=1)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        Q = (np.asarray(X) > 0).astype(np.float32)
        inter = Q @ self.X_.T
        union = Q.sum(axis=1)[:, None] + self.n_bits_[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        k = min(self.n_neighbors, sim.shape[1])
        idx = np.argsort(-sim, axis=1)[:, :k]
        return self.y_[idx].mean(axis=1)


# ----------------------------------------------------------------
# Ligand PDB writer (strict column layout)
# ----------------------------------------------------------------
_ELEMENT = {"A": "C", "C": "C", "N": "N", "NA": "N", "OA": "O", "O": "O",
            "S": "S", "SA": "S", "P": "P", "F": "F", "CL": "Cl", "BR": "Br",
            "I": "I", "FE": "Fe", "ZN": "Zn", "MG": "Mg", "CA": "Ca", "MN": "Mn"}


def write_ligand_pdb(pdbqt_path, out_pdb, resname: str = "LIG",
                     chain: str = "Z", resnum: int = 1001) -> int:
    """First MODEL of a Vina PDBQT -> heavy-atom PDB, strict column layout.

    Columns: name 13-16, resname 18-20, chain 22, resSeq 23-26,
    x 31-38, y 39-46, z 47-54, element 77-78. Returns atom count.
    """
    lines, per_element = [], {}
    started = False
    for line in Path(pdbqt_path).read_text(errors="replace").splitlines():
        if line.startswith("MODEL"):
            if started:
                break
            started = True
            continue
        if line.startswith("ENDMDL") and started:
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        atype = line[77:79].strip().upper()
        if atype in ("H", "HD", "HS"):
            continue
        el = _ELEMENT.get(atype, atype[:1] or "C")
        per_element[el] = per_element.get(el, 0) + 1
        aname = f"{el.upper()}{per_element[el]}"[:4]
        name_field = f" {aname:<3s}" if len(el) == 1 and len(aname) < 4 else f"{aname:<4s}"
        lines.append(
            f"HETATM{len(lines) + 1:5d} {name_field} {resname:>3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {el:>2s}"
        )
    Path(out_pdb).write_text("\n".join(lines) + "\nEND\n")
    return len(lines)


# ----------------------------------------------------------------
# RMSD: direct (receptor frame) and Kabsch (superposed)
# ----------------------------------------------------------------
def direct_rmsd(A, B) -> float:
    """Heavy-atom RMSD in the receptor frame. Use this for docking poses."""
    d = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def kabsch_rmsd(A, B) -> float:
    """Superposed RMSD. Secondary 'internal conformer' descriptor only."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    corr = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        corr[-1, -1] = -1
    R = Vt.T @ corr @ U.T
    d = Ac @ R.T - Bc
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# ----------------------------------------------------------------
# SMILES standardization (salt stripping + charge neutralization)
# ----------------------------------------------------------------
def strip_salts_and_neutralize(smi: str) -> Optional[str]:
    """Return the largest organic fragment with charges neutralized.

    Uses RDKit's MolStandardize. Returns None if the SMILES cannot be parsed.
    """
    from rdkit import Chem
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


# ----------------------------------------------------------------
# Path configuration
# ----------------------------------------------------------------
class DDR1Config:
    """Central configuration for the DDR1 workflow."""

    def __init__(self):
        self.is_colab = self._detect_colab()
        self.drive_root = None
        self.project_root = None
        self.execution_env = "colab" if self.is_colab else "local"
        self._paths: Dict[str, Path] = {}

    def _detect_colab(self) -> bool:
        try:
            from google.colab import drive  # noqa
            return True
        except ImportError:
            return False

    def _get_drive_root(self) -> Path:
        return Path("/content/drive/MyDrive") if self.is_colab else Path.cwd().resolve()

    def _resolve_project_root(self, project_name: str) -> Path:
        if self.is_colab:
            return self._get_drive_root() / project_name
        current_dir = Path.cwd().resolve()
        for d in (current_dir, *current_dir.parents):
            if d.name == project_name:
                return d
        for d in (current_dir, *current_dir.parents):
            if (d / "0-DDR1-CONFIG.ipynb").exists() or (d / "0_DDR1_CONFIG_SETUP.ipynb").exists():
                return d
            if (d / "data" / "raw").is_dir():
                return d
        return current_dir

    def setup(self, project_name: str = "DDR1_Workflow") -> Dict[str, Path]:
        self.drive_root = self._get_drive_root()
        self.project_root = self._resolve_project_root(project_name)
        self.project_root.mkdir(parents=True, exist_ok=True)
        manuscript = self.project_root / "candidate_docking" / "MANUSCRIPT_GRADE_DDR1_ANALYSIS"
        paths = {
            "PROJECT": self.project_root,
            "DRIVE_ROOT": self.drive_root,
            "DATA": self.project_root / "data",
            "DATA_RAW": self.project_root / "data" / "raw",
            "DATA_PROCESSED": self.project_root / "data" / "processed",
            "ML_MODEL": self.project_root / "ml_model",
            "ML_RESULTS": self.project_root / "ml_model" / "results",
            "ML_REPORTS": self.project_root / "ml_model" / "reports",
            "REDOCKING": self.project_root / "ddr1_redocking",
            "REDOCKING_RAW_STRUCTURE": self.project_root / "ddr1_redocking" / "raw_structure",
            "REDOCKING_RECEPTOR": self.project_root / "ddr1_redocking" / "receptor",
            "REDOCKING_LIGAND": self.project_root / "ddr1_redocking" / "ligand",
            "REDOCKING_DOCKING": self.project_root / "ddr1_redocking" / "docking",
            "REDOCKING_ANALYSIS": self.project_root / "ddr1_redocking" / "analysis",
            "REDOCKING_REPORTS": self.project_root / "ddr1_redocking" / "reports",
            "CANDIDATE": self.project_root / "candidate_docking",
            "CANDIDATE_MANUSCRIPT": manuscript,
            "CANDIDATE_LIGAND_PREP": manuscript / "ligand_preparation",
            "CANDIDATE_LIGAND_PDBQT": manuscript / "ligand_preparation" / "pdbqt",
            "CANDIDATE_DOCKING": manuscript / "candidate_docking",
            "CANDIDATE_POSE_REPRO": manuscript / "pose_reproducibility",
            "CANDIDATE_REPRESENTATIVE": manuscript / "representative_poses",
            "CANDIDATE_COMPLEXES": manuscript / "complexes",
            "CANDIDATE_PLIP": manuscript / "plip",
            "CANDIDATE_ANALYSIS": manuscript / "reports",
            "CANDIDATE_REPORTS": manuscript / "reports",
            "VISUALIZATION": self.project_root / "visualization",
            "VIS_FIGURES": self.project_root / "visualization" / "figures",
            "VIS_3D": self.project_root / "visualization" / "3D",
            "VIS_REPORTS": self.project_root / "visualization" / "reports",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        self._paths = paths
        return paths

    def mount_drive(self):
        if not self.is_colab:
            return self.drive_root
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            return self.drive_root
        except Exception as exc:
            print(f"Failed to mount Google Drive: {exc}")
            return None

    def get_config_dict(self) -> Dict[str, str]:
        if not self._paths:
            self.setup()
        return {k: str(v) for k, v in self._paths.items()}


CONFIG = DDR1Config()


def setup_paths(project_name: str = "DDR1_Workflow") -> Dict[str, Path]:
    if CONFIG.is_colab:
        CONFIG.mount_drive()
    paths = CONFIG.setup(project_name=project_name)
    print(f"Environment : {CONFIG.execution_env.upper()}")
    print(f"Project Root: {CONFIG.project_root}")
    print(f"Directories : {len(paths)}")
    return paths


def verify_paths(paths, required_keys=None) -> bool:
    keys = required_keys if required_keys else list(paths.keys())
    ok = True
    for k in keys:
        if k not in paths:
            print(f"  {k}: NOT FOUND")
            ok = False
            continue
        p = paths[k]
        exists = p.exists()
        print(f"  {k}: {'OK' if exists else 'MISSING'} - {p}")
        ok = ok and exists
    return ok


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_table(df, path_stem, also_csv: bool = False) -> Dict[str, str]:
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path = path_stem.with_suffix(".xlsx")
    df.to_excel(xlsx_path, index=False)
    written = {"xlsx": str(xlsx_path)}
    if also_csv:
        csv_path = path_stem.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)
    return written
