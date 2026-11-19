"""Plot the representability change produced by IPSp pair relabeling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def _read_scan(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["orientation"] == "x"]
    if not rows:
        raise ValueError(f"no x-oriented scan rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "distance_A",
            "physical_energy_eV",
            "physical_radial_force_eV_A",
            "target_energy_eV",
            "target_radial_force_eV_A",
            "predicted_interaction_energy_eV",
            "predicted_radial_force_eV_A",
        )
    }


def _prediction_envelope(
    scans: Sequence[dict[str, np.ndarray]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack([scan[key] for scan in scans])
    return np.mean(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)


def plot(raw_path: Path, ips_paths: Sequence[Path], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _read_scan(raw_path)
    ips = [_read_scan(path) for path in ips_paths]
    distance = raw["distance_A"]
    if any(not np.array_equal(distance, scan["distance_A"]) for scan in ips):
        raise ValueError("raw and IPSp scans use different distance grids")

    ips_energy, ips_energy_min, ips_energy_max = _prediction_envelope(
        ips, "predicted_interaction_energy_eV"
    )
    ips_force, ips_force_min, ips_force_max = _prediction_envelope(
        ips, "predicted_radial_force_eV_A"
    )

    raw_color = "#A65141"
    ips_color = "#1F6E9C"
    target_color = "#171717"
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 5.8))

    axes[0, 0].plot(
        distance,
        ips[0]["physical_energy_eV"],
        color="#8A8A8A",
        linestyle=":",
        linewidth=2,
        label="Coulomb + LJ",
    )
    axes[0, 0].plot(
        distance,
        ips[0]["target_energy_eV"],
        color=ips_color,
        linewidth=2,
        label="IPSp-completed target",
    )
    axes[0, 0].set(xlim=(2.2, 8.0), ylabel="interaction energy (eV)")
    axes[0, 0].set_title("a  IPSp rewrites the pair target")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        distance,
        raw["target_energy_eV"],
        color=target_color,
        linestyle="--",
        linewidth=1.5,
        label="target",
    )
    axes[0, 1].plot(
        distance,
        raw["predicted_interaction_energy_eV"],
        color=raw_color,
        linewidth=1.8,
        label="MACE",
    )
    axes[0, 1].set(xlim=(3.5, 7.0), ylim=(-4.3, 0.2), ylabel="interaction energy (eV)")
    axes[0, 1].set_title(r"b  Raw labels, 5 $\mathrm{\AA}$ graph")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(
        distance,
        ips[0]["target_energy_eV"],
        color=target_color,
        linestyle="--",
        linewidth=1.5,
        label="target",
    )
    axes[1, 0].plot(distance, ips_energy, color=ips_color, linewidth=1.8, label="MACE")
    axes[1, 0].fill_between(
        distance,
        ips_energy_min,
        ips_energy_max,
        color=ips_color,
        alpha=0.2,
        linewidth=0,
        label="two-seed range",
    )
    axes[1, 0].set(
        xlim=(3.5, 5.5),
        xlabel=r"Na--Cl distance ($\mathrm{\AA}$)",
        ylabel="interaction energy (eV)",
    )
    axes[1, 0].set_title(r"c  IPSp labels, 5 $\mathrm{\AA}$ graph")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(
        distance,
        raw["target_radial_force_eV_A"],
        color=raw_color,
        linestyle="--",
        linewidth=1.4,
        label="raw target",
    )
    axes[1, 1].plot(
        distance,
        raw["predicted_radial_force_eV_A"],
        color=raw_color,
        linewidth=1.8,
        label="raw-label MACE",
    )
    axes[1, 1].plot(
        distance,
        ips[0]["target_radial_force_eV_A"],
        color=target_color,
        linestyle="--",
        linewidth=1.4,
        label="IPSp target",
    )
    axes[1, 1].plot(
        distance,
        ips_force,
        color=ips_color,
        linewidth=1.8,
        label="IPSp-label MACE",
    )
    axes[1, 1].fill_between(
        distance,
        ips_force_min,
        ips_force_max,
        color=ips_color,
        alpha=0.2,
        linewidth=0,
    )
    axes[1, 1].set(
        xlim=(4.0, 5.5),
        xlabel=r"Na--Cl distance ($\mathrm{\AA}$)",
        ylabel=r"radial force on Na (eV/$\mathrm{\AA}$)",
    )
    axes[1, 1].set_title("d  Force at edge removal")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.axvline(5.0, color="#666666", linestyle=":", linewidth=1.2)
        axis.tick_params(direction="out")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "mace_nacl_ips_pair_rewrite.pdf")
    figure.savefig(output_dir / "mace_nacl_ips_pair_rewrite.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--ips", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.raw, args.ips, args.output_dir)


if __name__ == "__main__":
    main()
