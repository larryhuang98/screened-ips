#!/usr/bin/env python
"""Train MACE (or MACE-LES) for a fixed number of optimizer updates with the settings of the paper.

The run directory must contain data/{train,valid}.extxyz with REF_energy / REF_forces.
Defaults: r_max 5 A, two interaction layers, 32 channels, L_max 2, correlation order 3, 8 radial and 5 cutoff
basis functions, Adam (lr 3e-3, weight decay 5e-7), energy/force loss weights 10/50, batch 32, EMA 0.99,
ReduceLROnPlateau (factor 0.7), float32, no scaling, E0 = 0 for all elements, validation-selected model.
For NaCl at r_max = 5 A the average neighbour number was fixed to 16.30147 (measured on the training set);
otherwise it is computed from the training set.

usage: train_mace.py RUN_DIR SEED UPDATES [--device cpu|cuda] [--rmax 5] [--layers 2] [--channels 32]
                     [--batch 32] [--zbl] [--les] [--avg-neighbors auto|VALUE]
The final model is RUN_DIR/models/m.model.
"""
import argparse
import glob
import math

from ase.io import read

NACL_R5_NEIGHBORS = 16.301470588235293


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("seed", type=int)
    p.add_argument("updates", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rmax", type=float, default=5.0)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--zbl", action="store_true", help="add MACE's ZBL pair repulsion (models used for dynamics)")
    p.add_argument("--les", action="store_true", help="train MACE-LES (requires the LES-enabled MACE fork)")
    p.add_argument("--avg-neighbors", default=None,
                   help="'auto' to compute; default: 16.30147 for Na/Cl data at r_max = 5 A, else auto")
    a = p.parse_args()

    from mace.cli.run_train import run
    from mace.tools.arg_parser import build_default_arg_parser

    d = a.run_dir
    train = read(f"{d}/data/train.extxyz", ":")
    Z = sorted({int(z) for x in train for z in x.numbers})
    per = math.ceil(len(train) / a.batch)
    epochs = math.ceil(a.updates / per)
    ev = max(1, epochs // 200)
    avg = a.avg_neighbors
    if avg is None:
        avg = NACL_R5_NEIGHBORS if (Z == [11, 17] and a.rmax == 5.0) else "auto"
    opts = dict(
        name="m", seed=a.seed, work_dir=d, log_dir=f"{d}/logs", model_dir=f"{d}/models",
        checkpoints_dir=f"{d}/checkpoints", results_dir=f"{d}/results", downloads_dir=f"{d}/dl",
        train_file=f"{d}/data/train.extxyz", valid_file=f"{d}/data/valid.extxyz",
        energy_key="REF_energy", forces_key="REF_forces", model="MACELES" if a.les else "MACE",
        r_max=a.rmax, num_interactions=a.layers, max_L=2, correlation=3, num_channels=a.channels,
        num_radial_basis=8, num_cutoff_basis=5, atomic_numbers=str(Z),
        E0s="{" + ", ".join(f"{z}: 0.0" for z in Z) + "}", scaling="no_scaling", mean=0, std=1,
        optimizer="adam", lr=0.003, weight_decay=5e-7, loss="weighted", energy_weight=10, forces_weight=50,
        ema_decay=0.99, clip_grad=10, scheduler="ReduceLROnPlateau", scheduler_patience=max(20, 20 * ev),
        lr_factor=0.7, batch_size=a.batch, valid_batch_size=a.batch, max_num_epochs=epochs, patience=10 ** 6,
        eval_interval=ev, default_dtype="float32", device=a.device, compute_stress="False",
        compute_forces="True", error_table="PerAtomRMSE", num_workers=0, plot="False")
    if avg == "auto":
        opts["compute_avg_num_neighbors"] = "True"
    else:
        opts["avg_num_neighbors"] = float(avg)
        opts["compute_avg_num_neighbors"] = "False"
    parser = build_default_arg_parser()
    parser.set_defaults(amsgrad=False)
    flags = ["--ema", "--save_cpu"] + (["--pair_repulsion"] if a.zbl else [])
    if glob.glob(f"{d}/checkpoints/*.pt"):
        flags.append("--restart_latest")
    args = parser.parse_args([f"--{k}={v}" for k, v in opts.items()] + flags)
    print("n_train", len(train), "epochs", epochs, "eval_interval", ev, flush=True)
    run(args)


if __name__ == "__main__":
    main()
