# Trained models (molten NaCl, PBE-D3)

MACE 0.3.16 models (float32, CPU-saved), seed 20260850, trained on the 640 NaCl training configurations.
Load with `mace.calculators.MACECalculator(model_paths=..., device="cpu", default_dtype="float32")`.
Models trained on an IPS target must be deployed with the exterior restored, E_theta - lambda * Delta:
use `screened_ips.md.ScreenedIPSCalculator(base, lam)` (or `scripts/run_md.py --lam`).

| File | Target | Deploy with | r_max | Updates | Notes |
|---|---|---|---|---|---|
| `nacl_plain_mace_r5.model` | E_ref | lam = 0 | 5 A | 20,000 | plain MACE |
| `nacl_screened_ips_lam0.52_r5.model` | T_0.5205 | lam = 0.5205 | 5 A | 20,000 | MACE + screened IPS, a priori strength 1/eps_inf |
| `nacl_screened_ips_lam0.44_r5.model` | T_0.44 | lam = 0.44 | 5 A | 20,000 | exploratory strength |
| `nacl_mace_les_r5.model` | E_ref | lam = 0 | 5 A | 20,000 | MACE + LES; needs the LES-enabled MACE fork |
| `nacl_plain_mace_r5_zbl.model` | E_ref | lam = 0 | 5 A | 20,000 | with ZBL repulsion; used for dynamics |
| `nacl_screened_ips_lam0.52_r5_zbl.model` | T_0.5205 | lam = 0.5205 | 5 A | 20,000 | with ZBL repulsion; used for dynamics |
| `nacl_plain_mace_r6_5k.model` | E_ref | lam = 0 | 6 A | 5,000 | cutoff control |
| `nacl_screened_ips_lam0.52_r6_5k.model` | T_0.5205 with R_c = 6 A | lam = 0.5205, rc = 6 | 6 A | 5,000 | cutoff control |
| `nacl_plain_mace_r7_5k.model` | E_ref | lam = 0 | 7 A | 5,000 | cutoff control |

`SHA256SUMS` lists checksums. All other models of the paper (all seeds, scans, SuperSalt salts and mixtures,
water, NequIP) can be retrained with `scripts/train_mace.py`; they will be deposited on Zenodo.
