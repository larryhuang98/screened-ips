"""Long-wavelength charge diagnostics: S_ZZ(k), charge-current spectra and k-resolved force errors."""
import numpy as np

from .coulomb import charges_for


def charge_structure_factor(frames, kmax=2.0):
    """S_ZZ(k) = <|sum_i q_i exp(i k.r_i)|^2> / N for orthorhombic cells, formal charges.

    Returns sorted k (1/A, rounded to 1e-3) and lists of samples per k.
    """
    acc = {}
    for a in frames:
        L = np.diag(a.cell.array)
        q = charges_for(a)
        nmax = int(np.ceil(kmax * L.max() / (2 * np.pi)))
        g = np.arange(-nmax, nmax + 1)
        G = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        G = G[(G != 0).any(1)]
        K = 2 * np.pi * G / L
        km = np.linalg.norm(K, axis=1)
        m = km <= kmax
        K, km = K[m], km[m]
        rho = (q[None, :] * np.exp(1j * K @ a.positions.T)).sum(1)
        for kk, ss in zip(np.round(km, 3), np.abs(rho) ** 2 / len(a)):
            acc.setdefault(float(kk), []).append(float(ss))
    ks = sorted(acc)
    return ks, [acc[k] for k in ks]


def charge_current_spectra(positions, velocities, charges, box, dt_ps, nmax=4):
    """Longitudinal and transverse charge-current spectra C_L(k, f), C_T(k, f).

    positions, velocities: arrays (T, N, 3); box: orthorhombic lengths (3,); dt_ps: time between frames.
    J(k, t) = sum_i q_i v_i exp(i k.r_i) with k = 2 pi n / L along x, y, z (n = 1..nmax); correlation functions
    are averaged over time origins and directions, multiplied by a Hann window of half the trajectory length,
    and Fourier transformed. Returns {n: dict(k, f [THz], SL, ST)}.
    """
    R, V, q = np.asarray(positions), np.asarray(velocities), np.asarray(charges)
    T, N = R.shape[:2]

    def acf(x):
        x = x if x.ndim > 1 else x[:, None]
        f = np.fft.fft(x, n=2 * T, axis=0)
        c = np.fft.ifft(f * np.conj(f), axis=0)[:T].real.sum(1)
        return c / (T - np.arange(T))

    M = T // 2
    w = 0.5 * (1 + np.cos(np.pi * np.arange(M) / M))

    def spec(c):
        c = c[:M] * w
        return np.fft.rfft(np.concatenate([c, c[1:][::-1]])).real * dt_ps

    freq = np.fft.rfftfreq(2 * M - 1, d=dt_ps)
    out = {}
    for n in range(1, nmax + 1):
        CL, CT = 0, 0
        for ax in range(3):
            k = 2 * np.pi * n / box[ax]
            ph = np.exp(1j * k * R[:, :, ax])
            J = np.einsum("i,tic,ti->tc", q, V, ph)
            CL = CL + acf(J[:, ax]) / 3 / N
            CT = CT + acf(np.delete(J, ax, axis=1)) / 6 / N
        sL, sT = spec(CL), spec(CT)
        out[n] = {"k": float(2 * np.pi * n / np.mean(box)), "f": freq, "SL": sL, "ST": sT}
    return out


def force_error_spectrum(frames, dforces, nmax=7, bin_width=0.08):
    """k-resolved spectra of force errors dF (one (N, 3) array per frame).

    P_L(k) = <|sum_i q_i khat.dF_i exp(i k.r_i)|^2> / N (charge-weighted longitudinal), P_N (unweighted
    longitudinal) and P_T (charge-weighted transverse), binned in |k| (1/A). Values in (meV/A)^2.
    """
    acc = {}
    for a, dF in zip(frames, dforces):
        q = charges_for(a)
        r = a.get_positions()
        B = 2 * np.pi * np.linalg.inv(a.cell.array).T
        n = np.array([(i, j, k) for i in range(-nmax, nmax + 1) for j in range(-nmax, nmax + 1)
                      for k in range(0, nmax + 1) if (k > 0 or j > 0 or (j == 0 and i > 0))])
        K = n @ B
        km = np.linalg.norm(K, axis=1)
        kh = K / km[:, None]
        ph = np.exp(1j * r @ K.T)
        dl = dF @ kh.T
        AL = (q[:, None] * dl * ph).sum(0)
        AN = (dl * ph).sum(0)
        dt = dF[:, None, :] - dl[:, :, None] * kh[None]
        AT = np.einsum("i,ikc,ik->kc", q, dt, ph)
        N = len(a)
        for kk, pl, pn, pt in zip(km, np.abs(AL) ** 2 / N, np.abs(AN) ** 2 / N, (np.abs(AT) ** 2).sum(1) / (2 * N)):
            s = acc.setdefault(int(kk / bin_width), [0, 0.0, 0.0, 0.0, 0.0])
            s[0] += 1; s[1] += kk; s[2] += pl; s[3] += pn; s[4] += pt
    bs = sorted(acc)
    return {"k": [acc[b][1] / acc[b][0] for b in bs], "PL": [acc[b][2] / acc[b][0] * 1e6 for b in bs],
            "PN": [acc[b][3] / acc[b][0] * 1e6 for b in bs], "PT": [acc[b][4] / acc[b][0] * 1e6 for b in bs],
            "count": [acc[b][0] for b in bs]}
