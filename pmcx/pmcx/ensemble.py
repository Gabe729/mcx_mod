"""Host-side Mueller-table construction for mixtures of spherical scatterers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from _pmcx import mie_smatrix


def ensemble_smatrix(
    populations: Sequence[Mapping[str, float]],
    wavelength: float,
    structure_factors: Sequence[np.ndarray | None] | None = None,
) -> dict[str, np.ndarray | float]:
    """Build one medium's number-density-weighted spherical Mueller ensemble.

    Each population supplies ``radius`` (micrometres), ``rho``
    (particles/micrometre^3), ``nsph`` and ``nmed``.  The returned ``smatrix``
    has shape ``(1000, 4)`` and can be stacked by medium and passed to
    ``pmcx.run(..., smatrix=..., polmus=...)``.

    A structure factor, when supplied, multiplies all four raw Mueller elements
    at each angle.  Its integral rescales that population's scattering
    coefficient.  This is exact for identical spheres in the single-scattering
    with correlations approximation; it does not make non-spherical scattering
    representable.
    """
    if not populations:
        raise ValueError("at least one scatterer population is required")

    if structure_factors is None:
        structure_factors = [None] * len(populations)
    elif len(structure_factors) != len(populations):
        raise ValueError("structure_factors must match populations")

    nmed_values = np.asarray([float(pop["nmed"]) for pop in populations])
    if not np.allclose(nmed_values, nmed_values[0], rtol=0.0, atol=1e-7):
        raise ValueError("all populations in one medium must share nmed")

    mu = np.cos(np.linspace(0.0, np.pi, 1000, dtype=np.float64))
    order = np.argsort(mu)
    ensemble = np.zeros((1000, 4), dtype=np.float64)
    mus_total = 0.0
    monodisperse_matrix = None
    monodisperse_g = None

    for pop, structure_factor in zip(populations, structure_factors, strict=True):
        radius = float(pop["radius"])
        rho = float(pop["rho"])
        nsph = float(pop["nsph"])
        nmed = float(pop["nmed"])
        component = mie_smatrix(radius, rho, nsph, nmed, float(wavelength))
        matrix = np.asarray(component["smatrix"], dtype=np.float64)
        mus = float(component["mus"])
        if len(populations) == 1 and structure_factor is None:
            monodisperse_matrix = np.asarray(component["smatrix"], dtype=np.float32)
            monodisperse_g = float(component["g"])

        if structure_factor is not None:
            factor = np.asarray(structure_factor, dtype=np.float64)
            if factor.shape != (1000,) or np.any(~np.isfinite(factor)) or np.any(factor < 0.0):
                raise ValueError("each structure factor must be 1000 finite non-negative values")
            base_integral = np.trapz(matrix[order, 0], mu[order])
            adjusted_integral = np.trapz((matrix[:, 0] * factor)[order], mu[order])
            if base_integral <= 0.0 or adjusted_integral <= 0.0:
                raise ValueError("structure factor produced a non-positive scattering integral")
            matrix *= factor[:, None]
            mus *= adjusted_integral / base_integral

        ensemble += rho * matrix
        mus_total += mus

    if mus_total <= 0.0 or not np.all(np.isfinite(ensemble)):
        raise ValueError("ensemble must have positive scattering and finite Mueller elements")
    if monodisperse_matrix is not None:
        return {
            "smatrix": np.ascontiguousarray(monodisperse_matrix),
            "mus": float(mus_total),
            "g": float(monodisperse_g),
        }
    if ensemble[0, 0] + 1e-12 < np.max(ensemble[:, 0]):
        raise ValueError("custom phase function violates MCX's forward-angle rejection envelope")

    intensity = ensemble[:, 0]
    norm = np.trapz(intensity[order], mu[order])
    g = np.trapz((intensity * mu)[order], mu[order]) / norm

    return {
        "smatrix": np.ascontiguousarray(ensemble, dtype=np.float32),
        "mus": float(mus_total),
        "g": float(g),
    }
