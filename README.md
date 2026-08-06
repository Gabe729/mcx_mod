*Last update: 06/08/2026*

# mcx_mod

`mcx_mod` is a research fork of [Monte Carlo eXtreme (MCX)](https://github.com/fangq/mcx), the GPU-accelerated photon-transport simulator created by Qianqian Fang. It retains the upstream MCX v2025.10 interfaces while adding material-aware scattering and birefringent polarised transport.

Fork release: `material-optics-v1`
Upstream compatibility version: `v2025.10`

## What this fork adds

| Addition | Interface | Purpose |
|---|---|---|
| Per-medium inverse-CDF phase functions | `Domain.MediaInverseCDF`, `--mediainvcdf`, or PMCX `mediainvcdf` | Assign a distinct scalar phase function to each labelled medium instead of reducing all scattering to one Henyey–Greenstein `g` value. |
| Custom polarised Mueller tables | PMCX `smatrix` and `polmus` | Supply a precomputed `(nmedia, 1000, 4)` scattering matrix and the corresponding scattering coefficients. |
| Ensemble Mueller construction | `pmcx.mie_smatrix()` and `pmcx.ensemble_smatrix()` | Construct number-density-weighted spherical-scatterer mixtures, optionally modified by structure factors. |
| Jones propagation | PMCX `jonesprop` | Propagate polarisation through per-medium linear birefringence and optical rotation between scattering events. |
| Explicit fork identity | CLI banner/`--version`, PMCX `mod_version()` | Distinguish the extension release from the upstream compatibility version. |

The two custom-scattering paths serve different transport modes: per-medium inverse-CDF tables are scalar, while Mueller tables operate in MCX's polarised mode.

## Build

Initialise the upstream submodules, then compile for the CUDA architecture of the target GPU:

```bash
git submodule update --init --recursive
make -C src -j8 CUGENCODE=-arch=sm_89
```

The executable is written to `bin/mcx`. Replace `sm_89` with the target architecture; do not rely on the Makefile's detected default for a production build.

To build PMCX:

```bash
cd pmcx
CMAKE_ARGS="-DMCX_CUDA_ARCH=sm_89" python -m pip install --no-build-isolation .
```

For upstream installation, input, output, replay, detector, boundary-condition, and performance documentation, use [mcx.space](https://mcx.space) and the [upstream repository](https://github.com/fangq/mcx). This README documents only the fork-specific surface.

## Per-medium inverse-CDF phase functions

Each labelled medium may own an equal-length inverse-CDF table of `cos(theta)`. Media without a table use MCX's native Henyey–Greenstein branch. The diagnostic counter records table draws and native fallbacks separately for each medium.

### PMCX in-memory form

```python
import numpy as np
import pmcx

result = pmcx.run({
    # standard MCX configuration fields ...
    "mediainvcdf": {
        "tables": np.vstack([dermis_table, blood_table]).astype(np.float32),
        "media": np.array([1, 2], dtype=np.int32),
        "wavelength_nm": 650.0,
        "background_policy": "native_hg",
    },
    "invcdfcount": 1,
})

hits = result["stat"]["invcdfhits"]
# hits[:, 0]: inverse-CDF draws; hits[:, 1]: native-HG draws
```

The supplied arrays contain only the interior inverse-CDF values. MCX adds the `-1` and `+1` endpoints.

### JSON document form

`Domain.MediaInverseCDF`, the CLI flag `--mediainvcdf`, and PMCX may instead load a document with schema `mcx_mod.per_material_invcdf`:

```json
{
  "schema": "mcx_mod.per_material_invcdf",
  "schema_version": "1.0.0",
  "wavelength_nm": 650.0,
  "background_policy": "native_hg",
  "media": [
    {"medium_index": 1, "label": "medium_1", "table_id": "table_1"}
  ],
  "tables": {
    "table_1": {
      "nphase": 4096,
      "n_interior": 4094,
      "interior_binary": {
        "path": "table_1.f32",
        "dtype": "<f4",
        "count": 4094,
        "order": "C"
      },
      "sha256_interior_f32": "..."
    }
  }
}
```

Binary sidecars resolve relative to the document. Inline `interior` arrays are also supported. Tables must be finite, monotonic, within `[-1, 1]`, equal-length, and hash-consistent when a hash is declared.

CLI use:

```bash
bin/mcx -f simulation.json --mediainvcdf tables.json --invcdfcount 1
```

## Custom Mueller tables and mixtures

`pmcx.mie_smatrix()` returns the host-side Mie table, scattering coefficient, and anisotropy for one spherical population. `pmcx.ensemble_smatrix()` combines multiple populations and optional structure factors:

```python
mixture = pmcx.ensemble_smatrix(
    populations=[
        {"radius": 0.5, "rho": 0.01, "nsph": 1.45, "nmed": 1.33},
        {"radius": 1.0, "rho": 0.002, "nsph": 1.40, "nmed": 1.33},
    ],
    wavelength=0.65,
)

result = pmcx.run({
    # standard polarised MCX configuration fields ...
    "smatrix": np.stack([mixture["smatrix"]]),
    "polmus": np.array([mixture["mus"]], dtype=np.float32),
})
```

The mixture helper is exact only within its stated spherical, single-scattering-with-correlations approximation. A structure factor cannot make a non-spherical particle representable.

## Jones propagation

Supply `jonesprop` alongside MCX's existing `polprop` table:

```python
cfg["jonesprop"] = np.array([
    # ne,      chi (degrees/mm), Bx,  By,  Bz
    [0.0,      0.0,              0.0, 0.0, 0.0],
    [1.33005,  0.0,              0.0, 1.0, 0.0],
], dtype=np.float32)
```

Rows are indexed by medium label, including background row 0. The path propagates the Stokes state through a double-precision Jones/N-matrix calculation between scattering events. `lambda` is in nm; `chi` is scaled consistently with `unitinmm`.

## Important constraints

- Per-medium inverse-CDF tables and polarised mode are mutually exclusive in this release. Supplying both is a hard error.
- Per-medium inverse-CDF tables require labelled media (`mediabyte <= 4`) and are not implemented in MCXLAB. Unsupported MCXLAB fields fail explicitly.
- One table document represents one material set at one wavelength. Multi-wavelength studies require separate runs.
- All per-medium inverse-CDF rows must have the same length. Missing rows deliberately fall back to native Henyey–Greenstein scattering.
- `invcdfcount` is off by default because it adds one atomic operation per scattering event. Enable it for validation, not routine campaigns.
- The fork currently supports at most 1000 combined optical properties and detectors because of the enlarged birefringence property block.
- A custom Mueller table must have shape `(nmedia, 1000, 4)` and requires `polprop`; `polmus` must contain one finite, non-negative value per polarised medium.

## Validation and provenance

The release is built as a linear, auditable extension of upstream MCX:

- upstream base: `7826eb9`
- ensemble Mueller tables and double-precision Jones transport: `3cd0e9e`
- tagged fork release: `material-optics-v1`

The extension commits were built and exercised through both the CLI and PMCX on NVIDIA CUDA hardware. Validation included upstream scalar parity, analytic ballistic polarisation cases, host/device Mueller-table agreement, inverse-CDF sampling moments, explicit per-medium execution counters, failure-mode checks, and performance measurements. The diagnostic instrumentation is intentionally disabled by default.

## Licence and citation

This fork remains licensed under GNU GPL v3; see [LICENSE.txt](LICENSE.txt). MCX is authored and maintained upstream by Qianqian Fang and contributors. Cite the relevant MCX papers listed by the [upstream project](https://github.com/fangq/mcx#reference), plus any method-specific sources required by the phase functions or material models used in a study.
