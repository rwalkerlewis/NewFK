"""Layered velocity / Q model.

The model is a stack of horizontally homogeneous, isotropic, elastic
layers above a half-space.  Rows of the input array are
``[thickness_km, vp_kms, vs_kms, rho_gcc, Qp, Qs]``.  The thickness of
the last (half-space) row is ignored — Lupei Zhu's convention.

References
----------
Zhu & Rivera (2002) §2 — model definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from ._typing import F64Array
from .constants import EPSILON_THICKNESS_KM, EPSILON_VS_KMS


class ModelValidationError(ValueError):
    """Raised when a velocity model fails physical or geometric checks."""


@dataclass(frozen=True)
class LayeredModel:
    """1-D layered elastic model.

    Parameters
    ----------
    thickness_km
        Layer thicknesses in km.  The last entry is the half-space; its
        value is ignored (set to 0 internally).
    vp_kms, vs_kms
        P- and S-wave speeds, km/s.
    rho_gcc
        Density, g/cm³.
    qp, qs
        Quality factors for P and S.

    Notes
    -----
    Validity: thickness ≥ 0; ``vp_kms ≥ √2 · vs_kms`` (Poisson stable);
    ``rho_gcc > 0``; ``qp, qs ≥ 1`` (Futterman well-defined).  A liquid
    layer (vs == 0) is rounded up to ``EPSILON_VS_KMS`` to avoid SH
    division by zero, matching Zhu's `fk` convention.
    """

    thickness_km: F64Array
    vp_kms: F64Array
    vs_kms: F64Array
    rho_gcc: F64Array
    qp: F64Array
    qs: F64Array
    _frozen: bool = field(default=True, init=False, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_array(cls, arr: np.ndarray) -> Self:
        """Build a model from a ``(n_layers, 6)`` numpy array."""
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 6:
            msg = (
                "Model array must be 2-D with 6 columns "
                "[thickness_km, vp_kms, vs_kms, rho_gcc, Qp, Qs]; "
                f"got shape {a.shape}."
            )
            raise ModelValidationError(msg)
        if a.shape[0] < 1:
            raise ModelValidationError("Model must have at least one layer (the half-space).")
        thickness = a[:, 0].copy()
        thickness[-1] = 0.0  # half-space convention
        vs = np.where(a[:, 2] < EPSILON_VS_KMS, EPSILON_VS_KMS, a[:, 2])
        model = cls(
            thickness_km=thickness,
            vp_kms=a[:, 1].copy(),
            vs_kms=vs,
            rho_gcc=a[:, 3].copy(),
            qp=a[:, 4].copy(),
            qs=a[:, 5].copy(),
        )
        model._validate()
        return model

    @classmethod
    def from_text(cls, path: str | Path, model_format: str = "fkpy") -> Self:
        """Load from a whitespace-separated text file (``#`` comments).

        Parameters
        ----------
        path
            Path to the model file.
        model_format
            ``"fkpy"`` (default) — columns are
            ``[thickness_km, vp_kms, vs_kms, rho_gcc, Qp, Qs]``.
            ``"pyfk"`` — columns are
            ``[thickness_km, vs_kms, vp_kms, rho_gcc, Qs, Qp]``
            (Lupei Zhu's `fk` and ziyixi/pyfk convention).
        """
        arr = np.loadtxt(Path(path), comments="#", dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if model_format == "pyfk":
            # swap (vp, vs) and (Qp, Qs) columns
            arr = arr[:, [0, 2, 1, 3, 5, 4]]
        elif model_format != "fkpy":
            raise ValueError(
                f"Unknown model format {model_format!r}; expected 'fkpy' or 'pyfk'."
            )
        return cls.from_array(arr)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        """Number of layers (including the half-space)."""
        return int(self.thickness_km.size)

    @property
    def mu_gpa(self) -> F64Array:
        """Shear modulus per layer in GPa: ``μ = ρ vs²`` (units convert
        cleanly because g/cm³ × km²/s² = GPa)."""
        return self.rho_gcc * self.vs_kms**2

    @property
    def lam_plus_2mu_gpa(self) -> F64Array:
        """``λ + 2μ = ρ vp²`` in GPa."""
        return self.rho_gcc * self.vp_kms**2

    @property
    def xi(self) -> F64Array:
        """``ξ = vs² / vp²``.  Used in source jump (ZR-2002 eq. 16)."""
        return (self.vs_kms / self.vp_kms) ** 2

    def to_array(self) -> F64Array:
        """Return a ``(n_layers, 6)`` array round-trippable with
        :meth:`from_array`."""
        return np.column_stack(
            [
                self.thickness_km,
                self.vp_kms,
                self.vs_kms,
                self.rho_gcc,
                self.qp,
                self.qs,
            ]
        )

    def cumulative_depth_km(self) -> F64Array:
        """Depth (km) to the bottom interface of each layer; the last
        entry is the depth to the top of the half-space."""
        # The half-space thickness is 0, so cumsum naturally gives the
        # depth to the top of the half-space at index -1.
        return np.cumsum(self.thickness_km)

    # ------------------------------------------------------------------
    # Layer manipulation
    # ------------------------------------------------------------------
    def insert_interface(self, depth_km: float) -> tuple[Self, int]:
        """Return a copy with a zero-impedance interface at ``depth_km``.

        Returns
        -------
        new_model, layer_index
            ``layer_index`` is the index in the new model whose top is
            the inserted interface (i.e. the layer below the interface).

        Notes
        -----
        Inserting an interface at an existing boundary is a no-op
        (returns the same model unchanged) and gives back the matching
        layer index.  Used to coerce source/receiver depths to lie on a
        layer boundary, as in Lupei Zhu's `fk` (``add_layer`` in pyfk).
        """
        if depth_km < 0:
            raise ModelValidationError("depth_km must be ≥ 0")
        if depth_km == 0:
            return self, 0
        cum = self.cumulative_depth_km()
        # Find the layer index whose bottom is strictly below depth_km.
        idep = int(np.searchsorted(cum, depth_km, side="left"))
        if idep == self.n_layers:
            # Below the half-space top: just put the interface inside the
            # half-space.  The half-space remains, with a finite-thickness
            # layer of identical properties inserted above it.
            return self._do_insert(idep, depth_km - cum[-1])
        # If depth_km coincides with an existing boundary, no insertion.
        boundary = cum[idep - 1] if idep > 0 else 0.0
        if abs(depth_km - cum[idep]) < EPSILON_THICKNESS_KM:
            return self, idep + 1
        if abs(depth_km - boundary) < EPSILON_THICKNESS_KM:
            return self, idep
        # Split layer ``idep`` into two parts of thickness
        #   above  = depth_km - boundary
        #   below  = cum[idep] - depth_km
        above = depth_km - boundary
        below = self.thickness_km[idep] - above
        return self._do_split(idep, above, below)

    def _do_split(self, idep: int, above_km: float, below_km: float) -> tuple[Self, int]:
        new_thick = np.insert(self.thickness_km, idep + 1, below_km)
        new_thick[idep] = above_km
        new_thick[-1] = 0.0  # halfspace marker
        new_vp = np.insert(self.vp_kms, idep + 1, self.vp_kms[idep])
        new_vs = np.insert(self.vs_kms, idep + 1, self.vs_kms[idep])
        new_rho = np.insert(self.rho_gcc, idep + 1, self.rho_gcc[idep])
        new_qp = np.insert(self.qp, idep + 1, self.qp[idep])
        new_qs = np.insert(self.qs, idep + 1, self.qs[idep])
        return (
            type(self)(
                thickness_km=new_thick,
                vp_kms=new_vp,
                vs_kms=new_vs,
                rho_gcc=new_rho,
                qp=new_qp,
                qs=new_qs,
            ),
            idep + 1,
        )

    def _do_insert(self, idep: int, extra_km: float) -> tuple[Self, int]:
        # Insert a layer of finite thickness inside the half-space (which
        # carries the same physical properties).
        new_thick = np.append(self.thickness_km, 0.0)
        new_thick[-2] = extra_km
        new_vp = np.append(self.vp_kms, self.vp_kms[-1])
        new_vs = np.append(self.vs_kms, self.vs_kms[-1])
        new_rho = np.append(self.rho_gcc, self.rho_gcc[-1])
        new_qp = np.append(self.qp, self.qp[-1])
        new_qs = np.append(self.qs, self.qs[-1])
        return (
            type(self)(
                thickness_km=new_thick,
                vp_kms=new_vp,
                vs_kms=new_vs,
                rho_gcc=new_rho,
                qp=new_qp,
                qs=new_qs,
            ),
            idep + 1,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if np.any(self.thickness_km[:-1] < 0):
            raise ModelValidationError("Layer thicknesses must be non-negative.")
        if np.any(self.vp_kms <= 0) or np.any(self.vs_kms <= 0):
            raise ModelValidationError("Vp and Vs must be positive.")
        if np.any(self.rho_gcc <= 0):
            raise ModelValidationError("Density must be positive.")
        # Poisson stability: vp >= sqrt(2)*vs.  The factor 2/√3 is
        # cosmetic; we use the strict bound.
        if np.any(self.vp_kms < np.sqrt(2.0) * self.vs_kms - EPSILON_VS_KMS):
            raise ModelValidationError(
                "Vp must be ≥ √2 · Vs in every layer (Poisson stability)."
            )
        if np.any(self.qp < 1) or np.any(self.qs < 1):
            raise ModelValidationError("Q values must be ≥ 1 (Futterman).")

    def __repr__(self) -> str:
        return f"LayeredModel(n_layers={self.n_layers})"


__all__ = ["LayeredModel", "ModelValidationError"]
