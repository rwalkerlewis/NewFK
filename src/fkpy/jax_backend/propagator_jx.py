"""JAX-friendly reflection/transmission helpers.

This module is the JAX-numpy equivalent of :mod:`fkpy.propagator` —
the same algebra (Wang & Herrmann 1980 compound matrix /
Kennett–Kerry R/T recursion) but implemented with ``jax.numpy`` so
that the layer matrices can be ``vmap``-ed over batches of (ω, k)
pairs and ``jit``-compiled to XLA for GPU residency.

The current release exposes a thin wrapper that constructs the
compound matrix using :func:`jax.numpy` operations element-by-element,
which lets the user ``vmap`` over arbitrary batch axes.  Native
single-kernel JIT compilation of the full recursion is reserved for
v0.2.

Usage::

    import jax
    from fkpy.jax_backend.propagator_jx import compound_matrix_jx

    # vmap over k for a single layer:
    cmat = jax.vmap(
        lambda k: compound_matrix_jx(k, kp_sq, ks_sq, d, mu)
    )(k_array)
"""

from __future__ import annotations

from typing import Any


def compound_matrix_jx(
    k: float,
    kp_sq: complex,
    ks_sq: complex,
    d_km: float,
    mu_layer: float,
) -> Any:
    """Return the 7×7 compound matrix using ``jax.numpy``.

    Parameters identical to the Numba ``_compound_matrix`` in
    :mod:`fkpy.propagator`; the implementation here is mathematically
    identical (W&H 1980) but uses ``jax.numpy`` so that ``jax.vmap``
    and ``jax.jit`` can be applied.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "JAX backend requires `jax`; install with `pip install fkpy[jax]`."
        ) from exc

    # JAX defaults to complex64.  The compound matrix carries
    # cosh(ν·d) terms whose moderate-kd values exceed complex64
    # dynamic range; promote to complex128 if x64 is enabled,
    # otherwise fall back to complex64 with a debug log.
    cdtype = (
        jnp.complex128
        if jax.config.read("jax_enable_x64")  # type: ignore[no-untyped-call]
        else jnp.complex64
    )

    k2 = k * k
    kka = kp_sq / k2
    kkb = ks_sq / k2
    r = 2.0 / kkb
    kd = k * d_km
    mu2 = 2.0 * mu_layer
    ra = jnp.sqrt(1.0 - kka)
    rb = jnp.sqrt(1.0 - kkb)
    r1 = 1.0 - 1.0 / r

    # sh_ch with overflow-suppressing rescaling
    def _sh_ch(a: Any, kdv: float) -> tuple[Any, Any, Any, Any]:
        y = kdv * a
        re = jnp.real(y)
        im = jnp.imag(y)
        ex = jnp.exp(-re)
        yhalf = 0.5 * (jnp.cos(im) + 1j * jnp.sin(im))
        xhalf = ex * ex * jnp.conj(yhalf)
        c = yhalf + xhalf
        diff = yhalf - xhalf
        yy = diff / a
        xx = diff * a
        return c, yy, xx, ex

    Ca, Ya, Xa, exa = _sh_ch(ra, kd)
    Cb, Yb, Xb, exb = _sh_ch(rb, kd)

    CaCb = Ca * Cb
    CaYb = Ca * Yb
    CaXb = Ca * Xb
    XaCb = Xa * Cb
    XaXb = Xa * Xb
    YaCb = Ya * Cb
    YaYb = Ya * Yb
    ex = exa * exb
    r2 = r * r
    r3 = r1 * r1

    # Build the 7×7 in row-major order, then stack.
    rows = [
        [
            ((1 + r3) * CaCb - XaXb - r3 * YaYb - 2 * r1 * ex) * r2,
            (XaCb - CaYb) * r / mu2,
            ((1 + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2,
            (YaCb - CaXb) * r / mu2,
            (2 * (CaCb - ex) - XaXb - YaYb) * r2 / (mu2 * mu2),
            0.0 + 0j,
            0.0 + 0j,
        ],
        [
            (r3 * YaCb - CaXb) * r * mu2,
            CaCb,
            (r1 * YaCb - CaXb) * r,
            -Ya * Xb,
            (YaCb - CaXb) * r / mu2,
            0.0 + 0j,
            0.0 + 0j,
        ],
        [
            2 * mu2 * r2 * (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb),
            2 * r * (r1 * CaYb - XaCb),
            2 * (CaCb - ((1 + r3) * CaCb - XaXb - r3 * YaYb - 2 * r1 * ex) * r2) + ex,
            -2 * (r1 * YaCb - CaXb) * r,
            -2 * ((1 + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2,
            0.0 + 0j,
            0.0 + 0j,
        ],
        [
            mu2 * r * (XaCb - r3 * CaYb),
            -Xa * Yb,
            -2 * r * (r1 * CaYb - XaCb) / 2,
            CaCb,
            (XaCb - CaYb) * r / mu2,
            0.0 + 0j,
            0.0 + 0j,
        ],
        [
            mu2 * mu2 * r2 * (2 * (CaCb - ex) * r3 - XaXb - r3 * r3 * YaYb),
            mu2 * r * (XaCb - r3 * CaYb),
            -2 * mu2 * r2 * (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb) / 2,
            (r3 * YaCb - CaXb) * r * mu2,
            ((1 + r3) * CaCb - XaXb - r3 * YaYb - 2 * r1 * ex) * r2,
            0.0 + 0j,
            0.0 + 0j,
        ],
        [0.0 + 0j] * 5 + [Cb, -2.0 * Yb / mu2],
        [0.0 + 0j] * 5 + [-mu2 * Xb / 2.0, Cb],
    ]
    return jnp.array(rows, dtype=cdtype)


__all__ = ["compound_matrix_jx"]
