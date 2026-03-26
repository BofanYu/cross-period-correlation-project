# -*- coding: utf-8 -*-
"""
models_gpu.py - GPU-friendly probabilistic models for spatial correlation inference.

This module provides utility functions to precompute distance-based matrices and
NumPyro models for the E and EAS correlation structures.

Usage:
- For each event, pass X = [epi_dist, epi_azimuth, vs30] into precompute_mats(X)
  to obtain the matrices distE, angdeg, and soil.
- modelE uses distsE.
- modelEAS uses distsE, distsAdeg, and distsS.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Use float64 by default for numerical stability. Disable it in the run script if float32 is preferred.
jax.config.update("jax_enable_x64", True)


# -------------------------
# Distance / dissimilarity matrices (precomputed once)
# -------------------------

def getEucDistanceFromPolar(X):
    """
    X: (n, 3) -> [epi_dist (km), epi_azimuth (rad), vs30 (m/s)]
    Euclidean distance computed from polar coordinates (r, theta) using chord distance.
    """
    X = jnp.asarray(X)
    r = X[:, 0]
    theta = X[:, 1]
    r_i = r.reshape(-1, 1)
    r_j = r.reshape(1, -1)
    th_i = theta.reshape(-1, 1)
    th_j = theta.reshape(1, -1)
    sq = r_i ** 2 + r_j ** 2 - 2.0 * r_i * r_j * jnp.cos(jnp.abs(th_j - th_i))
    sq = jnp.clip(sq, 0.0, jnp.inf)
    dist = jnp.sqrt(sq)
    dist = dist.at[jnp.diag_indices_from(dist)].set(0.0)
    return dist


def getAngDistanceFromPolar(X):
    """
    Angular distance in radians.
    """
    X = jnp.asarray(X)
    theta = X[:, 1]
    th_i = theta.reshape(-1, 1)
    th_j = theta.reshape(1, -1)
    cos_angle = jnp.cos(jnp.abs(th_j - th_i))
    return jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))


def getSoilDissimilarity(X):
    """
    L2 distance of Vs30 differences, equivalent to |vs30_i - vs30_j|.
    """
    X = jnp.asarray(X)
    v = X[:, 2]
    v_i = v.reshape(-1, 1)
    v_j = v.reshape(1, -1)
    sq = (v_i - v_j) ** 2
    sq = jnp.clip(sq, 0.0, jnp.inf)
    return jnp.sqrt(sq)


def precompute_mats(X):
    """
    Precompute matrices that do not depend on model parameters.
    Returns: dict(distE, angdeg, soil)
    """
    X = jnp.asarray(X)
    distE = getEucDistanceFromPolar(X)
    angdeg = getAngDistanceFromPolar(X) * (180.0 / jnp.pi)  # [0, 180]
    soil = getSoilDissimilarity(X)
    return {"distE": distE, "angdeg": angdeg, "soil": soil}


# -------------------------
# Distance-based kernels (JIT)
# -------------------------

@jax.jit
def rhoE_from_dist(distE, LEt, gammaE, nugget: float = 1e-6):
    """
    Exponential-power kernel: exp( - distE^gammaE / LEt ), with a nugget added on the diagonal.
    """
    K = jnp.exp(- (jnp.power(distE, gammaE)) / LEt)
    n = distE.shape[0]
    K = K.at[jnp.diag_indices(n)].add(nugget)
    return K

@jax.jit
def rhoEAS_from_dists(distE, angdeg, soil, LEt, gammaE, LA, LS, w, nugget: float = 1e-6):
    """
    Combined kernel: K = KE * ( w*KA + (1-w)*KS )
    - KE: exponential-power kernel based on distE.
    - KA: angular kernel from the original formulation, ((1 + angdeg/LA) * (1 - angdeg/180)^(180/LA)).
    - KS: exponential kernel for Vs30 differences, exp(- soil / LS).
    """
    KE = jnp.exp(- (jnp.power(distE, gammaE)) / LEt)

    # Angular kernel with clipped base to avoid numerical issues.
    base = 1.0 - angdeg / 180.0
    base = jnp.clip(base, 1e-12, 1.0)
    KA = (1.0 + (angdeg / LA)) * jnp.power(base, 180.0 / LA)

    KS = jnp.exp(- soil / LS)

    K = KE * (w * KA + (1.0 - w) * KS)
    n = distE.shape[0]
    K = K.at[jnp.diag_indices(n)].add(nugget)
    return K


# -------------------------
# Probabilistic models (only E and EAS)
# -------------------------

def modelE(distsE, eqids, z):
    """
    Correlation model using only the spatial distance term E.
    Priors match the original script:
      LE ~ InvGamma(2, 30)
      gamma2 ~ Beta(2, 2), gammaE = 2*gamma2, LEt = LE^gammaE
    """
    LE = numpyro.sample("LE", dist.InverseGamma(concentration=2, rate=30))
    gamma2 = numpyro.sample("gamma2", dist.Beta(2, 2))

    gammaE = numpyro.deterministic("gammaE", 2.0 * gamma2)
    LEt = numpyro.deterministic("LEt", jnp.power(LE, gammaE))

    for i, eqid in enumerate(eqids):
        K = rhoE_from_dist(distsE[i], LEt, gammaE)
        numpyro.sample(f"z_{eqid}", dist.MultivariateNormal(loc=0.0, covariance_matrix=K), obs=z[i])


def modelEAS(distsE, distsAdeg, distsS, eqids, z):
    """
    Combined E + A + S kernel model, equivalent in form to the original script.
    Priors:
      LE ~ InvGamma(2, 30)
      gamma2 ~ Beta(2, 2), gammaE = 2*gamma2, LEt = LE^gammaE
      LAt ~ Gamma(2, 0.25), LA = 180 / (4 + LAt)
      LS ~ InvGamma(2, 100)
      w ~ Beta(2, 2)
    """
    LE   = numpyro.sample("LE", dist.InverseGamma(concentration=2, rate=30))
    gamma2 = numpyro.sample("gamma2", dist.Beta(2, 2))
    LAt  = numpyro.sample("LAt", dist.Gamma(concentration=2, rate=0.25))
    LS   = numpyro.sample("LS", dist.InverseGamma(concentration=2, rate=100))
    w    = numpyro.sample("w", dist.Beta(2, 2))

    gammaE = numpyro.deterministic("gammaE", 2.0 * gamma2)
    LEt    = numpyro.deterministic("LEt", jnp.power(LE, gammaE))
    LA     = numpyro.deterministic("LA", 180.0 / (4.0 + LAt))

    for i, eqid in enumerate(eqids):
        K = rhoEAS_from_dists(distsE[i], distsAdeg[i], distsS[i], LEt, gammaE, LA, LS, w)
        numpyro.sample(f"z_{eqid}", dist.MultivariateNormal(loc=0.0, covariance_matrix=K), obs=z[i])
