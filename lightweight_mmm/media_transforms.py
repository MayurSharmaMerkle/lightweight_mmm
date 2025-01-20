import jax
import jax.numpy as jnp
import functools

def adstock(data: jnp.ndarray,
            gamma_alpha: float = 0.5,
            gamma_beta: float = 0.5,
            max_lag: int = 13,
            normalise: bool = True) -> jnp.ndarray:
    """Calculates the adstock value of a given array with gamma parameters.

    Args:
        data: Input array.
        gamma_alpha: Alpha parameter for the gamma distribution.
        gamma_beta: Beta parameter for the gamma distribution.
        max_lag: Maximum lag to consider for the adstock calculation. Default is 13.
        normalise: Whether to normalise the output value.

    Returns:
        The adstock output of the input array.
    """

    # # Check gamma parameters
    # if gamma_alpha <= 0 or gamma_beta <= 0:
    #     raise ValueError("Gamma parameters must be positive.")

    # Calculate weights based on gamma parameters and lag values
    lags = jnp.arange(1, max_lag + 1)
    weights = jnp.power(lags, (20 * gamma_alpha) - 1) * jnp.exp(-(10 * gamma_beta) * lags)
    
    # Handle potential invalid values in weights
    weights = jnp.where(jnp.isnan(weights) | jnp.isinf(weights), 0, weights)
    weights /= jnp.sum(weights) + 1e-6

    # # Debugging: Print weights to check for any issues
    # print("Weights:", weights)

    def adstock_internal_convolve(data: jnp.ndarray,
                                  weights: jnp.ndarray, 
                                  max_lag: int) -> jnp.ndarray:
        """Applies the convolution between the data and the weights for the adstock.

        Args:
            data: Input data.
            weights: Window weights for the adstock.
            max_lag: Number of lags the window has.

        Returns:
            The result values from convolving the data and the weights with padding.
        """
        window = jnp.concatenate([jnp.zeros(max_lag - 1), weights])
        return jax.scipy.signal.convolve(data, window[:, None], mode="same") / weights.sum()

    convolve_func = adstock_internal_convolve
    if data.ndim == 3:
        convolve_func = jax.vmap(adstock_internal_convolve, in_axes=(2, None, None), out_axes=2)

    adstock_values = convolve_func(data, weights, max_lag)

    # # Debugging: Print adstock_values before normalization
    # print("Adstock Values (before normalization):", adstock_values)

    return jax.lax.cond(
        normalise,
        lambda adstock_values: adstock_values / jnp.sum(weights),
        lambda adstock_values: adstock_values,
        operand=adstock_values)