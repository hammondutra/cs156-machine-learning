"""CS156 Session 8: gradients and multivariate derivatives."""

import numpy as np


def scalar_backprop():
    """Reproduce the scalar sigmoid forward/backward pass from the workbook."""
    w0, w1, w2 = 2.00, -3.00, -3.00
    x0, x1 = -1.00, -2.00

    mul0 = w0 * x0
    mul1 = w1 * x1
    sum01 = mul0 + mul1
    sum012 = sum01 + w2

    neg = -sum012
    exp = np.exp(neg)
    plus1 = exp + 1
    invert = 1 / plus1

    d_invert = 1.00
    d_plus1 = d_invert * (-1 / plus1**2)
    d_exp = d_plus1
    d_neg = d_exp * exp
    d_sum012 = d_neg * (-1)

    d_w2, d_sum01 = d_sum012, d_sum012
    d_mul0, d_mul1 = d_sum01, d_sum01

    d_w1, d_x1 = d_mul1 * x1, d_mul1 * w1
    d_w0, d_x0 = d_mul0 * x0, d_mul0 * w0

    return {
        "output": invert,
        "d_w": np.array([d_w0, d_w1, d_w2]),
        "d_x": np.array([d_x0, d_x1]),
    }


def vector_backprop(w, x, b):
    """Vectorized forward/backward pass for sigmoid(w^T x + b)."""
    dot = w.T @ x
    summed = dot + b
    neg = -summed
    exp = np.exp(neg)
    plus1 = exp + 1
    invert = 1 / plus1

    d_invert = 1.00
    d_plus1 = d_invert * (-1 / plus1**2)
    d_exp = d_plus1
    d_neg = d_exp * exp
    d_summed = d_neg * (-1)

    d_b = d_summed
    d_dot = d_summed
    d_w = d_dot * x
    d_x = d_dot * w

    return invert, d_w, d_b, d_x


def sigmoid(w, x, b):
    """Sigmoid of a linear form."""
    z = w.T @ x + b
    return 1 / (1 + np.exp(-z))


def gradient(w, x, b):
    """Derivative of sigmoid(w^T x + b) with respect to w."""
    y = sigmoid(w, x, b)
    return y * (1 - y) * x


def gradient_descent(w, x, b, lr=0.1, iterations=1000):
    """Repeatedly step in the negative gradient direction."""
    w = w.copy()
    for _ in range(iterations):
        w = w - lr * gradient(w, x, b)
    return w


def jax_demo():
    """Return JAX gradients if JAX is installed."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed. Install jax and jaxlib to run this demo."
        ) from exc

    def sigmoid_example(params, x):
        dot = params["w"].T @ x
        summed = dot + params["b"]
        return 1 / (1 + jnp.exp(-summed))

    params = {"w": jnp.array([2.00, -3.00]), "b": -3.00}
    x = jnp.array([-1.00, -2.00])
    grad_sigmoid_example = jax.grad(sigmoid_example)

    return sigmoid_example(params, x), grad_sigmoid_example(params, x)


def main():
    scalar = scalar_backprop()
    print("Scalar sigmoid output:", scalar["output"])
    print("Scalar weight gradients:", scalar["d_w"])
    print("Scalar input gradients:", scalar["d_x"])

    w = np.array([2.00, -3.00])
    x = np.array([-1.00, -2.00])
    b = -3.00

    output, d_w, d_b, d_x = vector_backprop(w, x, b)
    print("\nVectorized sigmoid output:", output)
    print("d_w:", d_w)
    print("d_b:", d_b)
    print("d_x:", d_x)

    w_final = gradient_descent(w, x, b)
    print("\nFinal weights:", w_final)
    print("Final sigmoid output:", sigmoid(w_final, x, b))

    try:
        jax_output, jax_grads = jax_demo()
        print("\nJAX output:", jax_output)
        print("JAX gradients:", jax_grads)
    except RuntimeError as exc:
        print("\nJAX demo skipped:", exc)


if __name__ == "__main__":
    main()
