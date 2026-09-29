import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.kalman import kalman_filter
from src.likelihood import innovation_loglikelihood
from src.simulate import simulate_lds


SEED = 11
N_STEPS = 1000


def main():
    _, y, params = simulate_lds(n_steps=N_STEPS, seed=SEED)

    A_true = params["A0"]
    scales = np.linspace(-0.08, 0.08, 17)

    print("delta_A00, log_likelihood")

    for delta in scales:
        A_test = A_true.copy()
        A_test[0, 0] += delta

        result = kalman_filter(
            y,
            A_test,
            params["H"],
            params["Q"],
            params["R"],
        )

        ll = innovation_loglikelihood(y, result, params["H"], params["R"])
        print(f"{delta:+.4f}, {ll:.2f}")


if __name__ == "__main__":
    main()
