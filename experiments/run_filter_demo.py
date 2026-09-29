import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.kalman import kalman_filter, rts_smoother
from src.simulate import simulate_lds


SEED = 7
N_STEPS = 800
CHANGE_POINT = 450


def main():
    z, y, params = simulate_lds(
        n_steps=N_STEPS,
        seed=SEED,
        change_point=CHANGE_POINT,
    )

    result = kalman_filter(
        y,
        params["A0"],
        params["H"],
        params["Q"],
        params["R"],
    )
    smooth = rts_smoother(result, params["A0"])

    rmse_filter = np.sqrt(np.mean((result["filtered_mean"] - z) ** 2))
    rmse_smooth = np.sqrt(np.mean((smooth["smoothed_mean"] - z) ** 2))

    print(f"filter RMSE:   {rmse_filter:.4f}")
    print(f"smoother RMSE: {rmse_smooth:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(z[:, 0], label="true latent")
    plt.plot(result["filtered_mean"][:, 0], label="filtered", alpha=0.8)
    plt.plot(smooth["smoothed_mean"][:, 0], label="smoothed", alpha=0.8)
    plt.axvline(CHANGE_POINT, linestyle="--", label="change")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
