from raytrace_mirror import *


def main():
    gradient_descent(
        1e-7,
        5e-7,
        -0.5,
        0.0375,
        0.1125,
        0.05625,
        0.03,
        n_iter=5000,
        Nr=20,
        No=20,
        plt=True,
    )


if __name__ == "__main__":
    main()
