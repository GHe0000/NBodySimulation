#include "utils.h"
#include "parameters.h"

#include <vector>
#include <omp.h>
#include <cmath>

void utils::wave_number(std::vector<double> &kx, std::vector<double> &ky) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double val = (i > N / 2) ? (i - N) : i;
        val *= 2.0 * M_PI / L;
        kx[i] = val;
        ky[i] = val; // 方形盒子
    }
}

double utils::k_pow_safe(double k_sq, double n) {
    if (k_sq == 0.0) return 0.0;
    return std::pow(k_sq, n / 2.0);
}
