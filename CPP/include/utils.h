// 一些方便实用的小函数

#ifndef UTILS_H
#define UTILS_H

#include <vector>

namespace utils {
void wave_number(std::vector<double>& kx, std::vector<double>& ky);
double k_pow_safe(double k_sq, double n);
}
#endif
