#ifndef POISSON_H
#define POISSON_H

#include "parameters.h"

class PoissonSolver {
public:
    PoissonSolver();
    ~PoissonSolver();
    VectorOfVec2D calculate_acceleration(const VectorOfVec2D& pos, double a);
    static double da_dt(double a);

private:
    fftw_complex* delta_grid; // 真实空间密度网格
    fftw_complex* delta_f;    // 傅里叶空间密度网格
    fftw_complex* phi_f;      // 傅里叶空间引力势网格
    fftw_complex* phi_real;   // 真实空间引力势网格
    
    fftw_plan p_fwd; // 正向 FFT plan
    fftw_plan p_bwd; // 逆向 FFT plan

    std::vector<double> k_indices_x; // kx 波数
    std::vector<double> k_indices_y; // ky 波数
    
    // 辅助函数
    void cic(const VectorOfVec2D& pos_grid_units, fftw_complex* target); // 质量分配
    Vec2D interp(const VectorOfVec2D& data_grid, const Vec2D& x); // 插值
};

#endif
