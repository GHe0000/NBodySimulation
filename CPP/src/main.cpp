#include <iostream>
#include <cmath>
#include <omp.h>

#include "parameters.h"
#include "poisson.h"
#include "initial.h"
#include "io.h"


int main() {
    omp_set_num_threads(NUM_THREADS);
    VectorOfVec2D positions(TOTAL_PARTICLES);
    VectorOfVec2D momenta(TOTAL_PARTICLES);

    std::cout << "Generating initial conditions..." << std::endl;
    InitialConditions::generate(positions, momenta);
    std::cout << "Initial conditions generated." << std::endl;

    std::cout << "Starting N-body simulation..." << std::endl;
    
    SimIO data_io("data");
    
    PoissonSolver solver;
    double time = A_INIT;

    data_io.save_data(positions, momenta, time);
    
    int step_count = 0;
    while (time < A_FINAL) {
        // --- Leap-frog (Kick-Drift-Kick) ---
        VectorOfVec2D acc = solver.calculate_acceleration(positions, time);
        
        // Kick
        double dt_da = PoissonSolver::da_dt(time);
        double kick_factor = (DT / 2.0) / dt_da;
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            momenta[i] += acc[i] * kick_factor;
        }

        // Drift
        double dpos_factor = DT / (time * time * dt_da);
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            positions[i] += momenta[i] * dpos_factor;
            positions[i] = positions[i].unaryExpr([&](double val){ return fmod(fmod(val, L) + L, L); });
        }

        time += DT;

        acc = solver.calculate_acceleration(positions, time);
        
        // Kick
        dt_da = PoissonSolver::da_dt(time);
        kick_factor = (DT / 2.0) / dt_da;
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            momenta[i] += acc[i] * kick_factor;
        }

        data_io.save_data(positions, momenta, time);
        step_count++;
        printf("Step %d: Simulation time a = %.4f\n", step_count, time);
    }
    std::cout << "Simulation finished." << std::endl;
    return 0;
}
