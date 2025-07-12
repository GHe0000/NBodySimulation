#ifndef IO_H
#define IO_H

#include "parameters.h"

#include <string>

class SimIO {
    public:
        explicit SimIO(const std::string& data_dir = "./data");
        void save_data(const VectorOfVec2D& pos, const VectorOfVec2D& mom, double time) const;
    private:
        std::string data_dir_;
};

#endif
