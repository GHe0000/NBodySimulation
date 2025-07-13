#include "io.h"

#include <iostream>
#include <fstream>
#include <filesystem>

SimIO::SimIO(const std::string& data_dir) : data_dir_(data_dir) {
    if (!std::filesystem::exists(data_dir_)) {
        try {
            std::filesystem::create_directory(data_dir_);
            std::cout << "Created data dir: " << data_dir_ << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating data dir: " << e.what() << std::endl;
        }
    }
}

void SimIO::save_data(const VectorOfVec2D& pos, const VectorOfVec2D& mom, double time) const {
    int time_ms = static_cast<int>(round(time * 1000));
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%s/%05d.bin", data_dir_.c_str(), time_ms);

    std::ofstream outfile(buffer, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file: " << buffer << std::endl;
        return;
    }

    outfile.write(reinterpret_cast<const char*>(pos.data()), pos.size() * sizeof(Vec2D));
    outfile.write(reinterpret_cast<const char*>(mom.data()), mom.size() * sizeof(Vec2D));
    outfile.close();
}
