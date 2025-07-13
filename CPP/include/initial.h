#ifndef INITIAL_H
#define INITIAL_H

#include "parameters.h"

namespace InitialConditions {

// 初始条件参数
const double POWER_LAW_N = -0.5; // 功率谱指数
const double SCALE_SIGMA = 0.2;  // 平滑尺度
const double FIELD_AMPLITUDE = 10.0; // 场振幅
const unsigned int SEED = 4;     // 随机种子

// 生成初始条件函数
void generate(VectorOfVec2D& pos, VectorOfVec2D& mom);

}

#endif
