#ifndef METRICS_HPP
#define METRICS_HPP

#include <cstdint>
#include <vector>

float accuracy_score(const std::vector<uint8_t>& predictions,
                     const std::vector<uint8_t>& correct);

float f1_score(const std::vector<uint8_t>& predictions,
               const std::vector<uint8_t>& correct);

#endif
