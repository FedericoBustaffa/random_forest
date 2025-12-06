#ifndef METRICS_HPP
#define METRICS_HPP

#include <cstdint>
#include <vector>

double accuracy_score(const std::vector<uint32_t>& predictions,
                      const std::vector<uint32_t>& correct);

double f1_score(const std::vector<uint32_t>& predictions,
                const std::vector<uint32_t>& correct);

#endif
