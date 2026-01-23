#include "metrics.hpp"

#include <cassert>
#include <cstddef>
#include <set>

float accuracy_score(const std::vector<uint8_t>& predictions,
                     const std::vector<uint8_t>& correct)
{
    assert(predictions.size() == correct.size());
    float counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}

float f1_score(const std::vector<uint8_t>& predictions,
               const std::vector<uint8_t>& correct)
{
    assert(predictions.size() == correct.size());
    std::set<uint8_t> labels(correct.begin(), correct.end());
    labels.insert(predictions.begin(), predictions.end());

    float total_f1 = 0.0;
    size_t valid_labels = 0;

    for (const auto& label : labels)
    {
        size_t tp = 0;
        size_t fp = 0;
        size_t fn = 0;

        for (size_t i = 0; i < predictions.size(); i++)
        {
            if (label == predictions[i])
            {
                if (label == correct[i])
                    tp++; // true positive
                else
                    fp++; // false positive
            }
            else
            {
                if (label == correct[i])
                    fn++; // false negative
            }
        }

        if (tp == 0 && fp == 0 && fn == 0)
            continue;

        float precision = 0.0;
        if (tp + fp > 0)
            precision = (float)tp / (tp + fp);

        float recall = 0.0;
        if (tp + fn > 0)
            recall = (float)tp / (tp + fn);

        float f1 = 0.0;
        if (precision + recall > 0)
            f1 = (2 * precision * recall) / (precision + recall);

        total_f1 += f1;
        valid_labels++;
    }

    return valid_labels == 0 ? 0.0 : total_f1 / valid_labels;
}
