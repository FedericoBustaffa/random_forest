#include "metrics.hpp"

#include <cassert>
#include <cstddef>
#include <set>

double accuracy_score(const std::vector<uint32_t>& predictions,
                      const std::vector<uint32_t>& correct)
{
    assert(predictions.size() == correct.size());
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}

double f1_score(const std::vector<uint32_t>& predictions,
                const std::vector<uint32_t>& correct)
{
    assert(predictions.size() == correct.size());
    std::set<uint32_t> labels(correct.begin(), correct.end());
    double total_f1 = 0.0;

    for (const auto& label : labels)
    {
        size_t tp = 0;
        size_t fp = 0;
        size_t fn = 0;

        for (size_t i = 0; i < predictions.size(); i++)
        {
            if (predictions[i] == label)
            {
                // positives
                if (predictions[i] == correct[i])
                {
                    // true positive
                    tp++;
                }
                else
                {
                    // false positive
                    fp++;
                }
            }
            else
            {
                // negatives
                if (predictions[i] != correct[i])
                {
                    // false negative
                    fn++;
                }
            }
        }

        double precision = (double)tp / (tp + fp);
        double recall = (double)tp / (tp + fn);
        total_f1 += (2 * precision * recall) / (precision + recall);
    }

    return total_f1 / labels.size();
}
