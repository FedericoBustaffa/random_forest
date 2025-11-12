#include "utils.hpp"

double accuracy(const Tensor& predictions, const Tensor& correct)
{
    double counter = 0.0;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            counter++;
    }

    return counter / predictions.size();
}
