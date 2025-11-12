#include "dataframe.hpp"

#include <regex>

#include "utils.hpp"

DataFrame::DataFrame(const std::vector<std::string>& content,
                     const std::vector<std::string>& headers)
    : m_Rows(content.size() / headers.size()), m_Cols(headers.size()),
      m_Headers(std::move(headers)), m_Content(std::move(content))
{
    // numerical fields regex
    std::regex numerical(
        R"(^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$)");

    for (size_t i = 0; i < m_Cols; i++)
    {
        // infer columns types
        if (std::regex_search(content[i], numerical))
            m_DataTypes.push_back(DataType::numerical);
        else
            m_DataTypes.push_back(DataType::categorical);
    }
}

std::pair<Matrix, Vector> DataFrame::toData() const
{
    std::vector<double> features(m_Rows * (m_Cols - 1));
    std::vector<double> targets(m_Rows);

    std::vector<double> tmp;

    for (size_t i = 0; i < m_Cols - 1; i++)
    {
        if (m_DataTypes[i] == DataType::categorical)
            tmp = encode(this, i);
        else if (m_DataTypes[i] == DataType::numerical)
            tmp = convert(this, i);

        for (size_t j = 0; j < m_Rows; j++)
            features[j * (m_Cols - 1) + i] = tmp[j];
    }

    if (m_DataTypes[m_Cols - 1] == DataType::categorical)
        targets = encode(this, m_Cols - 1);
    else if (m_DataTypes[m_Cols - 1] == DataType::numerical)
        targets = convert(this, m_Cols - 1);

    Matrix feature_matrix(features.data(), m_Rows, m_Cols - 1);
    Vector target_vector(targets.data(), m_Rows);

    return {feature_matrix, target_vector};
}

DataFrame::~DataFrame() {}
