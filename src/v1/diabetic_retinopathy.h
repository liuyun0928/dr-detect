#pragma once

#include <string>
#include <mlpack/core.hpp>

class DiabeticRetinopathy {
public:
    // image width pixes
    static const int WIDTH = 128;
    // image height pixels
    static const int HEIGHT = 128;
    // image depth
    static const int DEPTH = 3;

public:
    /**
     * @brief train model with data in {path} 
     * 
     * @param path 
     * @return int 
     */
    int Train(std::string path);

private: 
    /**
     * @brief convert human readable text to level(int) 
     * 
     * @param label 
     * @return int 
     */
    int convertLabelToLevel(std::string label);

    /**
     * @brief convert level to human readable text
     * 
     * @param level 
     * @return std::string 
     */
    std::string convertLevelToClass(int level);

    /**
     * @brief read training data(meta data stored in csv file) from specified path 
     * 
     * @param path 
     * @return int 
     */
    int readTrainingMeta(std::string path);

    /**
     * @brief read training data(images) from specfied path
     * 
     * @param path 
     * @return int 
     */
    int readTrainingData(std::string path);

private:
    arma::mat trainDataset;
};