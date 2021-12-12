#pragma once

#include <string>
#include <mlpack/core.hpp>

using namespace std;

class DiabeticRetinopathy {
public:
    // image width pixes
    static const int WIDTH = 128;
    // image height pixels
    static const int HEIGHT = 128;
    // image depth
    static const int DEPTH = 3;
    // training labels file name
    inline static const string TRAINING_LABEL_FILE = "trainLabels.csv";

public:
    /**
     * @brief train model with data in {path} 
     * 
     * @param path 
     * @return int 
     */
    int Train(string path);

private: 
    /**
     * @brief convert human readable text to level(int) 
     * 
     * @param label 
     * @return int 
     */
    int convertLabelToLevel(string label);

    /**
     * @brief convert level to human readable text
     * 
     * @param level 
     * @return std::string 
     */
    string convertLevelToClass(int level);

    /**
     * @brief read training data(meta data stored in csv file) from specified path 
     * 
     * @param path 
     * @return int 
     */
    int readTrainingMeta(string path);

    /**
     * @brief read training data(images) from specfied path
     * 
     * @param path 
     * @return int 
     */
    int readTrainingData(string path);

private:
    arma::mat trainDataset;
};