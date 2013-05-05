#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <opencv2/ml/ml.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class classification
{
public:
    classification();
    void useSVM(Mat& trainData, Mat& trainLabels, Mat& testData, Mat& responses);
};

#endif // CLASSIFICATION_H
