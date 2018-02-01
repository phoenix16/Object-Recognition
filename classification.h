//
// Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
//

#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <opencv2/ml/ml.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class classification
{
public:
    void useSVM(const Mat& trainData, const Mat& trainLabels, const Mat& testData, Mat& responses);
};

#endif // CLASSIFICATION_H
