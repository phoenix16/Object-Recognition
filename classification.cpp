/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "classification.h"

void classification::useSVM(const Mat& trainData, const Mat& trainLabels, const Mat& testData, Mat& responses)
{
    // Set up SVM parameters
    CvSVMParams SVM_params;

    // Train the SVM
    CvSVM SVM;
    cout << "\nTraining SVM Classifier..." << endl;
    SVM.train_auto(trainData, trainLabels, Mat(), Mat(), SVM_params);

    // Test the SVM
    int testSize = testData.rows;
    cout << "\nPredicting Test Labels using SVM..." << endl;
    for (int i = 0; i < testSize; i++)
    {
        // Classify each row of the testData matrix
        cout << SVM.predict(testData.row(i), true) << endl;
    }
}


//void useNormalBayes()
//{

//NormalBayesClassifier classifier;
//cout << "\nTraining classifier..." << endl;
//classifier.train(trainData, trainLabels);

//cout << "\nEvaluating classifier..." << endl;
//Mat responses;
//classifier.predict(testData, &responses);

//double errorRate = (double) countNonZero(testLabels - responses) / testData.rows;
//cout << "\nError rate: " << errorRate << endl;

//}
