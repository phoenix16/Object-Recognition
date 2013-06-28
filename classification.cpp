/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "classification.h"

classification::classification()
{
}

void classification::useSVM(const Mat& trainData, const Mat& trainLabels, const Mat& testData, Mat& responses)
{
    // Set up SVM parameters
    CvSVMParams SVM_params;
//    SVM_params.svm_type = CvSVM::C_SVC;
//    SVM_params.kernel_type = CvSVM::LINEAR;
//    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    // If non-linear kernel:
    //SVM_params.degree = 0;
    //SVM_params.gamma = 1;
    //SVM_params.coef0 = 0;
    //SVM_params.C = 1;
    //SVM_params.nu = 0;
    //SVM_params.p = 0;

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
