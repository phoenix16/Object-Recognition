#include "bowfeatures.h"
#include "classification.h"

#define TRAINING_DATA_DIR "train/"   //location of the training data
#define TEST_DATA_DIR "test/"        //location of the testing data
#define DICTIONARY_SIZE 1000


int main()
{

    BOWfeatures feat_obj(path(TRAINING_DATA_DIR), path(TEST_DATA_DIR), DICTIONARY_SIZE);
    classification classify_obj;

    // Find the BoW features of the training and test images
    feat_obj.findBoWFeatures();

    Mat responses;
    classify_obj.useSVM(feat_obj.getTrainData(), feat_obj.getTrainLabels(), feat_obj.getTestData(), responses);

    return 0;
}
