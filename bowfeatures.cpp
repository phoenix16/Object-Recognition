/*
 * Developer : Prakriti Chintalapoodi - c.prakriti@gmail.com 
*/

#include "bowfeatures.h"


// Bag of Words Implementation:

// Step 1: Extract the SURF local feature vectors from each of training images.
//         Put all the local feature vectors extracted into a single set,
//         doesn't matter which feature vector came from which training image
// Step 2: Apply a clustering algorithm (e.g. k-means) over the set of local feature vectors and find N centroid coordinates
//         for dictionary size N and assign an id to each centroid.
//         This set of centroids = BoW vocabulary
// Step 3: Find the nearest centroid for each local feature vector.
//         Global feature vector of each image = normalized histogram where
//         i-th bin of the histogram = frequency of i-th word of the vocabulary in the given image
//                                   = how many times ith centroid occurred in that image
// DictionarySize = number of centroids for K means clustering = number of bins in BoW histogram = size of global feature vector of image


// Constructor
BOWfeatures::BOWfeatures(path trainpath, path testpath, int dictionarySize)
     :trainpath(trainpath),
      testpath(testpath),
      tc(CV_TERMCRIT_ITER, 10, 0.001),
      bowTrainer(dictionarySize, tc, 1, KMEANS_PP_CENTERS), // retries = 1
      SURFdetector(400),
      detector(FeatureDetector::create("SURF")),
      extractor(DescriptorExtractor::create("SURF")),
      matcher(DescriptorMatcher::create("FlannBased")),
      bowDE(extractor, matcher),
      trainData(0, dictionarySize, CV_32FC1),
      trainLabels(0, 1, CV_32FC1),
      testData(0, dictionarySize, CV_32FC1),
      testLabels(0, 1, CV_32FC1)
{
}

const Mat& BOWfeatures::getTrainData() const
{
    return trainData;
}

const Mat& BOWfeatures::getTrainLabels() const
{
    return trainLabels;
}


const Mat& BOWfeatures::getTestData() const
{
    return testData;
}


const Mat& BOWfeatures::getTestLabels() const
{
    return testLabels;
}


// Private function
// Recursively traverses a folder hierarchy, extracts features from the training images and adds them to the bowTrainer
void BOWfeatures::extractTrainingVocabulary(path basepath)
{
    for (directory_iterator it = directory_iterator(basepath); it != directory_iterator(); it++)
    {
        directory_entry entry = *it;

        if (is_directory(entry.path()))
        {
            cout << "\nProcessing directory " << entry.path().string() << endl;
            extractTrainingVocabulary(entry.path());
        }
        else
        {
            path entryPath = entry.path();
            if (entryPath.extension() == ".jpg")
            {
                cout << "\tProcessing file " << entryPath.string();
                Mat image = imread(entryPath.string());
                if (!image.empty())
                {
                    // Detect the SURF Keypoints
                    vector<KeyPoint> keypoints;
                    detector->detect(image, keypoints);
//                    SURFdetector.detect(image, keypoints);
                    cout << "...\tFound " << keypoints.size() << " keypoints" << endl;

                    if (keypoints.empty())
                    {
                        cerr << "Warning: Could not find keypoints in image: " << entryPath.string() << endl;
                    }
                    else
                    {
                        // Get the SURF Descriptors
                        Mat features;
                        extractor->compute(image, keypoints, features);   // features dim = 64 for SURF
                        bowTrainer.add(features);           // throw each feature vector into the bag
                    }
                }
                else
                {
                    cerr << "Warning: Could not read image: " << entryPath.string() << endl;
                }
            }
        }
    }
}

// Private function
// Recursively traverses a folder hierarchy, creates a BoW global feature vector (normalized histogram) for each image encountered
// After the dictionary has been constructed, images (training or test) can be described by extracting
// features from them and matching them with the features in the dictionary which are closest.
void BOWfeatures::extractBOWFeatures(path basepath, Mat& featureMat, Mat& labels)
{
    for (directory_iterator it = directory_iterator(basepath); it != directory_iterator(); it++)
    {
        directory_entry entry = *it;

        if (is_directory(entry.path()))
        {
            className = entry.path().filename().string();
            cout << "\nProcessing directory " << entry.path().string() << endl;
            extractBOWFeatures(entry.path(), featureMat, labels);
        }
        else
        {
            path entryPath = entry.path();
            if (entryPath.extension() == ".jpg")
            {
                cout << "\tProcessing file " << entryPath.string();
                Mat image = imread(entryPath.string());
                if (!image.empty())
                {
                    // Detect the SURF Keypoints
                    vector<KeyPoint> keypoints;
                    detector->detect(image, keypoints);
                    cout << "...\tFound " << keypoints.size() << " keypoints" << endl;

                    if (keypoints.empty())
                    {
                        cerr << "Warning: Could not find keypoints in image: " << entryPath.string() << endl;
                    }
                    else
                    {
                        Mat bowFeature; float label;
                        // does NOT compute SURF descriptors, finds global feature vector of image by finding nearest centroid, normalized histogram etc..
                        bowDE.compute(image, keypoints, bowFeature);
                        featureMat.push_back(bowFeature);

                        switch(hashit(className))
                        {
                            case _accordion:
                            label = 1.0; break;
                              case _barrel:
                            label = 2.0; break;
                              case _bonsai:
                            label = 3.0; break;
                        }
                        labels.push_back(label);
                    }
                }
                else
                {
                    cerr << "Warning: Could not read image: " << entryPath.string() << endl;
                }
            }
        }
    }
}

// Private function
// Maps the class names (so that the switch statement in the function extractBOWFeatures can work on strings)
BOWfeatures::classNameCode BOWfeatures::hashit(string const& inString)
{
    if (inString == "accordion") return _accordion;
    if (inString == "barrel") return _barrel;
    if (inString == "bonsai") return _bonsai;
}


// Public function
// Find the global features and their labels for the training and test data
void BOWfeatures::findBoWFeatures()
{
    cout << "Creating dictionary..." << endl;
    extractTrainingVocabulary(trainpath);

    cout << "\nClustering " << bowTrainer.descripotorsCount() << " features to form dictionary..." << endl;
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);
    // Dictionary contains the centroids of the training set features. This is NOT the training data to be fed into the classifier
    cout << "\nDictionary size = [Number of Centroids]x[Feature Dimension] = " << dictionary.rows << " x " << dictionary.cols << endl;

    cout << "\nProcessing Training data..." << endl;
    extractBOWFeatures(trainpath, trainData, trainLabels);
    cout << "\nTraining Data size = [Number of Training images]x[Dictionary size] = " << trainData.rows << " x " << trainData.cols << endl;

    cout << "\nProcessing Test data..." << endl;
    extractBOWFeatures(testpath, testData, testLabels);
    cout << "\nTest Data size = [Number of Test images]x[Dictionary size] = " << testData.rows << " x " << testData.cols << endl;
}


