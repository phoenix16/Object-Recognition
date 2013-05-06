#ifndef BOWFEATURES_H
#define BOWFEATURES_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

using namespace boost::filesystem;
using namespace std;
using namespace cv;

class BOWfeatures
{
private:
    path trainpath;
    path testpath;
    TermCriteria tc;
    BOWKMeansTrainer bowTrainer;
    SurfFeatureDetector SURFdetector;
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Ptr<DescriptorMatcher> matcher;
    BOWImgDescriptorExtractor bowDE;
    string className;
    enum classNameCode { _accordion, _barrel, _bonsai};
    Mat trainData, trainLabels, testData, testLabels;

    void extractTrainingVocabulary(path basepath);
    void extractBOWFeatures(path basepath, Mat& featureMat, Mat& labels);
    classNameCode hashit(string const& inString);
public:
    BOWfeatures(path trainpath, path testpath, int dictionarySize);
    void findBoWFeatures();
    const Mat& getTrainData() const;
    const Mat& getTrainLabels() const;
    const Mat& getTestData() const;
    const Mat& getTestLabels() const;
};

#endif // BOWFEATURES_H
