#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <tclap/CmdLine.h>
#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "common_code.hpp"
#include "funcionesAuxiliares.hpp"
#include "opencv2/imgcodecs.hpp"

#define IMG_WIDTH 400

void kmeans(std::vector<int> ndescs_per_sample, std::vector<std::string> categories, int ntrain, int keywords, int dict_runs,  cv::Mat train_descs, std::vector<std::vector<int>> train_samples, std::vector<int> &train_labels_v, cv::Ptr<cv::ml::SVM> &classifier, cv::Mat &train_bovw, cv::Ptr<cv::ml::KNearest> &dict, cv::Mat &keyws, int vecinos, std::string opcion){
    CV_Assert(ndescs_per_sample.size() == (categories.size()*ntrain));
    std::clog << "\t\tDescriptors size = " << train_descs.rows*train_descs.cols * sizeof(float) / (1024.0 *1024.0) << " MiB." << std::endl;
    std::clog << "\tGenerating " << keywords << " keywords ..." << std::endl;
    cv::Mat labels;
    double compactness = cv::kmeans(train_descs, keywords, labels,
                                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
                                    dict_runs,
                                    cv::KmeansFlags::KMEANS_PP_CENTERS, //cv::KMEANS_RANDOM_CENTERS,
                                    keyws);
    CV_Assert(keywords == keyws.rows);
    //free not needed memory
    labels.release();

    std::clog << "\tGenerating the dictionary ... " << std::endl;
    dict = cv::ml::KNearest::create();
    dict->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    dict->setDefaultK(vecinos);
    dict->setIsClassifier(true);
    cv::Mat indexes(keyws.rows, 1, CV_32S);
    for (int i = 0; i < keyws.rows; ++i)
        indexes.at<int>(i) = i;
    dict->train(keyws, cv::ml::ROW_SAMPLE, indexes);
    std::clog << "\tDictionary compactness " << compactness << std::endl;

    std::clog << "\tTrain classifier ... " << std::endl;

    //For each train image, compute the corresponding bovw.
    std::clog << "\t\tGenerating the a bovw descriptor per train image." << std::endl;
    int row_start = 0;
    train_labels_v.resize(0);
    for (size_t c = 0, i = 0; c < train_samples.size(); ++c)
        for (size_t s = 0; s < train_samples[c].size(); ++s, ++i)
        {
            cv::Mat descriptors = train_descs.rowRange(row_start, row_start + ndescs_per_sample[i]);
            row_start += ndescs_per_sample[i];
            cv::Mat bovw = compute_bovw(dict, keyws.rows, descriptors);
            train_labels_v.push_back(c);
            if (train_bovw.empty())
                train_bovw = bovw;
            else
            {
                cv::Mat dst;
                cv::vconcat(train_bovw, bovw, dst);
                train_bovw = dst;
            }
        }

    //free not needed memory
    train_descs.release();

    //Create the classifier.
    //Train a KNN classifier using the training bovws like patterns.

    if(opcion=="SVM" || opcion == "svm"){
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::CHI2);
        svm->setC(1);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
        classifier = svm;
    }
}


void kmenores(std::vector<int> ndescs_per_sample, std::vector<std::string> categories, int ntrain, int keywords, int dict_runs,  cv::Mat train_descs, std::vector<std::vector<int>> train_samples, std::vector<float> &train_labels_v, cv::Ptr<cv::ml::StatModel> &classifier, cv::Mat &train_bovw, cv::Ptr<cv::ml::KNearest> &dict, cv::Mat &keyws, int vecinos, std::string opcion){
    CV_Assert(ndescs_per_sample.size() == (categories.size()*ntrain));
    std::clog << "\t\tDescriptors size = " << train_descs.rows*train_descs.cols * sizeof(float) / (1024.0 *1024.0) << " MiB." << std::endl;
    std::clog << "\tGenerating " << keywords << " keywords ..." << std::endl;
    cv::Mat labels;
    double compactness = cv::kmeans(train_descs, keywords, labels,
                                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
                                    dict_runs,
                                    cv::KmeansFlags::KMEANS_PP_CENTERS, //cv::KMEANS_RANDOM_CENTERS,
                                    keyws);
    CV_Assert(keywords == keyws.rows);
    //free not needed memory
    labels.release();

    std::clog << "\tGenerating the dictionary ... " << std::endl;
    dict = cv::ml::KNearest::create();
    dict->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    dict->setDefaultK(vecinos);
    dict->setIsClassifier(true);
    cv::Mat indexes(keyws.rows, 1, CV_32S);
    for (int i = 0; i < keyws.rows; ++i)
        indexes.at<int>(i) = i;
    dict->train(keyws, cv::ml::ROW_SAMPLE, indexes);
    std::clog << "\tDictionary compactness " << compactness << std::endl;

    std::clog << "\tTrain classifier ... " << std::endl;

    //For each train image, compute the corresponding bovw.
    std::clog << "\t\tGenerating the a bovw descriptor per train image." << std::endl;
    int row_start = 0;
    train_labels_v.resize(0);
    for (size_t c = 0, i = 0; c < train_samples.size(); ++c)
        for (size_t s = 0; s < train_samples[c].size(); ++s, ++i)
        {
            cv::Mat descriptors = train_descs.rowRange(row_start, row_start + ndescs_per_sample[i]);
            row_start += ndescs_per_sample[i];
            cv::Mat bovw = compute_bovw(dict, keyws.rows, descriptors);
            train_labels_v.push_back(c);
            if (train_bovw.empty())
                train_bovw = bovw;
            else
            {
                cv::Mat dst;
                cv::vconcat(train_bovw, bovw, dst);
                train_bovw = dst;
            }
        }

    //free not needed memory
    train_descs.release();

    //Create the classifier.
        //Train a KNN classifier using the training bovws like patterns.

    if(opcion=="KNN" || opcion == "knn"){
        cv::Ptr<cv::ml::KNearest> knnClassifier = cv::ml::KNearest::create();
        knnClassifier->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);            
        knnClassifier->setDefaultK(vecinos);
        knnClassifier->setIsClassifier(true);
        classifier = knnClassifier;
    }
}

/*
void SURF(std::vector<std::vector<int>> train_samples, std::vector<std::string> categories){
    TCLAP::CmdLine cmd("Train and test a BoVW model", ' ', "0.0");
    TCLAP::ValueArg<std::string> basenameArg("", "basename", "basename for the dataset.", false, "./data", "pathname");
    cmd.add(basenameArg);
    for (size_t c = 0; c < train_samples.size(); ++c){
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < train_samples[c].size(); ++s){
            std::string filename = compute_sample_filename(basenameArg.getValue(), categories[c], train_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            int minHessian = 400;
            cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
            std::vector<cv::KeyPoint> keypoints_1;
            cv::Mat img_keypoints_1;
            detector->detect(img, keypoints_1);
            cv::drawKeypoints(img, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
            cv::imshow("Keypoints", img_keypoints_1);
            cv::waitKey(0);
        }
    }
}*/


bool digito(std::string nombre){
    if(nombre[0]=='0'){
        return true;   
    }else {
        return false;
    }
}

void SURF(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm){
    rRates.resize(n_runsArg, 0.0);
    cv::Mat train_bovw;
    cv::Ptr<cv::ml::KNearest> dict;
    cv::Mat keyws;

    int sift_type = 0;

    double best_rRate = 0.0;
    for (size_t c = 0; c < train_samples.size(); ++c){
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < train_samples[c].size(); ++s){
            
            std::string filename = compute_sample_filename(basenameArg, categories[c], train_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty()){
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
                exit(-1);
            }else{
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));                    
                
                int minHessian = 400;
                cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);
                std::vector<cv::KeyPoint> keypoint;
                cv::Mat descriptor;
                surf->detect(img, keypoint);
                surf->compute(img, keypoint, descriptor);
                cv::Mat descs = descriptor;
                if (train_descs.empty())
                    train_descs = descriptor;
                else{
                    cv::Mat dst;
                    cv::vconcat(train_descs, descs, dst);
                    train_descs = dst;
                }
                ndescs_per_sample.push_back(descs.rows);
                
            }
        }
    }

    std::clog << std::endl;
    cv::Ptr<cv::ml::StatModel> classifier;

    //free not needed memory.
    if(opcion == "SVM" || opcion == "svm"){
        std::vector<int> train_labels_v;
        train_labels_v.resize(0);
        kmeans(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, svm, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
        svm->train(td);
    }else if(opcion == "knn" || opcion == "KNN"){
        std::vector<float> train_labels_v;
        train_labels_v.resize(0);
        kmenores(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, classifier, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        classifier->train(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
    }
    train_bovw.release();
    std::clog << "Testing .... " << std::endl;

    //Para cada prueba se realiza la comprobación con su descriptor
    std::clog << "\tCompute image descriptors for test images..." << std::endl;
    cv::Mat test_bovw;
    std::vector<float> true_labels;
    true_labels.resize(0);
    for (size_t c = 0; c < test_samples.size(); ++c)
    {
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < test_samples[c].size(); ++s)
        {
            std::string filename = compute_sample_filename(basenameArg, categories[c], test_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty())
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
            else
            {
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));

                int minHessian = 400;
                cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);
                std::vector<cv::KeyPoint> keypoint;
                cv::Mat descriptor;
                surf->detect(img, keypoint);
                surf->compute(img, keypoint, descriptor);
                cv::Mat descs = descriptor;
                
                cv::Mat bovw = compute_bovw(dict, keyws.rows, descs);
                if (test_bovw.empty())
                    test_bovw = bovw;
                else
                {
                    cv::Mat dst;
                    cv::vconcat(test_bovw, bovw, dst);
                    test_bovw = dst;
                }
                true_labels.push_back(c);
            }
        }
    }
    std::clog << std::endl;
    std::clog << "\tThere are " << test_bovw.rows << " test images." << std::endl;

    //Clasificación del test de las muestras
    std::clog << "\tClassifying test images." << std::endl;
    cv::Mat predicted_labels;

    if(opcion == "SVM" || opcion == "svm"){
        svm->predict(test_bovw, predicted_labels);
    }else if(opcion == "knn" || opcion == "KNN"){
        classifier->predict(test_bovw, predicted_labels);
    }
    CV_Assert(predicted_labels.depth() == CV_32F);
    CV_Assert(predicted_labels.rows == test_bovw.rows);
    CV_Assert(predicted_labels.rows == true_labels.size());

    //Computa la matrzi de confusión
    std::clog << "\tComputing confusion matrix." << std::endl;
    cv::Mat confusion_mat = compute_confusion_matrix(categories.size(), cv::Mat(true_labels), predicted_labels);
    CV_Assert(int(cv::sum(confusion_mat)[0]) == test_bovw.rows);
    double rRate_mean, rRate_dev;
    compute_recognition_rate(confusion_mat, rRate_mean, rRate_dev);
    std::cerr << "Recognition rate mean = " << rRate_mean * 100 << "% dev " << rRate_dev * 100 << std::endl;
    rRates[trail]=rRate_mean;

    if (trail==0 || rRate_mean > best_rRate )
    {
        best_dictionary = dict;
        best_classifier = classifier;
        best_rRate = rRate_mean;
    }
}


void SIFT(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm){
    rRates.resize(n_runsArg, 0.0);
    cv::Ptr<cv::ml::StatModel> classifier;
    cv::Mat train_bovw;
    cv::Ptr<cv::ml::KNearest> dict;
    cv::Mat keyws;

    int sift_type = 0;

    std::vector<int> siftScales{ 9, 13 }; // 5 , 9
    double best_rRate = 0.0;
    for (size_t c = 0; c < train_samples.size(); ++c){   
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < train_samples[c].size(); ++s)
        {
            
            std::string filename = compute_sample_filename(basenameArg, categories[c], train_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty())
            {
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
                exit(-1);
            }
            else
            {
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));                    
                
                cv::Mat descs;
                descs = extractSIFTDescriptors(img, ndesc);

                if (train_descs.empty())
                    train_descs = descs;
                else
                {
                    cv::Mat dst;
                    cv::vconcat(train_descs, descs, dst);
                    train_descs = dst;
                }
                ndescs_per_sample.push_back(descs.rows); //we could really have less of wished descriptors.
            }
        }
    }
    std::clog << std::endl;

    //free not needed memory.
    if(opcion == "SVM" || opcion == "svm"){
        std::vector<int> train_labels_v;
        train_labels_v.resize(0);
        kmeans(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, svm, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
        svm->train(td);
        //svm->train(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
    }else if(opcion == "knn" || opcion == "KNN"){
        std::vector<float> train_labels_v;
        train_labels_v.resize(0);
        kmenores(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, classifier, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        classifier->train(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
    }
    train_bovw.release();
    std::clog << "Testing .... " << std::endl;

    //load test images, generate SIFT descriptors and quantize getting a bovw for each image.
    //classify and compute errors.

    //For each test image, compute the corresponding bovw.
    std::clog << "\tCompute image descriptors for test images..." << std::endl;
    cv::Mat test_bovw;
    std::vector<float> true_labels;
    true_labels.resize(0);
    for (size_t c = 0; c < test_samples.size(); ++c)
    {
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < test_samples[c].size(); ++s)
        {
            std::string filename = compute_sample_filename(basenameArg, categories[c], test_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty())
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
            else{
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));

                //cv::Mat descs = extractSIFTDescriptors(img, ndesc);
                cv::Mat descs;

                descs = extractSIFTDescriptors(img, ndesc);

                cv::Mat bovw = compute_bovw(dict, keyws.rows, descs);
                if (test_bovw.empty())
                    test_bovw = bovw;
                else
                {
                    cv::Mat dst;
                    cv::vconcat(test_bovw, bovw, dst);
                    test_bovw = dst;
                }
                true_labels.push_back(c);
            }
        }
    }
    std::clog << std::endl;
    std::clog << "\tThere are " << test_bovw.rows << " test images." << std::endl;

    //classify the test samples.
    std::clog << "\tClassifying test images." << std::endl;
    cv::Mat predicted_labels;

    if(opcion == "SVM" || opcion == "svm"){
        svm->predict(test_bovw, predicted_labels);
    }else if(opcion == "knn" || opcion == "KNN"){
        classifier->predict(test_bovw, predicted_labels);
    }

    CV_Assert(predicted_labels.depth() == CV_32F);
    CV_Assert(predicted_labels.rows == test_bovw.rows);
    CV_Assert(predicted_labels.rows == true_labels.size());

    //compute the classifier's confusion matrix.
    std::clog << "\tComputing confusion matrix." << std::endl;
    cv::Mat confusion_mat = compute_confusion_matrix(categories.size(), cv::Mat(true_labels), predicted_labels);
    CV_Assert(int(cv::sum(confusion_mat)[0]) == test_bovw.rows);
    double rRate_mean, rRate_dev;
    compute_recognition_rate(confusion_mat, rRate_mean, rRate_dev);
    std::cerr << "Recognition rate mean = " << rRate_mean * 100 << "% dev " << rRate_dev * 100 << std::endl;
    rRates[trail]=rRate_mean;

    if (trail==0 || rRate_mean > best_rRate )
    {
        best_dictionary = dict;
        best_classifier = classifier;
        best_rRate = rRate_mean;
    }
}

void DenseSIFT(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm){
    rRates.resize(n_runsArg, 0.0);
    cv::Ptr<cv::ml::StatModel> classifier;
    cv::Mat train_bovw;
    cv::Ptr<cv::ml::KNearest> dict;
    cv::Mat keyws;

    int sift_type = 0;

    std::vector<int> siftScales{ 9, 13 }; // 5 , 9
    double best_rRate = 0.0;
    for (size_t c = 0; c < train_samples.size(); ++c){
            
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < train_samples[c].size(); ++s)
        {
            
            std::string filename = compute_sample_filename(basenameArg, categories[c], train_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty())
            {
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
                exit(-1);
            }
            else
            {
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));                    
                int step = 10; // 10 pixels spacing between kp's

                std::vector<cv::KeyPoint> kps;
                for (int i=step; i<img.rows-step; i+=step)
                {
                    for (int j=step; j<img.cols-step; j+=step)
                    {
                        // x,y,radius
                        kps.push_back(cv::KeyPoint(float(j), float(i), float(step)));
                    }
                }

                cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
                //sift->detect(img, kps, features);
                cv::Mat descriptor;
                sift->compute(img, kps, descriptor);
                cv::Mat descs = descriptor;
                if (train_descs.empty())
                    train_descs = descriptor;
                else{
                    cv::Mat dst;
                    cv::vconcat(train_descs, descs, dst);
                    train_descs = dst;
                }
                ndescs_per_sample.push_back(descs.rows); //we could really have less of wished descriptors.
                
            }
        }
    }

    std::clog << std::endl;
    std::vector<float> train_labels_v;
    
    //free not needed memory.
    std::clog << std::endl;

    //free not needed memory.
    if(opcion == "SVM" || opcion == "svm"){
        std::vector<int> train_labels_v;
        train_labels_v.resize(0);
        kmeans(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, svm, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
        svm->train(td);
    }else if(opcion == "knn" || opcion == "KNN"){
        std::vector<float> train_labels_v;
        train_labels_v.resize(0);
        kmenores(ndescs_per_sample, categories, ntrain, keywords, dict_runs, train_descs, train_samples, train_labels_v, classifier, train_bovw, dict, keyws, vecinos, opcion);
        cv::Mat train_labels(train_labels_v);
        classifier->train(train_bovw, cv::ml::ROW_SAMPLE, train_labels);
    }
    train_bovw.release();
    std::clog << "Testing .... " << std::endl;

    //load test images, generate SIFT descriptors and quantize getting a bovw for each image.
    //classify and compute errors.

    //For each test image, compute the corresponding bovw.
    std::clog << "\tCompute image descriptors for test images..." << std::endl;
    cv::Mat test_bovw;
    std::vector<float> true_labels;
    true_labels.resize(0);
    for (size_t c = 0; c < test_samples.size(); ++c)
    {
        std::clog << "  " << std::setfill(' ') << std::setw(3) << (c * 100) / train_samples.size() << " %   \015";
        for (size_t s = 0; s < test_samples[c].size(); ++s)
        {
            std::string filename = compute_sample_filename(basenameArg, categories[c], test_samples[c][s]);
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty())
                std::cerr << "Error: could not read image '" << filename << "'." << std::endl;
            else
            {
                // Fix size
                resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));
                int step = 10; // 10 pixels spacing between kp's
                std::vector<cv::KeyPoint> kps;
                for (int i=step; i<img.rows-step; i+=step)
                {
                    for (int j=step; j<img.cols-step; j+=step)
                    {
                        // x,y,radius
                        kps.push_back(cv::KeyPoint(float(j), float(i), float(step)));
                    }
                }
                cv::Mat descriptor;
                cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
                sift->compute(img, kps, descriptor);
                cv::Mat descs = descriptor;
                

                cv::Mat bovw = compute_bovw(dict, keyws.rows, descs);
                if (test_bovw.empty())
                    test_bovw = bovw;
                else
                {
                    cv::Mat dst;
                    cv::vconcat(test_bovw, bovw, dst);
                    test_bovw = dst;
                }
                true_labels.push_back(c);
            }
        }
    }
    std::clog << std::endl;
    std::clog << "\tThere are " << test_bovw.rows << " test images." << std::endl;

    //classify the test samples.
    std::clog << "\tClassifying test images." << std::endl;
    cv::Mat predicted_labels;

    if(opcion == "SVM" || opcion == "svm"){
        svm->predict(test_bovw, predicted_labels);
    }else if(opcion == "knn" || opcion == "KNN"){
        classifier->predict(test_bovw, predicted_labels);
    }

    CV_Assert(predicted_labels.depth() == CV_32F);
    CV_Assert(predicted_labels.rows == test_bovw.rows);
    CV_Assert(predicted_labels.rows == true_labels.size());

    //compute the classifier's confusion matrix.
    std::clog << "\tComputing confusion matrix." << std::endl;
    cv::Mat confusion_mat = compute_confusion_matrix(categories.size(), cv::Mat(true_labels), predicted_labels);
    CV_Assert(int(cv::sum(confusion_mat)[0]) == test_bovw.rows);
    double rRate_mean, rRate_dev;
    compute_recognition_rate(confusion_mat, rRate_mean, rRate_dev);
    std::cerr << "Recognition rate mean = " << rRate_mean * 100 << "% dev " << rRate_dev * 100 << std::endl;
    rRates[trail]=rRate_mean;

    if (trail==0 || rRate_mean > best_rRate )
    {
        best_dictionary = dict;
        best_classifier = classifier;
        best_rRate = rRate_mean;
    }
}
