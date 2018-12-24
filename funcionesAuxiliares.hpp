#ifndef FUNCIONESAUXILIARES_HPP
#define FUNCIONESAUXILIARES_HPP
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

/**
* @brief      Realiza la función kmenores
*
* @param[in]  ndescs_per_sample  The ndescs per sample
* @param[in]  categories         The categories
* @param[in]  ntrain             The ntrain
* @param[in]  keywords           The keywords
* @param[in]  dict_runs          The dictionary runs
* @param[in]  train_descs        The train descs
* @param[in]  train_samples      The train samples
* @param[in]  train_labels_v     The train labels v
* @param[in]  classifier         The classifier
* @param[in]  train_bovw         The train bovw
* @param[in]  dict               The dictionary
* @param[in]  keyws              The keyws
* @param[in]  opcion             The option svm or knn
*/
void kmenores(std::vector<int> ndescs_per_sample, std::vector<std::string> categories, int ntrain, int keywords, int dict_runs,  cv::Mat train_descs, std::vector<std::vector<int>> train_samples, std::vector<float> &train_labels_v, cv::Ptr<cv::ml::StatModel> &classifier, cv::Mat &train_bovw, cv::Ptr<cv::ml::KNearest> &dict, cv::Mat &keyws, int vecinos, std::string opcion);

void kmeans(std::vector<int> ndescs_per_sample, std::vector<std::string> categories, int ntrain, int keywords, int dict_runs,  cv::Mat train_descs, std::vector<std::vector<int>> train_samples, std::vector<int> &train_labels_v, cv::Ptr<cv::ml::SVM> &classifier, cv::Mat &train_bovw, cv::Ptr<cv::ml::KNearest> &dict, cv::Mat &keyws, int vecinos, std::string opcion);


/**
* @brief      Recognize if the argv is a camera or video flow
*
* @param[in]  nombre  The nombre
*
* @return     { description_of_the_return_value }
*/
bool digito(std::string nombre);

//void SURF(std::vector<std::vector<int>> train_samples, std::vector<std::string> categories);

/**
* @brief      Calcula los descriptores SURF
*
* @param[in]  basenameArg        The basename argument
* @param[in]  n_runsArg          The n runs argument
* @param[in]  dict_runs          The dictionary runs
* @param[in]  keywords           The keywords
* @param[in]  ntrain             The ntrain
* @param[in]  ntest              The ntest
* @param[in]  ndesc              The ndesc
* @param[in]  train_samples      The train samples
* @param[in]  categories         The categories
* @param[in]  train_descs        The train descs
* @param[in]  ndescs_per_sample  The ndescs per sample
* @param      best_dictionary    The best dictionary
* @param      best_classifier    The best classifier
* @param      rRates             The r rates
* @param[in]  test_samples       The test samples
* @param[in]  trail              The trail
* @param[in]  vecinos            The KNValue
* @param[in]  opcion             The option svm or knn
*/
void SURF(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm);

/**
* @brief      Calcula los descriptores SIFT
*
* @param[in]  basenameArg        The basename argument
* @param[in]  n_runsArg          The n runs argument
* @param[in]  dict_runs          The dictionary runs
* @param[in]  keywords           The keywords
* @param[in]  ntrain             The ntrain
* @param[in]  ntest              The ntest
* @param[in]  ndesc              The ndesc
* @param[in]  train_samples      The train samples
* @param[in]  categories         The categories
* @param[in]  train_descs        The train descs
* @param[in]  ndescs_per_sample  The ndescs per sample
* @param      best_dictionary    The best dictionary
* @param      best_classifier    The best classifier
* @param      rRates             The r rates
* @param[in]  test_samples       The test samples
* @param[in]  trail              The trail
* @param[in]  vecinos            The KNValue
* @param[in]  opcion             The option svm or knn
*/
void SIFT(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm);

/**
* @brief      Calcula los descriptores SIFT DENSO
*
* @param[in]  basenameArg        The basename argument
* @param[in]  n_runsArg          The n runs argument
* @param[in]  dict_runs          The dictionary runs
* @param[in]  keywords           The keywords
* @param[in]  ntrain             The ntrain
* @param[in]  ntest              The ntest
* @param[in]  ndesc              The ndesc
* @param[in]  train_samples      The train samples
* @param[in]  categories         The categories
* @param[in]  train_descs        The train descs
* @param[in]  ndescs_per_sample  The ndescs per sample
* @param      best_dictionary    The best dictionary
* @param      best_classifier    The best classifier
* @param      rRates             The r rates
* @param[in]  test_samples       The test samples
* @param[in]  trail              The trail
* @param[in]  vecinos            The KNValue
* @param[in]  opcion             The option svm or knn
*/
void DenseSIFT(std::string basenameArg, int n_runsArg, int dict_runs, int keywords, int ntrain, int ntest, int ndesc, std::vector<std::vector<int>> train_samples, std::vector<std::string> categories, cv::Mat train_descs, std::vector<int> ndescs_per_sample, cv::Ptr<cv::ml::KNearest> &best_dictionary, cv::Ptr<cv::ml::StatModel> &best_classifier, std::vector<float> &rRates, std::vector<std::vector<int>> test_samples, int trail, int vecinos, std::string opcion, cv::Ptr<cv::ml::SVM> &svm);

#endif