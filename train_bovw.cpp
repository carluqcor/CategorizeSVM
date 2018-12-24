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
#include "funcionesAuxiliares.cpp"

#define IMG_WIDTH 400

int main(int argc, char * argv[]){
	TCLAP::CmdLine cmd("Train and test a BoVW model", ' ', "0.0");

	TCLAP::ValueArg<std::string> basenameArg("", "basename", "basename for the dataset.", false, "./data", "pathname");
	cmd.add(basenameArg);
	TCLAP::ValueArg<std::string> configFile("", "config_file", "configuration file for the dataset.", false, "03_ObjectCategories_conf.txt", "pathname");
	cmd.add(configFile);
    TCLAP::ValueArg<int> tipoDescriptor("", "tipoDescriptor", "0 SURF, 1 SIFT, 2 SIFT Denso", false, 0, "int");
    cmd.add(tipoDescriptor);
    TCLAP::ValueArg<int> n_runsArg("", "n_runs", "Number of trials train/set to compute the recognition rate. Default 10.", false, 10, "int");
    cmd.add(n_runsArg);
	TCLAP::ValueArg<int> dict_runs("", "dict_runs", "[SIFT] Number of trials to select the best dictionary. Default 5.", false, 5, "int");
	cmd.add(dict_runs);
    TCLAP::ValueArg<int> ndesc("", "ndesc", "[SIFT] Number of descriptors per image. Value 0 means extract all. Default 0.", false, 0, "int");
	cmd.add(ndesc);
    TCLAP::ValueArg<int> keywords("", "keywords", "[SIFT] Number of keywords generated. Default 100.", false, 100, "int");
	cmd.add(keywords);
    TCLAP::ValueArg<int> ntrain("", "ntrain", "Number of samples per class used to train. Default 15.", false, 15, "int");
	cmd.add(ntrain);
	TCLAP::ValueArg<int> ntest("", "ntest", "Number of samples per class used to test. Default 50.", false, 50, "int");
	cmd.add(ntest);
    TCLAP::ValueArg<int> vecinos("", "vecinos", "K Vecinos. Default 1.", false, 1, "int");
    cmd.add(vecinos);
    TCLAP::ValueArg<std::string> opcion("", "opcion", "Opcion. Default KNN.", false, "KNN", "std::string");
    cmd.add(opcion);
	cmd.parse(argc, argv);

	std::vector<std::string> categories;
	std::vector<int> samples_per_cat;
    std::vector<float> rRates;
    cv::Ptr<cv::ml::KNearest> best_dictionary;
    cv::Ptr<cv::ml::StatModel> best_classifier;
	
	std::string dataset_desc_file = basenameArg.getValue() + "/" + configFile.getValue();
	
	int retCode;
	if ((retCode = load_dataset_information(dataset_desc_file, categories, samples_per_cat)) != 0){
		std::cerr << "Error: could not load dataset information from '"
			<< dataset_desc_file
			<< "' (" << retCode << ")." << std::endl;
		exit(-1);
	}

    std::cout << "Found " << categories.size() << " categories: ";
    std::ofstream fich;
    fich.open("categories.txt");
    for(int k=0;k<(int)categories.size();k++){
        fich <<categories[k]<<"\n";
    }

    if (categories.size()<2){
        std::cerr << "Error: at least two categories are needed." << std::endl;
        return -1;
    }

    for (size_t i=0;i<categories.size();++i)
        std::cout << categories[i] << ' ';
    std::cout << std::endl;
   
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    for (int trail=0; trail<n_runsArg.getValue(); trail++){
        std::clog << "######### TRIAL " << trail+1 << " ##########" << std::endl;

        std::vector<std::vector<int>> train_samples;
        std::vector<std::vector<int>> test_samples;
        create_train_test_datasets(samples_per_cat, ntrain.getValue(), ntest.getValue(), train_samples, test_samples);

        std::clog << "Training ..." << std::endl;
        std::clog << "\tCreating dictionary ... " << std::endl;
        std::clog << "\t\tComputing descriptors..." << std::endl;
        cv::Mat train_descs;
        std::vector<int> ndescs_per_sample;
        ndescs_per_sample.resize(0);
        if(tipoDescriptor.getValue()==0)
            SURF(basenameArg.getValue(), n_runsArg.getValue(), dict_runs.getValue(), keywords.getValue(), ntrain.getValue(), ntest.getValue(), ndesc.getValue(), train_samples, categories, train_descs, ndescs_per_sample, best_dictionary, best_classifier, rRates, test_samples, trail, vecinos.getValue(), opcion.getValue(), svm);
        else if(tipoDescriptor.getValue()==1)
            SIFT(basenameArg.getValue(), n_runsArg.getValue(), dict_runs.getValue(), keywords.getValue(), ntrain.getValue(), ntest.getValue(), ndesc.getValue(), train_samples, categories, train_descs, ndescs_per_sample, best_dictionary, best_classifier, rRates, test_samples, trail, vecinos.getValue(), opcion.getValue(), svm);
        else if (tipoDescriptor.getValue()==2)
            DenseSIFT(basenameArg.getValue(), n_runsArg.getValue(), dict_runs.getValue(), keywords.getValue(), ntrain.getValue(), ntest.getValue(), ndesc.getValue(), train_samples, categories, train_descs, ndescs_per_sample, best_dictionary, best_classifier, rRates, test_samples, trail, vecinos.getValue(), opcion.getValue(), svm);
    }
    if(opcion.getValue()=="SVM" || opcion.getValue()=="svm"){
        svm->save("svm_filename.yml"); // saving

    }else if(opcion.getValue() == "knn" || opcion.getValue() == "KNN"){
        best_classifier->save("classifier.yml");
    }
    //Saving the best models.
    cv::FileStorage dictFile;
    dictFile.open("dictionary.yml", cv::FileStorage::WRITE);
    dictFile << "keywords" << keywords.getValue();
    best_dictionary->write(dictFile);
    dictFile.release();

    std::clog << "###################### FINAL STATISTICS  ################################" << std::endl;

    double rRate_mean = 0.0;
    double rRate_dev = 0.0;

    for (size_t i=0; i<rRates.size();++i)
    {
        const float v=rRates[i];
        rRate_mean += v;
        rRate_dev += v*v;
    }
    rRate_mean /= double(rRates.size());
    rRate_dev = rRate_dev/double(rRates.size()) - rRate_mean*rRate_mean;
    rRate_dev = sqrt(rRate_dev);
    std::clog << "Recognition Rate mean " << rRate_mean*100.0 << "% dev " << rRate_dev*100.0 << std::endl;
    return 0;
}
