#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <tclap/CmdLine.h>
#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "common_code.hpp"
#include "macros.hpp"
#include "funcionesAuxiliares.cpp"

#define IMG_WIDTH 400

const cv::String keys =
    "{help h usage ?    |      | print this message        }"
    "{@dictionaryFile   |      | fichero de diccionario    }"
    "{@classifierFile   |      | fichero de clasificador   }"
    "{@image1           |      | imagen entrada            }"
    "{@opcion           |      | Descriptor                }"
    "{@classifier       |      | Which classifier?         }"
    "{@ficheroCategorias|      | Fichero de categorias }"
    ;

int main (int argc, char* argv[]){
  int retCode=EXIT_SUCCESS;
    try{    
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Application name v1.0.0");
        if (parser.has("help")){
            parser.printMessage();
            return 0;
        }

        //Parser de los argumentos
        cv::FileStorage dictionary(parser.get<cv::String>(0), cv::FileStorage::READ);
        std::string classifier=(parser.get<std::string>(1));
        int option = parser.get<int>(3);

        std::string opcion=(parser.get<std::string>(4));
        //Declaración de las variables necesarias
        std::vector<int> keywords;
        cv::Ptr<cv::ml::KNearest> best_dictionary;
        best_dictionary = cv::ml::KNearest::create();
        cv::FileStorage fsRead(classifier, cv::FileStorage::READ);
        cv::Ptr<cv::ml::KNearest> best_classifierKNN;
        cv::Ptr<cv::ml::SVM> best_classifierSVM;
        if(opcion == "SVM" || opcion == "svm"){
            best_classifierSVM = cv::Algorithm::load<cv::ml::SVM>(classifier);//Prueba a compilar now 
        }else if(opcion == "knn" || opcion == "KNN"){
            best_classifierKNN = cv::Algorithm::load<cv::ml::KNearest>(classifier);//Prueba a compilar now
        }
        best_dictionary->read(dictionary.root());
        std::ifstream fich;
        std::string f;
        std::vector<std::string> categorias;
        cv::Mat img;
        dictionary["keywords"] >> keywords; //Número de keywords
        fich.open(parser.get<std::string>(5)); //Categorias entrenadas
        while(!fich.eof()){
            //std::cout<<"Peta aqui\n";
            std::getline(fich, f, '\n');
            categorias.push_back(f);
        }

        if(!best_dictionary->isTrained()){ //Está entrenado el modelo?
            std::cout<<"No se carga bien el diccionario\n";
        }
        if(opcion == "knn" || opcion == "KNN"){
            if(!best_classifierKNN->isClassifier()){ //Está clasificado el modelo?
                std::cout<<"No se carga bien el clasificador\n";
            }
        }
        bool detail;
        int ndesc, dict_runs=10, step=10, minHessian=400;
        //step: 10 pixels spacing between keypoints SIFT DENSO
        //minHessian: Para la creación del descriptor SURF
        cv::Mat keyws, test_bovw, descs, labels, train_descs, descriptor;
        cv::VideoCapture captura;

        if(digito(parser.get<cv::String>(2)))
            captura.open(0);
        else
            captura.open(parser.get<cv::String>(2));
        
        if(!captura.isOpened()){
            std::cout<<"Error en la apertura del video\n";
        }
        for(;;){
            cv::Mat keyws, test_bovw, descs, labels, train_descs, descriptor;
            std::vector<cv::KeyPoint> keypoint;
            //Todas tendrán el mismo tamaño
            cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(minHessian);  //SURF
            cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(); //SIFT DENSO
            captura>>img;
            if (!captura.read(img))
                break;
            resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));
            
            switch(option){
                case 0: //Descriptor SURF
                    surf->detect(img, keypoint);
                    surf->compute(img, keypoint, descriptor);
                    descs = descriptor;
                    break;

                case 1: //Descriptor SIFT
                    descs = extractSIFTDescriptors(img, ndesc);
                    break;

                case 2: //Descriptor SIFT DENSO
                    for (int i=step; i<img.rows-step; i+=step){ //Va de 10 en 10 pixeles
                        for (int j=step; j<img.cols-step; j+=step){
                            // x,y,radio
                            keypoint.push_back(cv::KeyPoint(float(j), float(i), float(step)));
                        }
                    }
                    sift->compute(img, keypoint, descriptor);
                    descs = descriptor;
                    break;
                
                default:
                    std::cout<<BIRED<<"No existe este descriptor!"<<RESET<<std::endl;
                    break;
            }
        
            cv::Mat bovw = compute_bovw(best_dictionary, keywords[0], descs);
            if (test_bovw.empty())
                test_bovw = bovw;
            else{
                cv::Mat dst;
                cv::vconcat(test_bovw, bovw, dst);
                test_bovw = dst;
            }

            cv::Mat predicted_labels;
            int i, aux=0;

            if(opcion == "SVM" || opcion == "svm"){
                best_classifierSVM->predict(test_bovw, predicted_labels);
            }else if(opcion == "knn" || opcion == "KNN"){
                best_classifierKNN->predict(test_bovw, predicted_labels);
            }
            float value=predicted_labels.at<float>(0);
            cv::putText(img, categorias[value], cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255,255,255), 2);
            cv::imshow("Imagen Categorizada", img);
            
            char key = cv::waitKey(20);
            if (key == 27) // Escape
                break; //Salir
        }
    }catch (std::exception& e){
    std::cerr << "Capturada excepcion: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  	}
  	return retCode;
}