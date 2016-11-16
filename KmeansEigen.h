//
//  KmeansEigen.h
//  K_means_clustering
//
//  Created by 李一楠 on 30/5/15.
//  Copyright (c) 2015 yinan. All rights reserved.
//

#ifndef __K_means_clustering__KmeansEigen__
#define __K_means_clustering__KmeansEigen__
#include<map>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include <stdio.h>
#include<math.h>
#include<random>
#include<iostream>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include <stdio.h>
#include<queue>
#include<algorithm>
using namespace std;
bool cmp(const pair<int,double>& p1, const pair<int,double>& p2){
    return p1.first<p2.first;
}
typedef vector<pair<int,double>> vecTfidf;

class km{
public:
    km(int k, double distCrit);
    void inputData(string filename);
    void normlizeTfidf();
    void sortWordId();
    void cluster();
private:
    vector<vecTfidf> mat_tfidf;
    vector<int> vec_docTempLabel;  //store every doc's label when training
    vector<int> vec_docTlabel;  //store doc's true label
    vector<vecTfidf> vec_centerTfIdf;
    vector<int> vec_clusterSize; //store every cluster's doc count
    vector<double> vec_df;  //word's df vector
    int k;
    int docCount;
    int wordCount;
    double distCrit;
    double vecDist(const vecTfidf& vec1, const vecTfidf& vec2);
};
#endif /* defined(__K_means_clustering__KmeansEigen__) */
