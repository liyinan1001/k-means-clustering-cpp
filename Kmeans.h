//
//  Kmeans.h
//  K_means_clustering
//
//  Created by 李一楠 on 28/5/15.
//  Copyright (c) 2015 yinan. All rights reserved.
//

#ifndef __K_means_clustering__Kmeans__
#define __K_means_clustering__Kmeans__
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
#include<queue>
#include<iterator>
using namespace std;
inline bool cmp(const pair<int,double>& p1, const pair<int,double>& p2){
    return p1.second>p2.second;
}
class km{
public:
    km(int k, double distCrit);
    void inputData(string filename);
    void normlizeTfidf();
    void cluster();
    void initiateCenter();
private:
    
    vector<map<int,double>> vec_tfidf;  //store every doc's tfidf vector
    vector<int> vec_docTempLabel;  //store every doc's label when training
    vector<int> vec_docTlabel;  //store doc's true label
    vector<map<int,double>> vec_centerTfIdf; //store K means center's tfidf
    vector<int> vec_clusterSize; //store every cluster's doc count
    vector<int> vec_trueClusterSize;
    vector<double> vec_df;  //word's df vector
    int k;
    int truelabelmax;
    int docCount;
    int wordCount;
    double distCrit;
    double vecDist(map<int,double>& vec1, map<int,double>& vec2);
};

#endif /* defined(__K_means_clustering__Kmeans__) */

