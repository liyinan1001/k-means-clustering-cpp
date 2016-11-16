//
//  main.cpp
//  K_means_clustering
//
//  Created by 李一楠 on 24/5/15.
//  Copyright (c) 2015 yinan. All rights reserved.
//

#include <iostream>
#include "Kmeans.h"
int main(int argc, const char * argv[]) {
    km kmeans(20,0);
    string filename="/Users/liyinan/Documents/machineLearning/project2/data.txt";
    kmeans.inputData(filename);
    kmeans.normlizeTfidf();
    kmeans.cluster();
    return 0;
}
