//
//  KmeansEigen.cpp
//  K_means_clustering
//
//  Created by 李一楠 on 30/5/15.
//  Copyright (c) 2015 yinan. All rights reserved.
//

#include "KmeansEigen.h"
km::km(int k,double distCrit){
    this->k=k;
    this->distCrit=distCrit;
    for(int i=0;i<k;i++){
        vec_clusterSize.push_back(0);
    }
}
void km::inputData(string filename){
    ifstream ifs(filename);
    string line;
    getline(ifs,line);
    stringstream buf(line);
    string arg1;
    string arg2;
    buf>>arg1;
    buf>>arg2;
    int docCount=atoi(arg1.c_str());
    int wordCount=atoi(arg2.c_str());
    this->docCount=docCount;
    this->wordCount=wordCount;
    for(int i=0;i<docCount;i++){
        mat_tfidf.push_back(vecTfidf());
        vec_docTempLabel.push_back(int());
        vec_docTlabel.push_back(int());
    }
    for(int i=0;i<wordCount;i++){
        vec_df.push_back(int());
    }
    while(!ifs.eof()){
        string line;
        getline(ifs,line);
        stringstream buf(line);
        string arg1,arg2,arg3,arg4;
        buf>>arg1>>arg2>>arg3>>arg4;
        int label=atoi(arg1.c_str());
        int doc=atoi(arg2.c_str());
        int word=atoi(arg3.c_str());
        int wordtf=atoi(arg4.c_str());
        this->mat_tfidf[doc-1].push_back(make_pair(word-1,double(wordtf)));
       // this->mat_tfidf[doc-1].push(make_pair(word-1,double(wordtf)));
        this->vec_docTlabel[doc-1]=label-1;
        this->vec_df[word-1]++;
        ifs.peek();
    }
}
void km::normlizeTfidf(){
    for(int i=0;i<this->wordCount;i++){
        vec_df[i]=(this->docCount/vec_df[i]);
    }
    for(int i=0;i<this->docCount;i++){  //tf*idf
        int tfsum=0;
        for(auto iter=this->mat_tfidf[i].begin();iter!=mat_tfidf[i].end();iter++){
            tfsum+=iter->second;
        }
        for(auto iter=mat_tfidf[i].begin();iter!=mat_tfidf[i].end();iter++){
            iter->second=(iter->second/tfsum)*log(this->vec_df[iter->first]);
        }
    }
}
void km::sortWordId(){
    for(int i=0;i<this->wordCount;i++){
        sort(this->mat_tfidf[i].begin(),this->mat_tfidf[i].end(),cmp);
    }
}
double km::vecDist(const vecTfidf &vec1, const vecTfidf &vec2){
    double dist=0;
    auto iter1=vec1.begin();
    auto iter2=vec2.begin();
    while(iter1!=vec1.end()||iter2!=vec2.end()){
        if((iter1->first==iter2->first)&&iter1!=vec1.end()&&iter2!=vec2.end()){
            dist+=pow(iter1->second-iter2->second,2);
            iter1++;
            iter2++;
        }
        else if((iter1==vec1.end()||iter1->first>iter2->first)&&iter2!=vec2.end()){
            dist+=pow(iter2->second,2);
            iter2++;
        }
        else if((iter2==vec2.end()||iter1->first<iter2->first)&&iter1!=vec1.end()){
            dist+=pow(iter1->second,2);
            iter1++;
        }
    }
    dist=pow(dist,0.5);
    return dist;
}
void km::cluster(){
    int repeat=10;
    double Fmax=0,Emin=0;  //F bigger is better, E smaller is better
    ofstream ofs_result("/Users/liyinan/Documents/machineLearning/project2/result.txt");
    for(int i=0;i<repeat;i++){
        // initialize cluster center randomly  ********************************************************
        for(int j=0;j<this->k;j++){
            random_device rd;
            int initCentDoc=double(rd()-random_device::min())/(random_device::max()-random_device::min())*this->docCount;
            this->vec_centerTfIdf[j]=this->mat_tfidf[initCentDoc];
        }
        double distChange=numeric_limits<double>::max();
        while(distChange>this->distCrit){   //if some doc changes cluster, do update********************************************************
            distChange=0;
            for(auto iter=vec_clusterSize.begin();iter!=vec_clusterSize.end();iter++){
                *iter=0;
            }
            // assign nearest cluster to each doc
            // cout<<endl<<"assign"<<endl;
            for(int doc=0;doc<this->docCount;doc++){
                // cout<<doc<<".";
                double minDist=numeric_limits<double>::max();
                int bestCluster=0;
                for(int cent=0;cent<this->k;cent++){
                    double dist=this->vecDist(this->mat_tfidf[doc], this->vec_centerTfIdf[cent]);
                    if(dist<minDist){
                        minDist=dist;
                        bestCluster=cent;
                    }
                }
                this->vec_clusterSize[bestCluster]++;
                this->vec_docTempLabel[doc]=bestCluster;
            }
            // update each cluster's center
            
            
            
            vector<vecTfidf> mat_tempCentTfidf(this->k);
            // cout<<endl<<"update"<<endl;
            for(int doc=0;doc<this->docCount;doc++){
                //cout<<doc<<".";
                int cent=this->vec_docTempLabel[doc];
                for(auto iterDoc=this->mat_tfidf[doc].begin();iterDoc!=this->mat_tfidf[doc].end();iterDoc++){
                    auto iterCent=mat_tempCentTfidf[cent].find(iterDoc->first);
                    if(iterCent==mat_tempCentTfidf[cent].end()){
                        mat_tempCentTfidf[cent].insert(*iterDoc);
                    }
                    else{
                        iterCent->second+=iterDoc->second;
                    }
                }
            }
            //caculate mean tfidf of each cluster
            for(int cent=0;cent<this->k;cent++){
                for(auto iter=mat_tempCentTfidf[cent].begin();iter!=mat_tempCentTfidf[cent].end();iter++){
                    iter->second/=this->vec_clusterSize[cent];
                }
            }
            
            for(int cent=0;cent<this->k;cent++){
                distChange+=this->vecDist(mat_tempCentTfidf[cent], this->vec_centerTfIdf[cent]);
            }
            this->vec_centerTfIdf=mat_tempCentTfidf;
            cout<<"size: "<<this->vec_centerTfIdf.size();
            cout<<"distchange"<<distChange<<"..."<<endl;
        }
        //evaluate cluster quality
        vector<vector<int>> mat_true_trained;
        for(int i=0;i<this->k;i++){
            mat_true_trained.push_back(vector<int>());
            for(int j=0;j<this->k;j++){
                mat_true_trained[i].push_back(0);
            }
        }
        for(int doc=0;doc<this->docCount;doc++){
            int trueLabel=this->vec_docTlabel[doc];
            int trainedLabel=this->vec_docTempLabel[doc];
            mat_true_trained[trueLabel][trainedLabel]++;
        }
        vector<int> vec_trueClust_size;
        for(int trueL=0;trueL<this->k;trueL++){
            int clusSize=0;
            for(int trainL=0;trainL<this->k;trainL++){
                clusSize+=mat_true_trained[trueL][trainL];
            }
            vec_trueClust_size.push_back(clusSize);
        }
        double F=0;
        double E=0;
        
        vector<vector<double>> mat_p;
        for(int i=0;i<this->k;i++){
            mat_p.push_back(vector<double>());
            for(int j=0;j<this->k;j++){
                mat_p[i].push_back(0);
            }
        }
        for(int i=0;i<this->k;i++){
            double maxFij=0;
            for(int j=0;j<this->k;j++){
                double r=double(mat_true_trained[i][j])/vec_trueClust_size[i];
                double p=double(mat_true_trained[i][j])/this->vec_clusterSize[j];
                mat_p[i][j]=p;
                double Fij=2*r*p/(r+p);
                if(Fij>maxFij){
                    maxFij=Fij;
                }
            }
            F+=double(vec_trueClust_size[i])/this->docCount*maxFij;
        }
        for(int j=0;j<this->k;j++){
            double Ej=0;
            for(int i=0;i<this->k;i++){
                Ej+=mat_p[i][j]*log(mat_p[i][j]);
            }
            E+=double(this->vec_clusterSize[j])*-1*Ej/this->docCount;
        }
        cout<<"F: "<<F<<" E: "<<E<<endl;
        ofs_result<<"F: "<<F<<" E: "<<E<<endl;
        if(F>Fmax){
            Fmax=F;
        }
        if(E<Emin){
            Emin=E;
        }
    }
    cout<<"Fmax: "<<Fmax<<" Emin: "<<Emin<<endl;
    ofs_result<<"Fmax: "<<Fmax<<" Emin: "<<Emin<<endl;
    ofs_result.close();
}

