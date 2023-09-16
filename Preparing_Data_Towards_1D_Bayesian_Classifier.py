
'''Preparing_Data_Towards_1D_Bayesian_Classifier
   NEED TO BE RUN VIA GOOGLE COLLAB (ON GOOGLE GPU)'''

import requests


# Function Definitions
# Histogram Functions

def Build1DHistogramClassifier(X,T,B,minheight,maxheight):
    HF=np.zeros(B).astype('int32');
    HM=np.zeros(B).astype('int32');
    binindices=(np.round(((B-1)*(X-minheight)/(maxheight-minheight)))).astype('int32');
    for i,b in enumerate(binindices):
        if T[i]=='Female':
            HF[b]+=1;
        else:
            HM[b]+=1;
    return [HF, HM]


def Apply1DHistogramClassifier(queries,HF,HM,minheight,maxheight):
    B=np.alen(HF);
    binindices=np.clip((np.round(((B-1)*(queries-minheight)/(maxheight-minheight)))).astype('int32'),0,B-1);
    countF=HF[binindices];
    countM=HM[binindices];
    resultlabel=np.full(np.alen(binindices),"Indeterminate",dtype=object);
    resultprob=np.full(np.alen(binindices),np.nan,dtype=object);
    indicesF=countF>countM;
    indicesM=countM>countF;
    resultlabel[indicesF]="Female";
    resultlabel[indicesM]="Male";
    probF=countF/(countF+countM);
    probM=countM/(countF+countM);
    resultprob[indicesF]=probF[indicesF];
    resultprob[indicesM]=probM[indicesM];
    return resultlabel, resultprob


# Bayesian Functions
# Prepare Data

excelfile = '/content/drive/My Drive/Data/Assignment_1_Data_and_Template.xlsx'

data=readExcel(excelfile)
X=np.array(data[:,0]*12+data[:,1],dtype=float)
T=np.array(data[:,2])

queries=(readExcel(excelfile,
                  sheetname='Queries',
                  startrow=3,
                  endrow=8,
                  startcol=1,
                  endcol=1)).astype(float);queries




# Full Data
# Full Data Histogram Classifier

B=32
minheight=np.amin(X)
maxheight=np.amax(X)
[HF,HM]=Build1DHistogramClassifier(X,T,B,minheight,maxheight)

showHistograms(HF, HM, minheight, maxheight)

[GH, PH]=Apply1DHistogramClassifier(queries,HF,HM,minheight,maxheight)

showResult(queries, GH, PH)


# Partial Data Histogram Classifier
# Full Data Bayesian Classifier

def Build1DBayesianClassifier(X, T):
  muF = np.mean(X[T == 'Female'])
  muM = np.mean(X[T == 'Male'])
  sigmaF = np.std(X[T == 'Female'], ddof=1)
  sigmaM = np.std(X[T == 'Male'], ddof=1)
  NF = len(X[T == 'Female'])
  NM = len(X[T == 'Male'])
  return [muF, muM, sigmaF, sigmaM, NF, NM]

[muF, muM, sigmaF, sigmaM, NF, NM] = Build1DBayesianClassifier(X, T)

[muF, muM, sigmaF, sigmaM, NF, NM]

def pdf(x, mu, sigma):
  factor = 1/(np.sqrt(2*np.pi)*sigma)
  return factor*np.exp(-0.5*((x-mu)/sigma)**2)

pdf(0, 0, 1)

def Apply1DBayesianClassifier(queries,muF, muM, sigmaF, sigmaM, NF, NM):
    w=1
    countF = NF*w*pdf(queries, muF, sigmaF)
    countM = NM*w*pdf(queries, muM, sigmaM)
    resultlabel=np.full(np.alen(queries),"Indeterminate",dtype=object);
    resultprob=np.full(np.alen(queries),np.nan,dtype=object);
    indicesF=countF>countM;
    indicesM=countM>countF;
    resultlabel[indicesF]="Female";
    resultlabel[indicesM]="Male";
    probF=countF/(countF+countM);
    probM=countM/(countF+countM);
    resultprob[indicesF]=probF[indicesF];
    resultprob[indicesM]=probM[indicesM];
    return resultlabel, resultprob

[GB, PB]= Apply1DBayesianClassifier(queries,muF, muM, sigmaF, sigmaM, NF, NM)

showResult(queries, GB, PB)

# Partial Data Bayesian Classifier
[muF50, muM50, sigmaF50, sigmaM50, NF50, NM50] = Build1DBayesianClassifier(X50, T50)
[GB50, PB50]= Apply1DBayesianClassifier(queries,muF50, muM50, sigmaF50, sigmaM50, NF50, NM50)
showResult(queries, GB50, PB50)

# Summary
showAllResults(queries, GH, PH, GH50, PH50, GB, PB, GB50, PB50)

# Export Results
check_all_vars(all_vars)

print("Please wait. Writing to Excel ...")
writeExcelData([minheight,maxheight],excelfile,'Classifiers - Full Data',1,2)
writeExcelData([HF],excelfile,'Classifiers - Full Data',5,3)
writeExcelData([HM],excelfile,'Classifiers - Full Data',6,3)
writeExcelData([muF,muM],excelfile,'Classifiers - Full Data',8,3)
writeExcelData([sigmaF,sigmaM],excelfile,'Classifiers - Full Data',11,3)
writeExcelData([NF,NM],excelfile,'Classifiers - Full Data',14,3)
print("Written Sheet \'Classifiers - Full Data\'")

writeExcelData([minheight,maxheight],excelfile,'Classifiers - Partial Data',1,2)
writeExcelData([HF50],excelfile,'Classifiers - Partial Data',5,3)
writeExcelData([HM50],excelfile,'Classifiers - Partial Data',6,3)
writeExcelData([muF50,muM50],excelfile,'Classifiers - Partial Data',8,3)
writeExcelData([sigmaF50,sigmaM50],excelfile,'Classifiers - Partial Data',11,3)
writeExcelData([NF50,NM50],excelfile,'Classifiers - Partial Data',14,3)
print("Written Sheet \'Classifiers - Partial Data\'")

writeExcelData(list(zip(*[GH,PH,GB,PB])),excelfile,'Queries',3,2)
writeExcelData(list(zip(*[GH50,PH50,GB50,PB50])),excelfile,'Queries',12,2)
print("Written Sheet \'Queries\'")
closeExcelFile(excelfile)
print("DONE!")
