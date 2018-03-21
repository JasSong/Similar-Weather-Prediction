import numpy as np
import os
import itertools as itt
import collections as cl
from pprint import pprint
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import OrderedDict
import csv

#특정 순위 날짜의 파일 위치 탐색
def dir_search(dates_rank_list):
    # UTC기준으로 분리된 폴더의 위치
    rootdir = "C:/Users/korea/Desktop/project/ksc/data"

    # 찾을 고도장 온도장 UTC시간 설정
    Hs=['H100','H200','H300','H400','H500','H600','H700','H800','H900','H1000']
    Ts=['T100','T200','T300','T400','T500','T600','T700','T800','T900','T1000']
    utc=['00','06','12','18']

    #메인함수
    HTutc=list(itt.product(zip(Hs,Ts), utc))

    dir_list_ssim={}
    dir_list_hist={}

    for i in range(len(dates_rank_list)):
        date=dates_rank_list[i]
        year=date[:4]
        mth=date[4:6]
        dd=date[6:]
        datelist_ssim=[]
        datelist_hist=[]
        for i in range(len(HTutc)):
            H, T=HTutc[i][0]
            utc=HTutc[i][1]           
            era_H=os.path.join(rootdir,'ERA_INT_KRU_{}'.format(utc),H,year,mth,'era_easia_{0}_anal.{1}{2}{3}{4}_s2.bin'.format(H.lower(),year,mth,dd,utc))
            era_T=os.path.join(rootdir,'ERA_INT_KRU_{}'.format(utc),T,year,mth,'era_easia_{0}_anal.{1}{2}{3}{4}_s2.bin'.format(T.lower(),year,mth,dd,utc))

            datelist_ssim.append(era_H)
            datelist_ssim.append(era_T)
            datelist_hist.append((era_H, era_T))

        dir_list_ssim[date]=datelist_ssim
        dir_list_hist[date]=datelist_hist
  

    return dir_list_ssim, dir_list_hist

#특정 순위 날짜의 값들을 호출하여 전체(H,T), H, T array 생성
def make_array(dates_rank_list, raw=True, split=False):
    dirlistssim, dirlisthist=dir_search(dates_rank_list)
    #온도장 고도장 합친 부분의 array생성
    if raw==True and split==False:
        dates_raw_data_dic={}
        for i in range(len(dates_rank_list)):
            date=dates_rank_list[i]
            dirlists=dirlistssim[date]
            date_raw_data=[]
            for j in range(len(dirlists)):
                dirpath=dirlists[j]
                data=np.fromfile(dirpath, dtype=np.float32, count=-1)
                date_raw_data.append(data)
            date_raw_data=np.asarray(date_raw_data)
            dates_raw_data_dic[date]=date_raw_data.reshape(-1)
        
        return dates_raw_data_dic

    #온도장, 고도장 분리한 부분의 array 생성
    if split==True and raw==False:
        dates_rawdic_H={}
        dates_rawdic_T={}
        
        for i in range(len(dates_rank_list)):
            date=dates_rank_list[i]
            rawHTlists=dirlisthist[date]
            date_rawH=[]
            date_rawT=[]
            for j in range(len(rawHTlists)):
                rawHdir, rawTdir=rawHTlists[j]
                rawH=np.fromfile(rawHdir, dtype=np.float32, count=-1)
                rawT=np.fromfile(rawTdir, dtype=np.float32, count=-1)
                
                date_rawH.append(rawH)
                date_rawT.append(rawT)
            date_rawH=np.asarray(date_rawH)
            date_rawH=np.clip(date_rawH, -100, 16999)
            date_rawT=np.asarray(date_rawT)
            date_rawT=np.clip(date_rawT, -100, 40)
            dates_rawdic_H[date]=date_rawH.reshape(-1)
            dates_rawdic_T[date]=date_rawT.reshape(-1)
            
        return dates_rawdic_H, dates_rawdic_T       

#평균 계산 함수
def de_mean(x) :
    x_bar = np.mean(x) 
    return [x_i - x_bar for x_i in x]

#공분산 계산 함수
def covariance(x,y): 
    n = len(x) 
    return np.dot(de_mean(x), de_mean(y)) / (n-1)

#SSIM 비교 함수
def ssim_compare(x1, x2, k1 = 0.01, k2 = 0.03 ) :
    x1_mean = np.mean(x1,dtype = 'float32')
    x2_mean = np.mean(x2,dtype = 'float32')
    x1_var  = np.var(x1,dtype = 'float32')
    x2_var  = np.var(x2,dtype = 'float32')
    data_range=np.max(x2) - np.min(x2)
    data_range = round(data_range)
    c1 = (k1*data_range)**2
    c2 = (k2*data_range)**2
    ssim = ((2*x1_mean*x2_mean + c1)*(2*covariance(x1,x2)+c2))/((x1_mean**2+x2_mean**2+c1)*(x1_var+x2_var+c2))
    return ssim

#Histgram 비교 함수
def hist_compare(x1,x2) :
    summation = np.sum(np.abs(x1-x2))
    return summation

#MSE 비교 함수
def mse_compare(x1,x2) :
    x1=x1.reshape(1,-1)
    x2=x2.reshape(1,-1)
    MSE=np.mean((x1-x2)**2, axis=1)
    return MSE

#COS유사도 비교 함수
def cos_compare(x1,x2) :
    COS=cosine_similarity(x1.reshape(1,-1),x2.reshape(1,-1))
    return COS[0]

#Histogram측도내 계산 함수
def collecting_hist(arr):
    floor_value=np.floor(arr/100)
    clsdic=cl.Counter(floor_value)
    keys=list(clsdic.keys())
    xs=[i for i in range(-1,171)]
    for x in xs:
        if not x in keys:
            clsdic[x]=0
    count_arrays = np.asarray(list(clsdic.values()))
    return count_arrays       

#Histogrma측도내 계산 종합 함수
def total_hist(inputdate, ranklists):
    inputdate=str(inputdate)
    dateslist=[inputdate]+ranklists
    rawH, rawT=make_array(dateslist, False, True)
    i=0
    dicH={}
    dicT={}
    while i<11:
        valueH=collecting_hist(rawH[dateslist[i]])
        valueT=collecting_hist(rawT[dateslist[i]])
        dicH[dateslist[i]]=valueH        
        dicT[dateslist[i]]=valueT
        i+=1

    histH_list=[]
    histT_list=[]
    for i in range(len(ranklists)):
        date_to_be_compared=ranklists[i]
        histH_list.append(hist_compare(dicH[inputdate], dicH[date_to_be_compared]))
        histT_list.append(hist_compare(dicT[inputdate], dicT[date_to_be_compared]))
    histHarr=np.array(histH_list)
    histTarr=np.array(histT_list)
    histtotal=np.sum([histHarr, histTarr], axis=0)
    histtotal=histtotal.reshape(-1,1)
    return histtotal ,histHarr, histTarr

#SSIM측도내 계산 종합 함수
def total_ssim(inputdate, ranklists):
    inputdate=str(inputdate)
    dateslist=[inputdate]+ranklists
    rawH, rawT = make_array(dateslist, False, True)
    rawdatadic=make_array(dateslist, True, False)

    #온도장, 고도장 분리한 계산부분
    ssimH_list=[]
    ssimT_list=[]
    for i in range(len(ranklists)):
        date_to_be_compared=ranklists[i]
        ssimH_list.append(ssim_compare(rawH[inputdate], rawH[date_to_be_compared]))
        ssimT_list.append(ssim_compare(rawT[inputdate], rawT[date_to_be_compared]))

    #온도장, 고도장 합친 계산부분
    ssim_list = []
    for i in range(len(ranklists)):
        date_to_be_compared=ranklists[i]
        ssim_list.append(ssim_compare(rawdatadic[inputdate],rawdatadic[date_to_be_compared]))
    return ssim_list , ssimH_list, ssimT_list

#MSE측도내 계산 종합 함수
def raw_MSE(inputdate, ranklists):
    inputdate=str(inputdate)
    dates_rank_list=[inputdate]+ranklists
    rawdatadic=make_array(dates_rank_list, True, False)
    rawH, rawT = make_array(dates_rank_list, False, True)

    # 온도장, 고도장 합친 계산부분
    MSE_list=[]
    for i in range(len(ranklists)):
        date_to_be_compared=ranklists[i]
        MSE_list.append(mse_compare(rawdatadic[inputdate], rawdatadic[date_to_be_compared]))

    # 온도장, 고도장 분리한 계산부분
    MSEH_list = []
    MSET_list = []
    for i in range(len(ranklists)):
        date_to_be_compared = ranklists[i]
        MSEH_list.append(mse_compare(rawH[inputdate], rawH[date_to_be_compared]))
        MSET_list.append(mse_compare(rawT[inputdate], rawT[date_to_be_compared]))

    return MSE_list, MSEH_list, MSET_list

#COS유사도내 계산 종합 함수
def raw_COS(inputdate, ranklists):
    inputdate=str(inputdate)
    dates_rank_list=[inputdate]+ranklists
    rawdatadic=make_array(dates_rank_list, True, False)
    rawH, rawT = make_array(dates_rank_list, False, True)

    # 온도장, 고도장 합친 계산부분
    COS_list=[]
    for i in range(len(ranklists)):
        date_to_be_compared=ranklists[i]
        COS_list.append(cos_compare(rawdatadic[inputdate], rawdatadic[date_to_be_compared]))

    # 온도장, 고도장 분리한 계산부분
    COSH_list = []
    COST_list = []
    for i in range(len(ranklists)):
        date_to_be_compared = ranklists[i]
        COSH_list.append(cos_compare(rawH[inputdate], rawH[date_to_be_compared]))
        COST_list.append(cos_compare(rawT[inputdate], rawT[date_to_be_compared]))

    return COS_list, COSH_list, COST_list

#Re값 비교 계산 함수
def re_compare_formula(x,y):
    return np.sum(np.abs(x-y))/np.abs(np.sum(x))

#Re측도내 계산 종합 함수
def re_compare(inputdate,ranklists):
    inputdate=str(inputdate)
    dateslist=[inputdate]+ranklists
    rawH, rawT = make_array(dateslist, False, True)
    rawdatadic=make_array(dateslist, True, False)

    # 온도장, 고도장 분리한 계산부분
    reH_list = []
    reT_list = []
    for i in range(len(ranklists)):
        date_to_be_compared = ranklists[i]
        reH_list.append(re_compare_formula(rawH[inputdate], rawH[date_to_be_compared]))
        reT_list.append(re_compare_formula(rawT[inputdate], rawT[date_to_be_compared]))

    # 온도장, 고도장 합친 계산부분
    re_list = []
    for i in range(len(ranklists)):
        date_to_be_compared = ranklists[i]
        re_list.append(ssim_compare(rawdatadic[inputdate], rawdatadic[date_to_be_compared]))
    return re_list, reH_list, reT_list


#각 측도별 순위 Data Frame 생성 함수
def check_result(inputdate, ranklists):
    #MSE
    mse_list, mse_H_list, mse_T_list=np.asarray(raw_MSE(inputdate, ranklists))
    #cos
    cos_list, cos_H_list, cos_T_list=np.asarray(raw_COS(inputdate, ranklists))
    #histtottal
    histtotal=total_hist(inputdate, ranklists)[0]
    #mse+cos
    eval3_list=0.99*np.array(mse_list)+0.01*(1-(np.array(cos_list)))
    #mse+cos+hist
    eval4_list=0.99*np.array(mse_list)+0.01*(1-(np.array(cos_list)))+0.003*histtotal[0]
    #ssim
    ssim_list, ssimH_list , ssimT_list =total_ssim(inputdate, ranklists)
    #re
    re_list , reH_list, reT_list = re_compare(inputdate, ranklists)

    df_matrix = {}
    #rank
    df_matrix['df_rank'] = pd.Series(ranklists).as_matrix()
    #mse
    df_mse = pd.DataFrame(OrderedDict({'mse_ranks':ranklists,'mse':mse_list.reshape(-1)})).sort_values('mse',axis = 0, ascending=True)
    df_matrix['msranks'], df_matrix['mse_1']= pd.Series(df_mse['mse_ranks']).as_matrix(), pd.Series(df_mse['mse']).as_matrix()
    #cos
    df_cos = pd.DataFrame(OrderedDict({'cos_ranks':ranklists,'cos':cos_list.reshape(-1)})).sort_values('cos',axis = 0, ascending= False)
    df_matrix['cosranks'], df_matrix['cos_1']  = pd.Series(df_cos['cos_ranks']).as_matrix(), pd.Series(df_cos['cos']).as_matrix()
    #eval3
    df_eval3 = pd.DataFrame(OrderedDict({'eval3_ranks':ranklists,'eval3':eval3_list.reshape(-1)})).sort_values('eval3',axis = 0, ascending= True)
    df_matrix['ev3ranks'], df_matrix['ev3_1'] = pd.Series(df_eval3['eval3_ranks']).as_matrix(), pd.Series(df_eval3['eval3']).as_matrix()
    #eval4
    df_eval4 = pd.DataFrame(OrderedDict({'eval4_ranks':ranklists,'eval4':eval4_list.reshape(-1)})).sort_values('eval4',axis = 0, ascending= True)
    df_matrix['ev4ranks'], df_matrix['ev4_1'] = pd.Series(df_eval4['eval4_ranks']).as_matrix(), pd.Series(df_eval4['eval4']).as_matrix()
    #ssim
    df_ssim = pd.DataFrame(OrderedDict({'ssim_ranks':ranklists,'ssim': ssim_list})).sort_values('ssim',axis = 0, ascending= False)
    df_matrix['ssranks'], df_matrix['ss_1'] = pd.Series(df_ssim['ssim_ranks']).as_matrix(), pd.Series(df_ssim['ssim'].as_matrix())
    #re_total
    df_retotal = pd.DataFrame(OrderedDict({'retotal_ranks': ranklists, 'retotal': re_list})).sort_values('retotal', axis=0,ascending=True)
    df_matrix['retotal_ranks'], df_matrix['retotal'] = pd.Series(df_retotal['retotal_ranks']).as_matrix(), pd.Series(df_retotal['retotal'].as_matrix())
    # re_h
    df_reh = pd.DataFrame(OrderedDict({'reH_ranks': ranklists, 'reH': reH_list})).sort_values('reH',axis=0,ascending=True)
    df_matrix['reH_ranks'], df_matrix['reH'] = pd.Series(df_reh['reH_ranks']).as_matrix(), pd.Series(df_reh['reH'].as_matrix())
    # re_t
    df_ret = pd.DataFrame(OrderedDict({'reT_ranks': ranklists, 'reT': reT_list})).sort_values('reT', axis=0, ascending=True)
    df_matrix['reT_ranks'], df_matrix['reT'] = pd.Series(df_ret['reT_ranks']).as_matrix(), pd.Series(df_ret['reT'].as_matrix())
    # pe_total, pe_h, pe_t
    pe1 = pd.Series(df_retotal['retotal'])
    df_matrix['petotal'] = (pe1*100).as_matrix()
    pe2 = pd.Series(df_reh['reH'])
    df_matrix['peH'] = (pe2*100).as_matrix()
    pe3 = pd.Series(df_ret['reT'])
    df_matrix['peT'] = (pe3*100).as_matrix()

    df2 = pd.DataFrame(df_matrix)
    df2 = df2[['df_rank','msranks','mse_1','cosranks','cos_1','ev3ranks','ev3_1','ev4ranks','ev4_1','ssranks','ss_1','reH_ranks','reH','peH','reT_ranks','reT','peT']]
    df2.columns =['ranks', 'mse_ranks', 'mse', 'cos_ranks', 'cos', 'eval3_ranks', 'eval3', 'eval4_ranks', 'eval4', 'ssim_ranks','ssim','reH_ranks','reH','peH','reT_ranks','reT','peT']
    pprint(df2)

    return df2

if __name__=='__main__':
    dir = 'C://Users//korea//Desktop//project//ksc//ksc_1204//final_result//results' #각 날짜별 순위 리스트가 포함되어있는 폴더 위치(2016년 2월 1일 ~ 2016년 11월 30일)
    inception_list = os.listdir(dir)
    inception_dic = {}
    for j in inception_list:
        with open(dir + "//" + j, 'r') as r:
            reader = csv.reader(r, delimiter='\t')
            date_list = []
            for row in reader:
                date_list.append(row)
        test1 = []
        for i in range(1, 11):
            test1.append(date_list[i][0].split(',')[0])
        inception_dic[j.split('.')[0]] = test1
    print(inception_dic)
    inception_dic_key = sorted(list(inception_dic.keys()))
    for inception_rank in inception_dic_key:
        f = check_result(int(inception_rank), inception_dic[inception_rank])
        f.to_csv('./final_result/' + inception_rank + "/" + inception_rank + '_inception.csv') #csv 파일 저장위치
        print(inception_rank + '_inception.csv is completed!')
