# 迭代一轮 
# 读一个npy，输出一个npy
import csv
import pickle
import numpy as np
import random
import torch
import pandas as pd
# 方式1：设置三维图形模式
from matplotlib import pyplot as plt
from numpy import sqrt
import numpy as np
import math
import sys
import os
import scipy
import itertools
import argparse
import datetime
from itertools import groupby
from numba import jit
# from pdb2coor import pdb2coor
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from sklearn.mixture import GaussianMixture
import gmm


import json
import logging
from tools.pic import task1
import multiprocessing


logger=logging.getLogger('my_logger')

threhold=21.6875

with open('path.json','r') as json_file:
    path=json.load(json_file)

#@jit(nopython=False)
def save_list_to_pickle(lst,file_apth='/home/hp/falcon/evomatrix/'):
        with open(file_apth+'iterate_aft.pickle', 'wb') as f:
            pickle.dump(lst, f)

#@jit(nopython=False)
def denoise1(obj,bound=0.003):
    data = np.sort(np.array(obj))
    bin = [i*0.1 for i in range(0, 401)]

    inliers, _ = np.histogram(data, bin)

    shape = len(obj)

    mask = np.full(shape, False)
    
    boundary=shape * bound
    print(boundary)
    preserve_bin_mask = [i > boundary for i in inliers]
    # 使用repeat函数重复元素
    repeated_elements = np.repeat(preserve_bin_mask,inliers)
    res=[]
    for i in range (0,len(repeated_elements)):
        if repeated_elements[i]:
            res.append(data[i])  
    return res


            
#@jit(nopython=False)
def is_triangle(edge1, edge2, edge3):
    if threhold in [edge1,edge2,edge3]:
        return False
    if -1 in [edge1,edge2,edge3]:
        return False
    # 检查三角不等式
    if edge1 + edge2 > edge3 and edge1 + edge3 > edge2 and edge2 + edge3 > edge1:
        return True
    else:
        return False

#@jit(nopython=False)
def caldistance(pointa, pointb):
    result = math.pow(pointa[0] - pointb[0], 2) + math.pow(pointa[1] - pointb[1], 2) + math.pow(pointa[2] - pointb[2],2)
    
    return result



#@jit(nopython=False)
def find_max_two_indices(lst):
    max1 = max(lst)
    max1_index = lst.index(max1)
    lst[max1_index] = float('-inf')  # 将最大值设置为负无穷，以便找到次最大值
    max2 = max(lst)
    max2_index = lst.index(max2)
    return max1_index, max2_index

#@jit(nopython=False)
def data3weight(disab,weightab,savepath="./lj/"):
    # 先对weightab做处理，扩展到4w数据上面

    weight = np.array(weightab)
    weightab = np.exp(weight)/np.sum(np.exp(weight))
    #breaks = torch.linspace(2.3125, 21.6875, steps=37 - 1).tolist()
    #breaks.insert(0,0)
    #breaks.append(float('inf'))    

    weightab=weightab*40000 # 数据扩展
    # 使用格式化字符串将每个元素保留3位小数
    threshold = 0.01 # to change 0.01
    max_value=max(disab)
    
    bins=[round(i*threshold,2) for i in range((int(max_value/threshold))+1)]

    size_disab=len(disab)
    size_bins=len(bins)
    for i in range(size_disab):
        for j in range(size_bins):
            if bins[j]<= disab[i]<=bins[j]+threshold:
                disab[i]=bins[j]
                break
    
    combined_list = list(zip(disab, weightab))
    #print(combined_list)
    # 创建一个字典用于累加第二个值
    result_dict = {}
    for item in combined_list:
        key,value = item
        if key in result_dict:
            result_dict[key] += value
        else:
            result_dict[key] = value

    # 创建一个空列表用于存储还原后的一维列表
    restored_list = []
    #with open(savepath+'/bins_value.txt', 'w') as file:
    for key, value in result_dict.items():
        # 将每个键的值重复value次，并添加到restored_list中
        restored_list.extend([key] * int(value))
    
    # 使用float()函数将字符串列表中的每个元素转换为浮点数
    restored_list = [float(item) for item in restored_list]

    return restored_list

# 获取所有组合
input_lists=[[0,1],[0,1],[0,1],[0,1]]
#@jit(nopython=False)
def all_combinations(lists):
    return itertools.product(*lists)

#@jit(nopython=False)
def flatten(lst):
    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(flatten(i))
        else:
            flattened_list.append(i)
    return flattened_list

# 求高斯混合的峰
#@jit(nopython=False)
def calculate_gaussian_mixture_peak_height(means, stds, weights):
    means = np.array(means)
    stds = np.array(np.sqrt(stds))
    weights = np.array(weights)

    peak_heights = weights * (1 / (stds * np.sqrt(2 * np.pi)))
    print("cal:")
    print(means,stds,weights)
    print(peak_heights)
    return peak_heights


class MyIterate():
    def __init__(self, pdbname,basepath,beforepath,pdbname2=None):
        self.manager = multiprocessing.Manager()
        self.pdbname = pdbname
        self.pdbname2=pdbname2
        self.nat_npy1=np.load(f"{path['native_npy']}{pdbname}.npy")
        self.nat_npy2=np.load(f"{path['native_npy']}{pdbname2}.npy")
        
        
        #peak_npy,covs_npy,weight_npy
        self.data_npy=np.load(f"{beforepath}/data_res.npy") # 检查路径正确否
        self.weight_npy=np.load(f"{beforepath}/weights_res.npy")
        self.len_stru=len(self.data_npy)
        self.triad_n=self.triad()
         
        # 使用共享的二维列表
        self.data = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.covs = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.weights = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.basepath=basepath 
        
        
    def triad(self):
        triplets = []
        for x in range(0, self.len_stru):
            for y in range(x+1, self.len_stru):
                for z in range(y+1, self.len_stru):
                    comb=all_combinations(input_lists[0:3])
                    for item in comb:
                        edge1=self.data_npy[x][y][item[0]]
                        edge2=self.data_npy[y][z][item[1]]
                        edge3=self.data_npy[x][z][item[2]]
                        if is_triangle(edge1, edge2, edge3):
                            # 所以符合三角不等式的三元组
                            triplet = ([x, y, z],item)
                            triplets.append(triplet)
        logging.info(f"triplets len is {len(triplets)}")
        return triplets
    
    def update_list(self, pi, pj, mus, cov , weight):
        logger.info(f"update_iterate: {pi} {pj} {mus} {cov} {weight}")
        # 进程安全的操作
        sub_list = self.data[pi]
        sub_list[pj] = mus
        sub_list = self.data[pj]
        sub_list[pi] = mus
        
        sub_list = self.covs[pi]
        sub_list[pj] = cov
        sub_list = self.covs[pj]
        sub_list[pi] = cov
        
        sub_list = self.weights[pi]
        sub_list[pj] = weight
        sub_list = self.weights[pj]
        sub_list[pi] = weight
        

    '''
    调用iterate_plan.py
    '''
    
    def process_data(self,i, j):
        print(f"start cal {i},{j}")
        if abs(j - i) <= 6: #？
            if (i-j) ==0:
                mus=[0,-1]
            # 保存数据
            else:
                #mus = [self.data_npy[i][j][0],self.data_npy[i][j][0]]
                mus = [self.data_npy[i][j][0],-1]
            weights=[1,0]
            covs=[0,0]
        else:
            mus, covs, weights = self.adj_cal_iterate(i, j)
            mus = [mus[0],mus[1]]
            covs=[covs[0],covs[1]]
            #logger.info("yoriko: "+str(i)+" "+str(j)+" "+str(mus[0])+" "+str(mus[1])+" "+str(weights[0])+" "+str(weights[1]))
        
        self.update_list(i,j,mus,covs,weights)
        
    def iterate_once(self):
        num_processes = 33 # 根据需要调整进程数量
        
        processes = []

        #len_stru=10
        for i in range(self.len_stru):
            for j in range(i, self.len_stru):
                process = multiprocessing.Process(target=self.process_data, args=(i, j))
                processes.append(process)
                process.start()
                
                if len(processes) >= num_processes:
                    for process in processes:
                        process.join()
                    processes = []
                
        for process in processes:
            process.join()
            
        data = [list(row) for row in self.data]
        covs = [list(row) for row in self.covs]
        weight = [list(row) for row in self.weights]
        
        
        data1=np.array(data)
        covs1=np.array(covs)
        weight1=np.array(weight)
        np.save(self.basepath+'/data.npy', data1)
        np.save(self.basepath+'/covs.npy', covs1)
        np.save(self.basepath+'/weights.npy', weight1)
        return data1,weight1    

    def if_legitimate(self,p1,listp):
        # 遍历所有组合
        combinations3 = all_combinations(input_lists[:-1])
        useful_list3=[]
        useful_weight_3=[]
        for combination in combinations3:
            edge1=self.data_npy[p1][listp[0]][combination[0]]
            edge2=self.data_npy[p1][listp[1]][combination[1]]
            edge3=self.data_npy[listp[0]][listp[1]][combination[2]]
            if edge1<0 or edge2<0 or edge3<0:
                continue
            if not is_triangle(edge1, edge2, edge3):
                continue
            useful_list3.append(combination)
            weight_1=self.weight_npy[p1][listp[0]][combination[0]]
            weight_2=self.weight_npy[p1][listp[1]][combination[1]]
            weight_3=self.weight_npy[listp[0]][listp[1]][combination[2]]
            # log
            weight_all=np.array([weight_1, weight_2, weight_3])
            weight_all=np.sum(np.log(weight_all))
            weight_whole=weight_all.tolist()
            #weight_whole=weight_1*weight_2*weight_3
            useful_weight_3.append(weight_whole) 
            
            # print(combination,p1,listp[0],listp[1],edge1,edge2,edge3)
        return useful_list3,useful_weight_3
    
    def is_legit(self,p4,list3,useful_list3,useful_weight_3):
        useful_list4=[]
        useful_weight_4=[]
        # 遍历useful_list3
        for index,item in enumerate(useful_list3):
            # 边的对应关系
            edge1=self.data_npy[list3[0]][list3[1]][item[0]]
            edge2=self.data_npy[list3[0]][list3[2]][item[1]]
            edge3=self.data_npy[list3[1]][list3[2]][item[2]]
                
            combinations4 = all_combinations(input_lists[:-1])    
            for combination in combinations4:
                #print(item,combination)
                edge04=self.data_npy[p4][list3[0]][combination[0]]
                edge14=self.data_npy[p4][list3[1]][combination[1]]
                edge24=self.data_npy[p4][list3[2]][combination[2]]
                if edge04 <0 or edge14<0 or edge24<0:
                    continue
                # to check
                if not (is_triangle(edge04, edge14, edge1) or is_triangle(edge14, edge24, edge2) or is_triangle(edge04, edge24, edge3)):
                    #print("have a nice day")
                    continue
                # item的基础上，combination
                #print(item,combination)
                weight_1=self.weight_npy[p4][list3[0]][combination[0]]
                weight_2=self.weight_npy[p4][list3[1]][combination[1]]
                weight_3=self.weight_npy[p4][list3[2]][combination[2]]
                # log
                weight_all=np.array([weight_1, weight_2, weight_3])
                weight_all=np.sum(np.log(weight_all))
                weight_whole=weight_all.tolist()+useful_weight_3[index]
                #weight_whole=weight_1*weight_2*weight_3*useful_weight_3[index]
                useful_weight_4.append(weight_whole)
                useful_list4.append((item,combination))
        # 遍历p4 和其他的
        
        return useful_list4,useful_weight_4
    
    def partmatrix(self,anchors,anchors_choice):
        matrix = []  # 4个锚点的子矩阵
        row = 6
        col = 6
        for i in range(row):
            m = []
            for j in range(col):
                m.append(0)
            matrix.append(m)
        
        matrix[0][1]=matrix[1][0]=self.data_npy[anchors[0]][anchors[1]][anchors_choice[0]]
        matrix[0][2]=matrix[2][0]=self.data_npy[anchors[0]][anchors[2]][anchors_choice[1]]
        matrix[1][2]=matrix[2][1]=self.data_npy[anchors[1]][anchors[2]][anchors_choice[2]]

        
        return matrix

    def full_matirx(self,matrix,anchors,anchors_choice):
        matrix[3][0]=matrix[0][3]=self.data_npy[anchors[3]][anchors[0]][anchors_choice[0]]
        matrix[3][1]=matrix[1][3]=self.data_npy[anchors[3]][anchors[1]][anchors_choice[1]]
        matrix[3][2]=matrix[2][3]=self.data_npy[anchors[3]][anchors[2]][anchors_choice[2]]
        return matrix

    def partofmatrix(self,matrix,anchors,a,b,comba,combb):
        
        matrix[4][0]=matrix[0][4]=self.data_npy[a][anchors[0]][comba[0]]
        matrix[4][1]=matrix[1][4]=self.data_npy[a][anchors[1]][comba[1]]
        matrix[4][2]=matrix[2][4]=self.data_npy[a][anchors[2]][comba[2]]
        matrix[4][3]=matrix[3][4]=-1
        
        matrix[5][0]=matrix[0][5]=self.data_npy[b][anchors[0]][combb[0]]
        matrix[5][1]=matrix[1][5]=self.data_npy[b][anchors[1]][combb[1]]
        matrix[5][2]=matrix[2][5]=self.data_npy[b][anchors[2]][combb[2]]
        matrix[5][3]=matrix[3][5]=-1
        
        matrix[4][5]=matrix[5][4]=-1
        return matrix
    
    # 4维距离矩阵3d结构绘制
    def set3dpos(self,anchors=None,anchors_choice=None):
        
        pos3d=[[],[],[]]
        matrix=self.partmatrix(anchors,anchors_choice)
        
        # print(matrix)
        a = matrix[1][2]
        b = matrix[2][0]
        c = matrix[0][1]

        pos3d[0] = [0, 0, 0]
        pos3d[1] = [c, 0, 0]
        cos1 = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)

        x = b * cos1
        if (abs(cos1) > 1):
            # print("cos>1 error ", a, b, c, cos1)
            return False

        y = b * sqrt(1 - cos1 ** 2)
        pos3d[2] = [x, y, 0]
        
        return matrix,pos3d
    
    # 找到该点在坐标中的坐标点
    def find3dpos(self,tho,pos3d,flag=False):
        if (threhold in tho) or (-1 in tho):
            return False
        x = ((((tho[0]) ** 2 - (tho[1]) ** 2) / pos3d[1][0]) + pos3d[1][0]) / 2 
        y = ((tho[0] ** 2 - tho[2] ** 2 - x ** 2 + (x - pos3d[2][0]) ** 2) / pos3d[2][1] + pos3d[2][1]) / 2
        z = (tho[0]) ** 2 - x ** 2 - y ** 2
        if z < 0:
            return False 
        else:
            z = sqrt(z)
            if flag:
                pos_num=abs(sqrt(caldistance([x, y, z], pos3d[3])) - tho[3])
                neg_num=abs(sqrt(caldistance([x, y, -z], pos3d[3])) - tho[3])
                if pos_num > neg_num:
                    z = 0 - z
        return [x, y, z]

    def create_dict(self,anchors,edges,weights):
        dict={}
        dict['anchors'] = anchors
        dict['edges'] = edges
        dict['weights'] = weights
        #print(dict)
        return dict
    
    #第四个点,可能坐标
    def anchors_p4(self,p4,anchors,matrix=None,pos3d=None,flag=False):
        comb=all_combinations(input_lists[0:3])
        chioce_p4=[]
        for item in comb:
            edge1=self.data_npy[p4][anchors[0]][item[0]]
            edge2=self.data_npy[p4][anchors[1]][item[1]]
            edge3=self.data_npy[p4][anchors[2]][item[2]]
            tho=[edge1,edge2,edge3]
            if flag:
                # 0
                edge4=self.data_npy[p4][anchors[3]][0]
                weight=self.weight_npy[p4][anchors[3]][0]
                tho1=[edge1,edge2,edge3,edge4]
                pos_z=self.find3dpos(tho1,pos3d,flag)
                if pos_z is not False:
                    chioce_p4.append((item,pos_z,weight))
                # 1
                edge4=self.data_npy[p4][anchors[3]][1]
                weight=self.weight_npy[p4][anchors[3]][1]
                tho2=[edge1,edge2,edge3,edge4]
                pos_z=self.find3dpos(tho2,pos3d,flag)
                if pos_z is not False:
                    chioce_p4.append((item,pos_z,weight))
            else:
                pos_z=self.find3dpos(tho,pos3d,flag)
                if pos_z is False:
                    continue 
                else:
                    chioce_p4.append((item,pos_z,1))
        return chioce_p4
    
    def cal_weights(self,anchors,pi,pj,chioce1,chioce2,chioce3,chioce4,weightpi,weightpj):
        weight_all=[]
        weight_all.append(weightpi)
        weight_all.append(weightpj)
        
        weight_all.append(self.weight_npy[anchors[0]][anchors[1]][chioce1[0]])
        weight_all.append(self.weight_npy[anchors[1]][anchors[2]][chioce1[1]])
        weight_all.append(self.weight_npy[anchors[0]][anchors[3]][chioce1[2]])
        
        weight_all.append(self.weight_npy[anchors[3]][anchors[0]][chioce2[0]])
        weight_all.append(self.weight_npy[anchors[3]][anchors[1]][chioce2[1]])
        weight_all.append(self.weight_npy[anchors[3]][anchors[2]][chioce2[2]])
        
        weight_all.append(self.weight_npy[pi][anchors[0]][chioce3[0]])
        weight_all.append(self.weight_npy[pi][anchors[1]][chioce3[1]])
        weight_all.append(self.weight_npy[pi][anchors[2]][chioce3[2]])
        
        weight_all.append(self.weight_npy[pj][anchors[0]][chioce4[0]])
        weight_all.append(self.weight_npy[pj][anchors[1]][chioce4[1]])
        weight_all.append(self.weight_npy[pj][anchors[2]][chioce4[2]])
        
        weight_all=np.array(weight_all)
        weight_all=np.exp(np.sum(np.log(weight_all)))
        
        return weight_all
    
    def initialize_anchors(self,pi,pj):
        res_dij=[]
        res_weight=[]
        
        anchors_dict=[]
        for item in self.triad_n:
            anchors=item[0]
            anchors_choice=item[1]
            if pi in anchors or pj in anchors:
                continue
            is3dpos=self.set3dpos(anchors,anchors_choice)
            if is3dpos is False:
                continue
            matrix,pos3d=is3dpos
            for p4 in range (max(anchors)+1,self.len_stru):
                if pi ==p4 or pj ==p4 or p4 in anchors:
                    continue
                chioce_p4=self.anchors_p4(p4,anchors,matrix,pos3d,flag=False)
                
                if len(chioce_p4)==0:
                    continue
                else:
                    anchors.append(p4)
                    p4chioce=chioce_p4[0][0]
                    p4weight=chioce_p4[0][2]
                    pos3d.append(chioce_p4[0][1])
                    
                    matrix=self.full_matirx(matrix,anchors,p4chioce)
                    break
            if len(anchors)!=4:
                continue 
            chioce_pi=self.anchors_p4(pi,anchors,matrix,pos3d,flag=True)
            chioce_pj=self.anchors_p4(pj,anchors,matrix,pos3d,flag=True)
            if len(chioce_pi)==0 or len(chioce_pj)==0:
                continue
            
            # 开始计算距离
            for chiocepi in chioce_pi:
                for chiocepj in chioce_pj:
                    dij=sqrt(caldistance(chiocepi[1],chiocepj[1]))
                    res_dij.append(dij)
                    
                    weightpi=chiocepi[2]
                    weightpj=chiocepj[2]
                    weights=self.cal_weights(anchors,pi,pj,anchors_choice,p4chioce,chiocepi[0],chiocepj[0],weightpi,weightpj)
                    res_weight.append(weights)
                    
                    matrix=self.partofmatrix(matrix,anchors,pi,pj,chiocepi[0],chiocepj[0])
                    matrix[4][5]=matrix[5][4]=dij
                    anchors_dict.append(self.create_dict(anchors,matrix,weights))
            
        return res_dij,res_weight
            
    def adj_cal_iterate(self,pi,pj):
        
        save_path=os.path.join(self.basepath,"other",str(pi)+'_'+str(pj))
        res=[]
        try:
            sum=1/0
            with open(f"{self.basepath}/other/{pi}_{pj}/after_denoise.pkl", 'rb') as file:
                res = pickle.load(file)
        except Exception as e:
            res_dij=self.initialize_anchors(pi,pj)
            if len(res_dij) ==0:
                #print("nonono")
                return [self.pf_npy[pi][pj],-1],[0,0],[1,0]
            # myround=10000
            # 去噪
            res=denoise1(res_dij)

            with open(save_path+'/after_denoise.pkl', 'wb') as f:
                pickle.dump(res, f)
        if len(res) ==0:
            logging.info(f"after denoise {pi} {pj}  has no results")
            return [self.pf_npy[pi][pj],-1],[0,0],[1,0]
        if len(res) ==1:
            logging.info(f"after denoise {pi} {pj}  has 1 results")
            return [res[0],-1],[0,0],[1,0] 
        if len(res) ==2:
            logging.info(f"after denoise {pi} {pj}  has 2 results")
            return [res[0],res[1]],[0,0],[0.5,0.5]
        mus, covs, weights = gmm.get_gmm_para_mutil(np.array(res).reshape(-1, 1),save_path,pi,pj)
    
        muss=flatten(mus.tolist())
        covss=flatten(covs.tolist())
        weightss=flatten(weights)
        print(len(res),muss,covss,weightss)
        logger.info("yoriko0: "+str(pi)+" "+str(pj)+" "+str(len(res))+" "+str(muss)+str(covss)+str(weightss))
        
        nat_dis=[self.nat_npy1[pi][pj]]
        if self.pdbname!=self.pdbname2:
            nat_dis.append(self.nat_npy2[pi][pj])
        
        task1(res,pi,pj,muss,covss,weightss,self.basepath,self.pdbname,nat_dis=nat_dis,pf_dis=list(self.data_npy[pi][pj]))
        
        if len(muss)==3:
            height_peak=calculate_gaussian_mixture_peak_height(muss,covss,weightss)
            height_peak=height_peak.tolist()
            idx1,idx2=find_max_two_indices(height_peak)
            mus = [muss[idx1],muss[idx2]]
            covs = [covss[idx1],covss[idx2]]
            weights = [weightss[idx1],weights[idx2]]
        elif len(mus)==2:
            mus = [muss[0],muss[1]]
            covs = [covss[0],covss[1]]
            weights = [weights[0],weights[1]]
            
        else:
            mus = [mus[0][0],-1]
            covs = [0,0]
            weights = [1,0]
        return mus,covs,weights    
        
def main():
    parser = argparse.ArgumentParser(description='iterate')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--pdb_name1', dest='pdbname1', default="1m12", help='input pdb name')
    parser.add_argument('-p2', '--pdb_name2', dest='pdbname2', default="1sn6", help='input pdb name')
    args = parser.parse_args()
    
    pdbname1=args.pdbname1
    pdbname2=args.pdbname2
    obj=MyIterate(pdbname1,"/home/yuyue/evomatrix/evo/lj/end",f"/home/yuyue/evomatrix/evo/record/strategy5/{pdbname1}/initial",pdbname2)
    obj.adj_cal_iterate(0,30)
