# 不考虑loop区

import csv
import math
import numpy as np
import random

# 方式1：设置三维图形模式
from matplotlib import pyplot as plt
from numpy import sqrt
import numpy as np
import math
import os
import itertools

from itertools import groupby, product
from numba import jit
# from pdb2coor import pdb2coor
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from sklearn.mixture import GaussianMixture
import gmm
import pickle
import multiprocessing
import logging
import json
from tools.pic import task,task_bins
from sklearn.cluster import DBSCAN
import tools.config 

logger=logging.getLogger('my_logger')


#anchors_illegal={}
#anchors_pos3d={}

with open('path.json','r') as json_file:
    path=json.load(json_file)

@jit(nopython=False)
def denoise1(obj,bound=0.003):
    data = np.sort(np.array(obj))
    bin = [i*0.1 for i in range(0, 401)]

    inliers, _ = np.histogram(data, bin)

    shape = len(obj)

    mask = np.full(shape, False)
    
    boundary=shape * bound
    preserve_bin_mask = [i > boundary for i in inliers]
    # 使用repeat函数重复元素
    repeated_elements = np.repeat(preserve_bin_mask,inliers)
    res=[]
    for i in range (0,len(repeated_elements)):
        if repeated_elements[i]:
            res.append(data[i])  
    return res


def denoise2(obj):
    bound=int(len(obj)/8)
    data = np.sort(np.array(obj))
    # 现在，你可以使用读取的对象进行后续操作
    data=data.reshape(-1, 1)
    # 使用DBSCAN拟合数据
    dbscan = DBSCAN(eps=3.8, min_samples=bound) # 参数有待考量
    dbscan.fit(data)

    # 标记簇中的点和离群值
    labels = dbscan.labels_

    # 将标签为-1的点标记为离群值
    outliers = data[labels == -1]
    inliers=data[labels == 0]
    inliers=inliers.flatten().tolist()
    return inliers
    
@jit(nopython=False)
def is_triangle(edge1, edge2, edge3):
    if edge1 ==21.6875 or edge2 ==21.6875 or edge3==21.6875:
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

# 找到该点在坐标中的坐标点
def find3dpos(tho,pos3d,pos_z=None):
    threhold=9
    for item in tho:
        if item == 21.6875:
            return False
        
    x = ((((tho[0]) ** 2 - (tho[1]) ** 2) / pos3d[1][0]) + pos3d[1][0]) / 2 
    y = ((tho[0] ** 2 - tho[2] ** 2 - x ** 2 + (x - pos3d[2][0]) ** 2) / pos3d[2][1] + pos3d[2][1]) / 2
    z = (tho[0]) ** 2 - x ** 2 - y ** 2
    if z<0:
        return False
    # verify
    z=sqrt(z)
    
    v0=abs(caldistance([x, y, z], pos3d[0])-tho[0]**2)
    v1=abs(caldistance([x, y, z], pos3d[1])-tho[1]**2)
    v2=abs(caldistance([x, y, z], pos3d[2])-tho[2]**2)
    if v0>threhold or v1>threhold or v2>threhold:
        return False
    if pos_z is None:
        return [x,y,z]
    else:
        pos_num=abs(caldistance([x, y, z], pos3d[3])-tho[3]**2)
        neg_num=abs(caldistance([x, y, -z], pos3d[3])-tho[3]**2)
        
        if pos_num > neg_num:
            z = 0 - z    
    return [x, y, z]

@jit(nopython=False)
def find_max_two_indices(lst):
    max1 = max(lst)
    max1_index = lst.index(max1)
    lst[max1_index] = float('-inf')  # 将最大值设置为负无穷，以便找到次最大值
    max2 = max(lst)
    max2_index = lst.index(max2)
    return max1_index, max2_index

# 求高斯混合的峰
@jit(nopython=False)
def calculate_gaussian_mixture_peak_height(means, stds, weights):
    means = np.array(means)
    stds = np.array(sqrt(stds))
    weights = np.array(weights)

    peak_heights = weights * (1 / (stds * np.sqrt(2 * np.pi)))
    return peak_heights

@jit(nopython=False)
def flatten(lst):
    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(flatten(i))
        else:
            flattened_list.append(i)
    return flattened_list


class MyInitial():
    def __init__(self, pdbname,basepath,pdbname1=None):
        self.manager = multiprocessing.Manager()
        self.pdbname = pdbname
        self.pdbname1=pdbname1
        #peak_npy,covs_npy,weight_npy
        self.pf_npy=np.load(f"{path['alphafold_npy']}{pdbname}.npy")
        self.nat_npy1=np.load(f"{path['native_npy']}{pdbname}.npy")
        self.nat_npy2=np.load(f"{path['native_npy']}{pdbname1}.npy")
        self.len_stru=len(self.pf_npy)
        self.triad_n=self.triad()
        
        # 使用共享的二维列表
        self.data = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.covs = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.weights = self.manager.list([self.manager.list([0 for _ in range(self.len_stru)]) for _ in range(self.len_stru)])
        self.basepath=basepath
        
    def triad(self):
        triplets = []
        for x in range(0, self.len_stru-2):
            for y in range(x+1, self.len_stru-1):
                for z in range(y+1, self.len_stru):
                    edge1=self.pf_npy[x,y]
                    edge2=self.pf_npy[x,z]
                    edge3=self.pf_npy[y,z]
                    if is_triangle(edge1, edge2, edge3):
                        # 所以符合三角不等式的三元组
                        triplet = [x, y, z]
                        triplets.append(triplet)
        logging.info(f"triplets len is {len(triplets)}")
        return triplets
                    
    def update_list(self, pi, pj, mus, cov , weight):
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
        
    # anchors是否满足三角不等式
    @jit(nopython=False)
    def is_legit(self,anchors):
        for i in range(len(anchors)):
            a1=anchors[i]
            a2=anchors[(i+1) %3]
            a3=anchors[(i+2) %3]
            tocheck=[a1,a2,a3]
            tocheck.sort()
            if not tocheck in self.triad_n:
                return False
        return True

    # 根据子矩阵重建三维结构
    @jit(nopython=False)
    def partmatrix(self,py):
        matrix = []  # 4个锚点的子矩阵
        row = 4
        col = 4
        for i in range(row):
            m = []
            for j in range(col):
                m.append(0)
            matrix.append(m)
        
        matrix[0][1]=matrix[1][0]=self.pf_npy[py[0]][py[1]]
        matrix[0][2]=matrix[2][0]=self.pf_npy[py[0]][py[2]]
        matrix[1][2]=matrix[2][1]=self.pf_npy[py[1]][py[2]]
        
        matrix[3][0]=matrix[0][3]=self.pf_npy[py[3]][py[0]]
        matrix[3][1]=matrix[1][3]=self.pf_npy[py[3]][py[1]]
        matrix[3][2]=matrix[2][3]=self.pf_npy[py[3]][py[2]]
        
        return matrix
    @jit(nopython=False)
    def partofmatrix(self,matrixs,anchors,a,b):
        matrix = []  # 6个点的子矩阵
        row = 6
        col = 6
        for i in range(row):
            m = []
            for j in range(col):
                if i<4 and j <4:
                    m.append(matrixs[i][j])
                else:
                    m.append(0)
            
            matrix.append(m)
        
        matrix[4][0]=matrix[0][4]=self.pf_npy[a][anchors[0]]
        matrix[4][1]=matrix[1][4]=self.pf_npy[a][anchors[1]]
        matrix[4][2]=matrix[2][4]=self.pf_npy[a][anchors[2]]
        matrix[4][3]=matrix[3][4]=self.pf_npy[a][anchors[3]]
        
        matrix[5][0]=matrix[0][5]=self.pf_npy[b][anchors[0]]
        matrix[5][1]=matrix[1][5]=self.pf_npy[b][anchors[1]]
        matrix[5][2]=matrix[2][5]=self.pf_npy[b][anchors[2]]
        matrix[5][3]=matrix[3][5]=self.pf_npy[b][anchors[3]]
        
        matrix[4][5]=matrix[5][4]=-1
        return matrix
    
    # 4维距离矩阵3d结构绘制
    @jit(nopython=False)
    def set3dpos(self,matrix):
        pos3d=[[],[],[],[]]
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
        
        # print("set3dpos1: ",pos3d)
        tho=[matrix[0][3],matrix[1][3],matrix[2][3]]
        
        posz=find3dpos(tho,pos3d,pos_z=None)
        
        if posz is False:
            return False
        else:
            pos3d[3]=posz
            
        return pos3d
        
    def create_dict(self,anchors,edges):
        dict={}
        dict['anchors'] = anchors
        dict['edges'] = edges
        return dict
        
    def initialize_anchors(self,pi,pj):
        anchors_pi=[]
        res_dij=[]
        anchors_dict=[]
        sum_times=10000
        rand_len=len(self.triad_n)
        for item in self.triad_n:
            #anchors=self.triad_n[index]
            if pi in item or pj in item:
                continue
            if 21.6875 in [self.pf_npy[pi][item[0]],self.pf_npy[pi][item[1]],self.pf_npy[pi][item[2]]] or 21.6875 in [self.pf_npy[pj][item[0]],self.pf_npy[pj][item[1]],self.pf_npy[pj][item[2]]]:
                continue
            for idx in range(max(item)+1,self.len_stru):
                if idx in item or idx==pi or idx==pj:
                    continue
                pos4=idx
                anchors=item.copy()
                anchors.append(pos4)
                mymatrix = self.partmatrix(anchors)    
                pos3d=self.set3dpos(mymatrix)
                if not pos3d:
                    continue
                if self.pf_npy[pi][anchors[3]]==21.6875 or self.pf_npy[pj][anchors[3]]==21.6875:
                    continue
                else: break
            if idx==self.len_stru:
                continue
            if not pos3d:
                continue
                
            tho1=[self.pf_npy[pi][anchors[0]],self.pf_npy[pi][anchors[1]],self.pf_npy[pi][anchors[2]],self.pf_npy[pi][anchors[3]]]
            posa=find3dpos(tho1,pos3d,pos_z=1)
            if not posa :continue
            tho2=[self.pf_npy[pj][anchors[0]],self.pf_npy[pj][anchors[1]],self.pf_npy[pj][anchors[2]],self.pf_npy[pj][anchors[3]]]
            posb=find3dpos(tho2,pos3d,pos_z=1)
            if not posb :continue
            
            dij=sqrt(caldistance(posa,posb))
            #anchors_res.append(anchors)
            res_dij.append(dij)
            
            matrix=self.partofmatrix(mymatrix,anchors,pi,pj)
            matrix[4][5]=matrix[5][4]=dij
            anchors_dict.append(self.create_dict(anchors,matrix))
            if len(res_dij)==8000: 
                break
            
        # 保存计算过程    
        if not res_dij == []:
            # 保存计算过程    
            save_path=os.path.join(self.basepath,"other",str(pi)+'_'+str(pj))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            with open(save_path+'/anchors_dict.pkl', 'wb') as f:
                    pickle.dump(anchors_dict, f)
            
            with open(save_path+'/todraw.pkl', 'wb') as f:
                pickle.dump(res_dij, f)
            
        return res_dij
    
    def cal_dij_by_anchor(self,pi,pj):
        save_path=os.path.join(self.basepath,"other",str(pi)+'_'+str(pj))
        res=[]
        try:
            aa=1/0
            with open(f"{self.basepath}/other/{pi}_{pj}/after_denoise.pkl", 'rb') as file:
                res = pickle.load(file)
        except Exception as e:
            res_dij=self.initialize_anchors(pi,pj)
            logging.info(f"res_dij's length of {pi}_{pj}: len{res_dij}")
            if len(res_dij) ==0:
                #print("nonono")
                return [self.pf_npy[pi][pj],-1],[0,0],[1,0]
            # myround=10000
            # 去噪
            res=denoise1(res_dij)

            with open(save_path+'/after_denoise.pkl', 'wb') as f:
                pickle.dump(res, f)
        
        if len(res) ==0:
            logging.INFO("after denoise" +str(pi)+' '+str(pj)+' has no results')
            return [self.pf_npy[pi][pj],-1],[0,0],[1,0]
        
        mus, covs, weights = gmm.get_gmm_para_mutil(np.array(res).reshape(-1, 1),save_path,pi,pj)
    
        muss=flatten(mus.tolist())
        covss=flatten(covs.tolist())
        weightss=flatten(weights)
        print(len(res),muss,covss,weightss)
        logger.info("yoriko0: "+str(pi)+" "+str(pj)+" "+str(len(res))+" "+str(muss)+str(covss)+str(weightss))
        
        nat_dis=[self.nat_npy1[pi][pj]]
        if self.pdbname1!=self.pdbname:
            nat_dis.append(self.nat_npy2[pi][pj])
        
        task(res,pi,pj,muss,covss,weightss,self.basepath,self.pdbname,nat_dis=nat_dis,pf_dis=[self.pf_npy[pi][pj]])
        
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
    
    def process_data(self,i,j):
        #print(f"start cal {i},{j}")
        if abs(i-j)<=6:
            if i==j:
                mus=[0,-1]
            else:
                mus = [self.pf_npy[i][j],-1]
            covs=[0,0]
            weights=[1,0]
        else:
            mus,covs,weights=self.cal_dij_by_anchor(i,j)
        logger.info(f"update: {i} {j} {mus} {covs} {weights}")
        self.update_list(i,j,mus,covs,weights)
        
    def Interation(self):
        num_processes = 27
        processes = []
        for i in range(self.len_stru):
            for j in range(i,self.len_stru):
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
        
        with open(self.basepath+'/mus.pkl', 'wb') as file:
            pickle.dump(data, file)
        with open(self.basepath+'/covs.pkl', 'wb') as file:
            pickle.dump(data, file)
        with open(self.basepath+'/weight.pkl', 'wb') as file:
            pickle.dump(data, file)
        data1=np.array(data)
        covs1=np.array(covs)
        weight1=np.array(weight)
        
        np.save(self.basepath+'/data.npy', data1)
        np.save(self.basepath+'/covs.npy', covs1)
        np.save(self.basepath+'/weights.npy', weight1)
        return data1,weight1
    
    
if __name__ == "__main__":
    pdb_name="1ysy"
    pdb_name1="2kys"
    obj=MyInitial(pdb_name,'/home/yuyue/evomatrix/evo/lj/',pdb_name1)

    obj.cal_dij_by_anchor(5,10)

