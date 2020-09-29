#!/usr/bin/env python
"""参数化因子图测试：只含双变量因子。
   1.已知所有参数，推理所有变量
   2.借助第一步推理得到的所有变量作为证据节点，学习所有参数，并且比较学习前后参数
   3.借助第二步学习到的参数，重新推理一遍所有变量，并且和第一步推理得到的变量进行比较"""

import random
from collections import namedtuple
import numpy as np
import numbskull
from numbskull.numbskulltypes import *
import math
import pickle

unobserved_ratio = 0.2
# 创建因子图
def create_fg():
    # 定义完因子图之后，需要存成文件，以备后面验证
    weights = 50  # 50种因子，全部为双因子
    variables = 1000  # 1000 个变量
    #unary_factors = 0  # 没有单因子
    binary_factors = 50 * 10  # 500个双因子，意味着有500组变量相连
    edges = 2 * binary_factors
    factors = binary_factors

    weight = np.zeros(weights, Weight)
    variable = np.zeros(variables, Variable)
    factor = np.zeros(factors, Factor)
    fmap = np.zeros(edges, FactorToVar)
    domain_mask = np.zeros(variables, np.bool)

    # 一·初始化变量
    for i in range(variables):
        variable[i]["isEvidence"] = 0  # 都设置为隐变量
        variable[i]["initialValue"] = random.choice((1, 0))
        variable[i]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
        variable[i]["cardinality"] = 2

    # 二·随机初始化边: 此处必须保证所有的因子和变量都被用上
    binary_edge = namedtuple('binary_edge', ['index', 'factorId', 'varId1', 'varId2'])  # 双变量因子的边
    var_list = list(range(0, 1000))
    random.shuffle(var_list)  #打乱变量顺序
    binary_edges = list()
    #  随机初始化双因子的边存成list,元素类型为binary_edge
    for i in range(binary_factors): #500个双因子
        varId1 = 2 * i
        varId2 = 2 * i + 1
        edge = binary_edge(i, i, varId1, varId2)
        binary_edges.append(edge)

    # 三·初始化权重
    for i in range(weights):
        weight[i]["isFixed"] = True  # True表示是真实值，后期不用再学习
        #weight[i]["parameterize"] = False
        weight[i]["initialValue"] = random.uniform(-10, 10)
        weight[i]["parameterize"]= 0
        weight[i]["a"] = 0
        weight[i]["b"] = 0


   # 四·初始化因子：双因子因子函数为9; 500个因子
    factor_index = 0
    fmp_index = 0
    for i in range(binary_factors):  # 500个为双因子(50种)
        factor[factor_index]["factorFunction"] = 9
        factor[factor_index]["weightId"] = random.randint(0, 49)  # 随机分配50种权重到500个双因子
        factor[factor_index]["featureValue"] = 1
        factor[factor_index]["arity"] = 2  # 双因子度为2
        factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次要加2
        fmap[fmp_index]["vid"] = binary_edges[i][2]  # 双因子需要填两个vid和两个x
        fmap[fmp_index]["x"] = 0  # 双因子因为不需要函数化，所以将x赋值为0，反正也用不上
        fmp_index += 1
        fmap[fmp_index]["vid"] = binary_edges[i][3]
        fmap[fmp_index]["x"] = 0
        fmp_index += 1
        factor_index += 1

    return weight, variable, factor, fmap, domain_mask, edges

#1.已知所有参数，推理所有变量
learn = 10000
ns = numbskull.NumbSkull(n_inference_epoch=10,
                         n_learning_epoch=learn,
                         quiet=True,
                         learn_non_evidence=True,
                         stepsize=0.0001,
                         burn_in=10,
                         decay=0.001 ** (1.0 / learn),
                         regularization=1,
                         reg_param=0.01)
fg = create_fg()
ns.loadFactorGraph(*fg)
print("First Inference Finished!")
for i in range(1000):
    if ns.factorGraphs[0].marginals[i] > 0.5:   #推理出来的变量是概率值,保存在factorGraphs[0].marginals，现在写回到fg
        fg[1][i]['initialValue'] = 1
    else:
        fg[1][i]['initialValue'] = 0
# 存储第一遍推理后的因子图
with open('double_factor_first_inference.pkl', 'wb') as f:    #由参数推理变量后的图
    pickle.dump(fg, f)

#2.借助第一步推理得到的所有变量作为证据节点，学习所有参数，并且比较学习前后参数
ns1 = numbskull.NumbSkull(n_inference_epoch=10,
                         n_learning_epoch=learn,
                         quiet=True,
                         learn_non_evidence=False,
                         stepsize=0.0001,
                         burn_in=10,
                         decay=0.001 ** (1.0 / learn),
                         regularization=1,
                         reg_param=0.01)

#加载第一遍推理后的因子图
with open('double_factor_first_inference.pkl', 'rb') as f:
    fg1 = pickle.load(f)

TrueBinWeightValue = list() #双变量因子真实权重

#weight, variable, factor, fmap, domain_mask, edges
FirstInferenceVarValue = list()
for i in range(len(fg1[0])):
    fg1[0][i]['isFixed'] = False
    TrueBinWeightValue.append(fg1[0][i]["initialValue"])
    fg1[0][i]["initialValue"] = random.uniform(-10,10)


for i in range(len(fg1[1])):
    if i > int(len(fg1[1])*unobserved_ratio):   #取一部分变量作为隐变量
        fg1[1][i]['isEvidence'] = 1
    FirstInferenceVarValue.append(fg1[1][i]['initialValue'])

ns1.loadFactorGraph(*fg1)
ns1.learning()
print("Weights Learned Finished!")

# for i in range(len(fg1[0])):
#     print("weight", i, "True:", TrueBinWeightValue[i], "----------", "Learned:", ns1.factorGraphs[0].weight[i]['initialValue'])

with open('double_factor_Learned.pkl', 'wb') as f:
    pickle.dump(fg1, f)

#3.借助第二步学习到的参数，重新推理一遍所有变量，并且和第一步推理得到的变量进行比较"""
ns2 = numbskull.NumbSkull(n_inference_epoch=10,
                         n_learning_epoch=learn,
                         quiet=True,
                         learn_non_evidence=False,
                         stepsize=0.0001,
                         burn_in=10,
                         decay=0.001 ** (1.0 / learn),
                         regularization=1,
                         reg_param=0.01)
with open('double_factor_Learned.pkl', 'rb') as f:
    fg2 = pickle.load(f)

for i in range(len(fg2[0])):  # 权重设为已知
    fg2[0][i]['isFixed'] = True

for i in range(len(fg2[1])):  # 变量全部设为未知，并且重新赋值
    fg2[1][i]["isEvidence"] = 0
    fg2[1][i]["initialValue"] = random.choice((1, 0))

ns2.loadFactorGraph(*fg2)
ns2.inference()
print("Second Inference Finished!")
SecondInferencedVarValue = list()
for i in range(len(fg2[1])):
    if ns2.factorGraphs[0].marginals[i] > 0.5:   #推理出来的变量是概率值,保存在factorGraphs[0].marginals，现在写回到fg2
        fg2[1][i]['initialValue'] = 1
    else:
        fg2[1][i]['initialValue'] = 0
    SecondInferencedVarValue.append(fg2[1][i]['initialValue'])

# 计算隐变量推理准确率
unobserved_hit = 0
unobserved_num=int(len(fg2[1]) * unobserved_ratio)
observed_num = len(fg2[1]) - unobserved_num

observed_hit   = 0
for i in range(len(fg2[1])):
    if i < unobserved_num:  # 取一部分变量作为隐变量
        if FirstInferenceVarValue[i] == SecondInferencedVarValue[i]:
            unobserved_hit += 1
    else:
        if FirstInferenceVarValue[i] == SecondInferencedVarValue[i]:
            observed_hit += 1
print("变量个数：",len(fg2[1]))
print("隐变量比例",unobserved_ratio)

print("UnobservedVar Inferenced Accuracy:", unobserved_hit / unobserved_num)
print("ObservedVar Inferenced Accuracy:", observed_hit / observed_num)
with open('double_factor_SecondInference.pkl', 'wb') as f:
    pickle.dump(fg2, f)