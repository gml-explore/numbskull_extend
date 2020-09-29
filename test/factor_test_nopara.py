#!/usr/bin/env python
""" 非参数化因子图测试：含单变量因子和双变量因子
   1.已知所有参数，推理所有变量
   2.借助第一步推理得到的所有变量作为证据节点，学习所有参数，并且比较学习前后参数
   3.借助第二步学习到的参数，重新推理一遍所有变量，并且和第一步推理得到的变量进行比较"""

import pickle
# Author：hxl
import random
from collections import namedtuple

import numbskull
from numbskull.numbskulltypes import *


# 创建因子图
def create_fg():
    # 定义完因子图之后，需要存成文件，以备后面验证
    weights = 50  # 50种因子，10种单因子，40种双因子
    variables = 1000  # 1000 个变量
    unary_factors = 10 * 50  # 500个单因子，意味着只有部分变量有资格连单因子，意味着需要存unary_factors个x
    binary_factors = 40 * 10  # 400个双因子，意味着有400组变量相连
    edges = 2 * binary_factors + unary_factors
    factors = unary_factors + binary_factors

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
    unary_edge = namedtuple('unary_edge', ['index', 'factorId', 'varId'])  # 单变量因子的边
    binary_edge = namedtuple('binary_edge', ['index', 'factorId', 'varId1', 'varId2'])  # 双变量因子的边
    var_list = list(range(0, 1000))
    random.shuffle(var_list)  # 打乱变量顺序
    unary_edges = list()
    binary_edges = list()
    # 1. 随机初始化单因子的边存成list,元素类型为unary_edge
    unary_var = list()  # 连有单因子的变量ID
    for i in range(unary_factors):  # 500条边
        varId = var_list[i]
        unary_var.append(varId)
        edge = unary_edge(i, i, varId)
        unary_edges.append(edge)
    # 2. 随机初始化双因子的边存成list,元素类型为binary_edge
    no_unary_var = [var for var in var_list if var not in unary_var]  # 没有连接单因子的变量ID
    first_count = int((variables - unary_factors) / 2)
    for i in range(binary_factors):  # 400个双因子，即
        # 先从没连单因子的变量ID（还剩下variables-unary_factors=500个，250对）中一对对挑，，再从没有的和有的中各取一个变量组成一对，还需要挑150对。
        if i < first_count:  # 先从no_unary_var挑250对
            varId1 = no_unary_var[2 * i]
            varId2 = no_unary_var[2 * i + 1]
        else:  # 再从unary_var和no_unary_var挑150对
            varId1 = no_unary_var[i - first_count]
            varId2 = unary_var[i - first_count]
        edge = binary_edge(i + unary_factors, i + unary_factors, varId1, varId2)
        binary_edges.append(edge)

    # 三·初始化权重
    for i in range(weights):
        weight[i]["isFixed"] = True  # True表示是真实值，后期不用再学习
        #weight[i]["parameterize"] = False
        weight[i]["initialValue"] = random.uniform(-10, 10)   

    # 四·初始化因子：单因子因子函数为18，双因子为9;900个因子
    # 此处的索引需要细看
    factor_index = 0
    fmp_index = 0
    for i in range(unary_factors):  # 前500个为单因子（10种）
        factor[factor_index]["factorFunction"] = 18
        factor[factor_index]["weightId"] = i % 10  # 随机分配10种权重到500个单因子
        factor[factor_index]["featureValue"] = 1
        factor[factor_index]["arity"] = 1  # 单因子度为1
        factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
        fmap[fmp_index]["vid"] = unary_edges[factor_index][2]
        #fmap[fmp_index]["x"] = random.random()  # 单因子需要存储x
        fmp_index += 1
        factor_index += 1

    for i in range(binary_factors):  # 后400个为双因子(40种)
        factor[factor_index]["factorFunction"] = 9
        factor[factor_index]["weightId"] = i % 40 + 10  # 随机分配40种权重到400个双因子
        factor[factor_index]["featureValue"] = 1
        factor[factor_index]["arity"] = 2  # 双因子度为2
        factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次要加2
        fmap[fmp_index]["vid"] = binary_edges[factor_index - unary_factors][2]  # 双因子需要填两个vid和两个x
        #fmap[fmp_index]["x"] = 0  # 双因子因为不需要函数化，所以将x赋值为0，反正也用不上
        fmp_index += 1
        fmap[fmp_index]["vid"] = binary_edges[factor_index - unary_factors][3]
        #fmap[fmp_index]["x"] = 0
        fmp_index += 1
        factor_index += 1
    # 将定义好的因子图保存为factorGraph.pkl文件，以备后期验证使用
    #factorGraph = weight, variable, factor, fmap, domain_mask, edges
    # with open('factorGraph_original.pkl', 'wb') as f:    #原始的图
    #     pickle.dump(factorGraph, f)
    return weight, variable, factor, fmap, domain_mask, edges

#1.已知所有参数，推理所有变量
learn = 1000
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

ns.inference()
print("First Inference Finished!")
for i in range(1000):
    if ns.factorGraphs[0].marginals[i] > 0.5:   #推理出来的变量是概率值,保存在factorGraphs[0].marginals，现在写回到fg
        fg[1][i]['initialValue'] = 1
    else:
        fg[1][i]['initialValue'] = 0
# 存储第一遍推理后的因子图
with open('factorGraph_first_inference.pkl', 'wb') as f:    #由参数推理变量后的图
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
# weight, variable, factor, fmap, domain_mask, edges
#加载第一遍推理后的因子图
with open('factorGraph_first_inference.pkl', 'rb') as f:
    fg1 = pickle.load(f)
TrueWeightValue = list()  #真实权重
FirstInferenceVarValue = list()
for i in range(len(fg1[0])):  # 权重设为待推,并且全部重新赋初值
    fg1[0][i]['isFixed'] = False
    TrueWeightValue.append(fg1[0][i]["initialValue"])
    # print('TrueWeightValue',TrueWeightValue[i])
    fg1[0][i]["initialValue"] = random.uniform(-10,10)
for i in range(len(fg1[1])):  # 将全部变量作为证据节点
    fg1[1][i]['isEvidence'] = 1
    FirstInferenceVarValue.append(fg1[1][i]['initialValue'])

ns1.loadFactorGraph(*fg1)
ns1.learning()
print("Weights Learned Finished!")
#print("isFixed",ns.factorGraphs[0].weight[0]['isFixed'],fg1[0][0]['isFixed'])   #output:isFixed True False   表明学习完之后并不会修改isFixed标志位
for i in range(len(fg1[0])):
    # print("weight", i, "True:",TrueWeightValue[i],"----------","Learned:", ns1.factorGraphs[0].weight_value[0][i])
    print("weight", i, "True:",TrueWeightValue[i],"----------","Learned:", ns1.factorGraphs[0].weight[i]['initialValue'])   #上下两行输出一样，证明可以将weight写回
    # print("weight", i, "True:",TrueWeightValue[i],"----------","Learned:", fg1[0][i]['initialValue'])                       #输出同上两行
with open('factorGraph_Learned.pkl', 'wb') as f:    
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
with open('factorGraph_Learned.pkl', 'rb') as f:
    fg2 = pickle.load(f)
for i in range(len(fg2[0])):  # 权重设为已知
    fg2[0][i]['isFixed'] = True
for i in range(len(fg2[1])):  # 变量全部设为未知，并且重新赋值
    fg2[1][i]["isEvidence"] = 0
    fg2[1][i]["initialValue"] = random.choice((1, 0))
ns2.loadFactorGraph(*fg2)
ns2.inference()
print("Second Inference Finished!")
NewInferencedVarValue = list()
for i in range(len(fg2[1])):
    if ns2.factorGraphs[0].marginals[i] > 0.5:   #推理出来的变量是概率值,保存在factorGraphs[0].marginals，现在写回到fg1
        fg2[1][i]['initialValue'] = 1
    else:
        fg2[1][i]['initialValue'] = 0
    NewInferencedVarValue.append(fg2[1][i]['initialValue'])
# 计算隐变量推理准确率
hit = 0
for i in range(len(fg2[1])):
    if FirstInferenceVarValue[i] == NewInferencedVarValue[i]:
        hit += 1
print("Var Inferenced Accuracy:", hit / len(fg2[1]))
with open('factorGraph_SecondInference.pkl', 'wb') as f:
    pickle.dump(fg2, f)


