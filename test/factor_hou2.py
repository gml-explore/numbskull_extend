#!/usr/bin/env python
import copy
import random

import numbskull
from numbskull.numbskulltypes import *

'''Author : 徐驳'''
#设置了部分隐变量

# 创建因子图
def create_fg(unobserved_ratio):
    file = open('hou_data.txt')
    read = file.readline()
    line = eval(read)
    n = len(line)

    ratio = 1  # 0个数 : 1个数
    count_1 = 0
    count_0 = 0
    lines = list()

    for i in range(n):
        if line[i][1] == 1:
            count_1 += 1
    count_0 = ratio * count_1

    count0 = 0
    count1 = 0
    for i in range(n):
        if (count1 < count_1) & (line[i][1] == 1):
            lines.append(line[i])
            count1 += 1
        elif (count0 < count_0) & (line[i][1] == 0):
            lines.append(line[i])
            count0 += 1
    initial_b = 0
    for var in lines:
        initial_b += var[0]
    initial_b = initial_b/len(lines)
    random.shuffle(lines)
    #print(lines)
    print("变量总数：", len(lines))
    print("0变量的个数",count0)
    print("1变量的个数",count1)
    n = len(lines)
    weights = 1
    variables = n
    edges = n
    factors = n

    weight = np.zeros(weights, Weight)
    variable = np.zeros(variables, Variable)
    factor = np.zeros(factors, Factor)
    fmap = np.zeros(edges, FactorToVar)
    domain_mask = np.zeros(variables, np.bool)

    # 初始化变量
    for i in range(variables):
        if i < int(variables*unobserved_ratio):   #10%作为隐变量
            variable[i]["isEvidence"] = 0
        else:
            variable[i]["isEvidence"] = 1
        variable[i]["initialValue"] = lines[i][1]
        variable[i]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
        variable[i]["cardinality"] = 2

    # 初始化权重
    weight[0]["isFixed"] = False
    weight[0]["parameterize"] = True
    weight[0]["a"] = 0
    weight[0]["b"] = 0
    weight[0]["initialValue"] = 1

    # 初始化因子 第i个因子连在第i个变量上，所有因子共享0号权重
    for i in range(factors):
        factor[i]["factorFunction"] = 18
        factor[i]["weightId"] = 0  # 随机分配10种权重到500个单因子
        factor[i]["featureValue"] = 1
        factor[i]["arity"] = 1  # 单因子度为1
        factor[i]["ftv_offset"] = i  # 偏移量每次加1
        fmap[i]["vid"] = i
        fmap[i]["x"] = lines[i][0]  # 单因子需要存储x

    return weight, variable, factor, fmap, domain_mask, edges


unobserved_ratio = 0.2
learn = 10000
ns = numbskull.NumbSkull(n_inference_epoch=100,
                         n_learning_epoch=learn,
                         quiet=True,
                         learn_non_evidence=False,
                         stepsize=0.0001,
                         burn_in=100,
                         decay=0.001 ** (1.0 / learn),
                         regularization=1,
                         reg_param=0.01)
fg = create_fg(unobserved_ratio)
fg1 = copy.deepcopy(fg)

# 学习得到参数
ns.loadFactorGraph(*fg)
ns.learning()
print("Weights Learned Finished!")
print("Initial_weight:", fg1[0][0]['initialValue'], \
      "learned_weight:", ns.factorGraphs[0].weight[0]['initialValue'])
print("Initial_a:", fg1[0][0]['a'], "learned_a:", ns.factorGraphs[0].weight[0]['a'])
print("Initial_b:", fg1[0][0]['b'], "learned_b:", ns.factorGraphs[0].weight[0]['b'])

# 通过学习到的参数进行推理
fg1[0][0]["isFixed"] = True
fg1[0][0]["a"] = ns.factorGraphs[0].weight[0]['a']
fg1[0][0]["b"] = ns.factorGraphs[0].weight[0]['b']
fg1[0][0]["initialValue"] = ns.factorGraphs[0].weight[0]['initialValue']
for i in range(len(fg1[1])):
    fg1[1][i]["initialValue"] = random.choice((1, 0))
    fg1[1][i]["isEvidence"] = 0

ns1 = numbskull.NumbSkull(n_inference_epoch=100,
                          n_learning_epoch=learn,
                          quiet=True,
                          learn_non_evidence=True,
                          stepsize=0.0001,
                          burn_in=100,
                          decay=0.001 ** (1.0 / learn),
                          regularization=1,
                          reg_param=0.01)
ns1.loadFactorGraph(*fg1)
ns1.inference()
print("Inference Finished!")
for i in range(len(fg1[1])):
    if ns1.factorGraphs[0].marginals[i] > 0.5:
        fg1[1][i]['initialValue'] = 1
    else:
        fg1[1][i]['initialValue'] = 0

# 结果：
unob_hit = 0
ob_hit = 0

for i in range(len(fg1[1])):
    if i < int(len(fg1[1]) * unobserved_ratio):   #统计10%隐变量准确率
        if fg[1][i]["initialValue"] == fg1[1][i]["initialValue"]:
            unob_hit += 1
    else:
        if fg[1][i]["initialValue"] == fg1[1][i]["initialValue"]:
            ob_hit += 1


print(".........................................")
print("隐变量比例：",unobserved_ratio)
print("隐变量个数：",int(len(fg1[1]) * unobserved_ratio))
print("unobserved precision:", unob_hit / int(len(fg1[1]) *unobserved_ratio))
print("显变量个数：",len(fg1[1])- int(len(fg1[1]) * unobserved_ratio))
print("observed precision:",ob_hit /(len(fg1[1])- int(len(fg1[1]) * unobserved_ratio)))

