import random
from collections import namedtuple

from numbskull import *
from numbskulltypes import *


# 创建因子图
def create_fg():
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
    return weight, variable, factor, fmap, domain_mask, edges

