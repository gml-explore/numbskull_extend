import copy
import pickle
import random

import numpy as np

from numbskull import NumbSkull
from numbskull.numbskulltypes import *

'''情感分析真实数据测试'''
def fg_compile(dict_reltype2wid, parse_fg, rng):
    print('----------------------------------------------------------------')
    print('Compile the parse graph')
    print('----------------------------------------------------------------')

    aspect_nodes, feature_nodes, asp2asp_relations, fea2asp_relations, fea2fea_relations = parse_fg

    # variables and relations
    n_variables = len(aspect_nodes)
    n_binary_factors = len(asp2asp_relations)
    n_unary_factors = len(fea2asp_relations)
    n_factors = n_binary_factors + n_unary_factors
    n_edges = 2 * n_binary_factors + n_unary_factors

    # initilize
    variable = np.zeros(n_variables, Variable)
    factor = np.zeros(n_factors, Factor)
    ftv = np.zeros(n_edges, FactorToVar)
    domain_mask = np.zeros(n_variables, np.bool)

    ############################################################################################
    ## compile variable matrix
    ############################################################################################
    dict_asp2index = {}
    dict_index2asp = {}
    # aspect polarities(variables)
    var_index = 0
    for asp_node in aspect_nodes:
        name = asp_node.name
        polarity = asp_node.polarity
        variable[var_index]['isEvidence'] = asp_node.isEvidence

        if polarity == 'positive':
            variable[var_index]['initialValue'] = 1
        elif polarity == 'negative':
            variable[var_index]['initialValue'] = 0
        else:
            variable[var_index]['initialValue'] = rng.randint(2)
        variable[var_index]['dataType'] = 0
        variable[var_index]['cardinality'] = 2
        dict_asp2index[name] = var_index
        dict_index2asp[var_index] = name
        var_index += 1

    ############################################################################################
    ## compile factor and ftv matrix
    ############################################################################################
    factor_index = 0
    ftv_index = 0
    tmp = []
    # binary factors between aspect polarities
    for asp2asp in asp2asp_relations:
        asp1 = asp2asp.name1
        asp2 = asp2asp.name2
        type = asp2asp.rel_type
        factor[factor_index]['factorFunction'] = 9
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('dict_reltype2wid: ', dict_reltype2wid)
        # print('the number of dict_reltype2wid: ', len(dict_reltype2wid))
        factor[factor_index]['weightId'] = dict_reltype2wid[type]
        factor[factor_index]['featureValue'] = 1
        factor[factor_index]['arity'] = 2
        factor[factor_index]['ftv_offset'] = ftv_index
        ftv[factor_index]['x'] = 1

        factor_index += 1
        variables = [asp1, asp2]
        for vtid in variables:
            ftv[ftv_index]['vid'] = dict_asp2index[vtid]
            ftv_index += 1
        tmp.append((type, dict_reltype2wid[type]))

    # unary factors between aspect polarities and features
    for fea2asp in fea2asp_relations:
        fea_name = fea2asp.name1
        asp_name = fea2asp.name2
        type = fea2asp.rel_type
        factor[factor_index]['factorFunction'] = 18
        factor[factor_index]['weightId'] = dict_reltype2wid[str((fea_name, type))]
        factor[factor_index]['featureValue'] = 1
        factor[factor_index]['arity'] = 1
        factor[factor_index]['ftv_offset'] = ftv_index
        factor_index += 1

        ftv[ftv_index]['vid'] = dict_asp2index[asp_name]
        ftv[factor_index]['x'] = 1
        ftv_index += 1
        tmp.append((fea_name, dict_reltype2wid[str((fea_name, type))]))
    # print('the involved weight:', set(tmp))
    print('the number of involved weight: ', len(set(tmp)))
    return variable, factor, ftv, domain_mask, n_edges, dict_asp2index


def fg_learning(compile_fg):
    print('----------------------------------------------------------------')
    print('Weight learning')
    print('----------------------------------------------------------------')
    weight, variable, factor, ftv, domain_mask, n_edges = compile_fg
    fg = NumbSkull(
        n_inference_epoch=1000,
        n_learning_epoch=1000,
        stepsize=0.01,
        decay=0.95,
        reg_param=1e-6,
        regularization=2,
        truncation=10,
        quiet=(not False),
        verbose=False,
        learn_non_evidence=False,  # need to test
        sample_evidence=False,
        burn_in=10,
        nthreads=1
    )
    fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)
    fg.learning(out=True)
    weight_value = fg.factorGraphs[0].weight_value[0]
    return weight_value

def fg_inference(compile_fg):
    print('----------------------------------------------------------------')
    print('Weight inference')
    print('----------------------------------------------------------------')
    weight, variable, factor, ftv, domain_mask, n_edges = compile_fg
    fg = NumbSkull(
        n_inference_epoch=1000,
        n_learning_epoch=1000,
        stepsize=0.01,
        decay=0.95,
        reg_param=1e-6,
        regularization=2,
        truncation=10,
        quiet=(not False),
        verbose=False,
        learn_non_evidence=False,  # need to test
        sample_evidence=False,
        burn_in=10,
        nthreads=1
    )
    fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)
    fg.inference(out=True)
    for i in range(len(variable)):
        if fg.factorGraphs[0].marginals[i] > 0.5:
            variable[i]['initialValue'] = 1
        else:
            variable[i]['initialValue'] = 0

    weight_value = fg.factorGraphs[0].weight_value[0]
    return weight_value

if __name__ == "__main__":
    [oldweight, dict_reltype2wid, parse_fg, rng] = pickle.load(open("factor_graph.pkl", "rb"))
    variable, factor, ftv, domain_mask, n_edges, dict_asp2index = fg_compile(dict_reltype2wid, parse_fg, rng)
    weight = np.zeros(len(oldweight), Weight)
    for i in range(len(oldweight)):
        weight[i]["isFixed"] = oldweight[i]["isFixed"]
        weight[i]["initialValue"] = oldweight[i]["initialValue"]
        weight[i]["parameterize"] = 0
        weight[i]["a"] = 0
        weight[i]["b"] = 0


    compile_fg = [weight, variable, factor, ftv, domain_mask, n_edges]
    weight_value= fg_learning(compile_fg)

    print("before", compile_fg[0][0]["isFixed"])
    var0 = copy.deepcopy(variable)
    for var in variable:
        # print(i)
        var["isEvidence"] = 0
        var["initialValue"] =random.choice((1, 0))

    for i in weight:
        i["isFixed"] = 1

    print("after", compile_fg[0][0]["isFixed"])
    fg_inference(compile_fg)
    var1 = copy.deepcopy(variable)
    # print(var1)


    UnobservedVarNum = 0
    ObservedVarNum = 0
    AllVarNum = len(var0)
    UnobservedVarHit = 0
    ObservedVarHit = 0
    AllVarHit = 0
    for i in range(len(var1)):
        if var0[i]["initialValue"] == var1[i]["initialValue"]:
            AllVarHit += 1
            if var0[i]["isEvidence"]  == 0:
                UnobservedVarHit += 1
            else:
                ObservedVarHit += 1

    for var in var0:
        if var["isEvidence"] == 0:
            UnobservedVarNum += 1
        else:
            ObservedVarNum += 1


    print("AllVarNum", AllVarNum)
    print("UnobservedVarNum", UnobservedVarNum)
    print("ObservedVarNum", ObservedVarNum)
    # print("num_correct", num_correct)
    # print("num_correct0", num_correct0)
    # print("num_correct1", num_correct1)
    print("Total precision:", AllVarHit / AllVarNum)
    print("UnobservedVar Inference precision:", UnobservedVarHit / UnobservedVarNum)
    print("ObservedVar Inference precision:", ObservedVarHit / ObservedVarNum)
    # print(weight_value)

