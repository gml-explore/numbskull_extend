from create_factorgraph import create_fg
from numbskull import NumbSkull
weight, variable, factor, fmap, domain_mask, edges = create_fg()

ns_learing = NumbSkull(
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
subgraph = weight, variable, factor, fmap, domain_mask, edges
ns_learing.loadFactorGraph(*subgraph)
# 因子图参数学习
ns_learing.learning()
# 因子图推理
# 参数学习完成后将weight的isfixed属性置为true
for index,w in enumerate(weight):
    w["isFixed"] = True
    w['initialValue'] = ns_learing.factorGraphs[0].weight[index]['initialValue']
ns_inference = NumbSkull(
    n_inference_epoch=1000,
    n_learning_epoch=1000,
    stepsize=0.001,
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
ns_inference.loadFactorGraph(*subgraph)
# 因子图推理
ns_inference.inference()
#获取变量推理结果
print(ns_inference.factorGraphs[0].marginals)
