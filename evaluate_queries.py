from search_engine_evaluate import *
import matplotlib.pyplot as plt
import math

rankings = [
    {"engine": "vector", "query": "engine", "retrievals":
        ["sbr18995.txt",
         "inf17195.txt",
         "str03295.txt",
         "sbr17695.txt",
         "emt21795.txt",
         "emt14395.txt",
         "ins18795.txt",
         "emt15895.txt",
         "emt13495.txt",
         "emt11895.txt", ]
        , "relevant":
         {"emt01995.txt",
          "emt13295.txt",
          "sbr18995.txt",
          "sbr18995.txt",
          "sbr21495.txt",
          "str03295.txt",
          "str03295.txt",
          "str03295.txt",
          "str03295.txt"}},
    {"engine": "vector", "query": "analysis", "retrievals":
        ["emt01995.txt",
         "sbr15695.txt",
         "emt11895.txt",
         "sbr17995.txt",
         "emt04895.txt",
         "emt04795.txt",
         "emt05995.txt",
         "emt04395.txt",
         "emt02495.txt",
         "emt04995.txt", ]
        , "relevant":
         {"emt01995.txt",
          "emt01995.txt",
          "emt01995.txt",
          "emt01995.txt",
          "emt04395.txt",
          "emt04395.txt",
          "emt04395.txt",
          "emt04895.txt",
          "emt04895.txt",
          "emt04995.txt",
          "emt05095.txt",
          "emt05995.txt",
          "emt05995.txt",
          "emt10395.txt",
          "emt10695.txt",
          "emt13495.txt",
          "emt15895.txt",
          "emt15895.txt",
          "eos16995.txt",
          "inf19695.txt",
          "inf21695.txt",
          "mip14195.txt",
          "sbr15695.txt",
          "sbr17995.txt",
          "sbr18995.txt",
          "str10095.txt"}},
    {"engine": "lsi", "query": "engine", "retrievals":
        ["inf17195.txt",
         "sbr18995.txt",
         "str03295.txt",
         "sbr17895.txt",
         "emt14395.txt",
         "emt21795.txt",
         "emt01995.txt",
         "eos00395.txt",
         "ins15795.txt",
         "emt15895.txt"]
        , "relevant":
         {
             "emt01995.txt",
             "emt13295.txt",
             "sbr18995.txt",
             "sbr18995.txt",
             "sbr21495.txt",
             "str03295.txt",
             "str03295.txt",
             "str03295.txt",
             "str03295.txt", }
     },
    {"engine": "lsi", "query": "analysis", "retrievals":
        ["emt01995.txt",
         "sbr15695.txt",
         "sbr17995.txt",
         "emt11895.txt",
         "emt10695.txt",
         "emt13495.txt",
         "emt04595.txt",
         "eos11695.txt",
         "emt14395.txt",
         "emt02495.txt", ]
        , "relevant":
         {
             "emt01995.txt",
             "emt01995.txt",
             "emt01995.txt",
             "emt01995.txt",
             "emt04395.txt",
             "emt04395.txt",
             "emt04395.txt",
             "emt04895.txt",
             "emt04895.txt",
             "emt04995.txt",
             "emt05095.txt",
             "emt05995.txt",
             "emt05995.txt",
             "emt10395.txt",
             "emt10695.txt",
             "emt13495.txt",
             "emt15895.txt",
             "emt15895.txt",
             "eos16995.txt",
             "inf19695.txt",
             "inf21695.txt",
             "mip14195.txt",
             "sbr15695.txt",
             "sbr17995.txt",
             "sbr18995.txt",
             "str10095.txt",
         }
     },
]

# plot p-r curves

num_subplots = len(rankings)
fig, axs = plt.subplots(2, 2, figsize=(15, 8))

average_precisions = {}
for i, ranking in enumerate(rankings[:4]):
    ax = axs[i // 2, i % 2]
    precisions = [ranked_list_precision(ranking["retrievals"][:i + 1], ranking["relevant"]) for i in
                  range(len(ranking["retrievals"]))]

    recalls = [ranked_list_recall(ranking["retrievals"][:i + 1], ranking["relevant"]) for i in
               range(len(ranking["retrievals"]))]

    ax.plot(range(1, 11), precisions, marker='o', label=f"{ranking['engine']} - Precision")
    ax.plot(range(1, 11), recalls, marker='o', label=f"{ranking['engine']} - Recall")
    ax.set_xlabel("k")
    ax.set_ylabel("Precision and Recall at k")
    ax.legend()
    ax.grid()
    ax.set_title(f"Precision and Recall at k for {ranking['engine']} - query : {ranking['query']}")
    average_precisions.setdefault(ranking['engine'], [])
    average_precisions[ranking['engine']].append(average_precision(ranking["retrievals"], ranking["relevant"]))
    print(
        f"Engine: {ranking['engine']}, Query: {ranking['query']}, nDCG: {nDCG(ranking['retrievals'], {item: 1 for item in ranking['relevant']})}, microF1: {micro_f1(ranking['retrievals'], ranking['relevant'])}, macroF1: {macro_f1(ranking['retrievals'], ranking['relevant'])}, spearman_r: {spearman_correlation(ranking['retrievals'], ranking['relevant'])}, kendall_tau: {kendall_tau(ranking['retrievals'], ranking['relevant'])}")


for engine, aps in average_precisions.items():
    print(f"{engine} mAP: {sum(aps) / len(aps)}")

plt.tight_layout()
plt.show()
