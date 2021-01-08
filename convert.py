from ckiptagger import WS
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ws = WS("./ckipdata", disable_cuda=False)  # , disable_cuda=not GPU)
sentences = [
    "但美国并没有因此而遭遇超越国界的系统性危机",
    "没有大规模基础设施投资，特别是交通投资，就不可能产生让美国跻身世界工业强国的生产率的提升。"
]
word_sentence_list = ws(sentences)
print(word_sentence_list)
