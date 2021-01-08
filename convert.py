from ckiptagger import WS
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ws = WS("./ckipdata", disable_cuda=False)  # , disable_cuda=not GPU)
# sentences = [
#     "但美国并没有因此而遭遇超越国界的系统性危机",
#     "没有大规模基础设施投资，特别是交通投资，就不可能产生让美国跻身世界工业强国的生产率的提升。"
# ]
with open("data/train.tsv") as fp:
    lines = fp.readlines()

lefts, rights = [], []
for i, line in enumerate(lines):
    ss = line.split("\t")
    lefts.append(ss[0])
    rights.append(ss[1])

right_segments = ws(rights)

out_lines = []
for i, left in enumerate(lefts):
    ww = " ".join(right_segments[i]).strip()
    out_lines.append("%s\t%s\n" % (left, ww))

with open("data/train-seg.tsv", "w") as fp:
    fp.writelines(out_lines)

