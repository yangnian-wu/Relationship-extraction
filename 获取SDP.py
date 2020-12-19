import os
import jieba
from pyltp import Postagger, Parser
LTP_DATA_DIR = 'F:\PycharmProjects\ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
postagger = Postagger()
postagger.load(pos_model_path)
parser = Parser()
parser.load(par_model_path)
f = open('./data/train1.txt', 'r', encoding='utf-8')
f1=open('./data/train-s.txt','w',encoding='utf-8')
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().strip('\n').split()
    # get entity name
    en1 = content[0]
    en2 = content[1]
    sentence=content[3]
    jieba.add_word(en1)
    jieba.add_word(en2)
    words = list(jieba.cut(sentence))
    print(words)
    # 词性标注
    postags = postagger.postag(words)
    # 依存句法分析
    arcs = parser.parse(words, postags)
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    import networkx as nx
    import matplotlib.pyplot as plt
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
    G = nx.Graph()  # 建立无向图G
    # 添加节点
    for word in words:
        G.add_node(word)
    G.add_node('Root')
    # 添加边
    for i in range(len(words)):
        G.add_edge(words[i], heads[i])
    source = en1
    target1 = en2
    if source in words and target1 in words:
        distance1 = nx.shortest_path(G, source=source, target=target1)
        print(''.join(distance1))
        f1.write(content[0]+' '+content[1]+' '+content[2]+' '+''.join(distance1))
        f1.write('\n')
    else:
        f1.write(content[0]+' '+content[1]+' '+content[2]+' '+sentence)
        f1.write('\n')
f.close()
f1.close()
