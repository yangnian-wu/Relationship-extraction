def labelch_to_labelen():
    '''
    将英文的标签转换成中文，影响了训练结果（未找到原因）
    :return:
    '''
    f = open(r'./data/test.txt', 'r', encoding='utf-8')
    h=open('./data/test1.txt', 'w', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        # get entity name
        en1 = content[0]
        en2 = content[1]
        if content[2]=='company/company/cooperate':
            relation = '合作'
        elif content[2] == 'company/company/invest':
            relation = '投资'
        elif content[2]=='company/company/compete':
            relation = '竞争'
        elif content[2]=='company/company/acquisition':
            relation = '收购'
        elif content[2]=='company/company/chairman':
            relation = '董事'
        content_end=content[0]+' '+content[1]+' '+relation+' '+content[3]+'\n'
        h.write(content_end)
def daluantxt():
    '''
    打乱训练集和测试集的顺序
    f:输入文件
    h:输出文件
    :return:
    '''
    import random
    f=open('all-datav4.txt','r',encoding='utf-8')
    h=open('all-datav6.txt','w',encoding='utf-8')
    lines=[]
    readlines=f.readlines()
    for line in readlines:
        lines.append(line)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    for line in lines:
        h.write(line)
    f.close()
    h.close()
def splitdata():
    '''
    划分数据集
    :return:
    '''
    f=open('all-datav6.txt','r',encoding='utf-8')
    h=open('trainv2.txt','a',encoding='utf-8')
    k=open('testv2.txt','a',encoding='utf-8')
    readlines=f.readlines()
    count=len(readlines)
    f.seek(0)
    for i in range(count):
        line=f.readline()
        if i%5==0:
            k.write(line)
        else:
            h.write(line)
    f.close()
    h.close()
    k.close()

def staticsnum():
    f=open('all-datav6.txt','r',encoding='utf-8')
    readlines=f.readlines()
    rel_list=[]
    value={}
    j=0
    for line in readlines:
        j=j+1
        content=line.strip().split('\t')
        rel_list.append(content[2])
    for i in rel_list:
        if i in value.keys():
            value[i]=value[i]+1
        else:
            value[i]=1
    print(value)
def banjiandu():
    f=open('F:\PycharmProjects\三元组获取\金融界爬虫\预处理后.txt','r',encoding='utf-8')
    h=open('半监督产生数据.txt','a',encoding='utf-8')
    keyword=[['大连重工', '华塑股份']]
    readlines=f.readlines()
    for j in keyword:
        for line in readlines:
            if j[1] in line and j[0] in line:
                h.write(j[0]+'\t'+j[1]+'\t'+'unknown'+'\t'+line)
    f.close()
    h.close()
def quchong():
    '''
    去除txt中的重复语句
    :return:
    '''
    f = open('半监督产生数据', 'r', encoding='utf-8')
    h=open('all-datav4.txt','a',encoding='utf-8')
    lines=set()
    for line in f:
        line=line.strip('')
        if line not in lines:
            h.write(line)
            lines.add(line)
    f.close()
    h.close()
if __name__ == '__main__':
    daluantxt()
    splitdata()
    staticsnum()
      # banjiandu()
    # quchong()




