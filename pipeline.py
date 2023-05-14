#%%

#%%
import requests
from datetime import datetime
from sklearn.cluster import KMeans

import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

import chardet
import re
import pandas as pd
import numpy as np
from pprint import pprint
import requests
from bixin import predict
import snownlp
import dm_pb2 as Danmaku
import matplotlib.pyplot as plt
import sklearn
#%%
# 1.根据bvid请求得到cid
bvid = "BV1bc411H7Gv"
def get_cid(bvid):
    url = 'https://api.bilibili.com/x/player/pagelist?bvid=' + bvid
    res = requests.get(url).text
    json_dict = json.loads(res)
    pprint(json_dict)
    return (json_dict["data"][0]["duration"], json_dict["data"][0]["cid"], json_dict["data"][0]["part"])
(max_time, cid, part) = get_cid(bvid)
max_time = max_time * 1000

print(cid)


#%%
# 更新：新版api需要使用proto获取弹幕文件
def get_prot_dm(cid):
    url = 'https://api.bilibili.com/x/v2/dm/web/seg.so'
    params = {
        'type': 1,         # 弹幕类型
        'oid': cid,    # cid
        'segment_index': 1 # 弹幕分段
    }
    resp = requests.get(url, params)
    data = resp.content

    danmaku_seg = Danmaku.DmSegMobileReply()
    danmaku_seg.ParseFromString(data)

    return danmaku_seg.elems

danmu_proto = get_prot_dm(cid)
#%%
#得到弹幕 array
def to_list(prot):
    l = []
    for i in range(len(prot)):
        l.append({})
        l[i]["id"] = prot[i].id
        l[i]["progress"] = prot[i].progress
        l[i]["content"] = prot[i].content

    return l
listed_danmu = to_list(danmu_proto)
df = pd.DataFrame(listed_danmu)
mask = df['content'].str.len() > 10
df = df[~mask]
df['content'] = df['content'].str.replace('[^\w\s]', '')
mask = df['content'].str.len() < 1
df = df[~mask]
df.dropna(subset=['content'], inplace=True)
mask = df['content'].str.len() == 0
df = df[~mask]
print(df)
#%%
#divide into 6 parts according to time variation
def classify_sentiment(s):
    if(s<0.166):
        return 0
    if(s<0.333):
        return 1
    if(s<0.5):
        return 2
    if(s<0.666):
        return 3
    if(s<0.833):
        return 4
    if(s<=1):
        return 5
def get_sentiment(text):
    s = snownlp.SnowNLP(text)
    return classify_sentiment(s.sentiments)
index = 0
leftop = pd.DataFrame()

for i in range(6):
    start_time = max_time * i / 6
    end_time = max_time * (i+1) / 6
    tmp = df[(df['progress'] >= start_time) & (df['progress'] <= end_time)]
    tmp = tmp['content'].value_counts()
    tmp = pd.DataFrame({'content': tmp.index, 'count': tmp.values})
    mask = tmp['count'].values <= 3
    tmp = tmp[~mask]
    tmp['emo'] = tmp['content'].apply(get_sentiment)
    tmp['label'] = index
    print(tmp)
    leftop = pd.concat([leftop, tmp], axis=0)

    index+=1

leftop.to_csv("video.csv",index=False)
counts = df['content'].value_counts()
leftop
emo = leftop['emo'].value_counts()
emo
emo_df = pd.DataFrame(emo, index = emo.index)
emo_df['label'] = emo.index

emo_df.to_csv("emo.csv", index=False)
emo_df
#%%
# 2.根据cid请求弹幕，解析弹幕得到最终的数据

# def get_data(cid):
#     final_url = "https://api.bilibili.com/x/v1/dm/list.so?oid=" + str(cid)
#     final_res = requests.get(final_url)
#     final_res.encoding = chardet.detect(final_res.content)['encoding']
#     final_res = final_res.text
#     pattern = re.compile('<d.*?>(.*?)</d>')
#     match = re.compile(r'<d\s+[^>]*\bp="([^"]*)"')
#     pa = match.findall(final_res)
#
#     data = pattern.findall(final_res)
#     danmu_time = [float(item.split(',')[0]) for item in pa]
#     danmu_mode = [float(item.split(',')[1]) for item in pa]
#     danmu_size = [float(item.split(',')[2]) for item in pa]
#     danmu_color = [float(item.split(',')[3]) for item in pa]
#     danmu_abstime = [float(item.split(',')[4]) for item in pa]
#     danmu_pool = [float(item.split(',')[5]) for item in pa]
#     danmu_id = [(item.split(',')[6]) for item in pa]
#     danmu_rowid = [float(item.split(',')[7]) for item in pa]
#     print(len(data))
#     return (data, danmu_time, danmu_mode, danmu_size, danmu_color, danmu_abstime, danmu_pool, danmu_id, danmu_rowid)
#
#
# (danmu_text, danmu_time, danmu_mode, danmu_size, danmu_color, danmu_abstime, danmu_pool, danmu_id, danmu_rowid) = get_data(cid)
#%%
# # 分割弹幕为八个部分
# # 视频持续时间
# max_time = max(danmu_time)
# divided_danmu = []
# for i in range(8):
#     time_start = max_time * (i/8)
#     time_stop = max_time * ((i+1)/8)
#     for j in range(len(danmu_time)):
#         if time_start <= danmu_time[j] <= time_stop:
#             divided_danmu.append([i, danmu_text[j]])
# print(len(danmu_time))
#%%
# export csv
# arr = np.array(divided_danmu)
# df = pd.DataFrame(arr)
# df.to_csv("video.csv", header=["label", "text"])
#%%
def get_relate(bvid):
    # 设置请求参数
    params = {
        "bvid": bvid,  # 你想获取推荐视频的原始视频的bid
        "num": 1,  # 获取的推荐视频数量，最多为40
    }

    # 发送请求
    response = requests.get("https://api.bilibili.com/x/web-interface/archive/related", params=params)

    # 解析响应
    if response.status_code == 200:
        data = response.json()
        recommend_list = data.get("data")
        # 处理推荐视频列表数据
        return recommend_list
    else:
        print("请求失败")

relate_list = get_relate(bvid)
print(relate_list[0])
#%%
#迭代获取其他视频的推荐
index = 0
whole_related_df = pd.DataFrame()
for it in relate_list:
    danmu_relate = get_prot_dm(it['cid'])
    #preprocess
    listed_danmu = to_list(danmu_relate)
    df = pd.DataFrame(listed_danmu)
    mask = df['content'].str.len() > 10
    df = df[~mask]
    df['content'] = df['content'].str.replace('[^\w\s]', '')
    df['content'] = df['content'].str.replace('哈', '')
    df['content'] = df['content'].str.replace('嘿', '')
    df['content'] = df['content'].str.replace('啊', '')
    df['content'] = df['content'].str.replace('6', '')
    mask = df['content'].str.len() <= 1
    df = df[~mask]
    df.dropna(subset=['content'], inplace=True)
    #end of preprocess
    relate_tmp = df['content'].value_counts()
    relate_tmp = pd.DataFrame({'content': relate_tmp.index, 'count': relate_tmp.values})
    relate_tmp['emo'] = relate_tmp['content'].apply(get_sentiment)
    relate_tmp.to_csv("relate_video_"+str(index)+".csv")
    whole_related_df = pd.concat([df, whole_related_df], axis=0)

    index+=1

whole_related_df['id'] /= 100000000
whole_related_df
#%%
# whole_count = whole_related_df['content'].value_counts()
# whole_count = pd.DataFrame({'content': whole_count.index, 'count': whole_count.values})
filtered_whole = whole_related_df[whole_related_df['progress'] > 20000]
count = len(filtered_whole)
li = [len(whole_related_df), count]
li = pd.DataFrame(li)
li = li.T
li.columns=['sum', 'recent']
li.to_csv('datasum.csv', index=False)
whole_count = whole_related_df.groupby('content').agg({'id': 'sum', 'content': 'size'}).rename(columns={'content': 'count'}).reset_index()
mask = whole_count['count'].values <= 20
whole_count = whole_count[~mask]
whole_count
whole_count['emo'] = whole_count['content'].apply(get_sentiment)
#%%
whole_count.to_csv("main.csv")
print(whole_count)
whole_count

max_count = whole_count['count'].max()
max_content = whole_count.loc[whole_count['count'] == max_count, 'content'].values[0]
keyword = max_content
print(keyword)
#%%
# 降维 用于主图
# 使用t-SNE进行降维
mask = whole_count['content'].str.len() <= 1
whole_count = whole_count[~mask]
tsne = TSNE(n_components=2, random_state=2)
pca = PCA(n_components = 2)
embedding = tsne.fit_transform(whole_count.drop('content', axis=1))

# 创建降维后的DataFrame
df_tsne = pd.DataFrame(embedding, columns=['Dimension_1', 'Dimension_2'])
whole_count.index = df_tsne.index
df_tsne['content'] = whole_count['content']
df_tsne['count'] = whole_count['count']
df_tsne['emo'] = df_tsne['content'].apply(get_sentiment)
df_tsne.to_csv("tsne.csv")
print(df_tsne)
#%%
# 聚类
features = df_tsne[['Dimension_1', 'Dimension_2']]

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(features)

# 将标签添加到DataFrame的新列'label'中
df_tsne['label'] = labels
df_tsne
#%%
#河流图
index = 0
river = pd.DataFrame()
for i in range(10):
    start_time = max_time * i / 10
    end_time = max_time * (i+1) / 10
    tmp = whole_related_df[(whole_related_df['progress'] >= start_time) & (whole_related_df['progress'] <= end_time)]
    tmp = tmp['content'].value_counts()
    tmp = pd.DataFrame({'content': tmp.index, 'count': tmp.values})
    mask = tmp['count'].values <= 3
    tmp['emo'] = tmp['content'].apply(get_sentiment)
    tmp = tmp['emo'].value_counts()
    tmp = tmp.sort_index()
    tmp = pd.DataFrame(tmp)
    tmp = tmp.T
    tmp.columns = ['emo0', 'emo1', 'emo2', 'emo3', 'emo4', 'emo5']
    river = pd.concat([river, tmp], axis=0)
#%%
river.insert(0, "time", [0,1,2,3,4,5,6,7,8,9])
river['time'] = [0,1,2,3,4,5,6,7,8,9]
river.to_csv("river.csv", index=False)

#%%
#热度图
#从推荐视频api中截取视频cid并遍历，得到的数据需要有两个label，分别对应时间和分区
# 分区的tid 动画  游戏 知识 科技 运动 生活 美食 鬼畜
tids = [1, 4, 36, 188, 234, 160, 211, 119]
def get_video_ranking(tid):
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"
    params = {
        "rid": tid,
        "type": "all"
    }

    response = requests.get(url, params=params)
    data = response.json()
    cid_list = []
    if data["code"] == 0:
        video_list = data["data"]["list"]
        # 处理视频列表，进行进一步操作
        for i in video_list:
            cid_list.append(i['cid'])
    else:
        print("请求错误:", data["message"])
    return cid_list

def get_video_ranking_bid(tid):
    url = "https://api.bilibili.com/x/web-interface/ranking/v2"
    params = {
        "rid": tid,
        "type": "all"
    }

    response = requests.get(url, params=params)
    data = response.json()
    bvid_list = []
    if data["code"] == 0:
        video_list = data["data"]["list"]
        # 处理视频列表，进行进一步操作
        for i in video_list:
            bvid_list.append(i['bvid'])
    else:
        print("请求错误:", data["message"])
    return bvid_list
tid = 1
cids = get_video_ranking(tid)
print(cids)
#%%
#获取各个分区中 每个视频的 包含目标字符的数量
def to_list_heat(prot): #热度图所需的获取方式
    l = []
    for i in range(len(prot)):
        l.append({})
        l[i]["id"] = prot[i].id
        l[i]["ctime"] = prot[i].ctime
        l[i]["content"] = prot[i].content
    return l
#外层循环， 分区进行
heat = pd.DataFrame()
label = 0
for tid in tids:
    cids = get_video_ranking(tid)
    #内层循环，每个分区进行搜索抓取
    for cid in cids:
        danmu_proto = get_prot_dm(cid)
        listed_danmu = to_list_heat(danmu_proto)
        if(len(listed_danmu) == 0):
            continue
        df = pd.DataFrame(listed_danmu)
        mask = df['content'].str.len() > 10
        df = df[~mask]
        df['content'] = df['content'].str.replace('[^\w\s]', '')
        mask = df['content'].str.len() < 1
        df = df[~mask]
        df.dropna(subset=['content'], inplace=True)
        mask = df['content'].str.len() == 0
        df = df[~mask]
        df['label'] = label
        heat = pd.concat([heat, df], axis=0)
    label += 1
print(heat)
#%%

#%%
heat
#%%
# 筛选目标词汇
given_string = keyword
# 保留包含给定字符串的行
filtered_heat = heat[heat['content'].str.contains(given_string)]
filtered_heat

#%%

min_time = filtered_heat['ctime'].min()
max_time = filtered_heat['ctime'].max()
print(min_time, max_time)
filtered_heat['ctime'] -= min_time
#%%
print(filtered_heat)
#%%
#用相同方法把ctime分成8份
max_time = filtered_heat['ctime'].max()
range_size = max_time / 6

# 创建新的"partition"列
filtered_heat['ctime'] = pd.cut(filtered_heat['ctime'], bins=[0, range_size, 2*range_size, 3*range_size, 4*range_size, 5*range_size, max_time], labels=[0, 1, 2, 3, 4, 5], include_lowest=True)

# 更新"ctime"列的值

filtered_heat
#%%
heat_counts = filtered_heat['label'].value_counts()
heat_counts
#%%
#制作扩散图
label_counts = filtered_heat.groupby('ctime')['label'].value_counts().reset_index(name='count')
label_counts.to_csv('heat.csv', index=False)
cur = 0
cou = 5
label_counts['pos'] = label_counts['count']

for c in label_counts.index:
    if label_counts.iloc[c][0]>cur:
        cou = 5
        cur += 1
    label_counts.at[c, 'pos'] = cou
    cou += label_counts.iloc[c][2]
label_counts.to_csv('rightbot.csv', index=False)
#%%
#折线图
filtered_heat['ctime'] = filtered_heat['ctime'].astype(int)
print(filtered_heat)
filtered_heat['ctime'] += 1
filtered_heat['ctime'] *= max_time/6
filtered_heat['ctime'] += min_time

# filtered_heat['ctime'] = (filtered_heat['ctime']+1) * max_time / 6 + min_time
filtered_heat['ctime'] = filtered_heat['ctime'].apply(datetime.fromtimestamp)
filtered_heat['ctime'] = filtered_heat['ctime'].apply(lambda x: x.strftime('%Y-%m-%d'))

heat_line = filtered_heat['ctime'].value_counts().reset_index().rename(columns={'index': 'ctime', 'ctime': 'count'})

print(heat_line)
#%%
#差分器
maximum = -9999
cur = heat_line.at[0, 'count']
for i in range(len(heat_line['count'])):
    delta = abs(cur - heat_line.at[i, 'count'])
    cur = heat_line.at[i, 'count']
    if delta > maximum:
        max = delta
        max_lab = i
print(max_lab)
#find the greatest video
bvids = get_video_ranking_bid(tids[max_lab+1])
print(bvids[max_lab])
(m, cid, part) = get_cid(bvids[max_lab])
print(part)
greatest = pd.DataFrame({'part':part, 'bvid':bvids[max_lab]}, index = [0])
greatest.to_csv('greatest.csv', index = False)
#%%
heat_line.sort_values(by='ctime', inplace=True)
filtered_heat
heat_line.to_csv('line.csv', index = False)
#%%
#关系图
# 找到内容为keyword的行
selected_rows = df_tsne[df_tsne['content'] == keyword]

# 提取对应行的标签列
selected_labels = selected_rows['label']

# 保留只包含选定标签的行
filtered_df = df_tsne[df_tsne['label'].isin(selected_labels)]

# 打印筛选后的DataFrame
print(filtered_df)
#%%
json_data = []

# 对于每一行数据
for index, row in filtered_df.iterrows():
    # 创建字典存储每行的数据
    data = {
        "name": row['content'],
        "size": 127,
        "imports": []
    }

    # keyword，将所有其他内容添加到imports列表中
    if row['content'] == keyword:
        data['imports'] = filtered_df[filtered_df['content'] != keyword]['content'].tolist()
    else:
        data['imports'].append(keyword)

    # 将字典添加到json_data列表中
    json_data.append(data)

# 将json_data列表转换为JSON格式字符串
json_str = json.dumps(json_data, indent=4)

# 打印转换后的JSON格式字符串
print(json_str)
with open('data.json', 'w') as file:
    file.write(json_str)


