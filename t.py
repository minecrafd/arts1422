import requests
import google.protobuf.text_format as text_format
import dm_pb2 as Danmaku

url = 'https://api.bilibili.com/x/v2/dm/web/seg.so'
params = {
    'type': 1,         # 弹幕类型
    'oid': 1176840,    # cid
    'pid': 810872,     # avid
    'segment_index': 1 # 弹幕分段
}
resp = requests.get(url, params)
data = resp.content

danmaku_seg = Danmaku.DmSegMobileReply()
danmaku_seg.ParseFromString(data)

print(danmaku_seg.elems[2])
