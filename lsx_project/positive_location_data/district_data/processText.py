f=open("tmp.txt","r",encoding="utf-8")
text=f.readline()
pudong=text.count("浦东新区")
huangpu=text.count("黄浦区")
jingan=text.count("静安区")
xuhui=text.count("徐汇区")
changning=text.count("长宁区")
putuo=text.count("普陀区")
hongkou=text.count("虹口区")
yangpu=text.count("杨浦区")
baoshan=text.count("宝山区")
minhang=text.count("闵行区")
jiading=text.count("嘉定区")
jinshan=text.count("金山区")
songjiang=text.count("松江区")
qingpu=text.count("青浦区")
fengxian=text.count("奉贤区")
chongming=text.count("崇明区")
print([pudong,huangpu,jingan,xuhui,changning,putuo,hongkou,yangpu,baoshan,minhang,jiading,jinshan,songjiang,qingpu,fengxian,chongming])
'''
print(huangpu)
print(jingan)
print(xuhui)
print(changning)
print(putuo)
print(hongkou)
print(yangpu)
print(baoshan)
print(minhang)
print(jiading)
print(jinshan)
print(songjiang)
print(qingpu)
print(fengxian)
print(chongming)
'''
f.close()