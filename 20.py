#==========最后还需要四角的平行四边形修正!  ######=测速版本
#============我们改用新的直线融合算法.
import cv2
import math
import time
import matplotlib.pyplot as plt
import numpy as np
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import imutils
import cv2

import numpy as np
start=time.time()
# image = cv2.imread('测试100/0719113903_04/Z_B.bmp')
image = cv2.imread('F_IR_F.bmp')


print(time.time()-start,22)

print(time.time()-start,56)
import numpy as np
kernel = np.ones((1, 5), np.uint8)

img = image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


print(time.time()-start,66)

tiaokuan=8
image=image[:,tiaokuan:-tiaokuan]
binary=binary[:,tiaokuan:-tiaokuan]
old_image=image
print(time.time()-start,88)
#======反转颜色 =========归一化到黑底白色图片才行. 背景色要是黑色的!
if binary[0][0] == 255:
    binary=255-binary
    old_image=image
    image=255-image
    pass

# binary=255-binary
print(time.time()-start,95)
#cv2.imwrite("13里面二值化的图片.png", binary)   
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)  # 二值化.

contours = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)  # 参数说明;https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71 
contours = imutils.grab_contours(contours) #适配cv2各个版本.
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = contours[0]
# img=binary.copy()
# img999=binary.copy()
# binary222=cv2.drawContours(img,contours,-1,(0,255,255),1)  
#cv2.imwrite("13里面的findcountours边缘.png", binary222)


print(time.time()-start,109)

epsilon = 0.02 * cv2.arcLength(contours, True)
approx = cv2.approxPolyDP(contours, epsilon, True)
if 0:
    n = []
    for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
                n.append((x, y))
    n = sorted(n)
    #print('找到的四点', n)

    print(time.time()-start,119)
    #========画所有点



    tmp=image.copy()
    # tmp2=image.copy()
    # tmp3=image.copy()
    
    sort_point = []
    n_point1 = n[:2]
    n_point1.sort(key=lambda x: x[1])
    sort_point.extend(n_point1)
    n_point2 = n[2:4]
    n_point2.sort(key=lambda x: x[1])
    n_point2.reverse()
    sort_point.extend(n_point2)                     
    print(time.time()-start,136)
    # print('方法一找到的四个点',sort_point)
    # for i in sort_point:
    #     tmp2=cv2.circle(tmp,i,0,(255,255,0),1)
    #cv2.imwrite('13图片里面方法一找的四个角.png',tmp2)    
    p1 = np.array(sort_point, dtype=np.float32)
    h = (sort_point[1][1] - sort_point[0][1] )**2+ (sort_point[1][0] - sort_point[0][0] )**2# sort_point : 左上, 左下, 右下,右上.
    h=math.sqrt(h)
    w = (sort_point[2][0] - sort_point[1][0])**2+(sort_point[2][1] - sort_point[1][1])**2
    w = math.sqrt(w)


    # h = sort_point[1][1] - sort_point[0][1]
    # w = sort_point[2][0] - sort_point[1][0]
    h=int(h)
    w=int(w)
    pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
    print(time.time()-start,153)
    M = cv2.getPerspectiveTransform(p1, pts2)

    dst = cv2.warpPerspective(old_image, M, (w, h))
    # #print(dst.shape)
    def show(image, window_name):
        # cv2.namedWindow(window_name, 0)
        #cv2.imwrite(window_name+'.png', image)
        pass
    if w < h:
        dst = np.rot90(dst)
# show(dst, "13里面方法一生成的图片")
	
	
#==========================================
	
	
	
#===============================================
#print('下面用新方法来对比')


print(time.time()-start,176)

#就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
if 1:
    img, color, width=binary,(0,0,255),2
    helper=img.copy()
    import numpy as np
    kernel = np.ones((1, 5), np.uint8)
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
    else:
        gray=img
    #cv2.imwrite('gray.png',gray)
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    if 1:                     
    # 边缘检测, Sobel算子大小为3
        edges = cv2.Canny(binary, 100, 200, apertureSize=3)
        # 霍夫曼直线检测
        #cv2.imwrite('13里面的canny边缘化图片.png',edges )

        gao=edges.shape[0]
        chang=edges.shape[1]


        print(time.time()-start,212)

        lines = cv2.HoughLinesP(edges, 1, 0.1*np.pi / 180, int((gao+chang)/40), minLineLength=(gao+chang)/20, maxLineGap=(gao+chang)/20)

        #==========话lines

        print(time.time()-start,213)

        
        #================进行直线筛选.
        # panduanzhixiantupian=binary.copy()
        #cv2.imwrite('全部的粗糙直线.png',panduanzhixiantupian)
        #===========使用算法1里面生成的四边形, 如果我们的直线在四边形里面那么就是没必要的, 可以删除.!!!!!!!!!!!!筛选!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dexsave=[]
        for dex,i67 in enumerate(lines):
            pt=(i67[0][0]+i67[0][2])/2,(i67[0][1]+i67[0][3])/2
            pt2=(i67[0][0]+i67[0][0])/2,(i67[0][1]+i67[0][1])/2
            pt3=(i67[0][2]+i67[0][2])/2,(i67[0][3]+i67[0][3])/2
            a=cv2.pointPolygonTest(approx, pt, 1)
            b=cv2.pointPolygonTest(approx, pt2, 1)
            c=cv2.pointPolygonTest(approx, pt3, 1)
            # print(a<=0)
            if a<=0.5 and  b<0.5  and  c<0.5: #全部在四边形外面才行.
                 dexsave.append(dex)
            else:
                 pass
                #  print(a,b,c,'不行的线的距离!!!!!!!!!!!!!!!!!!!!!!!!!')
        lines=lines[dexsave]
        # approx


        # fffff=tmp.copy()
        # for line in lines:
        # # 获取坐标 
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
        # #cv2.imwrite('13方法2的画全部的线.png',fffff)


        print(time.time()-start,245)


        








        #print()
        import math
        #=========输入一个直线, 计算他跟x轴的夹角.
        def jiajiao(line):   # 4点决定一个线
                if (line[2]-line[0]) :
                    a=math.atan((line[3]-line[1])/(line[2]-line[0]))/math.pi*180 
                    # if a<0:
                    #     return 180+a
                    return a
                else:
                    return 90
        # a=jiajiao([0,0,-1,1])
        a=[jiajiao([i[0][0],i[0][1],i[0][2],i[0][3]]) for i in lines]
#================负90到90度.
        print(time.time()-start,271)


        # a=[round(i) for i in a]
        # #print(a)
        # plt.hist(a, density=False)
        # plt.savefig('aaaa.png')
        # #print()
        # 按照相差上下10度来分类.

        #============下面按照上下15度进行分类.因为每一个类别一定有一个中间轴.中间轴一定是这些店里面的值.我们来存索引.
        # yuzhi=15
        #==========先把角度变到0到180  
#         a=[i  if i>=0 else 180+i for i in a]
#         a=[i  if i>=0 else 180+i for i in a]
# #============变到-90 到90
#         a=[i-180 if i>90 else i for i in a]

        #============我们现在不管长短边.
        #=======现在a 里面角度是 -90到90
        #========先找到里面距离0度近的. 都变成负数. 距离90度近的都变成正数.
        a1=[]
        a2=[]
        for dex,i in enumerate(a):
             if abs(i)<45:
                  a1.append(dex)
             else:
                  a2.append(dex)
        
















        print(time.time()-start,321)
        jiajiaobaocun=a
        if a:#
                pass
                #根据我们身份证的理解.
                # 我们要的2个中州一定是距离最大的.一般是0和90度.
                juli=0#先算出距离最大值.
                for dex,i in enumerate(a):
                    for dex2,j in enumerate(a):
                        tmp=min(abs(i-j),abs(180+i-j),abs(180-i+j))
                        if tmp>juli:
                            baocun=dex,dex2
                            juli=tmp
                #print(baocun,juli)
                yuzhi=juli/3
        else:
             1111111
            #print('没有任何一个直线,所以算法不进行后续边界识别')
        #=============先横线,再竖线.
        
        if baocun:
            baocun=list(baocun)
            if a[baocun[0]]>a[baocun[1]]:
                 baocun[0],baocun[1]=baocun[1],baocun[0]
            j=a[baocun[0]]
            list1=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
            j=a[baocun[1]]
            list2=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
        #print('打印两组直线角度阵营',list1,list2)
        #=========分别算投影, 去掉一个方向分量之后我们进行第二次细分这2住店.这样就得到了4个边的阵营.
        list1=a1
        list2=a2
        print(time.time()-start,353)
        
        zhixianfenzu=[]
        a1=list1
        a2=list2
        for aaa in [a1,a2]:
            #======算出每个阵营的投影直线.
            #==先算每个阵营的中心直线
            zhenying=aaa
            zhenyingjiaodu=[jiajiaobaocun[i] for i in aaa]
            zhenyingzhixianjiaodu=sum(zhenyingjiaodu)/len(zhenyingjiaodu)
            #print(zhenyingzhixianjiaodu)
            chuizhijiaodu=zhenyingzhixianjiaodu+90


            a=math.tan(chuizhijiaodu/180*math.pi)
            xiangliang=(1,a*1)
            list1zhongdian=[[(lines[i][0][0]+lines[i][0][2])/2,(lines[i][0][1]+lines[i][0][3])/2] for i in aaa]
            touying=[(i[0]*xiangliang[0]+i[1]*xiangliang[1])/math.sqrt(xiangliang[0]**2+xiangliang[1]**2) for i in list1zhongdian]
            touying=[abs(i) for i in touying] # 因为涉及角度问题,所以abs才行.




            chuizhijiaodu2=zhenyingzhixianjiaodu


            a2=math.tan(chuizhijiaodu2/180*math.pi)
            xiangliang2=(1,a2*1)
            list1zhongdian2=[[(lines[i][0][0]+lines[i][0][2])/2,(lines[i][0][1]+lines[i][0][3])/2] for i in aaa]
            touying2=[(i[0]*xiangliang2[0]+i[1]*xiangliang2[1])/math.sqrt(xiangliang2[0]**2+xiangliang2[1]**2) for i in list1zhongdian2]
            touying2=[abs(i) for i in touying2] # 水平投影.












            #=========继续用间隔来分类
            juli=0#先算出距离最大值.
            for dex,i in enumerate(touying):
                for dex2,j in enumerate(touying):
                    tmp=abs(i-j)
                    if tmp>juli:
                        baocun=dex,dex2
                        juli=tmp
            #print(baocun,juli)
            yuzhi=juli/3
            #print(a)
            a=touying
            if baocun:
                baocun=list(baocun)
                if a[baocun[0]]<a[baocun[1]]:
                     baocun[0],baocun[1]=baocun[1],baocun[0]
                j=a[baocun[0]] # 跟第一点近的放list1里面
                list1=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]
                j=a[baocun[1]]  
                list2=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]
                list1.sort(key=lambda x:touying2[x])
                list2.sort(key=lambda x:touying2[x])
                #print(1)
                list1inalldex=[zhenying[i] for i in list1]
                list2inalldex=[zhenying[i] for i in list2]
                zhixianfenzu.append(list1inalldex)
                zhixianfenzu.append(list2inalldex)
            #print()
        #print()







        print(time.time()-start,433)

        zhixianfenzu# 里面有4个数组, 每个数组表示一个直线族.  数组里面的数据是: shang xia  you zuo     4条变.
        #=========下面把每组的直线拟合成一条直线
        #==================
        all_four_line=[]

        if 0:
            fffff=tmp3.copy()
            for dex in zhixianfenzu[0]:
            # 获取坐标 
                line=lines[dex]
                x1, y1, x2, y2 = line[0]
                cv2.line(fffff, (x1, y1), (x2, y2), (0, 255, 255), thickness=1)
            #cv2.imwrite('13方法2的画全部的线下.png',fffff)


            fffff=tmp3.copy()
            for dex in zhixianfenzu[1]:
            # 获取坐标 
                line=lines[dex]
                x1, y1, x2, y2 = line[0]
                cv2.line(fffff, (x1, y1), (x2, y2), (0, 255, 255), thickness=1)
            #cv2.imwrite('13方法2的画全部的线上.png',fffff)

            fffff=tmp3.copy()
            for dex in zhixianfenzu[2]:
            # 获取坐标 
                line=lines[dex]
                x1, y1, x2, y2 = line[0]
                cv2.line(fffff, (x1, y1), (x2, y2), (0, 255, 255), thickness=1)
            #cv2.imwrite('13方法2的画全部的线右.png',fffff)


            fffff=tmp3.copy()
            for dex in zhixianfenzu[3]:
            # 获取坐标 
                line=lines[dex]
                x1, y1, x2, y2 = line[0]
                cv2.line(fffff, (x1, y1), (x2, y2), (0, 255, 255), thickness=1)
            #cv2.imwrite('13方法2的画全部的线左.png',fffff)



        print(time.time()-start,477)







##############################################################################============下面进行四边融合算法.之前的平均值方法不好.

        #########=======把每条直线的所有点都找到.


        # #print('开始处理上面的直线')
        # upper=[]
        # for i in zhixianfenzu[0]:
        #      #print(lines[i])
        #      upper.append([lines[i][0][0],lines[i][0][1]])
        #      upper.append([lines[i][0][2],lines[i][0][3]])
        # #print('-'*30)










#2023-08-10,14点48    从结果看, 直线的居合之后的一组直线拟合成一条直线还是精度不够. 从几何上再优化一下算法. #  水平的内部按照水平大小排序, 垂直的内部按照垂直大小排序.
#====================================================
        def cross_point(line1, line2):  # 计算交点函数
            #是否存在交点
            point_is_exist=False
            x=0
            y=0
            x1 = line1[0]  # 取四点坐标
            y1 = line1[1]
            x2 = line1[2]
            y2 = line1[3]

            x3 = line2[0]
            y3 = line2[1]
            x4 = line2[2]
            y4 = line2[3]

            if (x2 - x1) == 0:
                k1 = None
            else:
                k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
                b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

            if (x4 - x3) == 0:  # L2直线斜率不存在操作
                k2 = None
                b2 = 0
            else:
                k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
                b2 = y3 * 1.0 - x3 * k2 * 1.0

            if k1 is None:
                if not k2 is None:
                    x = x1
                    y = k2 * x1 + b2
                    point_is_exist=True
            elif k2 is None:
                x=x3
                y=k1*x3+b1
            elif not k2==k1:
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
                point_is_exist=True
            return point_is_exist,[x, y]
        if len(zhixianfenzu)==4:
            #=====================处理下面的直线
            xiazhixian=[lines[i] for i  in zhixianfenzu[0]]
            print()
            #=========首先我们计算左下角:
            xiamianzuizuo=zhixianfenzu[0][0]
            chuizhizuoxia=zhixianfenzu[-1][-1]
            # print('左下角',)
            zuoxiajiao=cross_point(lines[xiamianzuizuo][0],lines[chuizhizuoxia][0])[1]



            xiamianzuizuo=zhixianfenzu[0][-1]
            chuizhizuoxia=zhixianfenzu[2][-1]
            # print('右下角',)
            youxiajiao=cross_point(lines[xiamianzuizuo][0],lines[chuizhizuoxia][0])[1]



            xiamianzuizuo=zhixianfenzu[1][0]
            chuizhizuoxia=zhixianfenzu[3][0]
            # print('左上角',)
            zuoshangjiao=cross_point(lines[xiamianzuizuo][0],lines[chuizhizuoxia][0])[1]

            xiamianzuizuo=zhixianfenzu[1][-1]
            chuizhizuoxia=zhixianfenzu[2][0]
            # print('右上角',)
            youshangjiao=cross_point(lines[xiamianzuizuo][0],lines[chuizhizuoxia][0])[1]



            # tmp4=image.copy()
            
            # tmp4=cv2.circle(tmp4,(int(zuoxiajiao[0]),int(zuoxiajiao[1])),00,(255,0,255),0)

            #cv2.imwrite('查看左下角.png',tmp4)
































            pass       # 打印一组线. [lines[i] for i  in zhixianfenzu[0]]
        #没太好思路, 就平均数吧
            for i in zhixianfenzu:
                tmpzhixian= np.squeeze(lines[i], axis = 1)
                tmpjiajiao=np.array(jiajiaobaocun)[i].mean()
                tmpzhongxindian=np.array([(tmpzhixian[:,0]+tmpzhixian[:,2])/2,(tmpzhixian[:,1]+tmpzhixian[:,3])/2]).T


                tmpzhongdian2=tmpzhixian.mean(axis=0)
                tmpzhongdian2=(tmpzhongdian2[0]+tmpzhongdian2[2])/2,(tmpzhongdian2[1]+tmpzhongdian2[3])/2
                #print(1)
                all_four_line.append([tmpzhongdian2,tmpjiajiao])
        #=======转化为双点是.
        all_four_line2=[]
        for i in all_four_line:
                dian=i[0]
                jiaodu=i[1]
                a=min(math.tan(jiaodu/180*math.pi),99999) # 90度时候会溢出,所以上线设置10w即可.
                all_four_line2.append([dian[0],dian[1],dian[0]+1,dian[1]+a])
        all_four_line=all_four_line2
        #点斜华为2点.
        h,w,_=image.shape
        dianxieshi=[[i[0],i[1],i[3]-i[1]] for i in all_four_line]
        #========转化为足够长的射线.
        if 0:
            fffff=tmp3.copy()

            for i89 in range(4):
                out2=[]
                out2.append(0)
                out2.append(dianxieshi[i89][1]-dianxieshi[i89][0]*dianxieshi[i89][2])
                out2.append(w)
                out2.append((w-dianxieshi[i89][0])*dianxieshi[i89][2]+dianxieshi[i89][1])
                out2=[int(i) for i in out2]
                # fffff=cv2.line(fffff, (out2[0], out2[1]), (out2[2], out2[3]), (0, 255, 255), thickness=1)
            #cv2.imwrite('融合的直线.png',fffff)
        # print(1)
        print(time.time()-start,655)




        if 0:
            #===========计算交点
            
            #=======算每一个直线跟其他直线的交点
            #diyige :
            all3=[]
            tmp=all_four_line[0]
            tmp2=all_four_line[1]
            jiaodian1=cross_point(tmp,tmp2)
            tmp=all_four_line[0]
            tmp2=all_four_line[2]
            jiaodian2=cross_point(tmp,tmp2)
            tmp=all_four_line[0]
            tmp2=all_four_line[3]
            jiaodian3=cross_point(tmp,tmp2)
            all2=[]
            if jiaodian1[0]:
                    all2.append(jiaodian1[1])
            if jiaodian2[0]:
                    all2.append(jiaodian2[1])   
            if jiaodian3[0]:
                    all2.append(jiaodian3[1])
            all2.sort(key=lambda x:abs(x[0])+abs(x[1]) )
            all2=all2[:2]
            all3+=all2


            tmp=all_four_line[1]
            tmp2=all_four_line[0]
            jiaodian1=cross_point(tmp,tmp2)
            tmp=all_four_line[1]
            tmp2=all_four_line[2]
            jiaodian2=cross_point(tmp,tmp2)
            tmp=all_four_line[1]
            tmp2=all_four_line[3]
            jiaodian3=cross_point(tmp,tmp2)
            all2=[]
            if jiaodian1[0]:
                    all2.append(jiaodian1[1])
            if jiaodian2[0]:
                    all2.append(jiaodian2[1])   
            if jiaodian3[0]:
                    all2.append(jiaodian3[1])
            all2.sort(key=lambda x:abs(x[0])+abs(x[1]) )
            all2=all2[:2]
            all3+=all2
            #print(1)

            for dex,i in enumerate(all3):
                all3[dex][0]=round(i[0])
                all3[dex][1]=round(i[1])
            # print('处理前的四点',all3)
            aaaaaaaaaa=all3
            #print(all3,'最后的四点!!!!!!!!!!!!!!!')
            #=====================check!!!!!!!!!!

            #====================最后找到的四点基本是我们要的边缘上的点的差2个坐标左右.
            #================在图片修复9个点
            for i8 in range(4):
                fffff=tmp3.copy()
                # #cv2.imwrite('24123j12lk3j1l23j2lkj31.png',fffff[:30]) #========windows画板上的坐标对应, (y,x)
                tmppoint=all3[i8]
                bianchang=5
                candidate=[[i,j] for i in range(tmppoint[0]-bianchang,tmppoint[0]+bianchang+1) for j in range(tmppoint[1]-bianchang,tmppoint[1]+bianchang+1)]
                candidate.sort(key=lambda x:(x[0]-tmppoint[0])**2+(x[1]-tmppoint[1])**2)

                for i3 in candidate:
                    round2=[[i,j] for i in range(i3[0]-1,i3[0]+2) for j in range(i3[1]-1,i3[1]+2)]
        
                    # sedu=[sum(fffff[i4[1],i4[0]])>100 if (i4[1]<=fffff.shape[0] and i4[0]<=fffff.shape[0]) else 0 for i4 in round2] # 周围9个像素的色度.
                    sedu=[]
                    for i4 in round2:
                        if (i4[1]<=fffff.shape[0] and i4[0]<=fffff.shape[1]):
                            sedu.append(sum(fffff[i4[1],i4[0]])>100)
                        else:
                            sedu.append(0)


                    all_sedu=sum(sedu)
                    if all_sedu>=3  and  sum(fffff[i3[1],i3[0]])>10: # 加判断, 候选点本身也要有亮度!, 我们让他正好踩到图像的边.
                        all3[i8]=i3
                        #print(i3,76867867867)
                        break
            
            #print()

    #         tmp5=image.copy()
    # #=============这个点, 表示 第一个是距离y轴的距离,第二个是距离屏幕最上边缘直线的距离.
    #         tmp4=cv2.circle(tmp5,(400,10),00,(255,255,255),0)

        





            #==============画点.
    
            binary=gray








            #=============切回不校准的模式. 也就是边缘可能多点毛刺.但是更垂直水平了.
            all3=aaaaaaaaaa








        #=======2023-08-11,16点07改用最近的四角算法
        all3=[zuoshangjiao,zuoxiajiao,youshangjiao,youxiajiao]
        all3=[[int(j) for j in i] for  i in all3]
        # tmp4=image.copy()
        # for i in all3:
        #     tmp4=cv2.circle(tmp4,(int(i[0]),int(i[1])),00,(255,0,255),0)

        #cv2.imwrite('13里面方法2处理前的四角.png',tmp4)


########=============再加上取毛边的算法:
        if 0:
            for i8 in range(4):
                fffff=tmp3.copy()
                # #cv2.imwrite('24123j12lk3j1l23j2lkj31.png',fffff[:30]) #========windows画板上的坐标对应, (y,x)
                tmppoint=all3[i8]
                bianchang=5
                candidate=[[i,j] for i in range(tmppoint[0]-bianchang,tmppoint[0]+bianchang+1) for j in range(tmppoint[1]-bianchang,tmppoint[1]+bianchang+1)]
                candidate.sort(key=lambda x:(x[0]-tmppoint[0])**2+(x[1]-tmppoint[1])**2)

                for i3 in candidate:
                    round2=[[i,j] for i in range(i3[0]-1,i3[0]+2) for j in range(i3[1]-1,i3[1]+2)]
        
                    # sedu=[sum(fffff[i4[1],i4[0]])>100 if (i4[1]<=fffff.shape[0] and i4[0]<=fffff.shape[0]) else 0 for i4 in round2] # 周围9个像素的色度.
                    sedu=[]
                    for i4 in round2:
                        if (i4[1]<=fffff.shape[0] and i4[0]<=fffff.shape[1]):
                            sedu.append(sum(fffff[i4[1],i4[0]])>100)
                        else:
                            sedu.append(0)


                    all_sedu=sum(sedu)
                    if all_sedu>=3  and  sum(fffff[i3[1],i3[0]])>10: # 加判断, 候选点本身也要有亮度!, 我们让他正好踩到图像的边.
                        all3[i8]=i3
                        #print(i3,76867867867)
                        break











        # for i in all3:
        #     tmp3=cv2.circle(tmp3,(int(i[0]),int(i[1])),00,(255,255,255),0)
        # print('处理后的四点',all3,)
        #cv2.imwrite('13里面方法2处理后的四角.png',tmp3)

        #=======因为平行肯定有一个线超长.
        # contours,hierarchy = cv2.findContours(binary2,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)
        # tuxingzhouchang=cv2.arcLength(contours[0], True)
        # #print(1)
        #paixu jike




        print(time.time()-start,841)
        # lines = cv2.HoughLines(edges,1,np.pi/180,100)
        #====================下面我们做仿射变换即可.
        all3=np.array(all3)[:,None,...]
        #print(1)
        approx=all3
        n = []
        for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
            n.append((x, y))
        n = sorted(n)
        sort_point = []
        n_point1 = n[:2]
        n_point1.sort(key=lambda x: x[1])
        sort_point.extend(n_point1)
        n_point2 = n[2:4]
        n_point2.sort(key=lambda x: x[1])
        n_point2.reverse()
        sort_point.extend(n_point2)
        p1 = np.array(sort_point, dtype=np.float32)
        h = (sort_point[1][1] - sort_point[0][1] )**2+ (sort_point[1][0] - sort_point[0][0] )**2# sort_point : 左上, 左下, 右下,右上.
        h=math.sqrt(h)
        w = (sort_point[2][0] - sort_point[1][0])**2+(sort_point[2][1] - sort_point[1][1])**2
        w=math.sqrt(w)
        pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
        h=round(h)
        w=round(w)
        print(time.time()-start,867)
        M = cv2.getPerspectiveTransform(p1, pts2)
        
        dst = cv2.warpPerspective(old_image, M, (w, h))
        # #print(dst.shape)
        print(time.time()-start,872)
        if 0:
            def show(image, window_name):
                # cv2.namedWindow(window_name, 0)
                cv2.imwrite(window_name+'.png', image)

            if w < h:
                dst = np.rot90(dst)

            show(dst, '13里面方法2的最后图片')
        cv2.imwrite('13里面方法2的最后图片.png', image)


        
        print(time.time()-start,881)
	
	
	
	