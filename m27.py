# ========去掉毛边修复, 改进hopf直线代码. 可能是参数不对.


# ==========最后还需要四角的平行四边形修正!  ######=测速版本
# ============我们改用新的直线融合算法.
import cv2
import imutils
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import builtins


def round(x):
    return builtins.round(x+1e-10)


# import libraries
debug = 1
start = time.time()
# image = cv2.imread('测试100/0719113903_04/Z_B.bmp')
# image = cv2.imread('F_IR_F.bmp')
# image = cv2.imread('jianwei/50/f3.png')
# image = cv2.imread('jianwei/50/f1.png')
# image = cv2.imread('jianwei/50/f2_1.png')
# image = cv2.imread('jianwei/50/f2.png')
# image = cv2.imread('jianwei/50/f3_1.jpg')
# image = cv2.imread('jianwei/50/t1.png')
# image = cv2.imread('jianwei/50/t2.png')
# image = cv2.imread('jianwei/50/t3.jpg')
# image = cv2.imread('测试100/0719113903_00/Z_B.bmp')
# image = cv2.imread('测试100/0719113903_00/Z_G_T.bmp')
# image = cv2.imread('测试100/0719113903_00/Z_R_F.bmp')
# image = cv2.imread('测试100/0719113903_03/Z_G.bmp')
# image = cv2.imread('测试100/0719113903_02/Z_R_F.bmp')
# image = cv2.imread('测试100/0719113903_03/Z_R_F.bmp')


def main(name, savename=None):
    print('当前图片名字', name)
    ori_image = cv2.imread(name)
    if debug:
        cv2.imwrite('原图.png', ori_image)
    # ==========进行边缘裁剪多种情况的适配.
    for yuzhi999 in [0, 20, 15, 10, 5]:  # ========0表示自由裁剪一次.
        try:
            image = ori_image

            import numpy as np
            kernel = np.ones((1, 5), np.uint8)

            img = image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            print(f'当前裁剪阈值是{yuzhi999}')
            if 1:
                # =======下面我们来计算左右裁剪的距离:
                # 投影 gray
                touying = np.average(gray, axis=0)
                memo = []  # ======找到足够亮的条.
                for i in range(gray.shape[1]):
                    if sum(abs(gray[:, i,] - touying[i]) < 10) > gray.shape[0]*0.8:
                        memo.append(i)

                # ======找到左右边
                if 0 in memo:
                    zuobian = 1
                    for i in range(gray.shape[0]):
                        if i in memo:
                            zuobian += 1
                else:
                    zuobian = 0

                if gray.shape[1]-1 in memo:
                    youbian = 1
                    for i in range(gray.shape[1]-1, 0, -1):
                        if i in memo:
                            youbian += 1
                else:
                    youbian = 0

                # yuzhi999=20#==================这里还是要继续优化!!!!!!!!!!
                zuobian = int(0.7*zuobian)
                youbian = int(0.7*youbian)
                zuobian = min(yuzhi999, zuobian)
                youbian = min(yuzhi999, youbian)
                # print(zuobian,youbian,'裁剪的最终值!!!!!!!!!!!!!')
                youbian = gray.shape[1]-youbian

                # ===================2023-10-09,16点59 设计一个新的边缘检测.
                t2 = np.abs(touying[:-1]-touying[1:])
                t3 = np.where(t2 > 30)[0].flatten()
                left = [im for im in t3 if im < 100]
                right = [im for im in t3 if im > 100]
                if left:
                    left = min(left)+2
                else:
                    left = 0
                if right:
                    right = max(right)-2
                else:
                    right = 0
                if not yuzhi999:
                    zuobian = left
                    youbian = right
                    youbian = gray.shape[1]-youbian

                shagnxia = 3

                # tiaokuan=8

                image = image[shagnxia:-shagnxia, zuobian:youbian]
                gray = gray[shagnxia:-shagnxia, zuobian:youbian]
                binary = binary[shagnxia:-shagnxia, zuobian:youbian]

                old_image = image.copy()
                if debug:
                    cv2.imwrite('binary1.png', binary)
                if debug:
                    cv2.imwrite('裁剪完的.png', image)
                # ======反转颜色 =========归一化到黑底白色图片才行. 背景色要是黑色的!
                if binary[0][0] == 255:
                    binary = 255-binary
                    old_image = image
                    image = 255-image
                    pass
                if debug:
                    cv2.imwrite('反色之后的.png', image)
                # binary=255-binary

            import cv2 as cv
            if 0:

                corners = cv2.goodFeaturesToTrack(
                    gray, 4, 0.0001, 50, useHarrisDetector=False)
                tmp10000 = image.copy()
                for i in corners:
                    tmp10000 = cv2.circle(
                        tmp10000, (int(i[0][0]), int(i[0][1])), 0, (255, 255, 0), 1)
                if debug:
                    cv2.imwrite('dsafjdsaklfjalsdkjf.png', tmp10000)
                # =======corner方法绘制box
                aaaaa3 = image.copy()
                aaaaa3 = cv2.rectangle(aaaaa3, (round(corners[0][0][0]), round(
                    corners[0][0][1])), (round(corners[3][0][0]), round(corners[3][0][1])), (0, 0, 255), 0)
                cv2.imwrite('corner_bounding.png', aaaaa3)

            if 0:
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray, None)

                img = cv2.drawKeypoints(gray, kp, img)

                cv2.imwrite('sift_keypoints.jpg', img)

            # cv2.imwrite("13里面二值化的图片.png", binary)
            # 二值化.
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)

            # 参数说明;https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
            contours = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)  # 适配cv2各个版本.
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = contours[0]
            # img=binary.copy()
            # img999=binary.copy()
            # binary222=cv2.drawContours(img,contours,-1,(0,255,255),1)
            # cv2.imwrite("13里面的findcountours边缘.png", binary222)

            epsilon = 0.02 * cv2.arcLength(contours, True)
            approx = cv2.approxPolyDP(contours, epsilon, True)
            tmp = image.copy()
            tmp2 = image.copy()
            tmp3 = image.copy()
            tmp4 = image.copy()

            # ==========================================

            # ===============================================
            # print('下面用新方法来对比')

            # 就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
            if 1:
                img, color, width = binary, (0, 0, 255), 2
                helper = img.copy()
                import numpy as np
                kernel = np.ones((1, 5), np.uint8)
                if len(img.shape) > 2:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰度图
                else:
                    gray = img
                # cv2.imwrite('gray.png',gray)
                ret, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
                if debug:
                    cv2.imwrite('binary.png', binary)

    # ====================binary没问题!
                if 1:
                    # 边缘检测, Sobel算子大小为3
                    # ==========背景色是白色, 东西是黑色.我们要的边缘是黑色的边,也就是边缘上的色素是黑色的.这样保证我们hopf得到的点也是踩在我们要的区域上的.
                    binary = 255-binary

                    res = cv2.Laplacian(binary, cv2.CV_8U)

                    # 3.格式转换格式
                    edges = cv2.convertScaleAbs(res)  # 转换为uint8
                    cv2.imwrite('13里面的canny边缘化图片2.png', edges)
                    if 0:
                        if debug:
                            cv2.imwrite('binary2.png', binary)
                        edges = cv2.Canny(binary, 100, 200)
                        # 霍夫曼直线检测
                        cv2.imwrite('13里面的canny边缘化图片.png', edges)

                    gao = edges.shape[0]
                    chang = edges.shape[1]

                    lines = cv2.HoughLinesP(edges, 1, 0.01*np.pi / 180, round((gao+chang)/40), minLineLength=(
                        gao+chang)/20, maxLineGap=(gao+chang)/20)  # 0.01*pi是调优过的参数!!!!!!!!
                    fffff = tmp.copy()
                    for line in lines:  # 4he5
                        # 获取坐标
                        x1, y1, x2, y2 = line[0]
                        cv2.line(fffff, (x1, y1), (x2, y2),
                                 (0, 170, 170), thickness=1)
                    if debug:
                        if 1:
                            cv2.imwrite(
                                '13方法2的画全部的线allllll_before_shaixuan.png', fffff)
                    # ==========话lines

                    # ================进行直线筛选.
                    # panduanzhixiantupian=binary.copy()

                    # ===========使用算法1里面生成的四边形, 如果我们的直线在四边形里面那么就是没必要的, 可以删除.!!!!!!!!!!!!筛选!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    dexsave = []
                    quchu = []
                    fsdlkafjlkf = []
                    juliyuzhi = 0.5
                    for dex, i67 in enumerate(lines):
                        pt = (i67[0][0]+i67[0][2])/2, (i67[0][1]+i67[0][3])/2
                        pt2 = (i67[0][0]+i67[0][0])/2, (i67[0][1]+i67[0][1])/2
                        pt3 = (i67[0][2]+i67[0][2])/2, (i67[0][3]+i67[0][3])/2
                        a = cv2.pointPolygonTest(approx, pt, 1)
                        b = cv2.pointPolygonTest(approx, pt2, 1)
                        c = cv2.pointPolygonTest(approx, pt3, 1)
                        fsdlkafjlkf.append([a, b, c])
# cv2.pointPolygonTest() 函数
#         函数定义：cv2.pointPolygonTest(contour, pt, measureDist)

#         函数功能：找到图像里的点和轮廓之间的最短距离. 它返回的距离当点在轮廓外的时候是负值，当点在轮廓内是正值，如果在轮廓上是0。

                        if a <= juliyuzhi and b <= juliyuzhi and c <= juliyuzhi:  # 全部在四边形外面才行.
                            dexsave.append(dex)
                        else:
                            quchu.append(dex)

                            # print(a, b, c, '不行的线的距离!!!!!!!!!!!!!!!!!!!!!!!!!')
                    lines = lines[dexsave]
                    # approx
    # 函数定义：cv2.pointPolygonTest(contour, pt, measureDist)
    # 函数功能：找到图像里的点和轮廓之间的最短距离. 它返回的距离当点在轮廓外的时候是负值，当点在轮廓内是正值，如果在轮廓上是0。

                    fffff999 = tmp.copy()
                    for line in lines:
                        # 获取坐标
                        x1, y1, x2, y2 = line[0]
                        cv2.line(fffff999, (x1, y1), (x2, y2),
                                 (0, 170, 170), thickness=1)
                    if debug:
                        if 1:
                            cv2.imwrite('13方法2的画全部的线原来.png', fffff999)

                    import math
                    # =========输入一个直线, 计算他跟x轴的夹角.

                    def jiajiao(line):   # 4点决定一个线 # 因为坐标跟直角坐标系不同, 还是要差一个负号!!!!!!!!!
                        if (line[2]-line[0]):
                            a = math.atan(
                                (-line[3]+line[1])/(line[2]-line[0]))/math.pi*180
                            # if a<0:
                            #     return 180+a
                            return a
                        else:
                            return 90
                    # a=jiajiao([0,0,-1,1])
                    a = [jiajiao([i[0][0], i[0][1], i[0][2], i[0][3]])
                         for i in lines]
                    jiaodusave = a

                    # ============重新算角度分配.

                    # ======先分开2组
                    diyizu = a[0]
                    a1 = []
                    a2 = []
                    for dex, i in enumerate(a):
                        if abs(i-diyizu) < 15 or abs(180+i-diyizu) < 15 or abs(180+diyizu-i) < 15:
                            a1.append(dex)
                        else:
                            a2.append(dex)

                    if sum([abs(a[a1[i]]) for i in range(len(a1))])/len(a1) > sum([abs(a[a2[i]]) for i in range(len(a2))])/len(a2):
                        a1, a2 = a2, a1  # 让a1接近水平, a2接近竖直.

                    for dex, i in enumerate(a):
                        if dex in a2 and i < 0:
                            a[dex] = 180+a[dex]
                        # 让水平的都在0左右, 统一化, 让垂直的都在90度左右.

                    jiajiaobaocun = a
                    zhixianfenzu = []

                    # ===========a1是水平组, a2是竖直组.
                    for dex999, aaa in enumerate([a1, a2]):
                        # ======算出每个阵营的投影直线.
                        # ==先算每个阵营的中心直线
                        zhenying = aaa
                        zhenyingjiaodu = [jiajiaobaocun[i] for i in aaa]
                        zhenyingzhixianjiaodu = sum(
                            zhenyingjiaodu)/len(zhenyingjiaodu)

                        chuizhijiaodu = zhenyingzhixianjiaodu+90

                        a = math.tan(chuizhijiaodu/180*math.pi)
                        xiangliang = (1, a*1)
                        list1zhongdian = [
                            [(lines[i][0][0]+lines[i][0][2])/2, (lines[i][0][1]+lines[i][0][3])/2] for i in aaa]
                        touying = [(i[0]*xiangliang[0]+i[1]*xiangliang[1])/math.sqrt(
                            xiangliang[0]**2+xiangliang[1]**2) for i in list1zhongdian]
                        # 因为涉及角度问题,所以abs才行.
                        touying = [abs(i) for i in touying]
# ===============组内再分类.
                        chuizhijiaodu2 = zhenyingzhixianjiaodu

                        a22 = math.tan(chuizhijiaodu2/180*math.pi)
                        xiangliang2 = (1, a22*1)
                        list1zhongdian2 = [
                            [(lines[i][0][0]+lines[i][0][2])/2, (lines[i][0][1]+lines[i][0][3])/2] for i in aaa]
                        touying2 = [(i[0]*xiangliang2[0]+i[1]*xiangliang2[1])/math.sqrt(
                            xiangliang2[0]**2+xiangliang2[1]**2) for i in list1zhongdian2]
                        touying2 = [abs(i) for i in touying2]  # 水平投影.
                        shuipingzuizuozhi = [lines[i] for i in aaa]
                        # =========继续用间隔来分类
                        juli = 0  # 先算出距离最大值.
                        for dex, i in enumerate(touying):
                            for dex2, j in enumerate(touying):
                                tmp = abs(i-j)
                                if tmp > juli:
                                    baocun = dex, dex2
                                    juli = tmp
                        # print(baocun,juli)
                        yuzhi = juli/3
                        # print(a)
                        a3 = touying
                        if baocun:
                            baocun = list(baocun)
                            if a3[baocun[0]] < a3[baocun[1]]:
                                baocun[0], baocun[1] = baocun[1], baocun[0]
                            j = a3[baocun[0]]  # 跟第一点近的放list1里面
                            list1 = [dex for dex, i in enumerate(
                                a3) if abs(i-j) < yuzhi]
                            j = a3[baocun[1]]
                            list2 = [dex for dex, i in enumerate(
                                a3) if abs(i-j) < yuzhi]
                            list1.sort(key=lambda x: touying2[x])
                            list2.sort(key=lambda x: touying2[x])

                            # print(1)
                            list1inalldex = [zhenying[i] for i in list1]
                            list2inalldex = [zhenying[i] for i in list2]

                            # =========我们更新排序算法.2023-10-12,14点37
                            if dex999 == 0:  # =======排序水平的.
                                list1inalldex.sort(key=lambda x: min(
                                    lines[x][0][0], lines[x][0][2]))
                                list2inalldex.sort(key=lambda x: min(
                                    lines[x][0][0], lines[x][0][2]))
                            else:
                                list1inalldex.sort(key=lambda x: min(
                                    lines[x][0][1], lines[x][0][3]))
                                list2inalldex.sort(key=lambda x: min(
                                    lines[x][0][1], lines[x][0][3]))

                            zhixianfenzu.append(list1inalldex)
                            zhixianfenzu.append(list2inalldex)
                        # print()

                    # 里面有4个数组, 每个数组表示一个直线族.  数组里面的数据是: shang xia  you zuo     4条变.
                    zhixianfenzu
                    # =========下面把每组的直线拟合成一条直线
                    # ==================
                    all_four_line = []

            # ====================================================

                    def cross_point(line1, line2):  # 计算交点函数======双点型直线.
                        # 是否存在交点
                        point_is_exist = False
                        x = 0
                        y = 0
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
                            # 计算k1,由于点均为整数，需要进行浮点数转化
                            k1 = (y2 - y1) * 1.0 / (x2 - x1)
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
                                point_is_exist = True
                        elif k2 is None:
                            x = x3
                            y = k1*x3+b1
                        elif not k2 == k1:
                            x = (b2 - b1) * 1.0 / (k1 - k2)
                            y = k1 * x * 1.0 + b1 * 1.0
                            point_is_exist = True
                        return point_is_exist, [x, y]
# 2023-10-09,10点45 刚才才发现, cv里面高度是距离图片左上角的高度差.不是我们数学上的针对左下角坐标原点的坐标.所以直接带入原始的点斜式方程求解出来的是错误的. 惊了.需要重新做坐标的归一化啊. 应该加一个负号即可,求解完再换算回来.

                    # 计算交点函数====点斜型直线. +==========斜率最大是99999, 也就是垂直直线的斜率用这个数值表示.
                    def cross_point_dianxie(line1, line2):
                        # 是否存在交点
                        point_is_exist = False
                        x = 0
                        y = 0
                        x1 = line1[0]  # 取四点坐标
                        y1 = -line1[1]
                        k1 = line1[2]

                        x2 = line2[0]
                        y2 = -line2[1]
                        k2 = line2[2]

                        if k1 == k2:
                            return False
                        else:
                            x = (k1*x1-k2*x2+y2-y1)/(k1-k2)
                            y = k1*(x-x1)+y1
                            y = -y
                            return True, [x, y]

                    lines_2 = []
                    for dex, i in enumerate(lines):
                        jiajiaobaocun
                        # dian=(i[0][0]+i[0][2])/2,(i[0][1]+i[0][3])/2,
                        # 90度时候会溢出,所以上线设置10w即可.
                        a = min(
                            math.tan(jiajiaobaocun[dex]/180*math.pi), 99999)
                        lines_2.append(a)
                    lines_2 = np.array(lines_2)

                    def jiaoduhuaxielv(a):
                        return min(math.tan(a/180*math.pi), 99999)

                    baxian = []
                    if len(zhixianfenzu) == 4:
                        houxuanjidaxiao = 2

                        quanzhong1 = 0.9
                        quanzhong2 = 0.1
                        # =====================处理下面的直线
                        xiazhixian = [lines[i] for i in zhixianfenzu[0]]

                        # =========首先我们计算左下角:
                        xiamianzuizuo2 = zhixianfenzu[0][0]
                        chuizhizuoxia2 = zhixianfenzu[-1][-1]
                        baxian.append(xiamianzuizuo2)
                        baxian.append(chuizhizuoxia2)

                        # ===============左下角!!!!!!!!!!
                        yuzhi9991 = 100
                        zhixian999 = lines[zhixianfenzu[0]][0][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > yuzhi9991:
                            xiamianzuizuo = zhixianfenzu[0][0:1]
                        else:
                            xiamianzuizuo = zhixianfenzu[0][0:2]

                        zhixian999 = lines[zhixianfenzu[-1]][-1][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > 47:
                            chuizhizuoxia = zhixianfenzu[-1][-1:]
                        else:
                            chuizhizuoxia = zhixianfenzu[-1][-2:]
                        # chuizhizuoxia=zhixianfenzu[-1][-houxuanjidaxiao:]

                        # 2023-10-09,10点10 加上权重分配比例.
                        if len(xiamianzuizuo) == 2:
                            wt1 = [quanzhong1, quanzhong2]
                        else:
                            wt1 = [1]
                        if len(chuizhizuoxia) == 2:
                            wt2 = [quanzhong1, quanzhong2]
                        else:
                            wt2 = [1]

                        zhongdian = np.average(
                            lines[xiamianzuizuo], axis=0, weights=wt1)
                        zhongdian = (
                            zhongdian[0][0]+zhongdian[0][2])/2, (zhongdian[0][1]+zhongdian[0][3])/2,
                        zhongdian2 = np.average(
                            lines[chuizhizuoxia], axis=0, weights=wt2)
                        zhongdian2 = (
                            zhongdian2[0][0]+zhongdian2[0][2])/2, (zhongdian2[0][1]+zhongdian2[0][3])/2,
                        zuoxiajiao = cross_point_dianxie([zhongdian[0], zhongdian[1], jiaoduhuaxielv(np.average(np.array(jiajiaobaocun)[xiamianzuizuo], weights=wt1))],
                                                         [zhongdian2[0], zhongdian2[1], jiaoduhuaxielv(np.average(
                                                             np.array(jiajiaobaocun)[chuizhizuoxia], weights=wt2))]

                                                         )[1]

                        xiamianzuizuo2 = zhixianfenzu[0][-1]
                        chuizhizuoxia2 = zhixianfenzu[2][-1]
                        baxian.append(xiamianzuizuo2)
                        baxian.append(chuizhizuoxia2)
                        # print('右下角',)
                        xiamianzuizuo = zhixianfenzu[0][:houxuanjidaxiao]
                        chuizhizuoxia = zhixianfenzu[2][-houxuanjidaxiao:]

                        zhixian999 = lines[zhixianfenzu[0]][-1][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > yuzhi9991:
                            xiamianzuizuo = zhixianfenzu[0][-1:]
                        else:
                            xiamianzuizuo = zhixianfenzu[0][-2:]

                        zhixian999 = lines[zhixianfenzu[2]][-1][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > 47:
                            chuizhizuoxia = zhixianfenzu[2][-1:]
                        else:
                            chuizhizuoxia = zhixianfenzu[2][-2:]
                        # chuizhizuoxia=zhixianfenzu[-1][-houxuanjidaxiao:]

                        # 2023-10-09,10点10 加上权重分配比例.
                        if len(xiamianzuizuo) == 2:
                            wt1 = [quanzhong1, quanzhong2]
                        else:
                            wt1 = [1]
                        if len(chuizhizuoxia) == 2:
                            wt2 = [quanzhong1, quanzhong2]
                        else:
                            wt2 = [1]

                        zhongdian = np.average(
                            lines[xiamianzuizuo], axis=0, weights=wt1)
                        zhongdian = (
                            zhongdian[0][0]+zhongdian[0][2])/2, (zhongdian[0][1]+zhongdian[0][3])/2,
                        zhongdian2 = np.average(
                            lines[chuizhizuoxia], axis=0, weights=wt2)
                        zhongdian2 = (
                            zhongdian2[0][0]+zhongdian2[0][2])/2, (zhongdian2[0][1]+zhongdian2[0][3])/2,
                        youxiajiao = cross_point_dianxie(

                            [zhongdian[0], zhongdian[1], jiaoduhuaxielv(np.average(
                                np.array(jiajiaobaocun)[xiamianzuizuo], weights=wt1))],
                            [zhongdian2[0], zhongdian2[1], jiaoduhuaxielv(np.average(
                                np.array(jiajiaobaocun)[chuizhizuoxia], weights=wt2))]

                        )[1]

                        xiamianzuizuo2 = zhixianfenzu[1][0]
                        chuizhizuoxia2 = zhixianfenzu[3][0]
                        baxian.append(xiamianzuizuo2)
                        baxian.append(chuizhizuoxia2)
                        # print('左上角',)
                        xiamianzuizuo = zhixianfenzu[1][:houxuanjidaxiao]
                        chuizhizuoxia = zhixianfenzu[3][-houxuanjidaxiao:]

                        zhixian999 = lines[zhixianfenzu[1]][0][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > yuzhi9991:
                            xiamianzuizuo = zhixianfenzu[1][0:1]
                        else:
                            xiamianzuizuo = zhixianfenzu[1][0:2]

                        zhixian999 = lines[zhixianfenzu[3]][0][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > 47:
                            chuizhizuoxia = zhixianfenzu[3][0:1]
                        else:
                            chuizhizuoxia = zhixianfenzu[3][0:2]

                        # 2023-10-09,10点10 加上权重分配比例.计算直线归一化.
                        if len(xiamianzuizuo) == 2:
                            wt1 = [quanzhong1, quanzhong2]
                        else:
                            wt1 = [1]
                        if len(chuizhizuoxia) == 2:
                            wt2 = [quanzhong1, quanzhong2]
                        else:
                            wt2 = [1]

                        zhongdian = np.average(
                            lines[xiamianzuizuo], axis=0, weights=wt1)
                        zhongdian = (
                            zhongdian[0][0]+zhongdian[0][2])/2, (zhongdian[0][1]+zhongdian[0][3])/2,
                        zhongdian2 = np.average(
                            lines[chuizhizuoxia], axis=0, weights=wt2)
                        zhongdian2 = (
                            zhongdian2[0][0]+zhongdian2[0][2])/2, (zhongdian2[0][1]+zhongdian2[0][3])/2,
                        zuoshangjiao = cross_point_dianxie([zhongdian[0], zhongdian[1], jiaoduhuaxielv(np.average(np.array(jiajiaobaocun)[xiamianzuizuo], weights=wt1))],
                                                           [zhongdian2[0], zhongdian2[1], jiaoduhuaxielv(np.average(
                                                               np.array(jiajiaobaocun)[chuizhizuoxia], weights=wt2))]

                                                           )[1]

                        xiamianzuizuo2 = zhixianfenzu[1][-1]
                        chuizhizuoxia2 = zhixianfenzu[2][0]
                        baxian.append(xiamianzuizuo2)
                        baxian.append(chuizhizuoxia2)
                        # print('右上角',)
                        # xiamianzuizuo=zhixianfenzu[1][houxuanjidaxiao]
                        # chuizhizuoxia=zhixianfenzu[2][-houxuanjidaxiao:]

                        zhixian999 = lines[zhixianfenzu[1]][-1][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > yuzhi9991:
                            xiamianzuizuo = zhixianfenzu[1][-1:]
                        else:
                            xiamianzuizuo = zhixianfenzu[1][-2:]

                        zhixian999 = lines[zhixianfenzu[2]][0][0]
                        zhixianchagndu = (
                            (zhixian999[2]-zhixian999[0])**2+(zhixian999[3]-zhixian999[1])**2)**0.5
                        if zhixianchagndu > 47:
                            chuizhizuoxia = zhixianfenzu[2][:1]
                        else:
                            chuizhizuoxia = zhixianfenzu[2][:2]

                        # 2023-10-09,10点10 加上权重分配比例.
                        if len(xiamianzuizuo) == 2:
                            wt1 = [quanzhong1, quanzhong2]
                        else:
                            wt1 = [1]
                        if len(chuizhizuoxia) == 2:
                            wt2 = [quanzhong1, quanzhong2]
                        else:
                            wt2 = [1]
                        zhongdian = np.average(
                            lines[xiamianzuizuo], axis=0, weights=wt1)
                        zhongdian = (
                            zhongdian[0][0]+zhongdian[0][2])/2, (zhongdian[0][1]+zhongdian[0][3])/2,

                        zhongdian2 = np.average(
                            lines[chuizhizuoxia], axis=0, weights=wt2)
                        zhongdian2 = (
                            zhongdian2[0][0]+zhongdian2[0][2])/2, (zhongdian2[0][1]+zhongdian2[0][3])/2,
                        youshangjiao = cross_point_dianxie(

                            [zhongdian[0], zhongdian[1], jiaoduhuaxielv(np.average(
                                np.array(jiajiaobaocun)[xiamianzuizuo], weights=wt1))],

                            [zhongdian2[0], zhongdian2[1], jiaoduhuaxielv(np.average(
                                np.array(jiajiaobaocun)[chuizhizuoxia], weights=wt2))]

                        )[1]
                        # print(youshangjiao)

                        # np.average(np.array(jiajiaobaocun)[xiamianzuizuo])
                        # jiaoduhuaxielv(np.average(np.array(jiajiaobaocun)[xiamianzuizuo]))
                        aaaaaaaaaaaa = 3
                        # print(1)

                        # 打印一组线. [lines[i] for i  in zhixianfenzu[0]]
                        pass
                    # 没太好思路, 就平均数吧
                        for i in zhixianfenzu:
                            tmpzhixian = np.squeeze(lines[i], axis=1)
                            tmpjiajiao = np.array(jiajiaobaocun)[i].mean()
                            tmpzhongxindian = np.array(
                                [(tmpzhixian[:, 0]+tmpzhixian[:, 2])/2, (tmpzhixian[:, 1]+tmpzhixian[:, 3])/2]).T

                            tmpzhongdian2 = tmpzhixian.mean(axis=0)
                            tmpzhongdian2 = (
                                tmpzhongdian2[0]+tmpzhongdian2[2])/2, (tmpzhongdian2[1]+tmpzhongdian2[3])/2
                            # print(1)
                            all_four_line.append([tmpzhongdian2, tmpjiajiao])
                    # =======转化为双点是.
                    # ============
                    # 画8条定位直线
                    fffff = tmp3.copy()
                    for line in baxian:
                        # 获取坐标
                        x1, y1, x2, y2 = lines[line][0]
                        fffff = cv2.line(
                            fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
                    if debug:
                        cv2.imwrite('baxian.png', fffff)
                    fffff = tmp3.copy()
                    for line in baxian[:2]:
                        # 获取坐标
                        x1, y1, x2, y2 = lines[line][0]
                        fffff = cv2.line(
                            fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
                    if debug:
                        cv2.imwrite('baxian1左下.png', fffff)

                    fffff = tmp3.copy()
                    for line in baxian[2:4]:
                        # 获取坐标
                        x1, y1, x2, y2 = lines[line][0]
                        fffff = cv2.line(
                            fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
                    if debug:
                        cv2.imwrite('baxian2右下.png', fffff)

                    fffff = tmp3.copy()
                    for line in baxian[4:6]:
                        # 获取坐标
                        x1, y1, x2, y2 = lines[line][0]
                        fffff = cv2.line(
                            fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
                    if debug:
                        cv2.imwrite('baxian3左上.png', fffff)

                    fffff = tmp3.copy()
                    for line in baxian[6:]:
                        # 获取坐标
                        x1, y1, x2, y2 = lines[line][0]
                        fffff = cv2.line(
                            fffff, (x1, y1), (x2, y2), (0, 170, 170), thickness=1)
                    if debug:
                        cv2.imwrite('baxian4右上.png', fffff)

                    all_four_line2 = []
                    for i in all_four_line:
                        dian = i[0]
                        jiaodu = i[1]
                        # 90度时候会溢出,所以上线设置10w即可.
                        a = min(math.tan(jiaodu/180*math.pi), 99999)
                        all_four_line2.append(
                            [dian[0], dian[1], dian[0]+1, dian[1]+a])
                    all_four_line = all_four_line2
                    # 点斜华为2点.
                    h, w, _ = image.shape
                    dianxieshi = [[i[0], i[1], i[3]-i[1]]
                                  for i in all_four_line]
                    # ========转化为足够长的射线.
                    if 0:
                        fffff = tmp3.copy()

                        for i89 in range(4):
                            out2 = []
                            out2.append(0)
                            out2.append(
                                dianxieshi[i89][1]-dianxieshi[i89][0]*dianxieshi[i89][2])
                            out2.append(w)
                            out2.append(
                                (w-dianxieshi[i89][0])*dianxieshi[i89][2]+dianxieshi[i89][1])
                            out2 = [round(i) for i in out2]
                            # fffff=cv2.line(fffff, (out2[0], out2[1]), (out2[2], out2[3]), (0, 255, 255), thickness=1)
                        # cv2.imwrite('融合的直线.png',fffff)
                    # print(1)
                    # print(time.time()-start,655)

                    # =======2023-08-11,16点07改用最近的四角算法
                    all3 = [zuoshangjiao, zuoxiajiao, youshangjiao, youxiajiao]
                    # print('处理后的四点bbbbb',all3,)
                    all3 = [[round(j) for j in i] for i in all3]
                    # print('处理后的四点整数化',all3,)

                    if debug:
                        tmp4 = image.copy()
                        for i in all3:
                            tmp4 = cv2.circle(
                                tmp4, (int(i[0]), int(i[1])), 00, (255, 0, 255), 0)
                        cv2.imwrite('13里面方法2处理前的四角.png', tmp4)

            # =============再加上取毛边的算法:

                    if 0:
                        fffff = tmp4.copy()
                        for i8 in range(4):

                            # #cv2.imwrite('24123j12lk3j1l23j2lkj31.png',fffff[:30]) #========windows画板上的坐标对应, (y,x)
                            tmppoint = all3[i8]
                            # =========2023-10-09,11点45超出变长的不讨论, 这样能保证缺角时候不会进行过度后处理.
                            bianchang = 3
                            candidate = [[i, j] for j in range(max(tmppoint[1]-bianchang, 0), min(tmppoint[1]+bianchang+1, tmp4.shape[0])) for i in range(max(tmppoint[0]-bianchang, 0), min(tmppoint[0]+bianchang+1, tmp4.shape[1]))

                                         ]  # =========这里面我们先左右, 后上下,排序的依据是.钞票左右是长边,更容易产生直线误差.所以左右偏移的candidate更容易得到更好效果!!!!!!经验上来说左右的偏移上出错更多.
                            candidate.sort(key=lambda x: (x[1]-tmppoint[1])**2)
                            candidate.sort(key=lambda x: (
                                x[0]-tmppoint[0])**2+(x[1]-tmppoint[1])**2)

                            for i3 in candidate:
                                round2 = [[i, j] for i in range(
                                    i3[0]-1, i3[0]+2) for j in range(i3[1]-1, i3[1]+2)]

                                # sedu=[sum(fffff[i4[1],i4[0]])>100 if (i4[1]<=fffff.shape[0] and i4[0]<=fffff.shape[0]) else 0 for i4 in round2] # 周围9个像素的色度.
                                sedu = []
                                for i4 in round2:
                                    if (i4[1] <= fffff.shape[0] and i4[0] <= fffff.shape[1]):
                                        # sedu.append(sum(fffff[i4[1],i4[0]])>50)
                                        # 使用binary来判断是否存在根准确.
                                        sedu.append(gray[i4[1], i4[0]] > 100)
                                    else:
                                        sedu.append(0)

                                all_sedu = sum(sedu)
                                # 加判断, 候选点本身也要有亮度!, 我们让他正好踩到图像的边.????????????????????????这个地方设置1,2,3????????看看实际效果吧, 总感觉2可能更好. 还是1好,测试了.
                                if all_sedu >= 1 and sum(fffff[i3[1], i3[0]]) >= 0:
                                    all3[i8] = i3
                                    # print(1)
                                    # print(i3,76867867867)
                                    break

                    # print('处理后的四点',all3,)

                    for i in all3:
                        tmp3 = cv2.circle(
                            tmp3, (round(i[0]), round(i[1])), 00, (255, 255, 255), 0)
                    if debug:
                        cv2.imwrite('13里面方法2处理后的四角.png', tmp3)

                    tmp3 = fffff999
                    for i in all3:
                        tmp3 = cv2.circle(
                            tmp3, (round(i[0]), round(i[1])), 00, (255, 255, 255), 0)

                    import time
                    import os
                    dex = int(time.time())

                    import os

                    a789 = os.path.basename(os.path.splitext(name)[0])
                    # print(a789)
                    if debug:
                        def makedir(path):
                            dir_path = os.path.dirname(path)  # 获取路径名，删掉文件名

                            os.makedirs(dir_path, exist_ok=True)
                        path = f'ceshijianwei/yuanshi_debug_qietu/{a789}.png'
                        makedir(path)  # ======归一化
                        cv2.imwrite(path, tmp3)

                    if debug:

                        cv2.imwrite('13里面方法2处理后的四角带着直线.png', tmp3)

                    # =======因为平行肯定有一个线超长.
                    # contours,hierarchy = cv2.findContours(binary2,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)
                    # tuxingzhouchang=cv2.arcLength(contours[0], True)
                    # #print(1)
                    # paixu jike

                    # print(time.time()-start,841)
                    # lines = cv2.HoughLines(edges,1,np.pi/180,100)
                    # ====================下面我们做仿射变换即可.
                    all3 = np.array(all3)[:, None, ...]
                    # print(1)
                    approx = all3
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
                    # sort_point : 左上, 左下, 右下,右上.
                    h = (sort_point[1][1] - sort_point[0][1])**2 + \
                        (sort_point[1][0] - sort_point[0][0])**2
                    h = math.sqrt(h)
                    w = (sort_point[2][0] - sort_point[1][0])**2 + \
                        (sort_point[2][1] - sort_point[1][1])**2
                    w = math.sqrt(w)
                    pts2 = np.array(
                        [[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
                    h = round(h)
                    w = round(w)
                    # ===========面积合法性检测:
                    if h*w < 125*290:
                        raise

                    # print(time.time()-start,867)
                    M = cv2.getPerspectiveTransform(p1, pts2)

                    dst = cv2.warpPerspective(old_image, M, (w, h))
                    # #print(dst.shape)
                    # print(time.time()-start,872)
                    if 0:
                        def show(image, window_name):
                            # cv2.namedWindow(window_name, 0)
                            cv2.imwrite(window_name+'.png', image)

                        if w < h:
                            dst = np.rot90(dst)

                        show(dst, '13里面方法2的最后图片')
                    houcaijian = 1
                    if houcaijian:
                        dst = dst[1:-1, 1:-1]
                    if savename:
                        cv2.imwrite(savename, dst)

                    # print(time.time()-start,881)
                    if dst.shape[0] < 125 or dst.shape[1] < 290:
                        raise
                    return dst  # 如果成功直接return了.
        except Exception as e:
            print(f'{yuzhi999}报错!!!!!!!!!!!')
            print(e)
            print('打印完报错')
        #
    print('如果运行到这里,说明2个阈值20,10都报错,说明算法有问题直接raise')
    raise


if __name__ == '__main__':
    from pathlib import Path
    x = 'saving100v15ZZtrue'
    # a=[y for y in Path(x).rglob('*R_F.bmp') if y.is_file()][:]
    a = [y for y in Path(x).rglob('*.bmp') if y.is_file()]
    print(len(a))
    main('saving100v15ZZtrue/12.bmp', 'output99999.png')
    # aaaaaa='100V15/2/1104133213_53/F_R_F.bmp'
