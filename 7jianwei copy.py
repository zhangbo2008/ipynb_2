# 2023-09-27,10点00 用图像做差来做真假鉴别.
from pathlib import Path
import os
import numpy as np
import cv2
from m26 import main
import builtins


def round(x):
    return builtins.round(x+1e-10)
# 真钱yangben :


x = 'saving100v15ZZtrue'
# a=[y for y in Path(x).rglob('*R_F.bmp') if y.is_file()][:]
a = [y for y in Path(x).rglob('*.bmp') if y.is_file()][7878:]
print(len(a))
# print(1)
# jibai:


x = 'jiabi_2/100V15'
b = [y for y in Path(x).rglob('*R_F.bmp') if y.is_file()][:]
print(len(b))
# print(1)

# print(1)

a1 = [main(str(i)) for i in a]
a2 = [i.shape for i in a1]
a3 = [cv2.resize(i, (304, 130)) for i in a1]
print(1)

b1 = [main(str(i)) for i in b]
b2 = [i.shape for i in a1]
b3 = [cv2.resize(i, (304, 130)) for i in b1]
print(1)


all1 = a1+b1
all1 = a3+b3

# 2023-09-29,8点24 我们取消二值化看看效果.


def makedir(dir_path):
    dir_path = os.path.dirname(dir_path)  # 获取路径名，删掉文件名

    os.makedirs(dir_path, exist_ok=True)


all2 = []
shuaijianall = []


# 预处理,去掉穿透线. 那块黑色的.
def quheitiao(aaa):

    a = aaa
    gaodu = a.shape[0]
    changdu = a.shape[1]
    # print(gaodu,changdu)

    import numpy as np
    maskxian = []
    heitiao = []
    heitiao2 = []
    heitiao3 = []
    gaodujilu = []
    for i3 in range(10, changdu-10):  # ======左右去掉10像素条.
        tmp = a[:, i3]
        # ========对这个数组进行分析.
        # print(1)
        alldexdayu0 = np.where(tmp < 50)[0]
        alldexdayu10 = np.where(tmp < 110)[0]
        alldexdayu15 = np.where(tmp < 92)[0]
        if len(alldexdayu0) > 40 and alldexdayu0[0] < 20:

            heitiao.append(i3)
        # 条符合足够长的灰色,并且不能有黑色. 小鱼50的部分.保证保留100这个数字.
        if len(alldexdayu10) > 130/2 and alldexdayu10[0] < 3 and alldexdayu10[-1] > gaodu-3 and not len(alldexdayu0) > 4:
            heitiao2.append(i3)

        if 2 < len(alldexdayu15) < 8 and alldexdayu15[0] > 90 and alldexdayu15[-1] < gaodu-20:
            heitiao3.append(i3)
            gaodujilu.append((np.max(alldexdayu15)+np.min(alldexdayu15))/2)
    shuaijian = 0
    junzhi = round(np.mean(a))
    if heitiao:
        a[:, min(heitiao)-4:max(heitiao)+4] = junzhi
    if heitiao2:  # ===============这里用max, min也不太对.
        if max(heitiao2)-min(heitiao2) >= 3:  # 保持细条!!!!!!!!!!!!!!!!!!!!!!
            # heitiao2继续处理. #去掉离中位数太远的.去噪
            heitiao2 = np.array(heitiao2)
            zhognweishu = np.median(heitiao2)
            heitiao2 = [i for i in heitiao2 if abs(i-zhognweishu) < 12]

            if heitiao2:

                a[:, min(heitiao2)-5:max(heitiao2)+5] = junzhi
                a10 = max(heitiao2)-min(heitiao2)+2
                shuaijian = min(heitiao2)-a10*2, max(heitiao2)+a10*2
    gaodu = 0
    if heitiao3 and gaodujilu:
        gaodu = sum(gaodujilu)/len(gaodujilu)
        if 25 <= gaodu <= 31 or 97 <= gaodu <= 103:
            shuchu = gaodu
        if gaodu < 25:
            gaodu = 28
        if 80 > gaodu >= 31:
            gaodu = 28
        if 97 > gaodu >= 80:
            gaodu = 100
        if gaodu > 103:
            gaodu = 100
        gaodu = round(gaodu)
        # 28 左右   或者100左右.
    if heitiao3:
        a[gaodu-5:gaodu+5, max(min(heitiao3)-24, 0):max(heitiao3)+14] = junzhi
    # cv2.imwrite('fffffffffffffffffffffffffffff.png',a)
    return a, shuaijian


for dex, i in enumerate(all1):
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    ffffffffffff = gray
    grayguiyihuahou = gray
    if 0:  # =========方差效果不好, 因为钱的新旧差别太大了.
        guiyi = (np.median(gray))

        gray = gray.astype(float)  # =======均值归一化到127=255/2
        gray = gray+127-guiyi  # ============会发生越界255的情况. 还是要转int才行
        # 下面我们加入方差优化.因为有的色重, 就需要全变轻.########后续可考虑直方图等更细致的归一化.
        gray = np.clip(gray, 0, 255)

        gray = gray.astype(np.uint8)
        grayguiyihuahou = gray
    if 1:  # ========进行方差均值归一化.
        grayf = gray.astype(np.float)
        # print('fangcha  ',np.std(grayf))
        graynormed = (grayf-np.mean(grayf))/np.std(grayf)
        graynormed2 = graynormed*20+127
        graynormed2 = np.clip(graynormed2, 0, 255)
        grayguiyihuahou = np.round(graynormed2).astype(np.uint8)
        # print(1)
    gray = grayguiyihuahou.copy()

    # gray = i
    if 0:
        # ========================再优化一下二值化!!!!!!!!! 图片6,8 效果去分部不高.
        ret, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ==========以后需要测试一下不用二值化是否会更好.
        # if dex%2==0:
        #     path=f'ceshijianwei/zhengmian/{dex}.png'
        # else:
        #             path=f'ceshijianwei/fanmian/{dex}.png'
        # makedir(path) # ======归一化
        gray = binary
    gray2 = quheitiao(gray)
    gray = gray2[0]
    shuaijian = gray2[1]
    all2.append(gray)
    shuaijianall.append(shuaijian)
    # cv2.imwrite(path,gray)

    # =======保存原始图debug
    path = f'ceshijianwei/yuanshi_after_quheitiaoandguiyihua/{dex}.png'
    makedir(path)  # ======归一化

    cv2.imwrite(path, gray)

    path = f'ceshijianwei/yuanshi_after_guiyihua/{dex}.png'
    makedir(path)  # ======归一化

    cv2.imwrite(path, grayguiyihuahou)

    path = f'ceshijianwei/yuanshi/{dex}.png'
    makedir(path)  # ======归一化

    cv2.imwrite(path, ffffffffffff)

    # cv2.imwrite('tmp.png',ffffffffffff)
    # cv2.imwrite('tmp1.png',grayguiyihuahou)
    # cv2.imwrite('tmp2.png',binary)

    # print(1)


# =======制作模板
bianyuanzhi = 6
zhengmianmuban = all2[0][bianyuanzhi:-bianyuanzhi, bianyuanzhi:-bianyuanzhi]
beimianmuban = all2[1][bianyuanzhi:-bianyuanzhi, bianyuanzhi:-bianyuanzhi]

cv2.imwrite('2222222222zhengmianmuban.png', zhengmianmuban)
cv2.imwrite('2222222222222beimianmuban.png', beimianmuban)


# 前多少个作为模板.
mubangeshu = 8
mubanall = []
for i in range(mubangeshu):
    mubanall.append(all2[i][bianyuanzhi:-bianyuanzhi,
                    bianyuanzhi:-bianyuanzhi])


for dex, i in enumerate(all2):
    # =计算i跟zhegnmianbeimianmuban的差.
    # 上4,下4,中间.
    cha = []
    chasave = []

    saveforminijuli = float('inf')
    saveforbestdifftupian = 0
    for muban in mubanall:
        for hang in range(10):
            for lie in range(10):

                # muban对齐i图片的(hang,lie)
                # yiqi qiege
                dangqian = i[hang:, lie:,]
                shape1 = min(dangqian.shape[0], muban.shape[0])
                shape2 = min(dangqian.shape[1], muban.shape[1])
                dangqian = dangqian[:shape1, :shape2]
                tmpmuban = muban[:shape1, :shape2]

            #   cv2.imwrite('aaaaaa.png',dangqian)
            #   cv2.imwrite('bbbbbb.png',tmpmuban)
                dangqian = dangqian.astype(int)
                tmpmuban = tmpmuban.astype(int)
                aaaa = np.abs(tmpmuban-dangqian)
                aaaa = aaaa.astype(np.uint8)
            #   cv2.imwrite('ccccc.png',aaaa)
            #   if dex==3:
            #       print(1)
                # ==================后处理!!!!!!!!!!!!!
                # ========处理aaaa图.
                if 0:
                    import cv2
                    a = aaaa
                    # print(a.shape[:2])
                    gaodu = a.shape[0]
                    changdu = a.shape[1]
                    # print(gaodu,changdu)

                    import numpy as np
                    maskxian = []
                    for i3 in range(changdu):
                        tmp = a[:, i3]
                        # ========对这个数组进行分析.
                        # print(1)
                        alldexdayu0 = np.where(tmp > 100)[0]
                        #
                        baitiaochangdu = len(alldexdayu0)
                        if baitiaochangdu:
                            zuishangmian = min(alldexdayu0)
                            zuixiamian = max(alldexdayu0)

                        # ==================这里加入我们的判定条件.
                        # ygie 光条是我们的mask的条件是, 足够长.并且首尾接近0和gaodu
                            yuzhi = 30
                            pandingyi = zuishangmian < yuzhi/118*gaodu
                            pandinger = abs(zuixiamian-gaodu) < yuzhi/118*gaodu
                            pandingsan = baitiaochangdu > 5*8/118*gaodu
                            # print(1)

                            if pandingyi and pandinger and pandingsan:
                                maskxian.append(i3)
                    # print(maskxian)

                    # ===========修改aaaa即可
                    aaaa[:, maskxian] = 0
                # ==============加入模糊去掉不中要的细节.
                # =================一圈的2个像素也mask掉, 毛边没意义讨论.
                # ================引入衰减系数!!!!!!!!!!!!!!!!!!!!!

                shuaijian = shuaijianall[dex]
                if shuaijian:
                    b = np.clip(aaaa[:, shuaijian[0]:shuaijian[1]]*0.2, 0, 255)
                    b = np.round(b).astype(np.uint8)
                    aaaa[:, shuaijian[0]:shuaijian[1]] = b

                aaaa[:2] = 0
                aaaa[-2:] = 0
                aaaa[:, :2] = 0
                aaaa[:, -2:] = 0

                aaaa = cv2.medianBlur(aaaa, 5)
            #   aaaa = cv2.medianBlur(aaaa, 5)
            #   if dex==3:
            #     cv2.imwrite('ccccc.png',aaaa)
            #     print(1)
            #   aaaa=aaaa>10

                # ===========差距太小的也干掉
                aaaa[aaaa < 10] = 0

                kkk = np.sum(aaaa)
                if kkk < saveforminijuli:
                    saveforminijuli = kkk
                    saveforbestdifftupian = aaaa
            #   cha.append(np.sum(aaaa))
            #   chasave.append(aaaa) # chasave 用来保存diff图片.

    # ===============用来查看最接近的那个diff图像的可视化.
    # chadex=min(cha)
    # if 1:
    #  for dex1,i1 in enumerate(cha):
    #      if i1==chadex:
    if 1:

        # ============这里进行尝试去噪!!!!!!!

        # ============删除竖直光线. 因为这个正样本也乱动. 没有参考意义.
        # =========目前我的建议是.diff图片再进行去噪和消除竖直光线.
        # ==============

        # print('选择的索引是',dex1)
        # cv2.imwrite(f'ceshijianwei/chaju/{dex}-0.png',chasave[dex1])
        # quzaoqian=chasave[dex1]
        # quzaoqian=cv2.medianBlur(quzaoqian, 3)
        path = f'ceshijianwei/chaju/{dex}.png'
        makedir(path)
        cv2.imwrite(path, saveforbestdifftupian)

    print(saveforminijuli, f'===============图片索引是{dex}')
print(1)


if 0:
    out = []
    for i in all1:
        for j in all1:
            out.append(np.sum(i-j) if np.sum(i-j) else float('inf'))
    np.set_printoptions(suppress=True)
    out = np.array(out).reshape(len(all1), -1)
    out2 = out.min(axis=1)
    print(out2)
    import numpy as np
    np.savetxt('test.csv', out, delimiter=",")
