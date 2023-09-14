#=============纠正图片.
#============效果很好, 但是生成的图片不止1个.因为毕竟直线识别跟背景会多识别.再后续加算法去除即可.
# 不如看ocr效果来去除.





# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import open_vocab_seg
import multiprocessing as mp

import numpy as np
from PIL import Image

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available()else "cpu") 

from torch.nn import functional as F
import cv2
try:
    import detectron2
except:
    import os
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling.postprocessing import sem_seg_postprocess

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
# from open_vocab_seg.utils import VisualizationDemo, SAMVisualizationDemo
import open_clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry 
from open_vocab_seg.modeling.clip_adapter.adapter import PIXEL_MEAN, PIXEL_STD
from open_vocab_seg.modeling.clip_adapter.utils import crop_with_mask

from detectron2.data import MetadataCatalog
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry 
import open_clip
#===========chognxie 



class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)

            from IPython.display import display
            from PIL import Image
            # #print('画mask图像')
            # display(Image.fromarray((sem_seg == label)))
  


            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output
import numpy as np

#======从mask 变到 图像的四角定位.

import cv2  #######边缘检测.
import argparse
import imutils
import cv2

import numpy as np




import argparse
import imutils
import cv2

import cv2
import numpy as np
class SAMVisualizationDemo(object):
    def __init__(self, cfg, granularity, sam_path, ovsegclip_path, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.granularity = granularity
        sam = sam_model_registry["vit_l"](checkpoint=sam_path).to(device)
        self.predictor = SamAutomaticMaskGenerator(sam, points_per_batch=16)#加载sam模型.
        # self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ovsegclip_path)#加载clip模型
        # #print(self.clip_model,'加载了clip模型')
    def run_on_image2(self, ori_image, class_names):#只返回mask
        height, width, _ = ori_image.shape
        if width > height:
            new_width = 1280
            new_height = int((new_width / width) * height)
        else:
            new_height = 1280
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('最终变换后的原图在上面可以画四角.png',image)
        # ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # ori_image原始图片, images: 1280的图片.
        # visualizer = OVSegVisualizer(ori_image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)#==============这个只是用来最后可视化图像的.
        #利用sam看来生成图片里面全部的maks.
        #print('运行网络')
        with torch.no_grad():
            masks = self.predictor.generate(image) # 必须放缩, 不然分辨率卡死了.
        # #print(masks,9999999999999999999999999999999999999999999999999999)
        #======去掉过小的部分.
        all_area=new_width*new_height
        zuixiao_area=max(all_area/20,1000)

        masks=[ masks[i]  for i in range(len(masks))  if masks[i]['area']>zuixiao_area]


        import numpy as np
        pred_masks = [masks[i]['segmentation'][None,:,:] for i in range(len(masks))]
        pred_masks = np.row_stack(pred_masks)
        # #print(pred_masks)
        pred_masks = BitMasks(pred_masks)
        # #print(pred_masks)
        bboxes = pred_masks.get_bounding_boxes() # 这是detectron自己的代码.

        # 4个mask 4个bboxes

        for i in range(len(bboxes)):
            tupiansuoyin=i
            tmpmask=[masks[i]['segmentation'] for i in range(len(masks))][i].astype(np.uint8)*255  
            tmp_bbox=bboxes[i]

            dummy_img= np.array([[False,False,False,False]
                ,[False,True,True,False]
                ,[False,True,True,False]
                ,[False,False,False,False]

                ]).astype(np.uint8)*1             #######输入自定义mask
            img=tmpmask
            if 1: #=====================自己写的后处理算法, 非常复杂.
                import numpy as np
                kernel = np.ones((1, 5), np.uint8)
                if len(img.shape)>2:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
                else:
                    gray=img
                cv2.imwrite('gray.png',gray)
                ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
                binary2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=22) # 二值化. 
                #模糊化.

                



                binary = cv2.blur(img,(5,5))
                binary = cv2.blur(binary,(5,5))
                binary = cv2.blur(binary,(5,5))
                binary = cv2.boxFilter(binary,-1,(5,5), normalize=True )
                cv2.imwrite(str(tupiansuoyin)+'分割算法之后的各个小图片.png',binary)
                if 1:                     
                # 边缘检测, Sobel算子大小为3
                    edges = cv2.Canny(binary, 100, 200, apertureSize=3)
                    # 霍夫曼直线检测
                    cv2.imwrite('binary997.png',edges )

                    gao=edges.shape[0]
                    chang=edges.shape[1]




                    lines = cv2.HoughLinesP(edges, 1, 1*np.pi / 180, int((gao+chang)/40), minLineLength=(gao+chang)/20, maxLineGap=(gao+chang)/20)


                    if len(lines)>=4:
                #setp1: 现在各个直线.进行分成4类. 然后再拟合4个直线.!!!!!!!!!!!!!!!
                        #=======首先算出 每个直线的旋转角度.
                        #====直接算斜率
                        # xielv=[]
                        # for i in lines:
                        #     xielv.append('inf' if not i[0][3]-i[0][1] else (i[0][3]-i[0][1])/(i[0][2]-i[0][0]))
                        #     #print()

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
                        jiajiaobaocun=a

                        import matplotlib.pyplot as plt
                        import numpy as np
                        # import libraries
                        import numpy as np
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        a=[round(i) for i in a]
                        #print(a)
                        plt.hist(a, density=False)
                        plt.savefig('aaaa.png')
                        #print()
                        # 按照相差上下10度来分类.

                        #============下面按照上下15度进行分类.因为每一个类别一定有一个中间轴.中间轴一定是这些店里面的值.我们来存索引.
                        baocun=0
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
                            pass
                            print('没有任何一个直线,所以算法不进行后续边界识别')
                        if baocun:
                            j=a[baocun[0]]
                            list1=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
                            j=a[baocun[1]]
                            list2=[dex for dex,i in enumerate(a) if min(abs(i-j),abs(180+i-j),abs(180-i+j))<yuzhi]
                        #print('打印两组直线角度阵营',list1,list2)
                        #=========分别算投影, 去掉一个方向分量之后我们进行第二次细分这2住店.这样就得到了4个边的阵营.


                        
                        
                        zhixianfenzu=[]
                        a1=list1
                        a2=list2
                        for aaa in [a1,a2]:
                            #======算出每个阵营的投影直线.
                            #==先算每个阵营的中心直线
                            zhenying=aaa
                            zhenyingjiaodu=[a[i] for i in list1]
                            zhenyingzhixianjiaodu=sum(zhenyingjiaodu)/len(zhenyingjiaodu)
                            #print(zhenyingzhixianjiaodu)
                            chuizhijiaodu=zhenyingzhixianjiaodu+90


                            a=math.tan(chuizhijiaodu/180*math.pi)
                            xiangliang=(1,a*1)
                            list1zhongdian=[[(lines[i][0][0]+lines[i][0][2])/2,(lines[i][0][1]+lines[i][0][3])/2] for i in list1]
                            touying=[(i[0]*xiangliang[0]+i[1]*xiangliang[1])/math.sqrt(xiangliang[0]**2+xiangliang[1]**2) for i in list1zhongdian]



                            baocun=0
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
                                j=a[baocun[0]]
                                list1=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]
                                j=a[baocun[1]]
                                list2=[dex for dex,i in enumerate(a) if abs(i-j)<yuzhi]

                                #print(1)
                                list1inalldex=[zhenying[i] for i in list1]
                                list2inalldex=[zhenying[i] for i in list2]
                                zhixianfenzu.append(list1inalldex)
                                zhixianfenzu.append(list2inalldex)
                            #print()
                        #print()
                        zhixianfenzu# 里面有4个数组, 每个数组表示一个直线族. 
                        #=========下面把每组的直线拟合成一条直线
                        #==================
                        all_four_line=[]
                        if len(zhixianfenzu)==4:
                            pass
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
                                a=math.tan(jiaodu/180*math.pi)
                                all_four_line2.append([dian[0],dian[1],dian[0]+1,dian[1]+a])
                        all_four_line=all_four_line2
                        #===========计算交点
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
                        #=======算每一个直线跟其他直线的交点
                        #diyige :
                        if len(all_four_line)>=4:
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

                            #=====================check!!!!!!!!!!
                            for line in lines:
                            # 获取坐标 
                                x1, y1, x2, y2 = line[0]
                                # cv2.line(binary, (x1, y1), (x2, y2), (0, 255, 255), thickness=25)
                            cv2.imwrite('binary999.png',binary)


                            #==============画点.
                            if 0: # 原图画点.
                                binary2=binary
                                binary=cv2.imread('debug99k.png')
                                binary=cv2.resize(binary,binary2.shape[:2][::-1])
                                cv2.imwrite('debug1003.png',binary)
                            binary=gray
                            if len(all3)==4:
                                # for i in all3:
                                #     binary=cv2.circle(binary,(int(i[0]),int(i[1])),40,(255,255,255),2)
                                print(all3,'所有的定位点!!!!!!!!!!!!最终的')
                                # cv2.imwrite('debug1002.png',binary)
                            #====================下面我们做仿射变换即可.
                            all3=np.array(all3)[:,None,...]
                            print(1)
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
                            h=int(h)
                            w=int(w)
                            M = cv2.getPerspectiveTransform(p1, pts2)
                            gray=cv2.imread('debug0.png')
                            dst = cv2.warpPerspective(gray, M, (w, h))
                            # print(dst.shape)
                            def show(image, window_name):
                                # cv2.namedWindow(window_name, 0)
                                cv2.imwrite(window_name+'.png', image)

                            if w < h:
                                dst = np.rot90(dst)
                            show(dst, str(tupiansuoyin)+"dst2")
                            print('最终我们图片保存在',str(tupiansuoyin)+"dst2")




















#             if 0:       
#                 #就是上一章的内容，具体就是会输出一个轮廓图像并返回一个轮廓数据
#                 def draw_contour(img,color,width):
#                     if len(img.shape)>2:
#                         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换灰度图
#                     else:
#                         gray=img
#                     ret , binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)#转换成二值图
#                     contour,hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#寻找轮廓，都储存在contour中
#                     copy = img.copy()
#                     resoult = cv2.drawContours(copy,contour[:],-1,color,width)
#                     # cv2.imwrite('resoult.png',resoult)
#                     return contour
                
                
#                 contour = draw_contour(img,(0,0,255),2)
#                 #print(type(contour))    
#                 epsilon = 0.01*cv2.arcLength(contour[0],True)
#                 approx = cv2.approxPolyDP(contour[0], epsilon, True)
#                 #print(type(approx),approx.shape)
                
#                 res = cv2.drawContours(img.copy(),[approx],-1,(0,255,255),2)
#                 cv2.imwrite('res.png',res) #打印最后的多边形矿.
#                 #print(1)




#         #print(1111111111)








# #=========image 上进行画画.
#         for  dex,i in enumerate(masks):

#             i=np.array(i['segmentation'])[:,:,None]
#             #print(i.shape)
#             tmp=1
#             cv2.imwrite('debug'+str(dex)+'.png',(image*i))

#         return masks


        
    def run_on_image(self, ori_image, class_names):
        height, width, _ = ori_image.shape
        if width > height:
            new_width = 1280
            new_height = int((new_width / width) * height)
        else:
            new_height = 1280
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # ori_image原始图片, images: 1280的图片.
        visualizer = OVSegVisualizer(ori_image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)#==============这个只是用来最后可视化图像的.
        #利用sam看来生成图片里面全部的maks.
        with torch.no_grad():
            masks = self.predictor.generate(image)
        # #print(masks,9999999999999999999999999999999999999999999999999999)
        #======去掉过小的部分.
        all_area=new_width*new_height
        zuixiao_area=max(all_area/20,1000)

        masks=[ masks[i]  for i in range(len(masks))  if masks[i]['area']>zuixiao_area]




        pred_masks = [masks[i]['segmentation'][None,:,:] for i in range(len(masks))]
        pred_masks = np.row_stack(pred_masks)
        # #print(pred_masks)
        pred_masks = BitMasks(pred_masks)
        # #print(pred_masks)
        bboxes = pred_masks.get_bounding_boxes()
        # #print(88888888888888888,bboxes)

        mask_fill = [255.0 * c for c in PIXEL_MEAN]
        # #print(111111111111111111111,mask_fill)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        regions = []
        for bbox, mask in zip(bboxes, pred_masks):
            region, _ = crop_with_mask(
                image,
                mask,
                bbox,
                fill=mask_fill,
            ) #用fill 填充新的图片,然后  根据box 抽取出来部分.
            #计算方法:#原图box部分乘以要的部分. 加上 原图box部分*maks不要的部分*填充值.
            regions.append(region.unsqueeze(0))
        regions = [F.interpolate(r.to(torch.float), size=(224, 224), mode="bicubic") for r in regions]












        pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
        imgs = [(r/255.0 - pixel_mean) / pixel_std for r in regions]
        imgs = torch.cat(imgs)#=============各个box的图片
        if len(class_names) == 1:
            class_names.append('others')
        txts = [f'a photo of {cls_name}' for cls_name in class_names]

        # txts=class_names
        text = open_clip.tokenize(txts)
        img_batches = torch.split(imgs, 32, dim=0)#0维度每32个分一组.










        with torch.no_grad():
            self.clip_model.to(device)
            text_features = self.clip_model.encode_text(text.to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = []
            for img_batch in img_batches:
 
                image_feat = self.clip_model.encode_image(img_batch.to(device))
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat.detach())
            image_features = torch.cat(image_features, dim=0)#image_features 得到所有的图片特征.
            tmp=image_features @ text_features.T#================            #=============保留绝对值大于16的
            # #print(tmp,'打印置信度!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            tmp[tmp<0.15]=0
            # #print('fdsklfjasdkljfadlsjfkladsjfkladsjflkajsd',tmp)
            class_preds = (100.0 * tmp).softmax(dim=-1)
            #=============保留绝对值大于16的






#================================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            class_preds[class_preds<0.7]=0 #===========只保留概率大于0.9的 # 阈值还是要给, 不然低准读的大量也能进行干扰.


            # #print(1, image_features.shape)
            # #print(2, class_preds.shape)
            # #print(3, text_features.shape)
            # 1 torch.Size([27, 768])
            # 2 torch.Size([27, 2])   class_preds
# 3 torch.Size([2, 768])
            #class_preds每行一个图片, 一行表示这个图片在各个class上的分类概率 .


        select_cls = torch.zeros_like(class_preds)


















        max_scores, select_mask = torch.max(class_preds, dim=0) #保持dim=0的维度. 所以
        # #print(4,max_scores.shape) # 这个是class_num. 
        if len(class_names) == 2 and class_names[-1] == 'others':#如果只有一个分类
            select_mask = select_mask[:-1] #那么select去掉最后一个.
        if self.granularity < 1: # 置信度.如果小于1, 那么就对结果进行拓展, 一般来说写1即可.
            thr_scores = max_scores * self.granularity #阈值.
            select_mask = []
            if len(class_names) == 2 and class_names[-1] == 'others':
                thr_scores = thr_scores[:-1]
            for i, thr in enumerate(thr_scores):
                cls_pred = class_preds[:,i]
                locs = torch.where(cls_pred > thr)
                select_mask.extend(locs[0].tolist()) #select_mask进行拓展.
        for idx in select_mask:
            select_cls[idx] = class_preds[idx]
        # #print(select_cls.shape, 666666666666) # 最新的分类概率.
        # #print(pred_masks.tensor.shape, 666666666666) # 

# torch.Size([27, 2]) 666666666666
# torch.Size([27, 960, 1280]) 666666666666  27个图片,每个图片是960 1280的mask
        #====================================
        #select_cls加一个过滤


        semseg = torch.einsum("qc,qhw->chw", select_cls.float(), pred_masks.tensor.float().to(device)) #得到每一个类别的mask. 也就是27张图片在每一个分类上的mask叠加.
        #print(semseg)
        # #print('把tensor结果写入txt中进行查看细节',np.savetxt('2222222',  semseg.cpu().detach().numpy()))
        
        #print(324234234234,semseg.shape) #torch.Size([2, 960, 1280])
        r = semseg
        blank_area = (r[0] == 0) #黑色区域.
        #print(blank_area.shape, 9999999999999) # torch.Size([960, 1280])
        #print(r.shape, 32423423423423423) # 
#         torch.Size([2, 960, 1280]) 32423423423423423

        pred_mask = r.argmax(dim=0).to('cpu') #========这里面写入的是0,1,2这种class信息.
        #print('把tensor结果写入txt中进行查看细节',np.savetxt('111111111111111111111111',  pred_mask.detach().numpy()))
        #print(len(np.array_str(pred_mask.detach().numpy())),3243242342343888888888888888888888888888888888888888888888888888888888888888888888)
        #print(np.array_str(pred_mask.detach().numpy())[:1000],43333333333333333333333333333333)
        #print(pred_mask,32222222222222222222222222222222222222222222222222222222222222222222)
        #print(pred_mask.shape, 32423423423423423) #   # torch.Size([960, 1280]) 32423423423423423
        #print(9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999)
        pred_mask[blank_area] = 255
        pred_mask = np.array(pred_mask, dtype=np.int)
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST) #还原到原始图片大小.
        #print(324234234234234,width,height)
        vis_output = visualizer.draw_sem_seg(
            pred_mask
        )





        from IPython.display import display
        from PIL import Image
        # display(Image.fromarray(pred_mask))







        display((vis_output.fig))
        # display(Image.fromarray(np.uint8(vis_output.get_image())).convert('RGB'))
        return None, vis_output


import gradio as gr

import gdown

# ckpt_url = 'https://drive.google.com/uc?id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy'
# output = './ovseg_swinbase_vitL14_ft_mpt.pth'
# gdown.download(ckpt_url, output, quiet=False)

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def inference(class_names,  granularity, input_img):
    mp.set_start_method("spawn", force=True) #设置多进程.
    config_file = './ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)
    # #print(cfg,'看看cfg')
    demo = SAMVisualizationDemo(cfg, granularity, '/mnt/e/sam_vit_l_0b3195.pth', '/mnt/e/ovseg_clip_l_9a1909.pth')#=======后续有机器可以改大模型.
    class_names = class_names.split(',')
    img = read_image(input_img, format="BGR")

    def  imgRotation(pathtoimg):
        #print('旋转修复的图片是',pathtoimg)
        #图片自动旋正
        from PIL import Image
        img = Image.open(pathtoimg) 
        path=pathtoimg
        new_img=cv2.imread(path)
        if hasattr(img, '_getexif') and img._getexif() != None:
            # 获取exif信息
            dict_exif = img._getexif()
            if 274 in dict_exif:
                if dict_exif[274] == 3:
                    #顺时针180
                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_180)
            
                elif dict_exif[274] == 6:
                    #顺时针90°
                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif dict_exif[274] == 8:
                    #逆时针90°

                    new_img = cv2.imread(path)
                    new_img=cv2.rotate(new_img,cv2.ROTATE_90_CLOCKWISE)
        return new_img









    image=imgRotation(input_img)



    a= demo.run_on_image2(img, class_names)

    print('over')

    #==========对a的每一个图像进行透视变换.
    # a=[i['segmentation'].view(dtype=np.uint8) for i in a]


    if 0:
        1
    #======保存成图片,再opencv方法.
    # for j in a:
    #     cv2.imwrite('tmp.png',j)
    #     a=cv2.imread('tmp.png')
    #     #print(1)
    #     kernel = np.ones((1, 5), np.uint8)
    #     import imutils
    #     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    #     ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE) 
    #     cv2.imwrite("img2.png", binary)   
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2) # 二值化.
    #     contours = cv2.findContours(binary,cv2.RETR_CCOMP  ,cv2.CHAIN_APPROX_SIMPLE)  # 参数说明;https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71 
    #     contours = imutils.grab_contours(contours) #适配cv2各个版本.
    #     contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
    #     # binary=cv2.drawContours(img,contours,-1,(0,255,255),1)  
    #     # cv2.imwrite("img.png", binary)




    #     epsilon = 0.02 * cv2.arcLength(contours, True)
    #     approx = cv2.approxPolyDP(contours, epsilon, True)
    #     n = []
    #     for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
    #         n.append((x, y))
    #     n = sorted(n)
    #     sort_point = []
    #     n_point1 = n[:2]
    #     n_point1.sort(key=lambda x: x[1])
    #     sort_point.extend(n_point1)
    #     n_point2 = n[2:4]
    #     n_point2.sort(key=lambda x: x[1])
    #     n_point2.reverse()
    #     sort_point.extend(n_point2)
    #     p1 = np.array(sort_point, dtype=np.float32)
    #     h = sort_point[1][1] - sort_point[0][1]
    #     w = sort_point[2][0] - sort_point[1][0]
    #     pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

    #     M = cv2.getPerspectiveTransform(p1, pts2)
    #     dst = cv2.warpPerspective(image, M, (w, h))
    #     cv2.imwrite('dst.png',dst)
    #     #print(1)





    return a

#==========开始调试模型
if 1:
  import requests
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#   image = Image.open(requests.get(url, stream=True).raw) #request方式加载数据.
  canshu=0.92 #这个数值0-1, 越大,返回的物品越少. 越小越容易都返回.




  # inference('cat,remote_controller', canshu , url) #用逗号切分classname
  url="https://img2.baidu.com/it/u=3577844526,3962936240&fm=253&fmt=auto&app=138&f=JPEG?w=551&h=500"




  #===============下面进行批量测试:!!!!!!!!!!!!!!!!!
  for url in [
      

# 'data/tt.jpg',
'tq2.png',

  ]:
        inference('gun,violence,women,hero,king', canshu , url) #用逗号切分classname












