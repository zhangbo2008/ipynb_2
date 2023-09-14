import os # gradio 3.24.0
import io
import json
import torch

from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

import os

os.makedirs('saveforsuperuser', exist_ok=True)
UPLOAD_ROOT_PATH='saveforsuperuser'
app = Flask(__name__)
CORS(app)  # 解决跨域问题

# weights_path = "./MobileNetV2(flower).pth"
# class_json_path = "./class_indices.json"

# # select device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# # create model
# model = eval(num_classes=5)
# # load model weights
# model.load_state_dict(torch.load(weights_path, map_location=device))
# model.to(device)
# model.eval()

# # load class info
# json_file = open(class_json_path, 'rb')
# class_indict = json.load(json_file)


def get_prediction(image_bytes):
    try:
        tensor = ""
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)









#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import defaultdict
import re

__all__ = ['NaiveFilter', 'BSFilter', 'DFAFilter']
__author__ = 'observer'
__date__ = '2012.01.05'


class NaiveFilter():

    '''Filter Messages from keywords
    very simple filter implementation
    >>> f = NaiveFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keywords = set([])

    def parse(self, path):
        for keyword in open(path):
            self.keywords.add(keyword.strip().decode('utf-8').lower())

    def filter(self, message, repl="*"):
        message = unicode(message).lower()
        for kw in self.keywords:
            message = message.replace(kw, repl)
        return message


class BSFilter:

    '''Filter Messages from keywords
    Use Back Sorted Mapping to reduce replacement times
    >>> f = BSFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keywords = []
        self.kwsets = set([])
        self.bsdict = defaultdict(set)
        self.pat_en = re.compile(r'^[0-9a-zA-Z]+$')  # english phrase or not

    def add(self, keyword):
        if not isinstance(keyword, unicode):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        if keyword not in self.kwsets:
            self.keywords.append(keyword)
            self.kwsets.add(keyword)
            index = len(self.keywords) - 1
            for word in keyword.split():
                if self.pat_en.search(word):
                    self.bsdict[word].add(index)
                else:
                    for char in word:
                        self.bsdict[char].add(index)

    def parse(self, path):
        with open(path, "r") as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, unicode):
            message = message.decode('utf-8')
        message = message.lower()
        for word in message.split():
            if self.pat_en.search(word):
                for index in self.bsdict[word]:
                    message = message.replace(self.keywords[index], repl)
            else:
                for char in word:
                    for index in self.bsdict[char]:
                        message = message.replace(self.keywords[index], repl)
        return message


class DFAFilter():

    '''Filter Messages from keywords
    Use DFA to keep algorithm perform constantly
    >>> f = DFAFilter()
    >>> f.add("sexy")
    >>> f.filter("hello sexy baby")
    hello **** baby
    '''

    def __init__(self):
        self.keyword_chains = {}
        self.delimit = '\x00'  #这是一个不可见字符,看起来跟''一样,但是实际上不一样! 他作为结尾符很适合!

    def add(self, keyword):
        if not isinstance(keyword, str):
            keyword = keyword.decode('utf-8')
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)): #对字符串里面每一个字符建立trie树.
            if chars[i] in level: #如果当前这个汉子存在,那么就level进入下一层.
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):#走到头了.说明已经存在这个level了.
                    break
                for j in range(i, len(chars)):#建立新的子字典.
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0} #然后写入结束符.
                break
        if i == len(chars) - 1: # 说明已经有过这个字符串,那么我们就写入结束符即可. 比如先add 啊啊啊啊, 再 add 啊, 那么这一样代码就会触发!
            level[self.delimit] = 0

    def parse(self, path):
        with open(path,encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())

    def filter(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return ''.join(ret)




#==============这个函数,输入message,返回 跟trie树中匹配上的所有字符串的start end索引,组成的二维数组.
#==============这个函数,输入message,返回 跟trie树中匹配上的所有字符串的start end索引,组成的二维数组. 这里面end是按照python规范来!==============这个是最短匹配,从头到尾找子串,只要匹配成功就跳过这个成功的. 找后面匹配的部分!
    #==========现在我们把这个最短匹配当做默认情况,因为这个算法是最快的.一般也足够用了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def pipei_shortest(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        out_debug=[]
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        ret.append(repl * step_ins)
                        old_start=start
                        start += step_ins - 1
                        out.append([old_start,start+1])
                        out_debug.append(message[old_start:start+1])
                        break
                else:#如果char不存在,

                    break
            else: # for else: 上面的break都没触发,就走这个else. 说明一直进入到了最后一层.并且里面一直都没有结束符!!!!!说明当前位置字符串只是一个前缀,不能成为单词.所以不是我们要的.
                pass

            start += 1#=========这里也是可以直接跳过.
        print(out_debug)
        return out



#最长匹配, 尽量找跟字典中最长的匹配, 尽可能让找到的字符串最长!!!!!!!!!!!!!!!!!!!性能会比上面的低很多!
    def pipei_longest(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        message = message.lower()
        ret = []
        start = 0

        while start < len(message):
            level = self.keyword_chains
            step_ins = 0#用来记录当前遍历到字典的第几层.
            start2 = None
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        level = level[char]
                        old_start=start
                        start2 =start+ step_ins - 1 #保证找到的不会重叠.


                else:#如果char不存在,
                    # ret.append(message[start])
                    break

            #=================遍历玩了当前字符为起始字符的全排列.
            if start2!=None:
                out.append([start, start2 + 1])
            if start2!=None:
                start=start2+1 #因为已经匹配最长了,直接跳过即可!!!!!!!!!!
            else:
                start += 1

        return out











#全匹配,也是最浪费性能的!!!!!!!!!!!!!!!!!!!!!!!!!!!!性能会比上面的低很多!
    def pipei_all(self, message, repl="*"):
        if not isinstance(message, str):
            message = message.decode('utf-8')
        out=[]
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0#用来记录当前遍历到字典的第几层.
            for index,char in enumerate( message[start:]):
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:#无脑进入.
                        level = level[char]
                    else:#===============匹配成功了!#===========这个地方好像有问题, 如果字典里面有a 也有ab,那么运行到a就停了,不会继续找更长的!!!!!!!!!!!!!!!!
                        level = level[char]
                        old_start=start
                        start2 =start+ step_ins - 1 #保证找到的不会重叠.
                        out.append([old_start,start2+1])

                else:#如果char不存在,
                    # ret.append(message[start])
                    break
            else: # for else: 上面的break都没触发,就走这个else. 说明一直进入到了最后一层.并且里面一直都没有结束符!!!!!说明当前位置字符串只是一个前缀,不能成为单词.所以不是我们要的.
                # ret.append(message[start])
                pass
            start += 1

        return out















def test_first_character():
    gfw = DFAFilter()
    gfw.add("1989年")
    assert gfw.filter("1989", "*") == "1989"




#==================================sb服务.




with open('1.txt' ,'w') as f:
    f.write("""
问：招待外部专家的费用标准？
答：邀请外部专家为公司提供劳务等产生的由公司负担的差旅费、餐费等费用，原则上按照不超过公司职级 P9 的标准执行。

问：对于已收到入职通知书但未入职的员工按照公司要求产生的费用是否可以报销。
答：对于已经接到入职通知的新员工，在接到入职通知后按照公司要求参与公司安排的各项活动所发生的差旅费，依据公司差旅费管理办法予以报销。

问：差旅费的定义。
答：差旅费是指工作人员临时到常驻地以外地区（北京除外）公务出差所发生的城市间交通费、住宿费、伙食费、市内交通费和外埠交通费等。

    """.replace(':',"："))

with open('1.txt'  ) as f:

    tmp=f.read()
print(1)
tmp=tmp.split('\n\n')
print(1)
all_question=[i[:i.find('答：')] for i in tmp]
all_answer=[i[i.find('答：'):] for i in tmp]

print(1)

# Prompt-based MLM fine-tuning
from transformers import BertForMaskedLM, BertTokenizer
import torch










# Prompt-based Sentence Similarity
# To extract sentence representations.
from transformers import BertForMaskedLM, BertTokenizer
import torch

# Loading models
tokenizer=BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese")
model=BertForMaskedLM.from_pretrained("IDEA-CCNL/Erlangshen-TCBert-330M-Sentence-Embedding-Chinese")

# Cosine similarity function
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
all_vec=[]
for dex,i in enumerate(all_question):
    
    with torch.no_grad():

        # To extract sentence representations for training data
        t = tokenizer(i, return_tensors="pt")
        print(t)
        training_outputs = model(**t, output_hidden_states=True)
        training_representation = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)
        print(training_representation.shape)
      

    all_vec.append(training_representation)
# all_vec=torch.vstack(all_vec)
# print(all_vec.shape)

































@app.route("/", methods=["GET", "POST"])
def root():
    txt=request.args.get('txt')
    print(txt,888888888888888888888)
    with torch.no_grad():

            # To extract sentence representations for training data
            t = tokenizer(txt, return_tensors="pt")
            training_outputs = model(**t, output_hidden_states=True)
            t2 = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)

        
    # Calculate similarity scores


    mini=-float('inf')
    dex=-1
    for i in range(len(all_vec)):
        t=cos(all_vec[i], t2)
        print('当前相似度',t)
        if t>mini:
            mini=t
            dex=i

    print('最相似的是',dex,mini)
    #==========大于0.6就返回答案.
    if mini>0.6:
      ans=all_answer[dex][2:]
      print('你的答案是',all_answer[dex][2:])
      return jsonify({'msg':ans})
    return jsonify({'msg':''})


#=========function for gradio




def fresh(filelist):
    global all_answer
    global all_question
    global all_vec
    all_answer=[]
    all_question=[]
    for i1 in filelist:
        with open(i1  ) as f:

            tmp=f.read()
        print(1)
        tmp=tmp.split('\n\n')
        print(1)
        question=[i[:i.find('答：')].replace('\n','')[2:] for i in tmp]
        answer=[i[i.find('答：'):].replace('\n','') for i in tmp]
    all_question+=question
    all_answer+=answer
    #===========去重
    all_question2=[]
    all_answer2=[]
    for i in range(len(all_question)):
        if all_question[i] not in all_question2:
            all_question2.append(all_question[i])
            all_answer2.append(all_answer[i])
    all_answer=all_answer2
    all_question=all_question2






    print('更新后全部问题是',all_question)
    all_vec=[]
    for dex,i in enumerate(all_question):
    
        with torch.no_grad():

            # To extract sentence representations for training data
            t = tokenizer(i, return_tensors="pt")
            
            training_outputs = model(**t, output_hidden_states=True)
            training_representation = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)
            
      

        all_vec.append(training_representation)


def upload(files,chatbot):
    for i in files:
        filename = os.path.split(i.name)[-1]
        shutil.move(i.name, os.path.join(UPLOAD_ROOT_PATH,  filename))
    import glob
    files=glob.glob(UPLOAD_ROOT_PATH+'/*.*')
    os.listdir(UPLOAD_ROOT_PATH)
    fresh(files)
    chatbot+=[[None,'上传完毕']]
    gr.update(choices=get_list())
    refresh_vs_list()
    return chatbot
    pass


def get_answer(query,chatbot):
   
    txt=query
    print(txt,888888888888888888888)
    with torch.no_grad():

            # To extract sentence representations for training data
            t = tokenizer(txt, return_tensors="pt")
            training_outputs = model(**t, output_hidden_states=True)
            t2 = torch.mean(training_outputs.hidden_states[-1].squeeze(), dim=0)

        
    # Calculate similarity scores


    mini=-float('inf')
    dex=-1
    for i in range(len(all_vec)):
        t=cos(all_vec[i], t2)
        print('当前相似度',t)
        if t>mini:
            mini=t
            dex=i

    print('最相似的是',dex,mini)
    #==========大于0.6就返回答案.
    chatbot.append([txt,''])
    if mini>0.6:
      ans=all_answer[dex][2:]
      print('你的答案是',all_answer[dex][2:])
      chatbot[-1][-1]+=ans
      
    chatbot[-1][-1]+=''
    return chatbot
    pass
# if __name__ == "__main__":
#     # gfw = NaiveFilter()
#     # gfw = BSFilter()
#     import time

    

#     app.run(host="0.0.0.0", port=7862)

def refresh_vs_list():
    return gr.update(choices=get_list())
def get_list():
    import glob
    aaa=glob.glob(UPLOAD_ROOT_PATH+'/*.*')
    return aaa

def file_del2(select_vs,chatbot):
    os.remove(select_vs)
    chatbot+=[[None,'删除完毕']]
    gr.update(choices=get_list())
    refresh_vs_list()
    return chatbot
#=======falsk 修改为webui界面.

import gradio as gr
import os
import shutil

import gradio as gr
def update(name):
    return f"Welcome to Gradio, {name}!"

with gr.Blocks() as demo:
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None,'开始使用']],
                                     elem_id="chat-box",
                                     show_label=False).style(height=500)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            if 1:
             with gr.Column(scale=5):
                
             
                    vs_refresh = gr.Button("更新文件列表,重置知识库向量计算")
                    
                    select_vs = gr.Dropdown(get_list(),
                                            label="显示文件列表并选择",
                                            interactive=True,
                                            value=get_list()[0] if len(get_list()) > 0 else None
                                            ) # value是dropdown下拉框的默认值.
                    

                    file2vs = gr.Column(visible=True)
                    with file2vs:
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg'],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件,重置知识库向量计算,(文件重名会覆盖旧文件)")
                    file_del = gr.Button(value="删除选择的文件并重新计算知识库", visible=True)









                    if 1:

                        vs_refresh.click(fn=refresh_vs_list,
                                        inputs=[],
                                        outputs=select_vs)                  

                        load_file_button.click(fn=upload,
                                            inputs=[files,  chatbot, ],
                                            outputs=[  chatbot], show_progress=True,)

                        file_del.click(fn=file_del2,
                                 inputs=[select_vs, chatbot],
                                 outputs=[ chatbot])



                  

                        stream=True
                        query.submit(get_answer,
                                    [query,  chatbot, ],
                                    [chatbot],api_name='aaa')

demo.launch(inbrowser=True,height=300,server_port=7862,server_name='0.0.0.0').queue()