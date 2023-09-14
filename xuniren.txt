import os
import time
from textwrap import dedent

import gradio as gr
import mdtex2html
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer
#====================# 预加载一些数据 放这里.
from models import Wav2Lip
import pickle

with open("./marks", 'rb') as fr:
    face_det_results = pickle.load(fr)
face_det_results=face_det_results
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

modelw = load_model('checkpoints/wav2lip_gan.pth')
print ("Modelw loaded")






# fix timezone in Linux
os.environ["TZ"] = "Asia/Shanghai"
try:
    time.tzset()  # type: ignore # pylint: disable=no-member
except Exception:
    # Windows
    logger.warning("Windows, cant run time.tzset()")

model_name = "THUDM/chatglm2-6b"  # 7x?G
model_name = "THUDM/chatglm2-6b-int4"  # 7x?G
model_name = "THUDM/chatglm-6b-int8"  # 3.9G

RETRY_FLAG = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
has_cuda = torch.cuda.is_available()
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
model = model.eval()
print('语言model loaded')#============服务启动时间都花这上了.
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """Copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/."""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text





response2=''
def predict(
    RETRY_FLAG, input, chatbot, max_length, top_p, temperature, history, past_key_values
):  
    global response2
    try:
        chatbot.append((parse_text(input), ""))
    except Exception as exc:
        logger.error(exc)
        logger.debug(f"{chatbot=}")
        _ = """
        if chatbot:
            chatbot[-1] = (parse_text(input), str(exc))
            yield chatbot, history, past_key_values
        # """
        yield chatbot, history, past_key_values

    for response, history in model.stream_chat(
        tokenizer,
        input,
        history,
        past_key_values=past_key_values,
        return_past_key_values=True,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    ):
        # print('得到的response',response[-1])
        chatbot[-1] = (parse_text(input), parse_text(response))
        past_key_values=None
        yield chatbot, history, past_key_values,None
    print('回答完所有之后函数的结果是',response)#=========在生成yield的循环之后获取即可.
    response2=response
    #==============这个函数后面加上数字人.

    # print('切换视频')
    # # gr.update(value='test6.mp4')
    # gr.videostate.update(value='test6.mp4')
    # # videostate.update(value='test6.mp4')
    # print('切完')
    # print('切完')
    # print('切完')
    # print('切完')




    #
    if 0:
        from main9 import main
        main(response)
        videostate.autoplay=True
    
    # print('运行videochufa')
    # videostate.play() # ========这种对象,不能传参,只能用全局变量.
    # videostate.autoplay=True
    # videostate.value='test6.mp4'
    # # print(videostate.autoplay)
    # # from main9 import main
    # print('运行完')
    # return None,None,None,videostate.update('test6.mp4')







def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], [], None


# Delete last turn
def delete_last_turn(chat, history):
    if chat and history:
        chat.pop(-1)
        history.pop(-1)
    return chat, history


# Regenerate response
def retry_last_answer(
    user_input, chatbot, max_length, top_p, temperature, history, past_key_values
):
    if chatbot and history:
        # Removing the previous conversation from chat
        chatbot.pop(-1)
        # Setting up a flag to capture a retry
        RETRY_FLAG = True
        # Getting last message from user
        user_input = history[-1][0]
        # Removing bot response from the history
        history.pop(-1)

    yield from predict(
        RETRY_FLAG,  # type: ignore
        user_input,
        chatbot,
        max_length,
        top_p,
        temperature,
        history,
        past_key_values,
    )










with gr.Blocks(title="ChatGLM2-6B-int8", theme=gr.themes.Soft(text_size="sm")) as demo:
    with gr.Column(scale=4):
        with gr.Row():
            chatbot = gr.Chatbot()
            videostate =gr.Video(value='result99999999.mp4',width=400,height=400, autoplay=True)
            gr.videostate=videostate
            print(videostate,33333333333333)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Input...",
                ).style(container=False)
                RETRY_FLAG = gr.Checkbox(value=False, visible=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row():
                    submitBtn = gr.Button("Submit", variant="primary")
                    btn = gr.Button(value="生成视频")
                    deleteBtn = gr.Button("Delete last turn", variant="secondary")
                    retryBtn = gr.Button("Regenerate", variant="secondary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0,
                32768,
                value=8192,
                step=1.0,
                label="Maximum length",
                interactive=True,
            )
            top_p = gr.Slider(
                0, 1, value=0.85, step=0.01, label="Top P", interactive=True
            )
            temperature = gr.Slider(
                0.01, 1, value=0.95, step=0.01, label="Temperature", interactive=True
            )

    history = gr.State([])
    past_key_values = gr.State(None)

    user_input.submit(      # =======按回车相应
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values],
        show_progress="full",
    )












    _=gr.State(None)
    submitBtn.click(     #=========按submit按钮相应.
        predict,
        [
            RETRY_FLAG,
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        [chatbot, history, past_key_values,videostate],
        show_progress="full",
        api_name="predict",
    )











    # submitBtn.click(reset_user_input, [], [user_input])

    #===========添加submitBtn的相应为视频对应
    def videochufa():
        print('运行videochufa')
        # videostate.play() # ========这种对象,不能传参,只能用全局变量.
        videostate.autoplay=True
        videostate.value='test6.mp4'
        print(videostate.autoplay)
        from main9 import main
        # main('ffffff')
        # videostate.update(value='result99999999.mp4')
        return gr.videostate.update(value='test6.mp4') ########===========需要更新组件,要在返回函数里面写update,然后把值写上去.

    def combine():
        print('在切视频.')
        print()
        global response2
        print('接受到的文本是',response2)
        videostate.autoplay=True

        #========response2转化为视频即可.################################################################
        #=固定原始视频的情况下,如何中间保存变量,从而进行加速.
        # apt-get install ffmpeg
        # apt install nvidia-cuda-toolkit
        # pip install paddlepaddle-gpu==2.4.2
        #  pip install paddlepaddle==2.4.2
        # pip install typeguard==2.13.3

        # -*- coding: utf-8 -*-    python3 -m pip install paddlespeech==1.0.0
        # ========简化代码.     python3 -m pip install paddleaudio==1.0.1  
        # cp nltk_data  /root/nltk_data -r
        import nltk

        print(nltk.data.path)
        # ['/root/nltk_data', '/root/miniconda3/nltk_data', '/root/miniconda3/share/nltk_data', '/root/miniconda3/lib/nltk_data', '/usr/share/nltk_data', '/usr/local/share/nltk_data', '/usr/lib/nltk_data', '/usr/local/lib/nltk_data']  #==========去这些里面看文件是否下好.
        tttttt=response2
        # tttttt='掼蛋，一种从“跑得快”和“八十分”演化而来的升级类纸牌游戏，在这两年快速完成了“升级”，从民间牌桌一跃进入了竞技体育赛场。上个周末，上海市棋牌运动管理中心二楼大厅内，全国掼牌（掼蛋）公开赛上海站比赛落下大幕。这是继掼蛋被列为今年10月下旬在安徽合肥举办的智力运动会表演项目之后，全国大赛的第一站比赛。为什么掼蛋能够在过去两三年以江苏和安徽为中心，迅速风靡全国，渗透到不同的社交圈层和领域？用上海市休闲棋牌协会副会长金方伟的话来总结：“它（掼蛋）不像桥牌那么复杂，但又融合了几种牌类的玩法，而且具有社交属性。通过团队合作打升级的方式，体现了一种趣味性，也有竞技性，所以也符合体育精神。比赛现场。社交属性突显掼蛋魅力在上海举行的这场全国掼牌（掼蛋）公开赛第一站比赛，吸引了一共72支队伍144名选手参赛，其中包括了各区体育局、行业集团、上海市休闲棋牌协会等单位，以及外围赛晋级选手。'

        import os
        import argparse
        #================第一部分,从文字生成音频.
        #引入飞桨生态的语音和GAN依赖
        # from PaddleTools.TTS import TTSExecutor
        # from PaddleTools.GAN import wav2lip

        parser = argparse.ArgumentParser()
        parser.add_argument('--human', type=str,default='', help='human video', required=False)
        parser.add_argument('--output', type=str, default='output.mp4', help='output video')
        parser.add_argument('--text', type=str,default='', help='human video', required=False)
        import torch
        print(torch.cuda.is_available())
        print(11111111111111111111)
        import time

        # import paddle
        import time




        args = parser.parse_args()


        args.human='file/input/zimeng.mp4'
        args.human='file/input/test.png'
        args.text=tttttt

            
        # from paddlespeech.cli.tts import TTSExecutor

        # tts_executor = TTSExecutor()
        # print(paddle.get_device())
        # fdsaf=paddle.get_device()







        #====================


        wavfile='output.wav'
        # if 0:

        adsf=time.time()      #=============开始时间记录.
        #     wavfile = tts_executor(
        #         text=args.text,
        #         output='output.wav',
        #         # am='fastspeech2_csmsc',
        #         #am='fastspeech2_male',  #===========这里换性别.
        #         am='speedyspeech_csmsc',  #===========这里换性别.
        #         am_config=None,
        #         am_ckpt=None,
        #         am_stat=None,
        #         spk_id=174,
        #         phones_dict=None,
        #         tones_dict=None,
        #         speaker_dict=None,
        #         voc='pwgan_csmsc',
        #         voc_config=None,
        #         voc_ckpt=None,
        #         voc_stat=None,
        #         lang='zh',
        #         device=paddle.get_device())
        #     print('生成音频使用了',time.time()-adsf)
        args2=args
        import os
        os.system(f'edge-tts --voice zh-CN-XiaoxiaoNeural --text "{args.text}" --write-media output.wav')






        #=======推理代码. 参数都已经设置好了,直接跑即可. 结果再results/result_voice.mp4里面.







        from os import listdir, path
        import numpy as np
        import scipy, cv2, os, sys, argparse, audio
        import json, subprocess, random, string
        from tqdm import tqdm
        from glob import glob
        import torch, face_detection
        from models import Wav2Lip
        import platform

        parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

        parser.add_argument('--checkpoint_path', type=str, 
                            help='Name of saved checkpoint to load weights from', required=False)

        parser.add_argument('--face', type=str, 
                            help='Filepath of video/image that contains faces to use', required=False)
        parser.add_argument('--audio', type=str, 
                            help='Filepath of video/audio file to use as raw audio source', required=False)
        parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                        default='result99999999.mp4')

        parser.add_argument('--static', type=bool, 
                            help='If True, then use only first video frame for inference', default=False)
        parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                            default=25., required=False)

        parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                            help='Padding (top, bottom, left, right). Please adjust to include chin at least')

        parser.add_argument('--face_det_batch_size', type=int, 
                            help='Batch size for face detection', default=16)
        parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

        parser.add_argument('--resize_factor', default=1, type=int, 
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

        parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                            help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                            'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

        parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                            help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                            'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

        parser.add_argument('--rotate', default=False, action='store_true',
                            help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                            'Use if you get a flipped result, despite feeding a normal looking video')

        parser.add_argument('--nosmooth', default=False, action='store_true',
                            help='Prevent smoothing face detections over a short temporal window')

        args = parser.parse_args()
        args.img_size = 96
        args.face_det_batch_size = 4
        args.checkpoint_path = 'checkpoints/wav2lip_gan.pth'
        args.wav2lip_batch_size =40



        #========改这里啊就行.
        args.face = args2.human

        args.face = 'test3.mp4'
        args.audio = wavfile
        print(1)













        if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            args.static = True

        def get_smoothened_boxes(boxes, T):
            for i in range(len(boxes)):
                if i + T > len(boxes):
                    window = boxes[len(boxes) - T:]
                else:
                    window = boxes[i : i + T]
                boxes[i] = np.mean(window, axis=0)
            return boxes

        def face_detect(images):
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                    flip_input=False, device=device)

            batch_size = args.face_det_batch_size
            
            while 1:
                predictions = []
                try:
                    for i in tqdm(range(0, len(images), batch_size)):
                        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
                except RuntimeError:
                    if batch_size == 1: 
                        raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                    batch_size //= 2
                    print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                    continue
                break

            results = []
            pady1, pady2, padx1, padx2 = args.pads
            for rect, image in zip(predictions, images):
                if rect is None:
                    cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                    raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

                gaodu=rect[3]-rect[1] #========优化人脸位置.
                chagndu=rect[2]-rect[0]
                pady1=round(gaodu*0.1)
                pady2=round(gaodu*0.12)
                padx2=round(chagndu*0.1)
                padx1=round(chagndu*0.1)

                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)
                
                results.append([x1, y1, x2, y2])

            boxes = np.array(results)
            if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
            results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

            del detector
            return results 
        # a=3

        # print(face_det_results)
        def datagen(frames, mels):
            global face_det_results
            # print(face_det_results)
            # print(a)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []


            #=====写入:
            if 0:
                if args.box[0] == -1:
                    if not args.static: # 需要每一个帧进行处理.
                        face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
                    else:
                        face_det_results = face_detect([frames[0]])
                else:
                    print('Using the specified bounding box instead of face detection...')
                    y1, y2, x1, x2 = args.box
                    face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
                import pickle
                picklefile = open('marks', 'wb')
                # Pickle the dictionary and write it to file
                pickle.dump(face_det_results, picklefile)
                # Close the file
                picklefile.close()
                print('写入了新的mark')
                raise

            if 1:
                print(1)





            for i, m in enumerate(mels):
                idx = 0 if args.static else i%len(face_det_results) # 静态就是每个图片都是第一针.这里面已经保证了视频帧进行循环.
                frame_to_save = frames[idx].copy()
                face, coords = face_det_results[idx].copy()

                face = cv2.resize(face, (args.img_size, args.img_size))
                    
                img_batch.append(face)
                mel_batch.append(m)
                frame_batch.append(frame_to_save)
                coords_batch.append(coords)

                if len(img_batch) >= args.wav2lip_batch_size:#========一直往[]里面添加.数量够了就yield出去.
                    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                    img_masked = img_batch.copy()
                    img_masked[:, args.img_size//2:] = 0

                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                    yield img_batch, mel_batch, frame_batch, coords_batch
                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            if len(img_batch) > 0: #========剩余的最后一组,
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch

        mel_step_size = 16




        if 1:
            if not os.path.isfile(args.face):
                raise ValueError('--face argument must be a valid path to video/image file')

            elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
                full_frames = [cv2.imread(args.face)]
                fps = args.fps

            else:
                video_stream = cv2.VideoCapture(args.face) #cv2.videocapture作为opencv中常用的视频读取函数，其主要作用是从本地或者网络中读取视频帧，并预存储到内存中，便于图片处理或者特征提取等操作。
                fps = video_stream.get(cv2.CAP_PROP_FPS)
                print('是用视频的帧率是',fps)
                print('Reading video frames...')

                full_frames = []
                while 1:
                    still_reading, frame = video_stream.read()
                    if not still_reading:
                        video_stream.release()
                        break
                    if args.resize_factor > 1:
                        frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

                    if args.rotate:
                        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                    y1, y2, x1, x2 = args.crop
                    if x2 == -1: x2 = frame.shape[1]
                    if y2 == -1: y2 = frame.shape[0]

                    frame = frame[y1:y2, x1:x2]

                    full_frames.append(frame)

            print ("Number of frames available for inference: "+str(len(full_frames)))

            if not args.audio.endswith('.wav'):
                print('Extracting raw audio...')
                command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

                subprocess.call(command, shell=True)
                args.audio = 'temp/temp.wav'

            wav = audio.load_wav(args.audio, 16000)
            mel = audio.melspectrogram(wav)
            print(mel.shape)

            if np.isnan(mel.reshape(-1)).sum() > 0:
                raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

            mel_chunks = []
            mel_idx_multiplier = 80./fps  # 梅尔一秒是80, fps是我们采样率.
            i = 0
            while 1:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > len(mel[0]): # 每16个mel取一个区间.
                    mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
                i += 1

            print("Length of mel chunks: {}".format(len(mel_chunks)))

            # full_frames = full_frames[:len(mel_chunks)]
            import time
            start=time.time()
            batch_size = args.wav2lip_batch_size
            gen = datagen(full_frames.copy(), mel_chunks)

            for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                    total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
                global modelw
                if i == 0:
                    

                    frame_h, frame_w = full_frames[0].shape[:-1]
                    out = cv2.VideoWriter('temp/result.avi', 
                                            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.no_grad():
                    pred = modelw(mel_batch, img_batch)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                cnt=1
                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    # cv2.imwrite(f'temp/{cnt}.png',p)
                    f[y1:y2, x1:x2] = p #======新的脸部贴上.
                    out.write(f)
                    cnt+=1

            out.release()
            #-========给视频加上音频.
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
            subprocess.call(command, shell=platform.system() != 'Windows')
            print('文件保存在',args.outfile)
            print('总时间',time.time()-adsf)











        return  gr.videostate.update('result99999999.mp4')



    # submitBtn.click(videochufa,[],[videostate]) ######不触发.#======先写好切片函数.



    # btn2 = gr.Button(value="对话") #========作为对话用
    # btn2.click(duihua,inputs=[],outputs=[])

    # aaa=gr.PlayableVideo('test6.mp4')
    
    # aaa=gr.Video('test6.mp4',autoplay=True)
    # aaa.autoplay=True
    # gr.aaa=aaa
    btn.click(combine,inputs=[],outputs=[videostate])











    emptyBtn.click(
        reset_state, outputs=[chatbot, history, past_key_values], show_progress="full"
    )

    retryBtn.click(
        retry_last_answer,
        inputs=[
            user_input,
            chatbot,
            max_length,
            top_p,
            temperature,
            history,
            past_key_values,
        ],
        # outputs = [chatbot, history, last_user_message, user_message]
        outputs=[chatbot, history, past_key_values],
    )
    deleteBtn.click(delete_last_turn, [chatbot, history], [chatbot, history])

demo.queue(concurrency_count=3, max_size=30).launch(server_name='0.0.0.0',debug=True,show_api=True,inbrowser=True,share=True)
