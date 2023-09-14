# 下载:https://www.tingclass.net/down-5059-1135-1.html



import requests
for i in range(1,25):
    for j in range(1,3):
        tmp=str(j)+'_'+str(i)
        url=f'https://online1.tingclass.net/lesson/shi0529/0000/59/{tmp}.mp3'
        # myfile=requests.get(url)
        import wget
        wget.download(url)

        url=f'https://down11.tingclass.net/textrar/lesson/0000/59/{tmp}.lrc'
        wget.download(url)