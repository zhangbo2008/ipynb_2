import numpy as np
# 这个讲解很好.   O(1)复杂度来做非均匀抽样.
### https://zhuanlan.zhihu.com/p/54867139
def gen_prob_dist(N):
    p = np.random.randint(0,100,N)
    return p/np.sum(p)

def create_alias_table(area_ratio):

    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
#==========把每一个时间按照small , large进行切分.
    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
        if area_ratio[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
#=====可能是精度问题, 最后会有点偏差, 强制改结果为1即可.
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept,alias

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

def simulate(N=100,k=10000,):


    truth = gen_prob_dist(N)

    area_ratio = truth*N
    accept,alias = create_alias_table(area_ratio)

    ans = np.zeros(N)
    for _ in range(k):
        i = alias_sample(accept,alias)
        ans[i] += 1
    return ans/np.sum(ans),truth

if __name__ == "__main__":
    import time
    a=time.time()
    alias_result,truth = simulate()
    print(time.time()-a)