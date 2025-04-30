import os.path

import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils import mkdirs, loadObj
import multiprocessing as mp

wDir = r"E:\项目\NIID-Bench\Results-gradFix\noniid-#label1\rate_1\random-2_share-wLogDir"
gradShowDir = os.path.join(wDir, "modelShow")


def clientNetDeta(gMod, nets):
    netsDetaW = {
        netIdx: torch.cat([
            (nets[netIdx][lPara].cpu() - gMod[gPara].cpu()).view(-1)
            for gPara, lPara in zip(gMod, nets[netIdx])]
        ).cpu().numpy()
        for netIdx in nets
    }
    return netsDetaW


def netDetaNorm(netDict1, netDict2, norm=2):
    deta = torch.cat([
        (netDict2[lPara].cpu() - netDict1[gPara].cpu()).view(-1)
        for gPara, lPara in zip(netDict1, netDict2)]
    ).cpu().numpy()
    detaNorm = np.linalg.norm(deta, ord=norm, axis=-1)
    return detaNorm


def detaVec():
    pass


def show(fname, wLog, savePath):
    tsne = TSNE(perplexity=1, n_components=2, random_state=42)

    # vectorsDict = clientNetDeta(wLog["gOld"], wLog["client"])
    vectors = np.asarray([wLog["client"][netIdx] - wLog["gOld"] for netIdx in wLog["client"]])
    deta = wLog["gNew"] - wLog["gOld"]
    detaNorm = np.linalg.norm(deta, ord=2, axis=-1)

    vectors_2d = tsne.fit_transform(vectors)

    # Step 3: 创建子图，左边是t-SNE可视化，右边是余弦相似度热力图
    fig, axs = plt.subplots(1, 3, figsize=(30, 8))

    # 左图：t-SNE散点图
    axs[0].scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', label='Weight deta')
    axs[0].set_title("t-SNE 2D Visualization")
    axs[0].legend()

    # Step 2: 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(vectors)
    # 右图：余弦相似度热力图
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=axs[1],
                vmax=-1, vmin=1)
    axs[1].set_title("Cosine Similarity Heatmap")

    # Step 2: 计算余弦相似度矩阵
    sim_matrix = euclidean_distances(vectors)
    # 右图：余弦相似度热力图
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=axs[2],
                vmax=0, vmin=15)
    axs[2].set_title("Euclidean Similarity Heatmap")

    fig.suptitle(fname, fontsize=16)
    # 保存图像
    plt.savefig(savePath)
    plt.tight_layout()
    plt.clf()
    plt.close()

    return float(detaNorm), deta


if __name__ == '__main__':
    mkdirs(gradShowDir)
    gradFiles = sorted(list(filter(lambda f: f.endswith(".pickle"), os.listdir(wDir))),
                       key=lambda x: int(x.split("_")[0]))
    outPngs, jobs, allDeta, stepDetaVec = [], [], [], []
    mp.set_start_method("spawn")
    pool = mp.Pool(10)
    for gf in gradFiles:
        wLog = loadObj(os.path.join(wDir, gf))
        savePath = os.path.join(gradShowDir, os.path.splitext(gf)[0] + ".png")
        job = (pool.apply_async(show, args=(gf, wLog, savePath)), gf)
        outPngs.append(savePath)
        jobs.append(job)
    for job in tqdm(jobs):
        detaNorm, deta = job[0].get()
        allDeta.append("{}\t{}".format(job[1].split("_")[0], str(detaNorm)))
        stepDetaVec.append(deta)
    summaryGif = os.path.join(gradShowDir, "summary.gif")
    gradPngs = [Image.open(png_file) for png_file in outPngs]
    gradPngs[0].save(summaryGif, save_all=True, append_images=gradPngs[1:], optimize=False, duration=1000, loop=0)
    with open(os.path.join(gradShowDir, "detaNormList.tab"), "w", encoding="utf-8") as f:
        f.write("\n".join(allDeta))

    sim_matrix = cosine_similarity(np.asarray(stepDetaVec))
    plt.figure(figsize=(40, 32))
    heatmap = sns.heatmap(sim_matrix, fmt=".2f", cmap='coolwarm', cbar=True, vmax=-1, vmin=1)
    # 手动调整颜色条的刻度字体大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=32)  # 调整颜色条的刻度字体大小
    plt.suptitle("detaVec_Heatmap", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # 保存图像
    plt.savefig(os.path.join(gradShowDir, "detaVec_Heatmap.png"))

    # 清理图像
    plt.clf()
    plt.close()
