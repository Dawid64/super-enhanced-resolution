from math import sqrt
import numpy as np
from numpy import average
import pandas as pd
from tqdm import tqdm
from qsr.predictor import Upscaler
from glob import glob

testing_videos = [f'videos/HD/{i}.mp4' for i in range(100, 1001, 100)]
input_res = (640, 360)
output_res = (1280, 720)
models = sorted([x for x in glob("fine-models/*final.pt")])
results = {'model': models} | {f'{video}_psnr': [] for video in testing_videos} | {
    f'{video}_ssim': [] for video in testing_videos} | {f'{video}_qm': [] for video in testing_videos} | {'average_psnr': []} | {'average_ssim': []} | {'average_qm': []} | {'score': []}
mbar = tqdm(models, total=len(models), unit='model', leave=False)
for model in mbar:
    print(f"Testing model: {model}")
    psnr_edges = []
    ssim_edges = []
    qm_edges = []
    vbar = tqdm(testing_videos, total=len(testing_videos), unit='video', leave=False)
    for video in vbar:
        print(f"Video: {video}")
        upscaler = Upscaler(model, original_size=output_res, target_size=input_res,
                            listener=None, frames_backward=1, frames_forward=1, mode='test')
        psnr, ssim, baseline_psnr, baseline_ssim = upscaler.upscale(video)
        quality_measure = sqrt(psnr*ssim)
        baseline_quality_measure = sqrt(baseline_psnr*baseline_ssim)
        qm_edge = quality_measure - baseline_quality_measure
        psnr_edge = psnr - baseline_psnr
        ssim_edge = ssim - baseline_ssim
        print(f"PSNR: {psnr_edge}, SSIM: {ssim_edge} QM: {qm_edge}")
        results[f'{video}_psnr'].append(psnr_edge)
        results[f'{video}_ssim'].append(ssim_edge)
        results[f'{video}_qm'].append(qm_edge)
        psnr_edges.append(psnr_edge)
        ssim_edges.append(ssim_edge)
        qm_edges.append(qm_edge)
    mean_psnr = average(psnr_edges)
    mean_ssim = average(ssim_edges)
    mean_qm = average(qm_edges)
    score = average(np.array(qm_edges) > 0)
    print(f"Average PSNR: {mean_psnr}, Average SSIM: {mean_ssim} Average QM: {mean_qm} Score: {score}")
    results['average_psnr'].append(mean_psnr)
    results['average_ssim'].append(mean_ssim)
    results['average_qm'].append(mean_qm)
    results['score'].append(score)
df = pd.DataFrame(results)
df = df[['model'] + sorted([f'{video}_psnr' for video in testing_videos]+[f'{video}_ssim' for video in testing_videos] +
                           [f'{video}_qm' for video in testing_videos]) + ['average_psnr', 'average_ssim', 'average_qm', 'score']]
df.to_csv("results.csv", mode='a', header=False, index=False)
