# -*- coding: utf-8 -*-
"""VERA AI inference API: CLIP image embeddings + FFT/SRM + trained ensemble.

This file loads the trained bundle created by training_pipeline_colab.py.
It does not train a model when the API starts.
"""
from __future__ import annotations
import io, os
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

cv2.setNumThreads(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = Path(os.getenv("VERA_MODEL_BUNDLE", str(BASE_DIR / "models" / "ml_clip_bundle_final.pkl")))
MAX_UPLOAD_MB = int(os.getenv("VERA_MAX_UPLOAD_MB", "15"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="VERA AI API", version="1.0.0", description="Phân loại ảnh REAL/AI bằng CLIP + FFT Ensemble")
origins = [x.strip() for x in os.getenv("VERA_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

_FREQ_GEOM = {}
def _freq_geom(h: int, w: int):
    key=(h,w)
    if key not in _FREQ_GEOM:
        cy,cx=h//2,w//2; y,x=np.ogrid[:h,:w]
        r=np.sqrt((x-cx)**2+(y-cy)**2).astype(int); max_r=min(cy,cx); bin_size=max_r/48
        _FREQ_GEOM[key]=[np.flatnonzero(((r>=i*bin_size)&(r<(i+1)*bin_size)).ravel()) for i in range(48)]
    return _FREQ_GEOM[key]

def extract_frequency_features(img_pil: Image.Image) -> np.ndarray:
    """Same 64-dimensional FFT/SRM feature calculation as the training pipeline."""
    img_gray=np.array(img_pil.convert('L'),dtype=np.float32); h,w=img_gray.shape
    mag_spec=np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray)))+1e-8)
    cy,cx=h//2,w//2; flat=mag_spec.ravel(); radial_prof=np.zeros(48,dtype=np.float32)
    for i,ix in enumerate(_freq_geom(h,w)):
        if ix.size: radial_prof[i]=flat[ix].mean()
    radial_prof=(radial_prof-radial_prof.mean())/(radial_prof.std()+1e-8)
    srm_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float32)
    noise_res=cv2.filter2D(img_gray,-1,srm_kernel); q=np.percentile(noise_res,[10,25,50,75,90]); p95=np.percentile(mag_spec,95)
    noise_stats=np.array([noise_res.mean(),noise_res.std(),q[0],q[1],q[2],q[3],q[4],np.var(noise_res),
        mag_spec[:cy//2,:cx//2].mean(),mag_spec[cy//2:,:cx//2].mean(),mag_spec[:cy//2,cx//2:].mean(),mag_spec[cy//2:,cx//2:].mean(),
        mag_spec[cy-10:cy+10,cx-10:cx+10].mean(),np.std(mag_spec),np.mean(mag_spec>p95),
        np.sum(mag_spec>(mag_spec.mean()+2*mag_spec.std()))/(h*w)],dtype=np.float32)
    return np.concatenate([radial_prof,noise_stats]).astype(np.float32)

def prepare_image(img: Image.Image, size: int = 256) -> Image.Image:
    """Training app uses square_mode='squash'; preserve that preprocessing."""
    img=ImageOps.exif_transpose(img).convert('RGB')
    if img.size != (size,size): img=img.resize((size,size),Image.Resampling.BICUBIC)
    return img

@lru_cache(maxsize=1)
def load_pipeline():
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model bundle: {BUNDLE_PATH}. Hãy sao chép ml_clip_bundle_final.pkl vào backend/models/ hoặc đặt VERA_MODEL_BUNDLE.")
    bundle=joblib.load(BUNDLE_PATH)
    clip_name=bundle.get('clip_name','openai/clip-vit-base-patch32')
    processor=CLIPImageProcessor.from_pretrained(clip_name)
    clip=CLIPVisionModelWithProjection.from_pretrained(clip_name).to(DEVICE).eval()
    for p in clip.parameters(): p.requires_grad_(False)
    return bundle,processor,clip

def predict_image(img: Image.Image):
    bundle,processor,clip=load_pipeline()
    img=prepare_image(img)
    pv=processor(images=[img],return_tensors='pt')['pixel_values'].to(DEVICE)
    if DEVICE.type=='cuda': pv=pv.half(); clip=clip.half()
    with torch.inference_mode():
        emb=clip(pv).image_embeds
        emb=(emb/emb.norm(dim=-1,keepdim=True)).float().cpu().numpy()
    raw_freq=extract_frequency_features(img).reshape(1,-1)
    freq=bundle['scaler_freq'].transform(raw_freq)
    features=np.concatenate([emb,freq],axis=1).astype(np.float32)
    models=bundle['models']; weights=bundle.get('ensemble_weights',{})
    probabilities={}
    for name,model in models.items():
        classes=list(model.classes_)
        if 1 not in classes: raise ValueError(f"Model {name} không có class AI (1).")
        probabilities[name]=float(model.predict_proba(features)[:,classes.index(1)][0])
    if weights and all(name in probabilities for name in weights):
        ai_prob=float(sum(float(weights[n])*probabilities[n] for n in weights))
    else:
        best=bundle.get('best_model_name')
        if best not in probabilities: best=next(iter(probabilities))
        ai_prob=probabilities[best]
    threshold=float(bundle.get('threshold',0.5))
    label='AI' if ai_prob>=threshold else 'REAL'
    confidence=ai_prob if label=='AI' else 1.0-ai_prob
    return {'label':label,'ai_probability':round(ai_prob,6),'confidence':round(confidence,6),'threshold':round(threshold,6),
            'model':'CLIP + FFT Ensemble','device':str(DEVICE),'component_probabilities':{k:round(v,6) for k,v in probabilities.items()}}

@app.get('/')
def root(): return {'service':'VERA AI API','status':'running','model':'CLIP + FFT Ensemble','health':'/health'}

@app.get('/health')
def health():
    return {'status':'ok','bundle_exists':BUNDLE_PATH.is_file(),'bundle_path':str(BUNDLE_PATH),'device':str(DEVICE)}

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES: raise HTTPException(415,'Chỉ hỗ trợ JPG, PNG hoặc WEBP.')
    data=await file.read()
    if not data: raise HTTPException(400,'Tệp ảnh trống.')
    if len(data)>MAX_UPLOAD_MB*1024*1024: raise HTTPException(413,f'Ảnh vượt quá {MAX_UPLOAD_MB} MB.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load(); result=predict_image(im.copy())
        return result
    except FileNotFoundError as e: raise HTTPException(503,str(e))
    except UnidentifiedImageError: raise HTTPException(400,'Không đọc được định dạng ảnh.')
    except Exception as e: raise HTTPException(500,f'Lỗi xử lý mô hình: {type(e).__name__}: {e}')
