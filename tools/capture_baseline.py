"""Capture original BEATs scores and encoder tokens before extraction.

Reads: unmodified upstream source directory, pinned checkpoint, CC0 fixtures.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
sys.path.insert(0,a.upstream)
from BEATs import BEATs, BEATsConfig

torch.set_num_threads(4)
d=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
m=BEATs(BEATsConfig(d['cfg'])).eval().to(a.device);m.load_state_dict(d['model'],strict=True)
f=Path(a.fixtures); traces={}
hook=m.encoder.register_forward_hook(lambda mod,inputs,out:traces.update(tokens=out[0].detach().cpu().numpy().copy()))
for path in sorted(f.glob('*.wav')):
    data,sr=sf.read(path,dtype='float32',always_2d=True)
    x=torch.from_numpy(data.T).to(a.device)
    if sr!=16000:x=torchaudio.functional.resample(x,sr,16000)
    x=x.mean(0,keepdim=True)
    with torch.inference_mode():
        fb=m.preprocess(x);scores,_=m.extract_features(x)
        first={k:v.copy() for k,v in traces.items()};s=scores.cpu().numpy().copy()
        scores2,_=m.extract_features(x)
    assert np.array_equal(s,scores2.cpu().numpy()) and np.array_equal(first['tokens'],traces['tokens'])
    np.savez_compressed(f/(path.stem+'_'+a.device+'.npz'),waveform=x.cpu().numpy(),fbank=fb.cpu().numpy(),tokens=first['tokens'],scores=s)
(f/('environment_'+a.device+'.json')).write_text(json.dumps({'torch':torch.__version__,'torchaudio':torchaudio.__version__,'numpy':np.__version__,'device':a.device,'source_revision':'31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd','checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'cfg':d['cfg'],'label_dict':d['label_dict']},indent=2)+'\n')
print('Recorded repeatable native scores/fbank/encoder tokens',a.device)
