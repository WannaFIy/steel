# YODAS3 stereo spot check

Streams a few files from `espnet/yodas3` (no full 8 GB shard downloads), measures
L/R correlation and side/mid energy, labels dual-mono / near-mono / true stereo,
and writes 20 s example clips (stereo + side-only) with an HTML listening page.

```
pip install huggingface_hub soundfile numpy pandas pyarrow requests   # plus ffmpeg on PATH
python tools/yodas3_stereo/yodas3_stereo.py --out yodas3_out --langs ca fr --max-files 30 --max-bytes 1e9
```

Needs network access to `huggingface.co` and its LFS/Xet CDN hosts
(`cdn-lfs.huggingface.co`, `*.xethub.hf.co`). For offline testing:
`--local-tar ca=shard.tar --local-meta ca=meta.jsonl`.
