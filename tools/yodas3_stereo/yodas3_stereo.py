#!/usr/bin/env python3
"""Spot-check the "~70% true stereo" claim for espnet/yodas3.

Pipeline:
  1. Inspect data/<lang>/ in the HF repo (don't assume the layout), download the
     metadata that belongs to the chosen audio shard, report its columns.
  2. Stream audio tars over HTTP (never downloading whole 8 GB shards): stop at a
     byte cap or after N audio members.
  3. Decode each file at native rate/channels with ffmpeg, compute L/R Pearson
     correlation and side/mid energy ratio, and label dual-mono / near-mono /
     true stereo.
  4. Cut 20 s clips (48 kHz stereo FLAC) plus side-only (L-R) versions for
     6 examples.
  5. Write an HTML page with players, stats, transcript snippets.

Usage:
  python yodas3_stereo.py --out OUTDIR [--langs ca fr] [--max-files 30]
                          [--max-bytes 1e9] [--local-tar lang=path.tar ...]
"""
import argparse
import gzip
import html
import io
import json
import os
import re
import subprocess
import sys
import tarfile

import numpy as np
import soundfile as sf

REPO = "espnet/yodas3"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".opus", ".ogg", ".m4a", ".aac", ".webm", ".mka"}
META_HINT = re.compile(r"chan|band|stereo|sample_?rate|\bsr\b|codec", re.I)


# --------------------------------------------------------------------------- HF


def inspect_repo(lang, shard):
    """List data/<lang>, its subdirs, and pick the audio + metadata files for a shard."""
    from huggingface_hub import HfApi

    api = HfApi()
    tree = {}

    def ls(path):
        items = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=path))
        tree[path] = [(it.path, getattr(it, "size", None)) for it in items]
        return items

    top = ls(f"data/{lang}")
    print(f"\n== data/{lang}/")
    for p, s in tree[f"data/{lang}"]:
        print(f"   {p}  {fmt_size(s)}")
    subdirs = [it.path for it in top if getattr(it, "size", None) is None]
    for d in subdirs:
        ls(d)
        entries = tree[d]
        print(f"== {d}/  ({len(entries)} entries)")
        for p, s in entries[:8]:
            print(f"   {p}  {fmt_size(s)}")
        if len(entries) > 8:
            print("   ...")

    files = [(p, s) for d in subdirs for p, s in tree[d] if s is not None]
    audio = [(p, s) for p, s in files if p.endswith(".tar") and "/audio/" in p]
    meta = [(p, s) for p, s in files if "/metadata/" in p]
    audio.sort()
    pick = next(((p, s) for p, s in audio if os.path.basename(p).startswith(shard)), None)
    if pick is None and audio:
        pick = audio[0]
    meta_pick = [(p, s) for p, s in meta if pick and os.path.basename(p).split(".")[0] == os.path.basename(pick[0]).split(".")[0]]
    if not meta_pick:
        meta_pick = sorted(meta, key=lambda x: x[1] or 0)[:1]
    return pick, meta_pick


def fmt_size(s):
    if s is None:
        return "<dir>"
    for u in ["B", "KB", "MB", "GB"]:
        if s < 1024:
            return f"{s:.0f} {u}"
        s /= 1024
    return f"{s:.1f} TB"


def load_metadata_file(path):
    """Load parquet / json / jsonl (optionally .gz) / tsv / csv into list of dicts."""
    import pandas as pd

    name = path.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path).to_dict("records")
    opener = gzip.open if name.endswith(".gz") else open
    base = name[:-3] if name.endswith(".gz") else name
    with opener(path, "rt", encoding="utf-8") as f:
        if base.endswith((".tsv", ".csv")):
            return pd.read_csv(f, sep="\t" if base.endswith(".tsv") else ",").to_dict("records")
        text = f.read()
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    if text.startswith("{") and "\n{" not in text:
        obj = json.loads(text)
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            return [dict(id=k, **v) for k, v in obj.items()]
        return [obj]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def report_columns(rows, label):
    cols = {}
    for r in rows[:2000]:
        for k, v in r.items():
            cols.setdefault(k, v)
    print(f"\n== metadata columns ({label}, {len(rows)} rows)")
    for k, v in cols.items():
        flag = "   <-- channel/bandwidth-ish" if META_HINT.search(k) else ""
        print(f"   {k:24s} {type(v).__name__:8s} e.g. {short(v)}{flag}")
    hits = [k for k in cols if META_HINT.search(k)]
    print("   channel/bandwidth-like fields:", hits or "NONE")
    return list(cols)


def short(v, n=70):
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def index_metadata(rows):
    idx = {}
    for r in rows:
        for key in ("id", "audio_id", "utt_id", "key", "__key__", "video_id"):
            if key in r and r[key] is not None:
                idx[str(r[key])] = r
                break
    return idx


# ----------------------------------------------------------------------- stream


class CappedReader(io.RawIOBase):
    """File-like over an HTTP response that stops after max_bytes."""

    def __init__(self, resp, max_bytes):
        self.it = resp.iter_content(1 << 20)
        self.buf = b""
        self.read_total = 0
        self.max = max_bytes

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.read_total >= self.max:
                return 0
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
            self.read_total += len(self.buf)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def stream_tar_members(src, outdir, max_files, max_bytes):
    """Yield (stem, audio_path, sidecar_meta) from a tar (URL or local path), streaming."""
    os.makedirs(outdir, exist_ok=True)
    if src.startswith("http"):
        import requests
        from huggingface_hub.utils import build_hf_headers

        resp = requests.get(src, stream=True, headers=build_hf_headers(), timeout=60)
        resp.raise_for_status()
        fobj = io.BufferedReader(CappedReader(resp, max_bytes), buffer_size=1 << 20)
    else:
        resp = None
        fobj = open(src, "rb")
    sidecars = {}
    n = 0
    try:
        with tarfile.open(fileobj=fobj, mode="r|*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = os.path.basename(m.name)
                stem, ext = os.path.splitext(name)
                data = tf.extractfile(m).read()
                if ext.lower() in (".json", ".txt"):
                    try:
                        sidecars[stem] = json.loads(data) if ext == ".json" else {"transcript": data.decode()}
                    except Exception:
                        pass
                    continue
                if ext.lower() not in AUDIO_EXT:
                    continue
                path = os.path.join(outdir, name)
                with open(path, "wb") as f:
                    f.write(data)
                n += 1
                yield stem, path, sidecars
                if n >= max_files:
                    break
    except (tarfile.ReadError, EOFError) as e:
        print(f"   (stream ended: {e})")
    finally:
        if resp is not None:
            resp.close()
        fobj.close()


# ---------------------------------------------------------------------- analysis


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,channel_layout,bit_rate:format=bit_rate,duration",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    fm = j.get("format", {})
    return dict(codec=st.get("codec_name"), sr=int(st["sample_rate"]), ch=int(st["channels"]),
                layout=st.get("channel_layout"),
                bitrate=int(st.get("bit_rate") or fm.get("bit_rate") or 0) or None,
                duration=float(fm.get("duration") or 0))


def decode(path, sr, ch):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch),
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, ch)


def db(x):
    return 10 * np.log10(max(x, 1e-20))


def stereo_stats(x, sr, speech_segs=None):
    L = x[:, 0].astype(np.float64)
    R = x[:, 1].astype(np.float64)
    mid, side = (L + R) / 2, (L - R) / 2
    em, es = np.mean(mid ** 2), np.mean(side ** 2)
    corr = float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else 1.0
    st = dict(corr=corr, side_mid_db=db(es) - db(em), lr_bal_db=db(np.mean(L ** 2)) - db(np.mean(R ** 2)))

    # Inter-channel lag (spatially placed/delayed sources show a non-zero peak lag).
    seg = slice(0, min(len(L), sr * 30))
    a, b = L[seg] - L[seg].mean(), R[seg] - R[seg].mean()
    maxlag = int(sr * 0.002)
    if len(a) > 2 * maxlag and a.std() > 0 and b.std() > 0:
        n = 1 << int(np.ceil(np.log2(len(a) * 2)))
        cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
        cc = np.concatenate([cc[-maxlag:], cc[: maxlag + 1]])
        st["lag_ms"] = (int(np.argmax(cc)) - maxlag) / sr * 1000
    else:
        st["lag_ms"] = 0.0

    # Side/mid inside vs outside transcript speech, and low vs high band side share.
    if speech_segs:
        mask = np.zeros(len(L), bool)
        for s0, s1 in speech_segs:
            mask[int(s0 * sr): int(s1 * sr)] = True
        if 0 < mask.sum() < len(L):
            st["side_mid_speech_db"] = db(np.mean(side[mask] ** 2)) - db(np.mean(mid[mask] ** 2))
            st["side_mid_nonspeech_db"] = db(np.mean(side[~mask] ** 2)) - db(np.mean(mid[~mask] ** 2))
    S, M = np.abs(np.fft.rfft(side[seg])) ** 2, np.abs(np.fft.rfft(mid[seg])) ** 2
    f = np.fft.rfftfreq(len(side[seg]), 1 / sr)
    for name, lo, hi in (("lo", 0, 300), ("voice", 300, 3400), ("hi", 3400, sr / 2)):
        band = (f >= lo) & (f < hi)
        st[f"side_mid_{name}_db"] = db(S[band].sum()) - db(M[band].sum()) if band.any() else None
    return st


def label(ch, st):
    if ch == 1:
        return "mono"
    if st["corr"] > 0.999 or st["side_mid_db"] < -40:
        return "dual-mono"
    if st["side_mid_db"] < -20:
        return "near-mono"
    return "true stereo"


# ------------------------------------------------------------ transcript timing

SEG_KEYS = [("start", "end"), ("begin", "end"), ("start_time", "end_time"), ("offset", "duration"),
            ("s", "e")]


def speech_segments(meta):
    """Find [(start_s, end_s, text)] in whatever shape the metadata uses."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            for a, b in SEG_KEYS:
                if a in v and b in v:
                    try:
                        s0, s1 = float(v[a]), float(v[b])
                        if a == "offset":
                            s1 = s0 + s1
                        if s1 > s0:
                            txt = v.get("text") or v.get("transcript") or ""
                            out.append((s0, s1, str(txt)))
                            return
                    except (TypeError, ValueError):
                        pass
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)) or (hasattr(v, "tolist") and not isinstance(v, (str, bytes))):
            for x in list(v):
                walk(x)
        elif isinstance(v, str) and v[:1] in "[{":
            try:
                walk(json.loads(v))
            except Exception:
                pass

    walk(meta or {})
    # Whisper-style "<|1.23|> text <|4.56|>" timestamps in a plain transcript string.
    if not out:
        for v in (meta or {}).values():
            if isinstance(v, str) and "<|" in v:
                for m in re.finditer(r"<\|(\d+\.?\d*)\|>([^<]*)<\|(\d+\.?\d*)\|>", v):
                    out.append((float(m.group(1)), float(m.group(3)), m.group(2).strip()))
    return sorted(out)


def transcript_text(meta):
    for k in ("transcript", "text", "sentence", "caption"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return re.sub(r"<\|[^|]*\|>", " ", v).strip()
    segs = speech_segments(meta)
    return " ".join(t for _, _, t in segs)


def pick_window(x, sr, segs, dur=20.0):
    """Start time of a dur-second window with the most transcript speech (fallback: energy)."""
    total = len(x) / sr
    if total <= dur:
        return 0.0, "whole file"
    if segs:
        best, best_t = -1, 0.0
        for s0, _, _ in segs:
            t = min(max(0.0, s0 - 0.5), total - dur)
            cover = sum(max(0, min(s1, t + dur) - max(a, t)) for a, s1, _ in segs)
            if cover > best:
                best, best_t = cover, t
        return best_t, f"transcript ({best:.1f}s speech)"
    mid = x.mean(1)
    hop = sr
    rms = np.array([np.sqrt(np.mean(mid[i: i + hop] ** 2)) for i in range(0, len(mid) - hop, hop)])
    w = int(dur)
    thr = np.percentile(rms, 30)
    score = np.convolve((rms > thr).astype(float), np.ones(w), "valid")
    return float(np.argmax(score)), "energy fallback"


def save_clip(src, t0, dur, out_stereo, out_side):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur}", "-i", src,
                          "-f", "f32le", "-ac", "2", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    sf.write(out_stereo, x, 48000, subtype="PCM_24")
    side = (x[:, 0] - x[:, 1]) / 2
    mid_peak = np.max(np.abs(x)) + 1e-12
    peak = np.max(np.abs(side)) + 1e-12
    # Boost side to the same peak as the stereo clip so it is audible; report the gain.
    gain = min(mid_peak / peak, 10 ** (40 / 20))
    sf.write(out_side, np.stack([side * gain] * 2, 1), 48000, subtype="PCM_24")
    return 20 * np.log10(gain)


# --------------------------------------------------------------------- reporting


def print_table(results):
    hdr = f"{'lang':4} {'file':28} {'ch':>2} {'sr':>6} {'codec':6} {'kbps':>5} {'corr':>7} {'S/M dB':>7} {'lag ms':>6} {'meta ch/bw':14} label"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r['lang']:4} {r['stem'][:28]:28} {r['ch']:>2} {r['sr']:>6} {str(r['codec'])[:6]:6} "
              f"{(r['bitrate'] or 0) // 1000:>5} {r.get('corr', float('nan')):7.4f} "
              f"{r.get('side_mid_db', float('nan')):7.1f} {r.get('lag_ms', 0):6.2f} "
              f"{str(r['meta_chbw'])[:14]:14} {r['label']}")
    print()
    for grp in sorted({r["lang"] for r in results}) + ["ALL"]:
        rs = [r for r in results if grp == "ALL" or r["lang"] == grp]
        counts = {}
        for r in rs:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        s = ", ".join(f"{k}: {v} ({100 * v / len(rs):.0f}%)" for k, v in sorted(counts.items()))
        print(f"   {grp:4} n={len(rs):3d}  {s}")


def choose_examples(results):
    st = sorted([r for r in results if r["label"] == "true stereo"], key=lambda r: -r["side_mid_db"])
    picks = [("clearly stereo", r) for r in st[:3]]
    used = {id(r) for _, r in picks}
    border = sorted([r for r in results if r["ch"] == 2 and id(r) not in used],
                    key=lambda r: abs(r["side_mid_db"] + 20))
    picks += [("borderline", r) for r in border[:1]]
    used |= {id(r) for _, r in picks}
    dm = sorted([r for r in results if r["label"] == "dual-mono" and id(r) not in used],
                key=lambda r: r["side_mid_db"])
    picks += [("dual-mono", r) for r in dm[:2]]
    return picks


def fmt(v, spec=".1f"):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def write_html(path, results, picks):
    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    n = len(results)
    summary = " · ".join(f"{k}: {v}/{n} ({100 * v / n:.0f}%)" for k, v in sorted(counts.items()))
    cards = []
    for kind, r in picks:
        rows = [("Language", r["lang"]), ("File", r["stem"]), ("Codec / bitrate",
                f"{r['codec']} · {fmt((r['bitrate'] or 0) / 1000, '.0f')} kbps (meta: {r.get('meta_bitrate', '–')})"),
                ("Native", f"{r['ch']} ch · {r['sr']} Hz"), ("L/R corr", fmt(r["corr"], ".4f")),
                ("Side/mid", f"{fmt(r['side_mid_db'])} dB"),
                ("Side/mid speech vs non-speech",
                 f"{fmt(r.get('side_mid_speech_db'))} / {fmt(r.get('side_mid_nonspeech_db'))} dB"),
                ("Side/mid <300 Hz · 300–3400 · >3400",
                 f"{fmt(r.get('side_mid_lo_db'))} · {fmt(r.get('side_mid_voice_db'))} · {fmt(r.get('side_mid_hi_db'))} dB"),
                ("Inter-channel lag", f"{fmt(r.get('lag_ms'), '.2f')} ms"),
                ("Metadata ch/bw", r["meta_chbw"]), ("Clip", f"{r['clip_t0']:.1f}s +20s ({r['clip_how']})"),
                ("Side boost", f"+{r['side_gain_db']:.1f} dB")]
        table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
        cards.append(f"""
<section class="card"><header><span class="tag t-{kind.split()[0]}">{html.escape(kind)}</span>
<h2>{html.escape(r['stem'])}</h2><span class="lbl">{html.escape(r['label'])}</span></header>
<div class="players"><label>Stereo (48 kHz FLAC)<audio controls preload="none" src="{html.escape(r['clip'])}"></audio></label>
<label>Side only, (L−R)/2, boosted<audio controls preload="none" src="{html.escape(r['clip_side'])}"></audio></label></div>
<table>{table}</table>
<p class="tx">{html.escape(r['snippet'])}</p></section>""")
    allrows = "".join(
        f"<tr><td>{html.escape(r['lang'])}</td><td>{html.escape(r['stem'])}</td><td>{r['ch']}</td><td>{r['sr']}</td>"
        f"<td>{fmt(r.get('corr'), '.4f')}</td><td>{fmt(r.get('side_mid_db'))}</td><td>{html.escape(r['label'])}</td></tr>"
        for r in results)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>YODAS3 Stereo Check</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e4e4df;--st:#1f6feb;--bd:#b7791f;--dm:#6b6b66}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--muted:#9a9a94;--card:#1f1f1d;--line:#34342f;--st:#58a6ff;--bd:#e3b341;--dm:#9a9a94}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:900px;margin:auto}} h1{{margin:0 0 4px}} .sub{{color:var(--muted);margin:0 0 24px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}}
.card header{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}} h2{{font-size:15px;margin:0;font-family:ui-monospace,monospace;word-break:break-all}}
.tag{{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid currentColor}} .t-clearly{{color:var(--st)}} .t-borderline{{color:var(--bd)}} .t-dual-mono{{color:var(--dm)}}
.lbl{{color:var(--muted);font-size:13px}} .players{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}}
.players label{{font-size:13px;color:var(--muted)}} audio{{width:100%;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{text-align:left;padding:3px 8px 3px 0;border-top:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:500;width:40%}} .tx{{font-size:13px;color:var(--muted);font-style:italic;margin:10px 0 0}}
details{{margin-top:24px}} .scroll{{overflow-x:auto}}
@media (max-width:600px){{.players{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>YODAS3 stereo spot check</h1>
<p class="sub">{n} files · {html.escape(summary)}<br>Thresholds: dual-mono = corr&gt;0.999 or side/mid&lt;−40 dB · near-mono = −40…−20 dB · true stereo = side/mid&gt;−20 dB.</p>
{''.join(cards)}
<details><summary>All analysed files</summary><div class="scroll"><table>
<tr><th>lang</th><th>file</th><th>ch</th><th>sr</th><th>corr</th><th>S/M dB</th><th>label</th></tr>{allrows}</table></div></details>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# -------------------------------------------------------------------------- main


def meta_chbw(meta):
    if not meta:
        return "–"
    hits = {k: v for k, v in meta.items() if META_HINT.search(k)}
    return ", ".join(f"{k}={v}" for k, v in hits.items()) or "none in metadata"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="+", default=["ca", "fr"])
    ap.add_argument("--shard", default="0001")
    ap.add_argument("--max-files", type=int, default=30)
    ap.add_argument("--max-bytes", type=float, default=1e9)
    ap.add_argument("--local-tar", nargs="*", default=[], help="lang=path.tar (offline testing)")
    ap.add_argument("--local-meta", nargs="*", default=[], help="lang=path (offline testing)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    local_tar = dict(x.split("=", 1) for x in a.local_tar)
    local_meta = dict(x.split("=", 1) for x in a.local_meta)
    results = []
    for lang in a.langs:
        meta_rows = []
        if lang in local_tar:
            src = local_tar[lang]
            if lang in local_meta:
                meta_rows = load_metadata_file(local_meta[lang])
                report_columns(meta_rows, local_meta[lang])
        else:
            from huggingface_hub import hf_hub_download, hf_hub_url

            pick, metas = inspect_repo(lang, a.shard)
            if pick is None:
                print(f"no audio tars under data/{lang}")
                continue
            for mp, ms in metas:
                print(f"\n-> metadata {mp} ({fmt_size(ms)})")
                lp = hf_hub_download(REPO, mp, repo_type="dataset", cache_dir=os.path.join(a.out, ".hf"))
                rows = load_metadata_file(lp)
                report_columns(rows, mp)
                meta_rows += rows
            src = hf_hub_url(REPO, pick[0], repo_type="dataset")
            print(f"\n-> streaming {pick[0]} ({fmt_size(pick[1])}), cap {fmt_size(a.max_bytes)} / {a.max_files} files")
        idx = index_metadata(meta_rows)
        for stem, path, sidecars in stream_tar_members(src, os.path.join(a.out, "raw", lang), a.max_files, a.max_bytes):
            meta = idx.get(stem) or sidecars.get(stem) or next((v for k, v in idx.items() if stem.startswith(k) or k.startswith(stem)), None)
            p = probe(path)
            r = dict(lang=lang, stem=stem, path=path, meta=meta, meta_chbw=meta_chbw(meta), **p)
            r["meta_bitrate"] = (meta or {}).get("audio_bitrate", "–")
            segs = speech_segments(meta)
            r["segs"] = segs
            if p["ch"] >= 2:
                x = decode(path, p["sr"], p["ch"])
                r.update(stereo_stats(x, p["sr"], [(s0, s1) for s0, s1, _ in segs]))
            else:
                r.update(corr=float("nan"), side_mid_db=float("nan"))
            r["label"] = label(p["ch"], r)
            results.append(r)
            print(f"   {lang} {stem[:40]:40} {p['ch']}ch {p['sr']}Hz S/M={r['side_mid_db']:.1f} dB -> {r['label']}")

    if not results:
        sys.exit("no files analysed")
    print_table(results)

    clipdir = os.path.join(a.out, "clips")
    os.makedirs(clipdir, exist_ok=True)
    picks = choose_examples(results)
    for i, (kind, r) in enumerate(picks, 1):
        x = decode(r["path"], r["sr"], r["ch"])
        t0, how = pick_window(x, r["sr"], r["segs"])
        base = f"{i:02d}_{kind.replace(' ', '-')}_{r['lang']}_{r['stem'][:40]}"
        r["clip"] = f"clips/{base}.flac"
        r["clip_side"] = f"clips/{base}_side.flac"
        r["side_gain_db"] = save_clip(r["path"], t0, 20, os.path.join(a.out, r["clip"]), os.path.join(a.out, r["clip_side"]))
        r["clip_t0"], r["clip_how"] = t0, how
        inwin = " ".join(t for s0, s1, t in r["segs"] if s1 > t0 and s0 < t0 + 20)
        r["snippet"] = (inwin or transcript_text(r["meta"]) or "(no transcript)")[:400]
        print(f"   clip {base}: t0={t0:.1f}s ({how}), side boost +{r['side_gain_db']:.1f} dB")
    write_html(os.path.join(a.out, "index.html"), results, picks)

    slim = [{k: v for k, v in r.items() if k not in ("meta", "segs")} for r in results]
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(slim, f, indent=1, default=str)
    print(f"\nwrote {a.out}/index.html, {a.out}/clips/, {a.out}/results.json")


if __name__ == "__main__":
    main()
