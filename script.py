# -*- coding: utf-8 -*-
"""
Auto-news Urdu vertical video:
- Fetch latest news by topic (Google News RSS, locale-aware)
- Refine/condense with Ollama to Urdu headline + script (graceful fallbacks)
- TTS (edge-tts -> pyttsx3 WAV→MP3 fallback)
- Render robot video with Urdu subs

Env toggles (optional):
  NEWS_DEBUG=1     # verbose logging
  NEWS_NO_CACHE=1  # ignore & overwrite cache
"""

import os, sys, math, asyncio, random, shutil, re, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import feedparser
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, VideoClip
from pydub import AudioSegment

# ===================== CONFIG =====================
W, H, FPS = 1080, 1920, 30                   # vertical video
VOICE = "ur-PK-AsadNeural"                   # try "ur-PK-GulNeural" for female
OUTDIR = Path("output_videos"); OUTDIR.mkdir(parents=True, exist_ok=True)

# Urdu fonts (put .ttf files next to this script, or give full paths)
URDU_FONT_BOLD = "NotoNastaliqUrdu-Regular.ttf"   # or "Jameel Noori Nastaleeq.ttf"
URDU_FONT_REG  = "NotoNastaliqUrdu-Regular.ttf"

TITLE_SIZE = 64
SUB_SIZE   = 50
SUB_BOX_MARGINS = (60, 180)  # left/right padding for subtitle area

# Colors
BG_TOP = (18, 30, 55)
BG_BOT = (9, 16, 28)
ROBOT_BODY = (240, 242, 246)
ROBOT_OUTL = (40, 60, 90)
ACCENT     = (115, 187, 255)
SUB_BOX_BG = (15, 22, 38, 210)  # RGBA
TITLE_COLOR = (0, 0, 0)         # headline text BLACK
SUB_COLOR   = (230, 230, 230)

# Subtitles
MAX_SUB_CHARS = 42     # target characters per subtitle card line budget
SUB_LINES_PER_CARD = 2
MIN_SUB_DUR = 1.2      # seconds (minimum per subtitle card)

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# Cache
CACHE_DIR = OUTDIR / "cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
def _cache_path(topic: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", topic.strip() or "default")
    return CACHE_DIR / f"news_{safe}.json"

# Debug
DEBUG = os.environ.get("NEWS_DEBUG", "0") == "1"
FORCE_NO_CACHE = os.environ.get("NEWS_NO_CACHE", "0") == "1"
def log(*a):
    if DEBUG:
        print("[NEWS]", *a)

# Spyder-visible checkpoints
def checkpoint(msg):
    """Visible progress marker for Spyder console."""
    print(f"\n{'='*70}\n🟢 {msg}\n{'='*70}")
# ==================================================

def stamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        # fallback (not great for Urdu shaping, but avoids crash)
        win = os.environ.get("WINDIR", r"C:\Windows")
        for name in ["NotoNastaliqUrdu-Regular.ttf", "Jameel Noori Nastaleeq.ttf", "arial.ttf", "arialbd.ttf"]:
            for root in [Path("."), Path(win)/"Fonts"]:
                cand = root / name
                if cand.is_file():
                    try: return ImageFont.truetype(str(cand), size)
                    except: pass
        return ImageFont.load_default()

FONT_TITLE = load_font(URDU_FONT_BOLD, TITLE_SIZE)
FONT_SUB   = load_font(URDU_FONT_REG,  SUB_SIZE)

# ---------- TTS engines ----------
EDGE_OK = True
try:
    import edge_tts
except Exception:
    EDGE_OK = False

PYTTS_OK = True
try:
    import pyttsx3
except Exception:
    PYTTS_OK = False

# ---------- NEWS FETCH (robust, locale-aware) ----------
def fetch_google_news(topic: str, n=5, timeout=10, max_retries=3):
    """Fetch n recent items for a topic via Google News RSS with retries + cache fallback."""
    items, last_err = [], None

    # Urdu / Pakistan edition to improve Urdu-topic hits
    hl, gl, ceid = "ur", "PK", "PK:ur"
    q = requests.utils.quote(topic, safe="")
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PintosNewsBot/1.1)",
        "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    log("topic:", topic)
    log("rss url:", url)

    cache_file = _cache_path(topic)
    if FORCE_NO_CACHE and cache_file.exists():
        try:
            cache_file.unlink()
            log("cache deleted:", cache_file)
        except: pass

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            feed = feedparser.parse(r.content)
            entries = getattr(feed, "entries", []) or []
            log(f"rss entries: {len(entries)} (attempt {attempt+1})")
            for e in entries[:n]:
                items.append({
                    "title": (e.get("title") or "").strip(),
                    "summary": re.sub("<.*?>", "", e.get("summary", "") or "").strip(),
                    "link": (e.get("link", "") or "").strip(),
                })
            if items:
                try:
                    cache_file.write_text(json.dumps({"ts": time.time(), "items": items}, ensure_ascii=False), "utf-8")
                    log("cache wrote:", cache_file)
                except Exception as ce:
                    log("cache write failed:", ce)
                return items
            last_err = RuntimeError("empty RSS feed")
        except Exception as e:
            last_err = e
            log("rss error:", e, "| retrying…")
            time.sleep(1.2 * (attempt + 1))

    # cache fallback
    try:
        cached = json.loads(cache_file.read_text("utf-8"))
        cached_items = cached.get("items", [])[:n]
        if cached_items:
            log("using cached items:", len(cached_items))
            return cached_items
    except Exception as ce:
        log("cache read failed:", ce)

    log("fetch failed; returning []:", last_err)
    return []

def ollama_generate(prompt: str, model=OLLAMA_MODEL, max_retries=2, timeout=120):
    """Call Ollama local HTTP API (streaming). Returns text or '' on failure."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    for attempt in range(max_retries):
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                out = []
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line.decode("utf-8"))
                        if "response" in j:
                            out.append(j["response"])
                    except Exception:
                        continue
                txt = "".join(out).strip()
                if txt:
                    return txt
        except Exception as e:
            if attempt == max_retries - 1:
                print("⚠️  Ollama failed:", e)
            time.sleep(1.5 * (attempt + 1))
    return ""

def _fallback_urdu_from_titles(titles):
    """Simple rule-based Urdu headline + script when Ollama is unavailable."""
    titles = [t.strip(" .") for t in titles if t.strip()]
    if not titles:
        return ("آج کی اہم خبر", "آج کی سرخیوں میں ٹیکنالوجی اور کاروبار سے متعلق تازہ پیش رفت شامل ہیں۔ مزید تفصیلات کے لیے ہمارے ساتھ رہیے۔")
    h = f"اہم خبر: {titles[0][:48]}"
    body = "۔ ".join(titles[:3]) + "۔ مزید معلومات کے لیے ہمارے ساتھ رہیے۔"
    return (h, body)

def make_urdu_news(topic: str):
    """Fetch headlines → ask Ollama to craft Urdu headline + script, with cache and graceful fallbacks."""
    items = fetch_google_news(topic, n=6)
    if not items:
        return _fallback_urdu_from_titles([])

    context = "\n".join([f"- {it['title']}" for it in items if it.get("title")])
    prompt = f"""آپ ایک اردو نیوز اینکر کے رائٹر ہیں۔
نیچے دی گئی تازہ خبروں (ہیڈ لائنز) کی بنیاد پر:
{context}

ہدف:
1) ایک مختصر، توجہ کھینچنے والی اردو ہیڈ لائن (زیادہ سے زیادہ 16–18 الفاظ)۔
2) 3–4 جملوں میں اردو اسکرپٹ: خلاصہ، سیاق و سباق، اہم پوائنٹس۔ غیر مبہم اور باوقار لہجہ۔

آؤٹ پٹ صرف اس فارمیٹ میں دیں:
<HEADLINE_UR>
...ہیڈ لائن...
</HEADLINE_UR>
<SCRIPT_UR>
...۳–۴ جملوں کا اسکرپٹ...
</SCRIPT_UR>
"""
    text = ollama_generate(prompt).strip()
    h_match = re.search(r"<HEADLINE_UR>\s*(.*?)\s*</HEADLINE_UR>", text, re.S)
    s_match = re.search(r"<SCRIPT_UR>\s*(.*?)\s*</SCRIPT_UR>", text, re.S)
    headline_ur = (h_match.group(1).strip() if h_match else "")
    script_ur   = (s_match.group(1).strip() if s_match else "")

    if not headline_ur or not script_ur:
        headline_ur, script_ur = _fallback_urdu_from_titles([it["title"] for it in items])

    # sanitize/limit
    headline_ur = re.sub(r"\s+", " ", headline_ur)[:160]
    script_ur   = re.sub(r"\s+", " ", script_ur)[:800]

    # persist to cache
    try:
        _cache_path(topic).write_text(
            json.dumps({"ts": time.time(), "items": items, "headline_ur": headline_ur, "script_ur": script_ur}, ensure_ascii=False),
            "utf-8"
        )
    except Exception:
        pass

    return headline_ur, script_ur

# ---------- TTS ----------
def ensure_audio_urdu(text: str, out_mp3: Path):
    """Generate Urdu MP3 via edge-tts, fallback to pyttsx3 (WAV→MP3)."""
    text = re.sub(r"\s+", " ", text).strip()
    if EDGE_OK:
        try:
            async def gen():
                t = edge_tts.Communicate(text, voice=VOICE)
                await t.save(str(out_mp3))
            try:
                asyncio.run(gen())
            except RuntimeError as e:
                if "asyncio.run() cannot be called" in str(e):
                    import nest_asyncio, asyncio as _a
                    nest_asyncio.apply()
                    loop = _a.get_event_loop()
                    loop.run_until_complete(gen())
                else:
                    raise
        except Exception as e:
            print("edge-tts failed:", e)

    if (not out_mp3.exists()) or out_mp3.stat().st_size < 10_000:
        if not PYTTS_OK:
            raise RuntimeError("No TTS engine available. Install edge-tts or pyttsx3.")
        print("⚙️  Using pyttsx3 fallback…")
        tmp_wav = out_mp3.with_suffix(".wav")
        eng = pyttsx3.init()
        eng.setProperty('rate', 180)
        eng.save_to_file(text, str(tmp_wav))
        eng.runAndWait()
        seg = AudioSegment.from_wav(tmp_wav)
        seg.export(out_mp3, format="mp3")

    seg = AudioSegment.from_file(out_mp3)
    return out_mp3, seg.duration_seconds, seg

# ---------- Mouth sync from audio ----------
def mouth_curve_from_audio(seg: AudioSegment, fps=FPS):
    frame_ms = int(1000 / fps)
    vals = []
    for t in range(0, len(seg), frame_ms):
        chunk = seg[t:t+frame_ms]
        rms = chunk.rms or 1
        vals.append(rms)
    arr = np.array(vals, dtype=np.float32)
    if arr.max() > 0: arr = arr / arr.max()
    # light smoothing
    for _ in range(3):
        arr = (np.r_[arr[0], arr[:-1]] + arr + np.r_[arr[1:], arr[-1]]) / 3.0
    return arr.tolist()

# ---------- Background & robot ----------
def bg_gradient():
    img = Image.new("RGB", (W, H), BG_BOT)
    p = img.load()
    for y in range(H):
        t = y / max(1, H-1)
        r = int(BG_TOP[0]*(1-t) + BG_BOT[0]*t)
        g = int(BG_TOP[1]*(1-t) + BG_BOT[1]*t)
        b = int(BG_TOP[2]*(1-t) + BG_BOT[2]*t)
        for x in range(W):
            p[x, y] = (r, g, b)
    return img

def draw_stars(base_img, t):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    random.seed(12345)
    for layer, speed, size in [(0.35, 8, 2), (0.65, 14, 2), (0.95, 20, 3)]:
        step = max(40, int(150 * layer))
        for i in range(0, W, step):
            for j in range(0, H, max(60, int(180 * layer))):
                x = i + (j % step)//2
                y = j
                dx = int(math.sin((t + i*0.01 + j*0.01) * (1.0 + layer)) * speed)
                dy = int(math.cos((t + i*0.007 + j*0.013) * (1.0 + layer)) * speed/2)
                draw.ellipse((x+dx, y+dy, x+dx+size, y+dy+size), fill=(255, 255, 255))
    return img

def draw_robot(img, mouth, blink, bob):
    canvas = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    cx, cy = W//2, int(H*0.62) + bob

    body_w, body_h = 520, 560
    x1, y1 = cx - body_w//2, cy - body_h//2
    draw.rounded_rectangle((x1, y1, x1+body_w, y1+body_h), radius=60, fill=ROBOT_BODY, outline=ROBOT_OUTL, width=6)

    draw.line((cx, y1-90, cx, y1-10), fill=ROBOT_OUTL, width=6)
    draw.ellipse((cx-18, y1-110, cx+18, y1-74), fill=ACCENT, outline=ROBOT_OUTL, width=4)

    eye_w, eye_h = 90, int(90 * (1.0 - blink))
    ex1 = cx - 150 - eye_w//2
    ex2 = cx + 150 - eye_w//2
    ey  = y1 + 130
    for ex in [ex1, ex2]:
        draw.rounded_rectangle((ex, ey, ex+eye_w, ey+eye_h), radius=18, fill=(30,40,60), outline=ROBOT_OUTL, width=4)

    m_w = 220
    m_h = int(40 + 180 * mouth)
    mx1 = cx - m_w//2
    my1 = y1 + 330
    draw.rounded_rectangle((mx1, my1, mx1+m_w, my1+m_h), radius=18, fill=(20,28,45), outline=ROBOT_OUTL, width=4)

    draw.ellipse((x1+24, y1+24, x1+62, y1+62), fill=ACCENT, outline=ROBOT_OUTL, width=3)
    return canvas

# ---------- Title bubble ----------
def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def draw_title(img, text):
    canvas = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    max_w = int(W*0.86)
    lines = wrap_text(draw, text, FONT_TITLE, max_w)
    lh = int(FONT_TITLE.size*1.25)
    pad = 22
    bw = max(int(draw.textlength(ln, font=FONT_TITLE)) for ln in lines) + pad*2
    bh = lh*len(lines) + pad*2
    x = (W - bw)//2
    y = 120
    draw.rounded_rectangle((x, y, x+bw, y+bh), radius=28, fill=(255,255,255,235), outline=(10,25,40,255), width=4)
    draw.polygon([(x+bw//2-20, y+bh), (x+bw//2+20, y+bh), (x+bw//2, y+bh+26)], fill=(255,255,255,235), outline=(10,25,40,255))
    ty = y + pad
    for ln in lines:
        wlen = draw.textlength(ln, font=FONT_TITLE)
        draw.text((x + (bw - wlen)//2, ty), ln, fill=TITLE_COLOR, font=FONT_TITLE)
        ty += lh
    return canvas

# ---------- Subtitles ----------
def split_into_subs(text: str):
    parts = re.split(r"(?<=[۔!?])\s+", text.strip())
    raw = " ".join([p.strip() for p in parts if p.strip()])
    words = raw.split()

    cards, cur = [], []
    char_count = 0
    limit = MAX_SUB_CHARS * SUB_LINES_PER_CARD
    for w in words:
        if char_count + len(w) + 1 > limit and cur:
            cards.append(" ".join(cur))
            cur = [w]; char_count = len(w)
        else:
            cur.append(w); char_count += len(w) + 1
    if cur: cards.append(" ".join(cur))
    return cards or [raw]

def active_sub_index(t, timeline):
    for i, (ts, te, _) in enumerate(timeline):
        if ts <= t <= te: return i
    return -1

def build_sub_timeline(sub_cards, total_dur):
    start_at = 0.9
    tail = 0.3
    usable = max(0.1, total_dur - start_at - tail)
    n = max(1, len(sub_cards))
    per = max(MIN_SUB_DUR, usable / n)

    timeline = []
    t = start_at
    for card in sub_cards:
        timeline.append((t, min(t+per, total_dur - tail), card))
        t += per
    return timeline

def draw_subtitles(img, lines, box_bg=SUB_BOX_BG):
    canvas = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    L, R = SUB_BOX_MARGINS
    box_w = W - (L + R)
    # dynamic height
    line_imgs, total_h = [], 0
    for ln in lines:
        wrap = wrap_text(draw, ln, FONT_SUB, box_w)
        for wln in wrap:
            w = int(draw.textlength(wln, font=FONT_SUB))
            line_img = Image.new("RGBA", (box_w, int(FONT_SUB.size*1.3)), (0,0,0,0))
            ImageDraw.Draw(line_img).text(((box_w-w)//2, 0), wln, font=FONT_SUB, fill=SUB_COLOR)
            line_imgs.append(line_img); total_h += line_img.size[1]
    pad_y = 24
    panel_h = total_h + pad_y*2
    panel = Image.new("RGBA", (box_w, panel_h), box_bg)
    y = (H - panel_h - 120)
    canvas.alpha_composite(panel, (L, y))
    cy = y + pad_y
    for im in line_imgs:
        canvas.alpha_composite(im, (L, cy))
        cy += im.size[1]
    return canvas

# ---------- Build video ----------
def build_clip(audio_mp3, duration, mouth_curve, headline_ur, subs_timeline):
    base_grad = bg_gradient()
    def draw_stars_local(t):
        return draw_stars(base_grad, t).convert("RGBA")

    headline_on = (0.15, min(3.2, duration*0.35))

    def make_frame(t):
        bg = draw_stars_local(t)

        if headline_on[0] <= t <= headline_on[1]:
            overlay = draw_title(bg, headline_ur)  # no truncation; wrapping handles it
            bg = Image.alpha_composite(bg, overlay)

        idx = min(int(t*FPS), len(mouth_curve)-1)
        mouth = mouth_curve[idx] if idx >= 0 else 0.0
        blink = 1.0 if (t % 4.2) < 0.08 else 0.0
        bob = int(math.sin(t*2.0)*6)
        bot = draw_robot(bg, mouth, blink, bob)
        bg = Image.alpha_composite(bg, bot)

        k = active_sub_index(t, subs_timeline)
        if k != -1:
            _, _, txt = subs_timeline[k]
            lines = wrap_text(ImageDraw.Draw(bg), txt, FONT_SUB, W - sum(SUB_BOX_MARGINS))
            sub_canvas = draw_subtitles(bg, lines[:SUB_LINES_PER_CARD])
            bg = Image.alpha_composite(bg, sub_canvas)

        return np.array(bg.convert("RGB"))

    clip = VideoClip(make_frame, duration=duration)
    audio = AudioFileClip(str(audio_mp3))
    return clip.set_audio(audio)

# ---------- Main ----------
def main():
    # Combine CLI args (Windows sometimes splits Urdu)
    topic = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    log("CLI topic:", repr(topic))

    # 🔸 Auto-pick topic if none given
    if not topic:
        checkpoint("🔍 No topic given — fetching trending headlines automatically...")
        try:
            url = "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en&gl=US&ceid=US:en"
            r = requests.get(url, timeout=10)
            feed = feedparser.parse(r.content)
            titles = [e.title for e in feed.entries[:10] if getattr(e, "title", "")]
            if titles:
                topic = random.choice(titles)
                checkpoint(f"✅ Auto-selected trending topic: {topic}")
            else:
                checkpoint("⚠️ Could not fetch trending topics — using fallback.")
                topic = "Artificial Intelligence and Robotics"
        except Exception as e:
            checkpoint(f"⚠️ Auto-topic fetch failed ({e}); using fallback.")
            topic = "Artificial Intelligence and Robotics"

    # Brand intro
    intro_ur = "خوش آمدید! یہ ہے پِنٹوس، اورین اسِسٹنٹ کی طرف سے آپ کے لیے روزانہ کی ٹیکنالوجی نیوز۔"

    # ffmpeg check
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg not found. Install: conda install -c conda-forge ffmpeg -y")
        sys.exit(1)

    # 🔸 Live news generation
    checkpoint(f"Fetching LIVE Urdu news for topic: {topic}")
    try:
        headline_ur, script_ur = make_urdu_news(topic)
        if headline_ur and script_ur:
            checkpoint("✅ LIVE news fetched successfully from Google RSS + Ollama")
        else:
            checkpoint("⚠️ RSS/Ollama returned empty → using fallback text")
    except Exception as e:
        checkpoint(f"⚠️ Live news failed ({e}) → using fallback.")
        headline_ur = "آج کی اہم خبر: مصنوعی ذہانت میں نئی پیش رفت"
        script_ur   = ("دنیا بھر میں AI کے میدان میں اہم تبدیلیاں سامنے آ رہی ہیں۔ "
                       "نئے ماڈلز اور کم خرچ کمپیوٹنگ نے جدت کی رفتار تیز کر دی ہے۔ "
                       "تفصیلات کیلئے ہمیں فالو کریں۔")

    # final safety
    if not headline_ur.strip():
        headline_ur = "آج کی اہم خبر"
    if not script_ur.strip():
        script_ur = "تازہ ترین اپ ڈیٹس کے لیے ہمارے ساتھ رہیے۔"

    narration = f"{intro_ur}. {headline_ur}. {script_ur}"

    # 🔸 TTS
    audio_mp3 = OUTDIR / f"urdu_voice_{stamp()}.mp3"
    audio_mp3, dur, seg = ensure_audio_urdu(narration, audio_mp3)
    if dur < 1.0:
        raise RuntimeError("Audio too short; TTS likely failed.")

    # Subtitles + timeline
    curve = mouth_curve_from_audio(seg, fps=FPS)
    sub_cards = [intro_ur] + split_into_subs(script_ur)
    subs_timeline = build_sub_timeline(sub_cards, dur)

    # 🔸 Build video
    final = build_clip(str(audio_mp3), dur + 0.5, curve, headline_ur, subs_timeline)
    out_path = OUTDIR / f"robot_urdu_subs_{int(dur)}s_{stamp()}.mp4"
    final.write_videofile(str(out_path), fps=FPS, codec="libx264", audio_codec="aac", preset="medium")
    checkpoint(f"✅ Video saved successfully: {out_path}")


if __name__ == "__main__":
    main()
