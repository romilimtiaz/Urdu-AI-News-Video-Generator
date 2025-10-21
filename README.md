# Urdu-AI-News-Video-Generator
Urdu-AI-News-Video-Generator is an intelligent Python-based automation system that transforms real-time news into short, engaging Urdu-language AI videos.
📰 Urdu AI News Video Generator (Orion Pintos)
🎯 Overview

This project automatically fetches trending news, summarizes it into Urdu using Ollama, generates a human-sounding TTS narration, and renders a full vertical video featuring a friendly robot newscaster with subtitles — ideal for TikTok, Reels, or YouTube Shorts.

Built with Python + MoviePy + edge-tts + Ollama, the system can operate fully offline (with cached data) and requires no manual input — it can pick trending topics by itself.

🚀 Features

✅ Automatic Topic Detection

Fetches trending topics from Google News (Technology section by default)

Can run autonomously without user input

Optionally accepts a manual topic via command line

✅ AI-Generated Urdu News

Uses Ollama LLM (local) to generate a concise Urdu headline and script

Graceful fallback if Ollama or Internet unavailable

✅ TTS Narration

Default: Microsoft Edge-TTS (AsadNeural / GulNeural voices)

Fallback: Pyttsx3 (offline voice synthesis)

✅ Animated Video Rendering

Friendly robot character with mouth synced to speech

Animated eyes and subtle floating motion

Urdu subtitles with custom Nastaliq font

Vertical 1080×1920 layout for social media

✅ Exception Safe

Handles network, TTS, and LLM errors gracefully

Uses local cache if network is down
 ┌───────────────┐       ┌──────────────┐       ┌──────────────┐
 │ Google News   │──────▶│ Ollama (LLM) │──────▶│ Urdu Script  │
 └───────────────┘       └──────────────┘       └──────┬───────┘
                                                          │
                                                          ▼
                                                ┌──────────────────┐
                                                │ edge-tts / pyttsx3│
                                                └──────┬───────────┘
                                                       ▼
                                            ┌──────────────────────┐
                                            │  MoviePy Renderer    │
                                            │  Robot + Subtitles   │
                                            └──────────────────────┘
# 📰 Urdu AI News Video Generator

### 🎙️ Automatically generate short Urdu news videos with AI voice, subtitles, and an animated robot!

This project creates **vertical TikTok/YouTube-style Urdu news videos** powered by:
- 🧠 **Ollama LLM** for Urdu headline & script generation  
- 🗞️ **Google News RSS** for trending news topics  
- 🔊 **edge-tts** (or `pyttsx3`) for Urdu speech synthesis  
- 🎥 **MoviePy + Pillow** for animated robot video rendering  

All completely **offline-capable** (LLM via local Ollama) and optimized for **Spyder / Python environments**.

---

## 🚀 Features

✅ **Auto-topic detection** – if you don’t provide any topic, the system automatically picks a trending technology headline from Google News.  
✅ **Dynamic Urdu headline + script** generated using **Ollama (LLaMA-3)**  
✅ **AI Voice narration (Urdu)** using Microsoft Edge Neural TTS (`ur-PK-AsadNeural` / `ur-PK-GulNeural`)  
✅ **Animated robot** that lip-syncs with the TTS waveform  
✅ **Subtitles & headline overlay** in Urdu (using Nastaliq fonts)  
✅ **Vertical HD output** (1080×1920, ready for TikTok/Reels/YouTube Shorts)  
✅ **Robust fallbacks** (cached news, rule-based Urdu generation, local TTS)  
✅ **Spyder checkpoints** showing each stage of processing for easy debugging  

---

## 🧰 Requirements
1️⃣ Install Python Packages
pip install requests feedparser moviepy pillow pydub numpy edge-tts pyttsx3
conda install -c conda-forge ffmpeg -y

2️⃣ Install and Run Ollama

Ollama provides the local language model used for Urdu script generation.

For macOS / Linux:

curl -fsSL https://ollama.com/install.sh | sh


For Windows:

Download the installer from https://ollama.ai/download

Run Ollama once (it will start a local API server on port 11434)

3️⃣ Pull a Language Model
ollama pull llama3.1


You can also try smaller models like:

ollama pull mistral

4️⃣ Verify Installation
ollama run llama3.1 "Write a short Urdu headline about artificial intelligence."
### Python Dependencies
Install all required packages:
```bash
pip install requests feedparser moviepy pillow pydub numpy edge-tts pyttsx3
System Requirements

Python 3.9+

FFmpeg (for audio/video encoding)

conda install -c conda-forge ffmpeg -y


Optional: Ollama
 with any local model (llama3.1, mistral, etc.)

Fonts

Place a compatible Urdu font (NotoNastaliqUrdu-Regular.ttf or Jameel Noori Nastaleeq.ttf) next to the script.

⚙️ Configuration

You can customize key parameters in the script header:

Variable	Description	Default
VOICE	Urdu TTS voice	"ur-PK-AsadNeural"
OLLAMA_MODEL	Model for Urdu text generation	"llama3.1"
W, H	Video resolution	1080x1920
TITLE_COLOR	Urdu headline color	Black
OUTDIR	Output folder	output_videos/
🧠 How It Works

Fetch Trending Topics:
If no topic is given, it automatically retrieves trending Technology headlines from Google News RSS.

Generate Urdu Headline + Script:
It sends the selected topic headlines to Ollama, asking for a natural Urdu news headline and 3–4 sentence script.

Text-to-Speech (TTS):
Converts Urdu text to audio (edge-tts first, pyttsx3 fallback).

Mouth Animation Curve:
Extracts audio amplitude (RMS) per frame to animate mouth movement.

Rendering:
Builds a full 1080×1920 video with:

Gradient background + animated stars ✨

Friendly robot character with eyes, blinking, and lip-sync mouth

Urdu headline bubble 🗨️

Urdu subtitles synced with TTS

Export:
Produces a ready-to-upload .mp4 file with voice and subtitles.


🖥️ Usage
▶️ 1. Run with custom topic
python script.py "مصنوعی ذہانت اور روبوٹکس"

▶️ 2. Run without topic (auto-picks trending news)
python script.py

▶️ 3. Optional debug mode

To view detailed logs and checkpoints:

NEWS_DEBUG=1 NEWS_NO_CACHE=1 python script.py


You’ll see checkpoints like:

🟢 🔍 No topic given — fetching trending headlines automatically...
🟢 ✅ Auto-selected trending topic: Apple launches new M4 MacBooks
🟢 ✅ LIVE news fetched successfully from Google RSS + Ollama

📂 Output

Generated videos and TTS files are saved in:

output_videos/
│
├── urdu_voice_20251021_115253.mp3
├── robot_urdu_subs_26s_20251021_115253.mp4
└── cache/

🧩 Example Output (Concept)

Headline:
"اہم خبر: مصنوعی ذہانت اور روبوٹکس میں نئی جدت"

Video:
Robot character narrating Urdu news with subtitles
(You can attach your preview GIF or thumbnail here.)
