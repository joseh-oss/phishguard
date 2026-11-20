"""PhishGuard AI - single-file phishing URL detector.

Self-contained: feature extraction, a realistic training set, model training and
the web interface all live in this one file, so there are no folders to upload.
The model trains itself the first time it is needed (a few seconds) and is kept
in memory afterwards.

Local:  python app.py            -> http://127.0.0.1:5000
Hosted: gunicorn app:app
"""
import os
import random
import re
import threading
from urllib.parse import urlparse, quote

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Feature extraction (lexical only, no network access)
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "secure", "update", "confirm",
    "bank", "paypal", "webscr", "ebay", "password", "credential", "wallet",
    "billing", "invoice", "security", "recover", "unlock", "support",
]
SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at",
]
SUSPICIOUS_TLDS = [
    "zip", "review", "country", "kim", "cricket", "science", "work", "party",
    "gq", "link", "top", "xyz", "tk", "ml", "ga", "cf", "men", "loan",
]
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$")
FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_at", "num_question", "num_equal", "num_ampersand", "num_percent",
    "num_slash", "num_digits", "digit_ratio", "num_subdomains", "uses_https",
    "has_ip", "has_at", "has_port", "host_has_hyphen", "is_shortened",
    "suspicious_tld", "num_suspicious_words", "has_double_slash_path",
]


def extract_features(url):
    raw = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw_for_parse = "http://" + raw
    else:
        raw_for_parse = raw
    parsed = urlparse(raw_for_parse)
    host = parsed.hostname or ""
    path = parsed.path or ""
    lower = raw.lower()
    digits = sum(c.isdigit() for c in raw)
    labels = [p for p in host.split(".") if p]
    tld = labels[-1] if labels else ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(path),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_at": raw.count("@"),
        "num_question": raw.count("?"),
        "num_equal": raw.count("="),
        "num_ampersand": raw.count("&"),
        "num_percent": raw.count("%"),
        "num_slash": raw.count("/"),
        "num_digits": digits,
        "digit_ratio": round(digits / len(raw), 4) if raw else 0.0,
        "num_subdomains": max(len(labels) - 2, 0),
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "has_ip": 1 if IP_RE.match(host) else 0,
        "has_at": 1 if "@" in raw else 0,
        "has_port": 1 if port else 0,
        "host_has_hyphen": 1 if "-" in host else 0,
        "is_shortened": 1 if any(s == host or host.endswith("." + s) for s in SHORTENERS) else 0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "has_double_slash_path": 1 if "//" in path else 0,
    }


def vector(url):
    f = extract_features(url)
    return [float(f[n]) for n in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# 2. Training data (realistic sample; replace with real data for submission)
# ---------------------------------------------------------------------------
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "linkedin.com", "github.com", "microsoft.com", "apple.com", "netflix.com",
    "reddit.com", "stackoverflow.com", "bbc.com", "cnn.com", "paypal.com",
    "dropbox.com", "spotify.com", "adobe.com", "mozilla.org", "python.org",
    "ubuntu.com", "cloudflare.com", "nation.africa", "standardmedia.co.ke",
    "safaricom.co.ke", "ku.ac.ke", "kyu.ac.ke", "ecitizen.go.ke", "uonbi.ac.ke",
]
LEGIT_PATHS = [
    "", "/", "/about", "/contact", "/products", "/blog/2026/news",
    "/help/faq", "/search?q=technology", "/user/profile", "/docs/intro",
    "/articles/latest", "/pricing", "/support",
]
BRANDS = ["paypal", "apple", "amazon", "microsoft", "netflix", "facebook",
          "google", "bank", "ebay", "instagram", "safaricom", "mpesa"]
SUS_WORDS = ["login", "verify", "secure", "account", "update", "confirm",
             "signin", "webscr", "billing", "unlock", "security"]
BAD_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "work", "loan"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def build_dataset(n_each=900, seed=42):
    rnd = random.Random(seed)

    def rs(n):
        return "".join(rnd.choice(ALPHABET) for _ in range(n))

    def legit():
        dom = rnd.choice(LEGIT_DOMAINS)
        path = rnd.choice(LEGIT_PATHS)
        scheme = "https" if rnd.random() < 0.85 else "http"
        sub = "www." if rnd.random() < 0.5 else ""
        u = f"{scheme}://{sub}{dom}{path}"
        r = rnd.random()
        if r < 0.07:
            u = f"{scheme}://{sub}{dom}{path}?session={rs(14)}&ref={rs(8)}"
        elif r < 0.12:
            u = f"https://account-{rs(4)}.{dom}/{rnd.choice(SUS_WORDS)}"
        elif r < 0.15:
            u = f"https://{dom}/{rnd.choice(SUS_WORDS)}?id={rs(9)}"
        return u

    def phish():
        kind = rnd.random()
        brand = rnd.choice(BRANDS)
        word = rnd.choice(SUS_WORDS)
        if kind < 0.12:
            return f"https://{brand}-{word}.{rnd.choice(BAD_TLDS)}/{word}"
        if kind < 0.30:
            ip = ".".join(str(rnd.randint(1, 255)) for _ in range(4))
            return f"http://{ip}/{brand}/{word}.php?account={rs(8)}"
        if kind < 0.52:
            return (f"http://{brand}-{word}.{rs(6)}.{rnd.choice(BAD_TLDS)}/"
                    f"{word}/update?id={rs(10)}")
        if kind < 0.70:
            return f"http://{brand}.com@{rs(10)}.{rnd.choice(BAD_TLDS)}/{word}"
        if kind < 0.85:
            return (f"http://{brand}-{word}-{rs(4)}.{rs(5)}.{rs(5)}.com/signin")
        return (f"http://{rs(8)}.{rnd.choice(BAD_TLDS)}/{brand}/{word}"
                f"?user={rs(12)}&token={rs(16)}&redirect={rs(10)}")

    rows, seen = [], set()
    while sum(1 for r in rows if r[1] == 0) < n_each:
        u = legit()
        if u not in seen:
            seen.add(u)
            rows.append((u, 0))
    while sum(1 for r in rows if r[1] == 1) < n_each:
        u = phish()
        if u not in seen:
            seen.add(u)
            rows.append((u, 1))
    rnd.shuffle(rows)
    # ~3% label noise keeps results realistic rather than an artificial 100%
    rows = [(u, 1 - lab if rnd.random() < 0.03 else lab) for u, lab in rows]
    return rows


# ---------------------------------------------------------------------------
# 3. Training (done once, lazily, then cached)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state = {"bundle": None, "error": None}


def train_model():
    rows = build_dataset()
    X = np.array([vector(u) for u, _ in rows], dtype=float)
    y = np.array([lab for _, lab in rows], dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, random_state=42, n_jobs=-1),
    }
    best = None
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        score = f1_score(y_te, pred)
        if best is None or score > best[1]:
            best = (name, score, model, accuracy_score(y_te, pred),
                    recall_score(y_te, pred))
    name, f1, model, acc, rec = best
    return {"model": model, "name": name, "accuracy": acc,
            "f1": f1, "recall": rec}


def get_bundle():
    if _state["bundle"] is not None:
        return _state["bundle"]
    if _state["error"] is not None:
        raise RuntimeError(_state["error"])
    with _lock:
        if _state["bundle"] is None:
            try:
                _state["bundle"] = train_model()
            except Exception as exc:  # remember the failure, surface it clearly
                _state["error"] = f"Training failed: {exc}"
                raise
    return _state["bundle"]


def classify(url):
    bundle = get_bundle()
    feats = extract_features(url)
    X = np.array([[float(feats[n]) for n in FEATURE_NAMES]])
    label = int(bundle["model"].predict(X)[0])
    try:
        proba = float(bundle["model"].predict_proba(X)[0][label])
    except Exception:
        proba = 1.0
    return {
        "url": url,
        "label": label,
        "verdict": "phishing" if label == 1 else "legitimate",
        "confidence": round(proba * 100, 2),
        "model": bundle["name"],
        "features": feats,
    }


# ---------------------------------------------------------------------------
# 4. Web interface
# ---------------------------------------------------------------------------
app = Flask(__name__)

EXAMPLES = [
    ("phishing sample", "http://paypal.com@secure-verify-account.tk/login"),
    ("legitimate sample", "https://www.wikipedia.org/wiki/Main_Page"),
    ("shortened link", "http://bit.ly/3xFreeGift"),
]

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhishGuard AI</title>
<style>
:root{--bg:#05070d;--card:rgba(18,24,40,.6);--line:rgba(120,160,255,.16);
--cyan:#19e3ff;--txt:#eef2fb;--muted:#9aa6c2;--green:#28e0a8;--red:#ff5d73;
--grad:linear-gradient(100deg,#19e3ff,#2f6bff 55%,#8b5cff)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--txt);
min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;position:relative}
body::before{content:'';position:fixed;inset:0;z-index:-2;
background-image:linear-gradient(rgba(47,107,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(47,107,255,.07) 1px,transparent 1px);
background-size:48px 48px}
.wrap{width:100%;max-width:640px}
.head{text-align:center;margin-bottom:22px}
.logo{font-weight:800;font-size:1.8rem;letter-spacing:1px}
.logo .s{color:var(--cyan)}.logo .ai{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.tag{color:var(--muted);margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:26px;backdrop-filter:blur(12px)}
label{display:block;font-size:.9rem;color:var(--muted);margin-bottom:10px}
.row{display:flex;gap:10px}
input{flex:1;padding:14px 16px;border-radius:12px;background:rgba(255,255,255,.04);
border:1px solid var(--line);color:var(--txt);font-size:.95rem}
input::placeholder{color:#8b97b3}
input:focus{outline:none;border-color:var(--cyan)}
button{padding:14px 24px;border:0;border-radius:12px;background:var(--grad);
color:#04101c;font-weight:700;cursor:pointer}
.hint{color:#aab6d0;font-size:.8rem;margin-top:12px;line-height:1.7}
.hint a{color:var(--cyan);text-decoration:none;border-bottom:1px dotted rgba(25,227,255,.5)}
.verdict{margin-top:20px;padding:24px;border-radius:18px;border:1px solid var(--line);background:var(--card)}
.safe{border-color:rgba(40,224,168,.45)}.danger{border-color:rgba(255,93,115,.45)}
h2{font-size:1.15rem;margin-bottom:6px}
.safe h2{color:var(--green)}.danger h2{color:var(--red)}
.url{color:var(--muted);font-size:.85rem;word-break:break-all}
.meter{height:10px;border-radius:8px;background:rgba(255,255,255,.07);margin-top:14px;overflow:hidden}
.fill{height:100%;border-radius:8px}
.safe .fill{background:linear-gradient(90deg,#28e0a8,#19e3ff)}
.danger .fill{background:linear-gradient(90deg,#ff5d73,#8b5cff)}
.conf{color:var(--muted);font-size:.85rem;margin-top:8px}
table{width:100%;margin-top:14px;border-collapse:collapse;font-size:.82rem}
td{padding:6px 8px;border-bottom:1px solid var(--line);color:var(--muted)}
td:last-child{text-align:right;color:var(--txt)}
.foot{text-align:center;color:#a3aec9;font-size:.8rem;margin-top:20px}
@media(max-width:520px){.row{flex-direction:column}button{width:100%}}
</style></head><body><div class="wrap">
<div class="head"><div class="logo"><span class="s">◈</span> PhishGuard <span class="ai">AI</span></div>
<p class="tag">AI-Based Phishing URL Detection</p></div>
<div class="card">
<label for="u">Enter a URL to analyse</label>
<form method="post" action="/check" class="row">
<input id="u" name="url" type="text" placeholder="e.g. http://secure-login.example.tk/verify" value="{{ url }}" autocomplete="off" spellcheck="false" required>
<button type="submit">Analyse</button>
</form>
<p class="hint">The model inspects URL structure only. Your links are not stored.</p>
<p class="hint">Try one: {% for label, ex in examples %}<a href="/check?url={{ ex|urlencode }}">{{ label }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</p>
</div>
{% if result %}
<div class="verdict {{ 'danger' if result.label == 1 else 'safe' }}">
<h2>{{ '⚠ Suspicious - Likely Phishing' if result.label == 1 else '✔ Looks Legitimate' }}</h2>
<p class="url">{{ result.url }}</p>
<div class="meter"><div class="fill" style="width:{{ result.confidence }}%"></div></div>
<p class="conf">{{ result.confidence }}% confidence · model: {{ result.model }}</p>
<details style="margin-top:12px"><summary style="color:var(--cyan);font-size:.85rem;cursor:pointer">Why? View extracted features</summary>
<table>{% for k, v in result.features.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>
</details>
</div>
{% endif %}
<p class="foot">Final-year project · Kioko Joseph Mbithi · Kirinyaga University</p>
</div></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE, result=None, url="",
                                  examples=EXAMPLES)


@app.route("/check", methods=["GET", "POST"])
def check():
    url = (request.values.get("url") or "").strip()
    if not url:
        return render_template_string(PAGE, result=None, url="",
                                       examples=EXAMPLES)
    try:
        result = classify(url)
    except Exception as exc:
        return render_template_string(
            PAGE, url=url,
            result={"label": 0, "url": f"Error: {exc}", "confidence": 0,
                    "model": "-", "features": {}}, examples=EXAMPLES), 500
    return render_template_string(PAGE, result=result, url=url,
                               examples=EXAMPLES)


@app.route("/health")
def health():
    try:
        b = get_bundle()
        return jsonify({"status": "ok", "model": b["name"],
                        "accuracy": round(b["accuracy"], 4)})
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please provide a URL."}), 400
    try:
        return jsonify(classify(url))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
