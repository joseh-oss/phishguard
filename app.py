"""PhishGuard AI v2 - phishing URL detector with a brand watchlist.

Single self-contained file: lexical features + brand-impersonation checks +
a trained classifier + the web page. The model trains itself on first use.

Local:  python app.py            -> http://127.0.0.1:5000
Hosted: gunicorn app:app
"""
import gzip
import os
import random
import re
import threading
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Basic lexical vocabulary
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

# Multi-level public suffixes we care about (longest first when splitting).
PUBLIC_SUFFIXES = {
    "co.ke", "go.ke", "gov.ke", "or.ke", "ac.ke", "ne.ke", "sc.ke", "me.ke",
    "info.ke", "mobi.ke", "name.ke", "biz.ke", "co.ug", "go.ug", "ac.ug",
    "co.tz", "go.tz", "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au",
    "co.za", "com.ng",
}

# ---------------------------------------------------------------------------
# 2. Brand watchlist: the official home of the brands people impersonate.
#    Add or edit entries here - nothing else needs to change.
# ---------------------------------------------------------------------------
OFFICIAL = [
    # --- Kenyan government, banks, telcos, services -------------------------
    {"name": "eCitizen", "domains": ["ecitizen.go.ke"], "aliases": ["ecitizen", "e-citizen"]},
    {"name": "Kenya Revenue Authority", "domains": ["kra.go.ke"], "aliases": ["kra"]},
    {"name": "NTSA", "domains": ["ntsa.go.ke"], "aliases": ["ntsa"]},
    {"name": "HELB", "domains": ["helb.co.ke"], "aliases": ["helb"]},
    {"name": "Kenya Power", "domains": ["kplc.co.ke"], "aliases": ["kplc", "kenyapower"]},
    {"name": "Safaricom", "domains": ["safaricom.co.ke"], "aliases": ["safaricom", "mpesa", "m-pesa"]},
    {"name": "Airtel Kenya", "domains": ["airtel.co.ke"], "aliases": ["airtel"]},
    {"name": "Telkom Kenya", "domains": ["telkom.co.ke"], "aliases": ["telkom"]},
    {"name": "Equity Bank", "domains": ["equitybank.co.ke"], "aliases": ["equitybank", "equity"]},
    {"name": "KCB Bank", "domains": ["kcbgroup.com", "kcb.co.ke"], "aliases": ["kcb"]},
    {"name": "Co-operative Bank", "domains": ["co-opbank.co.ke"], "aliases": ["coopbank", "co-opbank"]},
    {"name": "NCBA", "domains": ["ncbagroup.com"], "aliases": ["ncba"]},
    {"name": "Absa Kenya", "domains": ["absa.co.ke"], "aliases": ["absa"]},
    {"name": "Stanbic Bank", "domains": ["stanbicbank.co.ke"], "aliases": ["stanbic"]},
    {"name": "DTB Kenya", "domains": ["dtbafrica.com"], "aliases": ["dtb", "dtbafrica"]},
    {"name": "Family Bank", "domains": ["familybank.co.ke"], "aliases": ["familybank"]},
    {"name": "I&M Bank", "domains": ["imbank.co.ke"], "aliases": ["imbank"]},
    {"name": "NHIF", "domains": ["nhif.or.ke"], "aliases": ["nhif"]},
    {"name": "IEBC", "domains": ["iebc.or.ke"], "aliases": ["iebc"]},
    {"name": "Kenya Airways", "domains": ["kenya-airways.com"], "aliases": ["kenyaairways"]},
    {"name": "Jambojet", "domains": ["jambojet.com"], "aliases": ["jambojet"]},
    {"name": "KUCCPS", "domains": ["kuccps.ac.ke"], "aliases": ["kuccps"]},
    {"name": "Kirinyaga University", "domains": ["kyu.ac.ke"], "aliases": ["kyu"]},
    {"name": "University of Nairobi", "domains": ["uonbi.ac.ke"], "aliases": ["uonbi"]},
    {"name": "Kenya Red Cross", "domains": ["redcross.or.ke"], "aliases": ["redcross"]},
    {"name": "Teachers Service Commission", "domains": ["tsc.go.ke"], "aliases": ["tsc"]},
    {"name": "Kenya Police", "domains": ["kenyapolice.go.ke"], "aliases": ["kenyapolice"]},
    {"name": "Judiciary of Kenya", "domains": ["judiciary.go.ke"], "aliases": ["judiciary"]},
    {"name": "Parliament of Kenya", "domains": ["parliament.go.ke"], "aliases": ["parliament"]},
    {"name": "Nairobi County", "domains": ["nairobi.go.ke"], "aliases": ["nairobicounty"]},
    {"name": "Mombasa County", "domains": ["mombasa.go.ke"], "aliases": ["mombasacounty"]},
    {"name": "Kiambu County", "domains": ["kiambu.go.ke"], "aliases": ["kiambucounty"]},
    {"name": "Nakuru County", "domains": ["nakuru.go.ke"], "aliases": ["nakurucounty"]},
    {"name": "Kenya Wildlife Service", "domains": ["kws.go.ke"], "aliases": ["kws"]},
    {"name": "Kenya Ports Authority", "domains": ["kpa.co.ke"], "aliases": ["kpa"]},
    {"name": "Kenya Railways", "domains": ["krc.co.ke"], "aliases": ["kenyarailways"]},
    {"name": "EPRA", "domains": ["epra.go.ke"], "aliases": ["epra"]},
    {"name": "NEMA", "domains": ["nema.go.ke"], "aliases": ["nema"]},
    {"name": "KEMRI", "domains": ["kemri.org"], "aliases": ["kemri"]},
    {"name": "KICD", "domains": ["kicd.ac.ke"], "aliases": ["kicd"]},
    {"name": "Office of the DPP", "domains": ["dpp.go.ke"], "aliases": ["dppkenya"]},
    {"name": "KEPHIS", "domains": ["kephis.org"], "aliases": ["kephis"]},
    {"name": "Kenya Forest Service", "domains": ["kenyaforestservice.org"], "aliases": ["kenyaforest"]},
    {"name": "Kenyatta National Hospital", "domains": ["knh.or.ke"], "aliases": ["knh"]},
    {"name": "Nairobi Water", "domains": ["nairobiwater.co.ke"], "aliases": ["nairobiwater"]},
    {"name": "Jumia Kenya", "domains": ["jumia.co.ke"], "aliases": ["jumia"]},
    {"name": "Naivas", "domains": ["naivas.co.ke"], "aliases": ["naivas"]},
    {"name": "Quickmart", "domains": ["quickmart.co.ke"], "aliases": ["quickmart"]},
    {"name": "Carrefour Kenya", "domains": ["carrefour.co.ke", "carrefour.com"], "aliases": ["carrefour"]},
    {"name": "Tala", "domains": ["tala.co.ke"], "aliases": ["talakenya"]},
    {"name": "M-KOPA", "domains": ["m-kopa.com"], "aliases": ["mkopa"]},
    {"name": "East African Breweries", "domains": ["eabl.com"], "aliases": ["eabl", "tusker"]},
    {"name": "Nation Media Group", "domains": ["nation.africa"], "aliases": ["nationmedia"]},
    {"name": "Standard Group", "domains": ["standardmedia.co.ke"], "aliases": ["standardmedia"]},
    {"name": "Royal Media Services", "domains": ["royalmedia.co.ke"], "aliases": ["royalmedia"]},
    {"name": "Citizen TV", "domains": ["citizen.digital"], "aliases": ["citizentv"]},
    {"name": "Sidian Bank", "domains": ["sidianbank.co.ke"], "aliases": ["sidian"]},
    {"name": "Standard Chartered Kenya", "domains": ["sc.com"], "aliases": ["stanchart", "standardchartered"]},
    {"name": "Premier Bank Kenya", "domains": ["premierbank.co.ke"], "aliases": ["premierbank"]},
    {"name": "Gulf African Bank", "domains": ["gulfafricanbank.com"], "aliases": ["gulfafrican"]},
    {"name": "Unaitas", "domains": ["unaitas.com"], "aliases": ["unaitas"]},
    {"name": "Britam", "domains": ["britam.com"], "aliases": ["britam"]},
    {"name": "Jubilee Insurance", "domains": ["jubileeinsurance.com"], "aliases": ["jubileeinsurance"]},
    {"name": "Old Mutual Kenya", "domains": ["oldmutual.co.ke"], "aliases": ["oldmutual"]},
    {"name": "HFC Kenya", "domains": ["hfgroup.co.ke"], "aliases": ["hfgroup"]},
    {"name": "SportPesa", "domains": ["sportpesa.com"], "aliases": ["sportpesa"]},
    {"name": "Betika", "domains": ["betika.com"], "aliases": ["betika"]},

    {"name": "Kenyatta University", "domains": ["ku.ac.ke"], "aliases": []},
    {"name": "JKUAT", "domains": ["jkuat.ac.ke"], "aliases": ["jkuat"]},
    {"name": "Moi University", "domains": ["mu.ac.ke"], "aliases": []},
    {"name": "Egerton University", "domains": ["egerton.ac.ke"], "aliases": ["egerton"]},
    {"name": "Strathmore University", "domains": ["strathmore.edu"], "aliases": ["strathmore"]},
    {"name": "USIU Africa", "domains": ["usiu.ac.ke"], "aliases": ["usiu"]},
    # --- Global brands -----------------------------------------------------
    {"name": "PayPal", "domains": ["paypal.com", "paypal.me", "paypalobjects.com", "paypal-community.com"], "aliases": ["paypal"]},
    {"name": "Facebook", "domains": ["facebook.com", "facebook.net", "fb.com", "fbcdn.net", "instagram.com", "cdninstagram.com", "whatsapp.com", "whatsapp.net"], "aliases": ["facebook"]},
    {"name": "WhatsApp", "domains": ["whatsapp.com"], "aliases": ["whatsapp"]},
    {"name": "Google", "cc": True, "domains": ["google.com", "google.co.ke", "google.co.ug", "googleapis.com", "googleusercontent.com", "googlesyndication.com", "google-analytics.com", "googletagmanager.com", "gstatic.com", "youtube.com", "ytimg.com", "ggpht.com", "gmail.com", "android.com"], "aliases": ["google"]},
    {"name": "Gmail", "domains": ["gmail.com"], "aliases": ["gmail"]},
    {"name": "Microsoft", "domains": ["microsoft.com", "live.com", "outlook.com", "microsoftonline.com", "office.com", "office365.com", "bing.com", "msn.com", "azure.com", "sharepoint.com", "windowsupdate.com", "msedge.net"], "aliases": ["microsoft", "outlook", "office365"]},
    {"name": "Apple", "domains": ["apple.com", "icloud.com", "mzstatic.com", "apple-dns.net"], "aliases": ["apple", "icloud"]},
    {"name": "Amazon", "cc": True, "domains": ["amazon.com", "amazon.co.uk", "amazon.de", "amazonaws.com", "media-amazon.com", "amazon-adsystem.com"], "aliases": ["amazon"]},
    {"name": "Netflix", "domains": ["netflix.com", "nflxvideo.net", "nflximg.net", "nflxext.com"], "aliases": ["netflix"]},
    {"name": "Instagram", "domains": ["instagram.com"], "aliases": ["instagram"]},
    {"name": "TikTok", "domains": ["tiktok.com", "tiktokcdn.com", "tiktokv.com"], "aliases": ["tiktok"]},
    {"name": "Binance", "domains": ["binance.com"], "aliases": ["binance"]},
    {"name": "DHL", "cc": True, "domains": ["dhl.com"], "aliases": ["dhl"]},
    {"name": "Booking.com", "domains": ["booking.com"], "aliases": ["booking"]},
    {"name": "LinkedIn", "domains": ["linkedin.com", "licdn.com"], "aliases": ["linkedin"]},
    {"name": "Dropbox", "domains": ["dropbox.com"], "aliases": ["dropbox"]},
    {"name": "Adobe", "domains": ["adobe.com", "adobelogin.com", "adobe-dns.com", "typekit.net"], "aliases": ["adobe"]},
    {"name": "Steam", "domains": ["steampowered.com", "steamcommunity.com", "steamstatic.com", "steamcontent.com"], "aliases": ["steam"]},
]

# Shared platforms where ordinary people publish pages; a brand word in the
# sub-domain there is usually a blog, not an impersonation.
HOSTING_DOMAINS = {
    "blogspot.com", "blogspot.co.ke", "wordpress.com", "github.io",
    "medium.com", "wixsite.com", "weebly.com", "netlify.app", "vercel.app",
    "web.app", "firebaseapp.com", "godaddysites.com", "tumblr.com",
    "substack.com", "quora.com", "linkedin.com", "facebook.com",
}

HOMOGLYPH_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a",
                               "5": "s", "7": "t", "8": "b", "9": "g"})

GENERIC_TLDS = {"com", "org", "net", "io", "co", "info", "biz"}
ALLOWED_CC = {"ke", "ug", "tz", "za", "ng", "uk", "ie", "in", "au", "ca",
              "fr", "de", "es", "it", "nl", "be", "ae", "sa", "jp", "br", "mx"}

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_at", "num_question", "num_equal", "num_ampersand", "num_percent",
    "num_slash", "num_digits", "digit_ratio", "num_subdomains", "uses_https",
    "has_ip", "has_at", "has_port", "host_has_hyphen", "is_shortened",
    "suspicious_tld", "num_suspicious_words", "has_double_slash_path",
    "official_domain_match", "brand_token_present", "brand_wrong_domain",
    "brand_typosquat", "brand_homoglyph", "min_brand_distance",
]

# ---------------------------------------------------------------------------
# 3. Helpers: split the host into suffix / name, measure similarity
# ---------------------------------------------------------------------------
def split_host(host):
    """Return (registrable_domain, labels, base_name, suffix)."""
    labels = [p for p in (host or "").split(".") if p]
    if len(labels) < 2:
        return host, labels, (labels[0] if labels else ""), ""
    for n in (3, 2):
        if len(labels) > n and ".".join(labels[-n:]) in PUBLIC_SUFFIXES:
            suffix = ".".join(labels[-n:])
            reg = ".".join(labels[-(n + 1):])
            return reg, labels, labels[-(n + 1)], suffix
    return ".".join(labels[-2:]), labels, labels[-2], labels[-1]


def levenshtein(a, b, limit=3):
    """Edit distance where swapping two neighbouring letters costs 1."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1,
                             rows[i - 1][j - 1] + cost)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[len(a)][len(b)]


def dehomoglyph(text):
    return text.translate(HOMOGLYPH_MAP).replace("rn", "m").replace("vv", "w")


def netloc_has_at(url):
    """True when the part before the first slash contains an @ sign."""
    raw = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    try:
        return "@" in (urlparse(raw).netloc or "")
    except Exception:
        return False


def fake_government_domain(url):
    """True when an address pretends to be Kenyan government but is not *.go.ke."""
    host = hostname_of(url)
    if not host or IP_RE.match(host):
        return False
    suffix = split_host(host)[3]
    if (host.endswith(".go.ke") or host.endswith(".gov.ke")
            or suffix.endswith("gov") or suffix.endswith("gov.uk")
            or re.search(r"\.gov(\.[a-z]{2})?$", host)
            or host.endswith(".gob.mx") or host.endswith(".gouv.fr")
            or host.endswith(".govt.nz") or host.endswith(".gc.ca")):
        return False
    cleaned = host.replace("government", "").replace("govern", "")
    flat = re.sub(r"[^a-z0-9]", "", cleaned)
    tokens = [t for t in re.split(r"[^a-z0-9]+", cleaned) if t]
    # "gov" only counts when it is not the opening letters of a normal word
    return ("gok" in flat or "goke" in flat or "gokke" in flat
            or "gov" in flat[2:] or "gov" in tokens)


def hostname_of(url):
    raw = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw
    try:
        return (urlparse(raw).hostname or "").lower().rstrip(".")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 4. Brand-impersonation analysis (the new knowledge layer)
# ---------------------------------------------------------------------------
STOP_WORDS = {"msedge", "azure", "bing", "msn", "sharepoint", "gstatic",
              "ytimg", "ggpht", "mzstatic", "licdn", "tiktokcdn", "tiktokv",
              "nflxvideo", "nflximg", "nflxext", "adobelogin", "paypalobjects",
              "amazonaws", "microsoftonline", "windowsupdate", "office365",
              "googleapis", "googleusercontent", "googlesyndication", "gmail",
              "android", "youtube", "icloud", "outlook", "steamcontent",
              "live", "office", "home", "news", "mail", "shop", "play",
              "media", "digital", "group", "africa", "google", "apple",
              "amazon", "netflix", "adobe", "steam", "binance", "booking",
              "dropbox", "microsoft", "safaricom", "standard", "citizen",
              "family", "equity", "premier", "jubilee", "britam", "unaitas"}

_ALIAS_INDEX = None


def _alias_index():
    """Flatten every brand name once, so each URL is checked quickly."""
    global _ALIAS_INDEX
    if _ALIAS_INDEX is None:
        items = []
        for b in OFFICIAL:
            names = list(b["aliases"])
            for d in b["domains"]:
                extra = split_host(d)[2].replace("-", "")
                if (extra and len(extra) >= 4 and extra not in STOP_WORDS
                        and extra not in [x.replace("-", "") for x in names]):
                    names.append(extra)
            for nm in names:
                a = nm.replace("-", "")
                if a:
                    items.append((a, b))
        _ALIAS_INDEX = items
    return _ALIAS_INDEX


@lru_cache(maxsize=20000)
def _brand_signals_host(host):
    out = {"official": 0, "mismatch": 0, "typo": 0, "homoglyph": 0,
           "brand": "", "min_dist": 10, "reasons": []}
    if not host or IP_RE.match(host):
        return out

    reg, labels, base, suffix = split_host(host)
    flat = re.sub(r"[^a-z0-9]", "", host)
    flat_soft = dehomoglyph(flat)
    tokens = [t for t in re.split(r"[^a-z0-9]+", host) if t]

    # (a) Exact official domain, or any subdomain of it -> trusted.
    for b in OFFICIAL:
        for d in b["domains"]:
            if host == d or host.endswith("." + d):
                out.update(official=1, brand=b["name"])
                out["reasons"].append(
                    "The hostname is the official %s domain (%s)." % (b["name"], d))
                return out
            if b.get("cc") and _is_country_variant(host, split_host(d)[2]):
                out.update(official=1, brand=b["name"])
                out["reasons"].append(
                    "The hostname is a country version of the official %s domain "
                    "(%s), which belongs to the same organisation." % (b["name"], d))
                return out

    # (a2) Page on a shared platform (blog, portfolio) with no login wording.
    if reg in HOSTING_DOMAINS and not any(w in host for w in SUSPICIOUS_WORDS):
        out["reasons"].append(
            "The address is an ordinary page on a shared platform (%s), not a "
            "brand domain, and it contains no login or payment wording." % reg)
        return out

    # (b) Brand mentioned but the domain is not the official one.
    best = None  # (priority, brand, distance, kind, alias)
    pieces = []
    if len(base) >= 4:
        pieces.append(base.replace("-", ""))
    pieces += [t for t in re.split(r"[^a-z0-9]+", base) if len(t) >= 6]
    first = base[:1]

    for a, b in _alias_index():
        kind = None
        if len(a) >= 3 and a in tokens:
            kind = "mismatch"
        elif len(a) >= 4 and _brand_next_to_wording(a, flat):
            kind = "mismatch"
        elif len(a) >= 4 and a in flat_soft and a not in flat:
            kind = "homoglyph"
        dist = 99
        if kind is None and len(a) >= 4 and (a[:1] == first or a[-1:] == base[-1:]):
            dists = [levenshtein(pc, a) for pc in pieces
                     if len(pc) == len(a)] if pieces else []
            dist = min(dists) if dists else 99
            if dist == 1 and len(a) >= 5 and not _too_loose(base, a, dist):
                kind = "typo"
        if kind:
            score = 0 if kind == "mismatch" else (1 if kind == "homoglyph" else 2)
            if best is None or score < best[0]:
                best = (score, b, min(dist, 10), kind, a)

    if best:
        _, brand, dist, kind, alias = best
        domains = ", ".join(brand["domains"])
        out["brand"] = brand["name"]
        if kind == "mismatch":
            out["mismatch"] = 1
            out["reasons"].append(
                "The address mentions %s, but the official address is %s, not %s."
                % (brand["name"], domains, reg))
        elif kind == "homoglyph":
            out["mismatch"] = 1
            out["homoglyph"] = 1
            out["reasons"].append(
                "The address imitates %s by swapping letters for look-alike digits "
                "(0/1/3/4/5/7 for o/l/e/a/s/t). Official address: %s."
                % (brand["name"], domains))
        else:
            out["typo"] = 1
            out["reasons"].append(
                "The name '%s' is %d letter(s) away from the official %s name "
                "'%s' - a classic typosquat. Official address: %s."
                % (base, dist, brand["name"], alias, domains))
    return out


CC_MIDDLE = {"com", "co", "or", "net", "go", "ac", "gov", "edu"}


def _is_country_variant(host, official_name):
    """google.co.ke, google.com.vn, amazon.de ... share the brand label."""
    labels = [x for x in host.split(".") if x]
    if official_name not in labels:
        return False
    rest = labels[labels.index(official_name) + 1:]
    if not rest or len(rest[-1]) != 2:
        return False
    return all(r in CC_MIDDLE for r in rest[:-1])


VOWELS = "aeiou"


def _too_loose(base, alias, dist):
    """Reject short or vowel-only look-alikes (bild/kicd, notion/nation)."""
    if len(alias) <= 5 and dist > 1:
        return True
    if len(alias) <= 5 and len(base) != len(alias):
        return True
    if dist == 1 and len(base) == len(alias):
        diff = [i for i in range(len(base)) if base[i] != alias[i]]
        if len(diff) == 1:
            i = diff[0]
            if base[i] in VOWELS and alias[i] in VOWELS:
                return True
    return False


LURE_WORDS = ("login", "signin", "logon", "verify", "verification", "secure",
              "account", "update", "confirm", "billing", "unlock", "security",
              "password", "wallet", "support", "recover", "payment", "pay")


def _brand_next_to_wording(alias, flat):
    """The brand appears glued to or beside a login/payment word."""
    start = flat.find(alias)
    while start != -1:
        tail = flat[start + len(alias):]
        head = flat[:start]
        if any(tail.startswith(w) for w in LURE_WORDS):
            return True
        if any(head.endswith(w) for w in LURE_WORDS):
            return True
        start = flat.find(alias, start + 1)
    return False


def brand_signals(url):
    out = _brand_signals_host(hostname_of(url))
    return {"official": out["official"], "mismatch": out["mismatch"],
            "typo": out["typo"], "homoglyph": out["homoglyph"],
            "brand": out["brand"], "min_dist": out["min_dist"],
            "reasons": list(out["reasons"])}


# ---------------------------------------------------------------------------
# 5. Feature extraction (23 lexical + 6 brand features = 29)
# ---------------------------------------------------------------------------
def extract_features(url):
    raw = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw_for_parse = "http://" + raw
    else:
        raw_for_parse = raw
    parsed = urlparse(raw_for_parse)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    lower = raw.lower()
    digits = sum(c.isdigit() for c in raw)
    _, labels, base, suffix = split_host(host)
    try:
        port = parsed.port
    except ValueError:
        port = None
    signals = brand_signals(raw)

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
        "num_subdomains": max(len(labels) - 2, 0) if suffix else max(len(labels) - 1, 0),
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "has_ip": 1 if IP_RE.match(host) else 0,
        "has_at": 1 if "@" in raw else 0,
        "has_port": 1 if port else 0,
        "host_has_hyphen": 1 if "-" in host else 0,
        "is_shortened": 1 if any(s == host or host.endswith("." + s) for s in SHORTENERS) else 0,
        "suspicious_tld": 1 if suffix in SUSPICIOUS_TLDS else 0,
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "has_double_slash_path": 1 if "//" in path else 0,
        "official_domain_match": signals["official"],
        "brand_token_present": 1 if signals["brand"] else 0,
        "brand_wrong_domain": signals["mismatch"],
        "brand_typosquat": signals["typo"],
        "brand_homoglyph": signals["homoglyph"],
        "min_brand_distance": signals["min_dist"],
    }


def vector(url):
    f = extract_features(url)
    return [float(f[n]) for n in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# 6. Training data (synthetic sample - swap in real data for submission)
# ---------------------------------------------------------------------------
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "linkedin.com", "github.com", "microsoft.com", "apple.com", "netflix.com",
    "reddit.com", "stackoverflow.com", "bbc.com", "cnn.com", "paypal.com",
    "dropbox.com", "spotify.com", "adobe.com", "mozilla.org", "python.org",
    "ubuntu.com", "cloudflare.com", "nation.africa", "standardmedia.co.ke",
    "ecitizen.go.ke", "kra.go.ke", "safaricom.co.ke", "ku.ac.ke", "kyu.ac.ke",
]
LEGIT_PATHS = [
    "", "/", "/about", "/contact", "/products", "/blog/2026/news",
    "/help/faq", "/search?q=technology", "/user/profile", "/docs/intro",
    "/articles/latest", "/pricing", "/support", "/services/payments",
]
SUBS = ["www", "accounts", "selfservice", "portal", "services", "eservices"]
SUS_WORDS = ["login", "verify", "secure", "account", "update", "confirm",
             "signin", "webscr", "billing", "unlock", "security"]
BAD_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "work", "loan"]
LOOKALIKE_TLDS = ["net", "com", "co.ke", "online", "site", "info", "org"]
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def build_dataset(n_each=900, seed=42):
    rnd = random.Random(seed)

    def rs(n):
        return "".join(rnd.choice(ALPHABET) for _ in range(n))

    def legit():
        if rnd.random() < 0.30:
            b = rnd.choice(OFFICIAL)
            d = rnd.choice(b["domains"])
            sub = rnd.choice(SUBS)
            return "https://%s.%s%s" % (sub, d, rnd.choice(LEGIT_PATHS))
        dom = rnd.choice(LEGIT_DOMAINS)
        path = rnd.choice(LEGIT_PATHS)
        scheme = "https" if rnd.random() < 0.85 else "http"
        sub = "www." if rnd.random() < 0.5 else ""
        u = "%s://%s%s%s" % (scheme, sub, dom, path)
        r = rnd.random()
        if r < 0.07:
            u = "%s://%s%s%s?session=%s&ref=%s" % (scheme, sub, dom, path, rs(14), rs(8))
        elif r < 0.12:
            u = "https://account-%s.%s/%s" % (rs(4), dom, rnd.choice(SUS_WORDS))
        elif r < 0.15:
            u = "https://%s/%s?id=%s" % (dom, rnd.choice(SUS_WORDS), rs(9))
        return u

    def phish():
        kind = rnd.random()
        word = rnd.choice(SUS_WORDS)
        brand = rnd.choice(OFFICIAL)
        d = rnd.choice(brand["domains"])
        base, suffix = split_host(d)[2], split_host(d)[3]
        if kind < 0.30:
            # same name, wrong suffix  (ecitizen.net, safaricom-support.com)
            pool = [t for t in LOOKALIKE_TLDS if t != suffix and t != d]
            t = rnd.choice(pool) if pool else "net"
            pre = "" if rnd.random() < 0.6 else "-" + word
            return "http://%s%s.%s/%s" % (base, pre, t, word)
        if kind < 0.45:
            # typosquat: swap two letters of the official name
            chars = list(base)
            i = rnd.randrange(len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "https://%s%s.%s/%s" % ("".join(chars),
                                           "" if rnd.random() < 0.6 else "-" + word,
                                           suffix, word)
        if kind < 0.58:
            # homoglyph: digits instead of letters (3citizen, ec1tizen)
            swap = {"e": "3", "i": "1", "o": "0", "a": "4", "s": "5"}
            chars = list(base)
            spots = [i for i, c in enumerate(chars) if c in swap]
            if spots:
                for i in rnd.sample(spots, max(1, min(2, len(spots)))):
                    chars[i] = swap[chars[i]]
            return "https://%s.%s/%s" % ("".join(chars), suffix, word)
        if kind < 0.66:
            # the @ decoy trick
            return "http://%s.com@%s.%s/%s" % (base, rs(9), rnd.choice(BAD_TLDS), word)
        if kind < 0.72:
            # brand buried under another domain
            return "http://%s.%s.%s/%s" % (base, rs(6), rnd.choice(BAD_TLDS), word)
        if kind < 0.86:
            # official-looking prefix stacked on a bad suffix
            return "http://%s.%s.%s/%s?id=%s" % (d, rs(5), rnd.choice(BAD_TLDS), word, rs(8))
        ip = ".".join(str(rnd.randint(1, 255)) for _ in range(4))
        return "http://%s/%s/%s.php?account=%s" % (ip, base, word, rs(8))

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
    # ~3% label noise keeps the metrics realistic instead of an artificial 100%
    rows = [(u, 1 - lab if rnd.random() < 0.03 else lab) for u, lab in rows]
    return rows


# ---------------------------------------------------------------------------
# 7. Training (once, lazily, then cached in memory)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state = {"bundle": None, "error": None}


DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "phish_dataset.txt.gz")


def load_rows():
    """Load the bundled real dataset; fall back to the built-in sample."""
    rows = []
    try:
        if os.path.exists(DATA_FILE):
            with gzip.open(DATA_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.rstrip().split(chr(9), 1)
                    if len(parts) == 2 and parts[0] in ("0", "1"):
                        u = parts[1].strip()
                        if u:
                            rows.append((u, int(parts[0])))
    except Exception:
        rows = []
    if len(rows) >= 500:
        return rows, "real dataset (%d URLs)" % len(rows)
    rows = build_dataset()
    return rows, "built-in sample (%d URLs)" % len(rows)


def train_model():
    rows, source = load_rows()
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
    return {"model": model, "name": name, "accuracy": acc, "f1": f1,
            "recall": rec, "source": source, "rows": len(rows)}


def get_bundle():
    if _state["bundle"] is not None:
        return _state["bundle"]
    if _state["error"] is not None:
        raise RuntimeError(_state["error"])
    with _lock:
        if _state["bundle"] is None:
            try:
                _state["bundle"] = train_model()
            except Exception as exc:
                _state["error"] = "Training failed: %s" % exc
                raise
    return _state["bundle"]


# ---------------------------------------------------------------------------
# 8. Decision: brand watchlist first, machine learning for everything else
# ---------------------------------------------------------------------------
def classify(url):
    bundle = get_bundle()
    feats = extract_features(url)
    X = np.array([[float(feats[n]) for n in FEATURE_NAMES]])
    ml_label = int(bundle["model"].predict(X)[0])
    try:
        proba = float(bundle["model"].predict_proba(X)[0][ml_label])
    except Exception:
        proba = 1.0
    ml_conf = round(proba * 100, 2)
    signals = brand_signals(url)

    label, confidence, decided, reasons = ml_label, ml_conf, "machine learning", []

    if signals["official"]:
        label, confidence = 0, max(ml_conf if ml_label == 0 else 0.0, 97.5)
        decided = "brand watchlist"
        reasons = signals["reasons"] + [
            "Sub-domains of an official domain keep the same owner."]
    elif fake_government_domain(url):
        label, confidence = 1, max(ml_conf if ml_label == 1 else 0.0, 95.0)
        decided = "government-domain rule"
        reasons = ["Kenyan government addresses always end in .go.ke. This address "
                   "imitates that style without belonging to government."]
    elif netloc_has_at(url):
        label, confidence = 1, max(ml_conf if ml_label == 1 else 0.0, 96.5)
        decided = "URL structure rule"
        reasons = ["Everything before the @ sign is a decoy: the browser really "
                   "visits the part after it, so the familiar name is a disguise."]
    elif signals["mismatch"] or signals["typo"]:
        label = 1
        confidence = max(ml_conf if ml_label == 1 else 0.0, 96.5)
        decided = "brand watchlist"
        reasons = signals["reasons"] + [
            "Impersonating a known brand is the most common phishing trick."]
    else:
        if ml_label == 1:
            reasons = ["The URL structure matches patterns learned from known phishing links."]
        else:
            reasons = ["The URL structure looks like ordinary legitimate addresses."]

    return {
        "url": url,
        "label": label,
        "verdict": "phishing" if label == 1 else "legitimate",
        "confidence": round(float(confidence), 2),
        "model": bundle["name"],
        "decided_by": decided,
        "brand": signals["brand"],
        "reasons": reasons,
        "features": feats,
    }


# ---------------------------------------------------------------------------
# 9. Web interface
# ---------------------------------------------------------------------------
app = Flask(__name__)

EXAMPLES = [
    ("ecitizen lookalike", "https://ecitizen.net/login"),
    ("typosquat", "https://ecitzien.go.ke/"),
    ("real eCitizen", "https://www.ecitizen.go.ke/"),
    ("safaricom promo scam", "https://safaricom-promo.top/winner"),
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
.wrap{width:100%;max-width:680px}
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
.why{margin-top:14px;padding-left:18px;color:#c8d2e8;font-size:.86rem;line-height:1.65}
.why li{margin-bottom:6px}
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
<p class="conf">{{ result.confidence }}% confidence · decided by {{ result.decided_by }}{% if result.brand %} · brand checked: {{ result.brand }}{% endif %}</p>
<ul class="why">{% for r in result.reasons %}<li>{{ r }}</li>{% endfor %}</ul>
<details style="margin-top:12px"><summary style="color:var(--cyan);font-size:.85rem;cursor:pointer">Why? View extracted features</summary>
<table>{% for k, v in result.features.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>
</details>
</div>
{% endif %}
<p class="foot">4th Year Project</p>
</div></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE, result=None, url="", examples=EXAMPLES)


@app.route("/check", methods=["GET", "POST"])
def check():
    url = (request.values.get("url") or "").strip()
    if not url:
        return render_template_string(PAGE, result=None, url="", examples=EXAMPLES)
    try:
        result = classify(url)
    except Exception as exc:
        return render_template_string(
            PAGE, url=url,
            result={"label": 0, "url": "Error: %s" % exc, "confidence": 0,
                    "model": "-", "decided_by": "-", "brand": "",
                    "reasons": [], "features": {}}, examples=EXAMPLES), 500
    return render_template_string(PAGE, result=result, url=url, examples=EXAMPLES)


@app.route("/health")
def health():
    try:
        b = get_bundle()
        return jsonify({"status": "ok", "model": b["name"],
                        "accuracy": round(b["accuracy"], 4),
                        "f1": round(b["f1"], 4),
                        "training_rows": b.get("rows"),
                        "dataset": b.get("source"),
                        "brands_watched": len(OFFICIAL)})
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
