from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlsplit

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u180e"), None)

DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹०१२३४५६७८९০১২৩৪৫৬৭৮৯",
    "0123456789012345678901234567890123456789",
)

COMMON_TLDS = {
    "com", "net", "org", "edu", "gov", "mil", "int", "io", "co", "ai", "app",
    "dev", "me", "info", "biz", "xyz", "top", "shop", "store", "online", "site",
    "website", "space", "live", "link", "click", "club", "vip", "icu", "buzz",
    "monster", "rest", "fun", "work", "life", "world", "today", "news", "blog",
    "wang", "win", "bid", "loan", "download", "stream", "racing", "party",
    "review", "date", "men", "ren", "mom", "dad", "prof", "quest", "cyou",
    "sbs", "cfd", "bond", "beauty", "hair", "skin", "makeup", "kred",
    "accountant", "science", "gdn", "agency", "solutions", "services", "support",
    "help", "center", "digital", "media", "network", "systems", "email", "mail",
    "tech", "technology", "finance", "financial", "capital", "markets", "trade",
    "trading", "exchange", "crypto", "coin", "wallet", "bank", "banking", "pay",
    "payments", "money", "cash", "credit", "loans", "insurance", "legal", "law",
    "tax", "travel", "tours", "holiday", "hotel", "booking", "delivery",
    "express", "logistics", "courier", "shipping", "track", "parcel", "secure",
    "security", "verify", "verification", "login", "signin", "auth", "auths",
    "update", "alert", "notify", "govt", "gob", "bd", "in", "pk", "uk", "us",
    "ca", "au", "de", "fr", "es", "it", "nl", "be", "ch", "at", "se", "no",
    "dk", "fi", "pl", "ru", "ua", "kz", "tr", "sa", "ae", "qa", "kw", "eg",
    "ma", "dz", "tn", "ng", "ke", "za", "gh", "br", "mx", "ar", "cl", "pe",
    "ve", "id", "my", "sg", "ph", "vn", "th", "jp", "kr", "cn", "hk", "tw",
    "ir", "iq", "il", "lb", "jo", "sy", "ye", "om", "bh", "af", "lk", "np",
    "mm", "kh", "la", "mn", "ms", "cc", "tv", "ws", "fm", "am", "pm", "is",
    "st", "to", "tk", "ml", "ga", "gq", "cf", "sz", "li", "nu", "vg", "tc",
    "gd", "vc", "kn", "ag", "sh", "ac", "gg", "je", "im", "su", "pw", "ly",
    "gl", "do", "gy", "pe", "xxx", "adult", "porn", "sex", "casino", "bet",
    "poker", "lottery", "prize", "gift", "free", "discount", "sale", "deals",
    "offer", "promo", "coupon", "cashback", "rewards",
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "info", "click", "link", "online", "site", "icu", "buzz",
    "rest", "monster", "cf", "tk", "ml", "ga", "gq", "work", "loan", "zip",
    "mov", "live", "vip", "win", "bid", "download", "stream", "racing",
    "party", "review", "date", "men", "ren", "mom", "dad", "prof", "quest",
    "cyou", "sbs", "cfd", "bond", "beauty", "hair", "skin", "makeup", "kred",
    "accountant", "science", "gdn", "agency", "solutions", "services", "support",
    "help", "center", "digital", "media", "network", "systems", "email", "mail",
    "tech", "technology", "finance", "financial", "capital", "markets", "trade",
    "trading", "exchange", "crypto", "coin", "wallet", "bank", "banking", "pay",
    "payments", "money", "cash", "credit", "loans", "insurance", "legal", "law",
    "tax", "travel", "tours", "holiday", "hotel", "booking", "delivery",
    "express", "logistics", "courier", "shipping", "track", "parcel", "secure",
    "security", "verify", "verification", "login", "signin", "auth", "auths",
    "update", "alert", "notify", "lottery", "prize", "gift", "free",
    "discount", "sale", "deals", "offer", "promo", "coupon", "cashback",
    "rewards", "casino", "bet", "poker", "adult", "porn", "sex", "xxx",
}

SHORTENER_DOMAINS = {
    "bit.ly", "bit.do", "tinyurl.com", "tiny.cc", "t.co", "goo.gl", "ow.ly",
    "cutt.ly", "rebrand.ly", "is.gd", "rb.gy", "shorturl.at", "t.ly", "s.id",
    "lnkd.in", "buff.ly", "adf.ly", "shorte.st", "bc.vc", "soo.gd", "tr.im",
    "v.gd", "x.co", "me2.do", "qr.ae", "clck.ru", "u.to", "tiny.one",
    "urlz.fr", "1url.com", "shrtco.de", "9qr.de", "rotf.lol", "rb.gy",
    "wa.me", "t.me", "telegram.me", "paypal.me", "cash.app", "linktr.ee",
    "linkin.bio", "bio.link", "snip.ly", "lnk.to", "trib.al", "dlvr.it",
}

LEGITIMATE_DOMAINS = {
    "amazon.com", "amazon.in", "amazon.co.uk", "flipkart.com", "myntra.com",
    "swiggy.com", "zomato.com", "uber.com", "ola.cab", "irctc.co.in",
    "sbi.co.in", "onlinesbi.sbi", "icicibank.com", "hdfcbank.com",
    "axisbank.com", "kotak.com", "paytm.com", "phonepe.com", "google.com",
    "gmail.com", "microsoft.com", "apple.com", "netflix.com", "whatsapp.com",
    "facebook.com", "instagram.com", "linkedin.com", "x.com", "twitter.com",
    "github.com", "google.co.in", "gov.in", "nic.in", "india.gov.in",
    "bengaluruurban.nic.in", "delhipolice.gov.in", "uidai.gov.in",
    "incometax.gov.in", "gst.gov.in", "epfindia.gov.in", "passportindia.gov.in",
    "bkash.com", "nagad.com.bd", "rocket.com.bd", "upay.com.bd", "bKash.com.bd",
    "grameenphone.com", "robi.com.bd", "banglalink.net", "teletalk.com.bd",
    "dhl.com", "fedex.com", "ups.com", "usps.com", "royal.mail",
    "yodel.co.uk", "dpd.co.uk", "evri.com", "delhivery.com", "bluedart.com",
    "indigo.com", "airindia.com", "emirates.com", "qatarairways.com",
    "meet.google.com", "zoom.us", "teams.microsoft.com", "webex.com",
}

URL_RE = re.compile(
    r"(?i)(?:(?:https?://|hxxps?://|www\.)[^\s<>\"']+|"
    r"(?<![\w@])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:"
    + "|".join(sorted(COMMON_TLDS, key=len, reverse=True))
    + r")\b(?:/[^\s<>\"']*)?)"
)

SPACED_LETTER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z][\s._-]+){3,}[A-Za-z](?![A-Za-z0-9])"
)

EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,24}\b")
IP_URL_RE = re.compile(r"(?i)\b(?:\d{1,3}\.){3}\d{1,3}\b")
ETH_ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
BTC_ADDRESS_RE = re.compile(r"\b(?:bc1[a-zA-HJ-NP-Z0-9]{25,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")

CURRENCY_TOKEN = (
    r"(?:\$|€|£|₹|₨|₺|₽|¥|₩|₪|৳|Rs\.?|INR|USD|EUR|GBP|BDT|PKR|NGN|"
    r"TK|Tk|taka|pesos?|euros?|dollars?|pounds?|rupees?|dirhams?|riyals?|"
    r"دولار|دولارات|ريال|درهم|টাকা|রুপি)"
)
AMOUNT_BEFORE_RE = re.compile(
    rf"(?i)({CURRENCY_TOKEN})\s*([0-9][0-9,.]{{0,18}})(?:\s*(lakh|lac|crore|k|m|million|billion))?"
)
AMOUNT_AFTER_RE = re.compile(
    rf"(?i)([0-9][0-9,.]{{0,18}})\s*(lakh|lac|crore|k|m|million|billion)?\s*({CURRENCY_TOKEN})"
)
CRYPTO_AMOUNT_RE = re.compile(
    r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*(BTC|ETH|USDT|USDC|BNB|SOL|XRP|DOGE|LTC|TRX)\b"
)
PHONE_RE = re.compile(r"(?<![\d+])(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
OTP_CONTEXT_RE = re.compile(
    r"(?i)\b(?:otp|o\.t\.p|verification\s+code|security\s+code|one[-\s]?time\s+password|"
    r"code|pin|password|passcode)\b[^\d]{0,28}(\d{4,8})\b"
)
OTP_CONTEXT_RE_2 = re.compile(
    r"\b(\d{4,8})\b[^\d]{0,18}\b(?:otp|code|pin|password|passcode)\b", re.I
)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_text(text: object) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH)
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf" or cat == "Cs":
            continue
        if cat == "Cc" and ch not in "\n\r\t":
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION)


def refang(text: str) -> str:
    t = clean_text(text)
    t = re.sub(r"\[\s*\.\s*\]", ".", t)
    t = re.sub(r"\(\s*\.\s*\)", ".", t)
    t = re.sub(r"\{\s*\.\s*\}", ".", t)
    t = re.sub(r"\(\s*dot\s*\)", ".", t, flags=re.I)
    t = re.sub(r"\[\s*dot\s*\]", ".", t, flags=re.I)
    t = re.sub(r"\s+dot\s+", ".", t, flags=re.I)
    t = re.sub(r"\bhxxps\b", "https", t, flags=re.I)
    t = re.sub(r"\bhxxp\b", "http", t, flags=re.I)

    def _join_spaced(match: re.Match) -> str:
        raw = match.group(0)
        joined = re.sub(r"[\s._-]", "", raw)
        return joined if len(joined) >= 3 else raw

    t = SPACED_LETTER_RE.sub(_join_spaced, t)
    return t


def canonical_for_dedupe(text: object) -> str:
    t = refang(str(text or "")).casefold()
    t = re.sub(r"(?:https?://|www\.)\S+", " url ", t)
    t = EMAIL_RE.sub(" email ", t)
    t = re.sub(r"\d+(?:[.,]\d+)*", " 0 ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    t = re.sub(r"[^\w\s\u0600-\u06ff\u0900-\u097f\u0980-\u09ff]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:600]


def text_hash(text: object) -> str:
    return hashlib.blake2b(
        canonical_for_dedupe(text).encode("utf-8"), digest_size=16
    ).hexdigest()


def normalize_url(raw: str) -> tuple[str | None, str | None, str | None]:
    value = clean_text(raw)
    if not value:
        return None, None, None
    candidate = value if re.match(r"(?i)^(?:https?|hxxps?)://", value) else "http://" + value
    candidate = re.sub(r"(?i)^hxxp", "http", candidate)
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None, None, None
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return None, None, None
    host = host[4:] if host.startswith("www.") else host
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    normalized = f"http://{host}{parts.path or ''}"
    return host, tld, normalized


@dataclass(frozen=True)
class UrlInfo:
    raw: str
    domain: str | None
    tld: str
    normalized: str | None
    is_shortener: bool
    is_suspicious_tld: bool
    is_ip: bool
    has_punycode: bool
    is_obfuscated: bool
    has_credentials: bool
    path_length: int

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "domain": self.domain,
            "tld": self.tld,
            "is_shortener": self.is_shortener,
            "is_suspicious_tld": self.is_suspicious_tld,
            "is_ip": self.is_ip,
            "has_punycode": self.has_punycode,
            "is_obfuscated": self.is_obfuscated,
            "has_credentials": self.has_credentials,
            "path_length": self.path_length,
        }


def extract_urls(text: object) -> list[UrlInfo]:
    original = clean_text(text)
    if not original:
        return []
    refanged = refang(original)
    lowered = original.casefold()
    obfuscation_present = any(
        marker in lowered for marker in ("[.]", "(.)", "{.}", " dot ", "hxxp")
    )
    results: list[UrlInfo] = []
    seen: set[str] = set()
    for match in URL_RE.finditer(refanged):
        raw = match.group(0).rstrip(".,;:!?)]}'\"")
        domain, tld, normalized = normalize_url(raw)
        if not domain:
            continue
        key = normalized or domain
        if key in seen:
            continue
        seen.add(key)
        is_ip = bool(IP_URL_RE.fullmatch(domain))
        is_obf = obfuscation_present or bool(re.search(r"[\[\]{}()]", raw))
        results.append(
            UrlInfo(
                raw=raw,
                domain=domain,
                tld=tld or "",
                normalized=normalized,
                is_shortener=domain in SHORTENER_DOMAINS
                or any(domain.endswith("." + d) for d in SHORTENER_DOMAINS),
                is_suspicious_tld=(tld or "") in SUSPICIOUS_TLDS,
                is_ip=is_ip,
                has_punycode="xn--" in domain,
                is_obfuscated=is_obf,
                has_credentials="@" in raw.split("://", 1)[-1].split("/", 1)[0],
                path_length=len(urlsplit(normalized or "").path),
            )
        )

    if SPACED_LETTER_RE.search(original):
        for match in SPACED_LETTER_RE.finditer(original):
            raw = match.group(0)
            joined = re.sub(r"[\s._-]", "", raw)
            if len(joined) < 4 or "." not in raw:
                continue
            key = "spaced:" + joined.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(
                UrlInfo(
                    raw=raw,
                    domain=None,
                    tld="",
                    normalized=None,
                    is_shortener=False,
                    is_suspicious_tld=False,
                    is_ip=False,
                    has_punycode=False,
                    is_obfuscated=True,
                    has_credentials=False,
                    path_length=0,
                )
            )
    return results


def _parse_number(value: str) -> float | None:
    value = normalize_digits(value).replace(",", "").strip().rstrip(".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _apply_multiplier(value: float | None, multiplier: str | None) -> float | None:
    if value is None:
        return None
    factors = {
        "k": 1_000, "m": 1_000_000, "million": 1_000_000,
        "billion": 1_000_000_000, "lakh": 100_000, "lac": 100_000,
        "crore": 10_000_000,
    }
    if multiplier:
        value *= factors.get(multiplier.casefold(), 1)
    return value


def extract_amounts(text: object) -> list[dict]:
    t = normalize_digits(refang(clean_text(text)))
    found: list[dict] = []
    seen: set[tuple[str, float | None]] = set()
    for match in AMOUNT_BEFORE_RE.finditer(t):
        currency, number, mult = match.group(1), match.group(2), match.group(3)
        value = _apply_multiplier(_parse_number(number), mult)
        key = (currency, value)
        if key in seen:
            continue
        seen.add(key)
        found.append({"raw": match.group(0), "currency": currency, "value": value})
    for match in AMOUNT_AFTER_RE.finditer(t):
        number, mult, currency = match.group(1), match.group(2), match.group(3)
        value = _apply_multiplier(_parse_number(number), mult)
        key = (currency, value)
        if key in seen:
            continue
        seen.add(key)
        found.append({"raw": match.group(0), "currency": currency, "value": value})
    for match in CRYPTO_AMOUNT_RE.finditer(t):
        number, currency = match.group(1), match.group(2)
        value = _parse_number(number)
        key = (currency, value)
        if key in seen:
            continue
        seen.add(key)
        found.append({"raw": match.group(0), "currency": currency, "value": value})
    return found[:12]


def extract_emails(text: object) -> list[str]:
    return sorted(set(EMAIL_RE.findall(clean_text(text))))[:8]


def extract_phones(text: object) -> list[str]:
    t = refang(clean_text(text))
    phones: list[str] = []
    for match in PHONE_RE.finditer(t):
        raw = match.group(0).strip()
        if any(sym in raw for sym in ("$", "€", "£", "₹", "৳")):
            continue
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15 and not re.fullmatch(r"0+", digits):
            phones.append("+" + digits if raw.strip().startswith("+") else digits)
    return sorted(set(phones))[:8]


def extract_otps(text: object) -> list[str]:
    t = normalize_digits(refang(clean_text(text)))
    found = set()
    for pattern in (OTP_CONTEXT_RE, OTP_CONTEXT_RE_2):
        for match in pattern.finditer(t):
            found.add(match.group(1))
    return sorted(found)[:5]


def extract_crypto_addresses(text: object) -> list[str]:
    t = refang(clean_text(text))
    return sorted(set(ETH_ADDRESS_RE.findall(t) + BTC_ADDRESS_RE.findall(t)))[:8]


def simhash64(text: object) -> int:
    canonical = canonical_for_dedupe(text)
    tokens = canonical.split()
    if not tokens:
        return 0
    shingles = set(tokens)
    shingles.update(" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1))
    shingles.update(tokens[i][:4] for i in range(len(tokens)) if len(tokens[i]) >= 4)
    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little"
            )
            for s in shingles
        ),
        dtype=np.uint64,
    )
    if hashes.size == 0:
        return 0
    shifts = np.arange(64, dtype=np.uint64)
    bits = ((hashes[:, None] >> shifts) & np.uint64(1)).astype(np.int64)
    votes = 2 * bits.sum(axis=0) - len(hashes)
    value = 0
    for bit in np.nonzero(votes >= 0)[0]:
        value |= 1 << int(bit)
    return value


def hamming64(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def dedupe_simhash(
    texts: Sequence[str],
    hamming_threshold: int = 3,
    bands: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    n = len(texts)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    hashes = np.empty(n, dtype=object)
    exact: dict[str, int] = {}
    uf = _UnionFind(n)
    for i, text in enumerate(texts):
        canonical = canonical_for_dedupe(text)
        if canonical in exact:
            uf.union(i, exact[canonical])
        else:
            exact[canonical] = i
        hashes[i] = simhash64(text)
        if show_progress and (i + 1) % 25000 == 0:
            print(f"  simhash {i + 1}/{n}", flush=True)

    band_bits = max(1, 64 // max(1, bands))
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(bands)]
    for i in range(n):
        h = int(hashes[i])
        for b in range(bands):
            key = (h >> (b * band_bits)) & ((1 << band_bits) - 1)
            buckets[b][key].append(i)

    comparisons = 0
    for b in range(bands):
        for ids in buckets[b].values():
            if len(ids) < 2:
                continue
            if len(ids) > 800:
                ids = ids[:800]
            for i in range(len(ids)):
                hi = int(hashes[ids[i]])
                for j in range(i + 1, len(ids)):
                    comparisons += 1
                    if hamming64(hi, int(hashes[ids[j]])) <= hamming_threshold:
                        uf.union(ids[i], ids[j])
        if show_progress:
            print(f"  lsh band {b + 1}/{bands} comparisons={comparisons}", flush=True)

    roots: dict[int, int] = {}
    cluster_ids = np.empty(n, dtype=np.int32)
    for i in range(n):
        root = uf.find(i)
        if root not in roots:
            roots[root] = len(roots)
        cluster_ids[i] = roots[root]
    return cluster_ids


def format_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = defaultdict(int)
    for ch in value:
        counts[ch] += 1
    total = len(value)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def is_legitimate_domain(url: UrlInfo) -> bool:
    if not url.domain:
        return False
    return url.domain in LEGITIMATE_DOMAINS or any(
        url.domain.endswith("." + d) for d in LEGITIMATE_DOMAINS
    )
