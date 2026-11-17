from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import (
    EMAIL_RE,
    SPACED_LETTER_RE,
    URL_RE,
    UrlInfo,
    clean_text,
    extract_amounts,
    extract_crypto_addresses,
    extract_emails,
    extract_otps,
    extract_phones,
    extract_urls,
    is_legitimate_domain,
    normalize_digits,
    refang,
)

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u180e]")
EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U0001f900-\U0001f9ff\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff\u2b00-\u2bff]"
)
LEET_RE = re.compile(r"\b(?=\w*[a-z])(?=\w*[0-9@$])\w{4,}\b", re.I)

LEXICONS: dict[str, tuple[str, ...]] = {
    "urgency": (
        "urgent", "urgently", "immediately", "right now", "act now", "act fast",
        "final notice", "last chance", "last warning", "expires today",
        "expiring today", "within 24 hours", "within 48 hours", "24 hours",
        "48 hours", "hurry", "today only", "limited time", "asap", "emergency",
        "don't delay", "do not delay", "deadline", "final reminder",
        "turant", "turanT", "abhi", "abhi tak", "jaldi", "jaldi karo",
        "aaj hi", "aaj tak", "fauran", "foran", "ekhoni", "sathe sathe",
        "deri korben na", "deri mat karo", "turant karein", "abhi karein",
        "ahora", "urgente", "inmediatamente", "hoy mismo", "hoy", "rapido",
        "aujourd'hui", "urgent", "immediatement", "tout de suite",
        "maintenant", "derniere chance", "فورا", "الان", "عاجل", "حالا",
        "بسرعة", "آخر فرصة", "এখনই", "দ্রুত", "আজই", "শেষ সুযোগ",
    ),
    "reward": (
        "won", "winner", "win", "prize", "lottery", "jackpot", "gift",
        "free", "cashback", "reward", "bonus", "voucher", "coupon",
        "giveaway", "sweepstakes", "iphone", "laptop", "ps5", "playstation",
        "rtx", "gift card", "amazon voucher", "claim now", "claim your",
        "you have been selected", "congratulations", "lucky winner",
        "jeeta", "jeet", "jeeti", "inaam", "inam", "jeet gaye", "jeet liya",
        "jitechen", "jitechhen", "inam", "upohar", "upahar", "puraskar",
        "আপনি জিতেছেন", "পুরস্কার", "উপহার", "লটারি", "ক্যাশব্যাক",
        "premio", "ganaste", "gano", "sorteo", "regalo", "gratis",
        "felicidades", "has ganado", "gagné", "gagnez", "prix", "cadeau",
        "gratuit", "felicitations", "vous avez gagne", "tirage",
        "جائزة", "ربحت", "هدية", "مجانا", "مسابقة", "مبروك",
    ),
    "threat": (
        "suspended", "blocked", "locked", "deactivated", "closed",
        "terminated", "blocked permanently", "legal action", "police",
        "arrest", "fine", "penalty", "account freeze", "frozen",
        "will be blocked", "will be suspended", "will be deactivated",
        "expire", "expires", "verify", "verification required", "kyc",
        "update your", "reactivate", "restricted", "limited access",
        "unusual activity", "unauthorized", "security alert", "final warning",
        "fraud alert", "attempted transfer", "transfer was attempted",
        "on hold", "avoid cancellation", "account limited", "access limited",
        "suspicious transfer", "unauthorized transfer",
        "band", "block", "suspend", "saspend", "saspand", "nilambit",
        "ब्लॉक", "सस्पेंड", "निलंबित", "बंद", "बंद हो", "खाता बंद",
        "সাসপেন্ড", "ব্লক", "বন্ধ", "বাতিল", "নিষিদ্ধ", "একাউন্ট বন্ধ",
        "suspendida", "bloqueada", "bloqueado", "verificar", "cuenta bloqueada",
        "suspendu", "bloque", "verifier", "compte suspendu", "desactive",
        "موقوف", "مغلق", "محظور", "تحقق", "تم تعليق", "حسابك",
    ),
    "authority": (
        "bank", "paypal", "amazon", "netflix", "hmrc", "irs", "government",
        "police", "customs", "delivery", "courier", "sbi", "icici", "hdfc",
        "axis", "kotak", "vodafone", "jio", "airtel", "bkash", "nagad",
        "rocket", "upay", "phonepe", "paytm", "google pay", "apple id",
        "microsoft", "facebook", "instagram", "whatsapp", "telegram",
        "binance", "coinbase", "metamask", "dhl", "fedex", "ups", "usps",
        "royal mail", "yodel", "dpd", "evri", "delhivery", "bluedart",
        "income tax", "gst", "uidai", "aadhaar", "aadhar", "pan card",
        "bank of america", "chase", "wells fargo", "citibank", "hsbc",
        "bbva", "santander", "caixa", "banorte", "itau", "bradesco", "nubank",
        "scotiabank", "rbc", "td bank", "barclays", "lloyds", "natwest",
        "bkash", "nagad", "rocket", "bikash", "nagad", "sonali bank",
        "account", "cuenta", "compte", "حساب", "একাউন্ট", "खाता",
        "brac bank", "dbbl", "city bank", "dutch bangla", "islami bank",
        "विभाग", "सरकार", "पुलिस", "बैंक", "কাস্টমস", "পুলিশ", "সরকার",
        "ব্যাংক", "বিকাশ", "নগদ", "بنك", "الجمارك", "الشرطة", "الحكومة",
        "مصرف", "بنك الراجحي", "البنك الأهلي",
    ),
    "payment": (
        "send money", "transfer", "wire transfer",
        "bank transfer", "pay now", "payment", "deposit", "pay fee",
        "processing fee", "shipping fee", "customs fee", "redelivery fee",
        "release fee", "clearance fee", "activation fee", "service fee",
        "bhej do", "bhejo", "bhej dena", "paise bhejo", "paisa bhej",
        "transfer karo", "pay karo", "payment karo", "pathao", "pathiye din",
        "taka pathao", "tk pathao", "taka bhej", "bhej", "de do", "de dena",
        "envoie", "envoyer", "envia", "enviar", "paga", "pagar", "deposita",
        "أرسل", "ارسال", "حوّل", "حول", "ادفع", "دفع", "পাঠান", "পাঠাও",
        "টাকা পাঠাও", "বিকাশ করুন",
    ),
    "investment": (
        "double your money", "double your", "guaranteed returns",
        "guaranteed profit", "risk free", "risk-free", "300%", "500%",
        "10x", "100x", "crypto", "bitcoin", "btc", "eth", "usdt", "usdc",
        "forex", "trading bot", "trading signal", "investment", "profit",
        "returns", "airdrop", "mining", "passive income", "financial freedom",
        "portfolio", "binance", "coinbase", "metamask", "defi", "nft",
        "paisa double", "paise double", "taka double", "invest karo",
        "profit guarantee", "guaranteed", "lakhpati", "crorepati",
        "din mein", "daily profit", "telegram par", "telegram join",
        "argent facile", "rendement garanti", "investissement", "crypto",
        "dinero facil", "rendimiento garantizado", "inversion", "cripto",
        "استثمار", "أرباح مضمونة", "عملات رقمية", "بيتكوين", "ربح",
    ),
    "job": (
        "work from home", "work-from-home", "daily payout", "daily payment",
        "no experience", "no experience needed", "part time job", "part-time",
        "telegram", "whatsapp", "hiring", "salary", "earn", "income",
        "job offer", "job", "vacancy", "apply now", "recruitment", "data entry",
        "typing job", "copy paste job", "online job", "ghar se kaam",
        "ghar baithe", "roz paisa", "daily income", "part time",
        "job chahiye", "naukri", "kaam karo", "telegram join karo",
        "travail a domicile", "revenu quotidien", "emploi", "gagnez",
        "trabajo desde casa", "ingresos diarios", "empleo", "gana dinero",
        "عمل من المنزل", "وظيفة", "دخل يومي", "ربح يومي",
    ),
    "romance": (
        "single", "dating", "lonely", "love", "marriage", "soldier",
        "army", "doctor", "widow", "visa", "ticket", "dear", "my love",
        "soulmate", "relationship", "girlfriend", "boyfriend", "hot singles",
        "meet you", "come to see you", "fallen in love", "i am a soldier",
        "military", "peacekeeping", "deep connection", "feel a connection",
        "connection with you", "my dear", "সেনাবাহিনী",
        "pyaar", "mohabbat", "shaadi",
        "dil", "jaan", "meri jaan", "tumse pyar", "valobasha", "bhalobasha",
        "biye", "amour", "rencontre", "celibataire", "amor", "soltero",
        "cita", "حب", "زواج", "عزباء", "أرملة",
    ),
    "delivery": (
        "parcel", "package", "courier", "delivery", "shipment", "tracking",
        "awb", "customs", "redelivery", "postal", "dpd", "fedex", "dhl",
        "usps", "royal mail", "delivery failed", "delivery attempt",
        "out for delivery", "consignment", "parcel pending", "held at customs",
        "delivery fee", "shipping fee", "track your", "courier service",
        "customs fee", "clearance fee", "redelivery fee", "duty",
        "envío", "envio", "aduana", "aduanas", "douane", "dédouanement",
        "dedouanement", "tasa", "taux", "frais", "شحنتك", "الجمارك",
        "parcel aapke", "delivery nahi", "courier aa raha", "parcel aayega",
        "livraison", "colis", "suivi", "entrega", "paquete", "envio",
        "توصيل", "شحنة", "طرد", "تتبع", "ডেলিভারি", "পার্সেল", "কুরিয়ার",
    ),
    "otp": (
        "otp", "o.t.p", "one time password", "one-time password",
        "verification code", "security code", "confirmation code",
        "pin", "password", "passcode", "do not share", "don't share",
        "never share", "share the code", "send the code", "tell me the code",
        "code batao", "otp batao", "otp bhejo", "otp share", "otp do",
        "code do", "code bhejo", "otp pathao", "code pathao", "otp bolun",
        "code partage", "ne partagez pas", "code de verification",
        "codigo de verificacion", "no compartas", "codigo",
        "رمز التحقق", "لا تشارك", "أرسل الرمز", "الكود", "ওটিপি", "কোড",
    ),
    "marketplace": (
        "selling", "buyer", "seller", "deposit", "zelle only",
        "cash app only", "shipping", "pickup", "item", "ticket", "concert",
        "ps5", "rtx", "iphone", "macbook", "role", "gpu", "graphics card",
        "playstation", "xbox", "nintendo", "sneakers", "watch", "concert ticket",
        "selling my", "for sale", "dm me", "inbox me", "serious buyer",
        "bechna hai", "bech raha", "bech rahi", "kharidna", "kharido",
        "item chahiye", "ticket chahiye", "vendre", "acheter", "vendo",
        "compro", "vendiendo", "للبيع", "أشتري", "بيع", "বিক্রি", "কিনব",
    ),
    "p2p_verbs": (
        "give me", "send me", "pay me", "transfer me", "give you",
        "send you", "i will give", "i'll give", "i will send", "i'll send",
        "you get", "you will get", "you'll get", "double", "return",
        "refund", "reimburse", "mujhe do", "mujhe bhej", "mujhe de",
        "amake dao", "amake de", "amake pathao", "ami debo", "ami dibo",
        "ami return korbo", "tui de", "tumi de", "bhai de", "de do",
        "dame", "dame eso", "te doy", "me das", "donne moi", "je te donne",
        "أعطني", "أرسل لي", "سأعطيك", "أعطيك",
    ),
    "click": (
        "click here", "click the link", "click this link", "tap here",
        "tap the link", "open this link", "visit this link", "follow this link",
        "link par click", "is link par click", "link kholo", "link kholun",
        "link par jao", "niche link", "haga clic", "cliquez ici",
        "cliquez sur", "انقر هنا", "اضغط هنا", "লিঙ্কে ক্লিক", "এখানে ক্লিক",
    ),
    "guarantee": (
        "guaranteed", "guarantee", "100% free", "100% safe", "risk free",
        "risk-free", "no risk", "double your", "triple your", "instant",
        "within minutes", "within hours", "no experience", "no investment",
        "free of cost", "absolutely free", "guaranteed return",
        "guaranteed profit", "pakka", "guaranteed", "100%", "নিশ্চিত",
        "গ্যারান্টি", "garantizado", "garantie", "garanti", "مضمون",
    ),
    "click_terms": (
        "click", "tap", "open", "visit", "follow", "download", "install",
        "claim", "verify", "update", "reactivate", "unlock", "confirm",
        "kholo", "kholun", "khol", "click karo", "click karein", "cliquez",
        "haga clic", "abra", "verifique", "confirme", "انقر", "اضغط",
        "افتح", "تحقق", "أكد", "ক্লিক", "খুলুন", "যাচাই", "নিশ্চিত",
    ),
}

LEXICON_PATTERNS: dict[str, re.Pattern] = {}


def _build_pattern(terms: Sequence[str]) -> re.Pattern:
    alternatives = []
    for term in sorted(set(terms), key=len, reverse=True):
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        left = r"(?<!\w)" if re.match(r"\w", term) else ""
        right = r"(?!\w)" if re.search(r"\w$", term) else ""
        alternatives.append(left + escaped + right)
    return re.compile("|".join(alternatives), re.I | re.U)


for _name, _terms in LEXICONS.items():
    LEXICON_PATTERNS[_name] = _build_pattern(_terms)

HINGLISH_MARKERS = (
    "hai", "hain", "kya", "aap", "aapka", "aapki", "tum", "tumhara",
    "tumhari", "bhai", "yaar", "nahi", "nahin", "kar", "karo", "karna",
    "karein", "mujhe", "mera", "meri", "paise", "paisa", "bhej", "bhejo",
    "turant", "abhi", "jaldi", "jeet", "inaam", "kholo", "band", "gaya",
    "raha", "rahi", "ka", "ki", "ke", "se", "ko", "mein", "par", "mat",
    "chahiye", "milega", "dega", "doge", "batao", "kaisa", "kaise",
    "karo", "kijiye", "kijiyega", "aapko", "hum", "ham", "ya", "bhi",
    "toh", "phir", "wala", "wale", "kuch", "bahut", "bohot", "acha",
    "theek", "kal", "aaj", "ab", "kab", "kaun", "kyun", "kyu", "kaam",
    "de do", "de dena", "de dijiye", "kar do", "kar dena",
)

BANGLISH_MARKERS = (
    "ami", "tumi", "apni", "apnar", "tomar", "amar", "bhai", "taka", "tk",
    "bdt", "bhej", "pathao", "pathan", "korbo", "korte", "korchen", "kore",
    "keno", "kemon", "acho", "ache", "hobe", "hoyeche", "jitechen",
    "jitben", "inam", "link", "kholun", "bondho", "na", "ki", "theke",
    "jonno", "sathe", "valo", "khub", "onek", "bhalo", "dhonnobad",
    "salam", "kemon acho", "bhai", "apa", "abu", "mama", "chacha",
    "tui", "tora", "amra", "tomra", "kotha", "bolo", "bolen", "den",
    "din", "dao", "dibo", "debo", "korlam", "korle", "korben", "hoy",
    "hoyna", "hobe na", "bikash", "nagad", "rocket", "upay", "grameenphone",
    "gp", "robi", "banglalink", "teletalk", "bangladesh",
)

SPANISH_MARKERS = (
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una",
    "para", "por", "con", "su", "sus", "tu", "tus", "cuenta", "premio",
    "ganaste", "gano", "banco", "urgente", "haga clic", "verifique",
    "bloqueada", "bloqueado", "suspendida", "dinero", "pago", "tarjeta",
    "envie", "envia", "reclame", "gratis", "oferta", "hoy", "ahora",
    "este", "esta", "seguridad", "codigo", "no comparta", "felicidades",
    "gracias", "por favor", "dame", "amigo", "hola", "buenos", "buenas",
    "como", "estas", "donde", "cuando", "porque", "pero", "muy", "mas",
)

FRENCH_MARKERS = (
    "le", "la", "les", "de", "des", "du", "et", "en", "un", "une", "pour",
    "par", "avec", "votre", "vos", "compte", "prix", "gagne", "gagnez",
    "banque", "urgent", "cliquez", "verifiez", "bloque", "suspendu",
    "argent", "paiement", "carte", "envoyez", "reclamez", "gratuit",
    "offre", "aujourd'hui", "maintenant", "ce", "cette", "securite",
    "code", "ne partagez pas", "felicitations", "merci", "bonjour",
    "salut", "je", "tu", "nous", "vous", "ils", "elles", "est", "sont",
    "avec", "pour", "dans", "sur", "pas", "plus", "tres", "bien",
)

ENGLISH_MARKERS = (
    "the", "and", "you", "your", "are", "is", "to", "for", "with",
    "have", "this", "that", "from", "will", "can", "please", "thanks",
    "hello", "my", "me", "we", "our", "not", "on", "at", "in", "of",
    "it", "be", "was", "were", "has", "had", "do", "does", "did",
    "but", "or", "if", "so", "just", "get", "got", "need", "want",
    "call", "text", "message", "today", "tomorrow", "now", "ok",
    "okay", "bro", "dear", "friend", "click", "here", "verify",
    "account", "payment", "won", "winner", "prize", "free", "support",
    "service", "delivery", "order", "confirm", "bank", "card", "otp",
    "money", "send", "pay", "gift", "offer", "limited", "time", "only",
)

MARKER_PATTERNS = {
    "English": _build_pattern(ENGLISH_MARKERS),
    "Hinglish": _build_pattern(HINGLISH_MARKERS),
    "Banglish": _build_pattern(BANGLISH_MARKERS),
    "Spanish": _build_pattern(SPANISH_MARKERS),
    "French": _build_pattern(FRENCH_MARKERS),
}

SCRIPT_NAMES = (
    "Latin", "Devanagari", "Bengali", "Arabic", "Cyrillic", "Greek",
    "Hebrew", "Han", "Hangul", "Thai", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Gujarati", "Gurmukhi", "Oriya", "Sinhala", "Other",
)


def script_of(ch: str) -> str | None:
    o = ord(ch)
    if 0x0041 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:
        return "Latin"
    if 0x0900 <= o <= 0x097F:
        return "Devanagari"
    if 0x0980 <= o <= 0x09FF:
        return "Bengali"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
        return "Arabic"
    if 0x0400 <= o <= 0x04FF:
        return "Cyrillic"
    if 0x0370 <= o <= 0x03FF:
        return "Greek"
    if 0x0590 <= o <= 0x05FF:
        return "Hebrew"
    if 0x4E00 <= o <= 0x9FFF:
        return "Han"
    if 0xAC00 <= o <= 0xD7AF:
        return "Hangul"
    if 0x0E00 <= o <= 0x0E7F:
        return "Thai"
    if 0x0B80 <= o <= 0x0BFF:
        return "Tamil"
    if 0x0C00 <= o <= 0x0C7F:
        return "Telugu"
    if 0x0C80 <= o <= 0x0CFF:
        return "Kannada"
    if 0x0D00 <= o <= 0x0D7F:
        return "Malayalam"
    if 0x0A80 <= o <= 0x0AFF:
        return "Gujarati"
    if 0x0A00 <= o <= 0x0A7F:
        return "Gurmukhi"
    if 0x0B00 <= o <= 0x0B7F:
        return "Oriya"
    if 0x0D80 <= o <= 0x0DFF:
        return "Sinhala"
    if ch.isalpha():
        return "Other"
    return None


def script_shares(text: str) -> dict[str, float]:
    t = EMAIL_RE.sub(" ", refang(text))
    t = URL_RE.sub(" ", t)
    counts: dict[str, int] = defaultdict(int)
    for ch in t:
        name = script_of(ch)
        if name:
            counts[name] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in counts.items()}


def _marker_score(text: str, lang: str) -> int:
    pattern = MARKER_PATTERNS[lang]
    return len({m.group(0).casefold() for m in pattern.finditer(text)})


def detect_language(text: str) -> tuple[str, float, dict[str, float]]:
    cleaned = refang(clean_text(text))
    shares = script_shares(cleaned)
    if not shares:
        return "unknown", 0.0, {}
    total = sum(shares.values())
    if total <= 0:
        return "unknown", 0.0, shares

    non_latin = {k: v for k, v in shares.items() if k != "Latin" and v >= 0.12}
    latin_share = shares.get("Latin", 0.0)
    if len(non_latin) >= 2 and max(non_latin.values()) < 0.7:
        return "mixed", float(max(non_latin.values())), shares

    if non_latin:
        dominant, share = max(non_latin.items(), key=lambda kv: kv[1])
        mapping = {
            "Devanagari": "Hindi", "Bengali": "Bengali", "Arabic": "Arabic",
            "Cyrillic": "Russian", "Greek": "Greek", "Hebrew": "Hebrew",
            "Han": "Chinese", "Hangul": "Korean", "Thai": "Thai",
            "Tamil": "Tamil", "Telugu": "Telugu", "Kannada": "Kannada",
            "Malayalam": "Malayalam", "Gujarati": "Gujarati",
            "Gurmukhi": "Punjabi", "Oriya": "Odia", "Sinhala": "Sinhala",
            "Other": "other",
        }
        lang = mapping.get(dominant, "other")
        if share >= 0.55 or latin_share < 0.15:
            return lang, float(share), shares
        return "mixed", float(share), shares

    lowered = cleaned.casefold()
    scores = {lang: _marker_score(lowered, lang) for lang in MARKER_PATTERNS}
    english_score = scores.get("English", 0)
    accented = {
        "Spanish": sum(1 for ch in cleaned if ch in "ñÑ¿¡áéíóúÁÉÍÓÚ"),
        "French": sum(1 for ch in cleaned if ch in "àâçéèêëîïôûùüÿœæÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ"),
    }
    has_spanish_accent = accented["Spanish"] > 0
    has_french_accent = accented["French"] > 0
    if has_spanish_accent and not has_french_accent:
        scores["Spanish"] += 3
    if has_french_accent and not has_spanish_accent:
        scores["French"] += 3
    if has_spanish_accent and has_french_accent:
        scores["French"] += 1

    non_english = {k: v for k, v in scores.items() if k != "English"}
    best_lang, best_score = max(non_english.items(), key=lambda kv: kv[1])
    tokens = len(re.findall(r"\w+", lowered))
    if best_lang in ("Hinglish", "Banglish"):
        if best_score >= 2 and best_score + 1 >= english_score:
            confidence = min(0.98, 0.58 + 0.05 * best_score)
            return best_lang, float(confidence), shares
        return "English", float(max(0.55, latin_share)), shares
    if best_lang in ("Spanish", "French"):
        if best_score >= 3 and best_score > english_score:
            confidence = min(0.98, 0.58 + 0.05 * best_score)
            return best_lang, float(confidence), shares
        if best_lang == "Spanish" and has_spanish_accent and best_score >= 2:
            return "Spanish", 0.8, shares
        if best_lang == "French" and has_french_accent and best_score >= 2:
            return "French", 0.8, shares
        return "English", float(max(0.55, latin_share)), shares
    return "English", float(max(0.55, latin_share)), shares


@dataclass
class TextAnalysis:
    text: str
    language: str
    language_confidence: float
    script_shares: dict[str, float]
    urls: list[UrlInfo]
    emails: list[str]
    phones: list[str]
    amounts: list[dict]
    otps: list[str]
    crypto_addresses: list[str]
    features: dict[str, float]
    signals: list[dict]
    matched_terms: dict[str, list[str]]

    @property
    def feature_vector(self) -> np.ndarray:
        return np.asarray([self.features[name] for name in FEATURE_NAMES], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_confidence": round(self.language_confidence, 4),
            "scripts": {k: round(v, 4) for k, v in self.script_shares.items()},
            "urls": [u.to_dict() for u in self.urls],
            "emails": self.emails,
            "phones": self.phones,
            "amounts": self.amounts[:6],
            "otps": [mask_secret(x) for x in self.otps],
            "crypto_addresses": [
                x[:4] + "..." + x[-4:] if len(x) > 10 else x
                for x in self.crypto_addresses
            ],
            "signals": self.signals,
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
        }


def mask_secret(value: str) -> str:
    if len(value) <= 2:
        return "*" * len(value)
    return "*" * (len(value) - 2) + value[-2:]


def _matches(text: str, name: str) -> list[str]:
    pattern = LEXICON_PATTERNS[name]
    return sorted({m.group(0).casefold() for m in pattern.finditer(text)})


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def prepare_for_model(text: object) -> str:
    raw = clean_text(text)
    if not raw:
        return ""
    refanged = refang(raw)
    urls = extract_urls(raw)
    t = refanged.casefold()
    for url in sorted(urls, key=lambda u: len(u.raw), reverse=True):
        if url.raw:
            t = t.replace(url.raw.casefold(), " __url__ ")
    t = URL_RE.sub(" __url__ ", t)
    t = EMAIL_RE.sub(" __email__ ", t)
    for amount in extract_amounts(raw):
        token = str(amount.get("raw", "")).casefold()
        if token:
            t = t.replace(token, " __amount__ ")
    for phone in extract_phones(raw):
        t = re.sub(r"(?<!\d)" + re.escape(phone) + r"(?!\d)", " __phone__ ", t)
    for otp in extract_otps(raw):
        t = re.sub(r"\b" + re.escape(otp) + r"\b", " __otp__ ", t)
    t = re.sub(r"\b\d{3,}\b", " __num__ ", t)
    t = re.sub(r"[^\w\s@#/$%!?.,:+\-]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    domains = " ".join(f"__url__ {u.domain or 'obf'}" for u in urls[:3])
    return f"{t} {domains}".strip()[:900]


P2P_EXCHANGE_RE = re.compile(
    r"(?is)(?:\b(?:give|send|pay|transfer|deposit)\b.{0,55}\b(?:me|mujhe|amake)\b"
    r".{0,80}\b(?:i(?:'ll| will| shall)?\s+(?:give|send|return)\s+you|"
    r"you(?:'ll| will)?\s+get|ami\s+(?:debo|dibo|return)|"
    r"main\s+(?:dunga|de dunga|return))\b)"
    r"|(?:\b(?:send|give|deposit|pay)\b.{0,40}\b(?:btc|eth|usdt|usdc|bnb|sol)\b"
    r".{0,60}\b(?:get|receive|return|double)\b)"
    r"|(?:\bdouble\b.{0,25}\b(?:money|btc|eth|usdt|investment|amount)\b)"
    r"|(?:\b(?:mujhe|amake|ama ke)\b.{0,40}\b\d{1,6}\b.{0,25}"
    r"\b(?:taka|tk|bdt)?\b.{0,15}\b(?:debo|dibo|dunga|return)\b)"
    r"|(?:\b\d{1,6}\s*(?:taka|tk|bdt)\b.{0,25}\b(?:bhej|patha|de)\b"
    r".{0,40}\b(?:debo|dibo|dunga|return|double)\b)"
    r"|(?:(?:give|send|pay|deposit|transfer|env[ií]a(?:me)?|mándame|"
    r"envoie[- ]moi|donne[- ]moi|أرسل لي|أعطني).{0,120}"
    r"(?:\$|€|₹|৳|ريال|دولار|درهم|\d{1,4}\s*(?:€|دولار|ريال|درهم)).{0,120}"
    r"(?:RTX|PS5|iPhone|MacBook|GPU|graphics card))"
    r"|(?:(?:accidentally|by mistake).{0,60}(?:sent|transferred)"
    r".{0,60}(?:send|return).{0,50}(?:back|me).{0,60}(?:keep|rest))"
    r"|(?:(?:seed phrase|recovery phrase|12-word|24-word).{0,80}"
    r"(?:reply|send|share|give|tell)|"
    r"(?:reply|send|share|give|tell).{0,80}"
    r"(?:seed phrase|recovery phrase|12-word|24-word))"
)

OTP_NEGATION_RE = re.compile(
    r"(?is)(?:do not share|don't share|never share|do not disclose|"
    r"never disclose|no compartas|no compartir|ne partagez pas|"
    r"ne jamais partager|لا تشارك|لا تعط|কারো সাথে শেয়ার করবেন না|"
    r"কারও সাথে শেয়ার করবেন না|কাউকে শেয়ার করবেন না|"
    r"किसी के साथ साझा न करें|किसी को न बताएं|"
    r"karor sathe share korben na|karo sathe share korben na|"
    r"share korben na|share koro na|kisi ko bhi mat batana|"
    r"kisi ke saath share na karein)"
)
OTP_REQUEST_RE = re.compile(
    r"(?is)(?:\b(?:share|send|tell|give|forward|reply|provide|confirm)\b"
    r".{0,30}\b(?:otp|code|pin|password|passcode)\b|"
    r"\b(?:otp|code|pin|password|passcode)\b.{0,30}"
    r"\b(?:share|send|tell|give|forward|reply|provide|confirm)\b|"
    r"\b(?:otp|code)\s*(?:batao|bhejo|bolo|do|pathao|bhej\s+do|bhej\s+dena|de\s+do)\b|"
    r"\b(?:batao|bhejo|bolo|pathao|bhej\s+do|bhej\s+dena|de\s+do)\b.{0,20}\b(?:otp|code)\b|"
    r"\b(?:otp|code)\b.{0,60}(?:bhej\s+do|bhej\s+dena|de\s+do|pathao|bhejo|batao|din|dao|dio)|"
    r"(?:otp|code)\s*(?:দিন|dio|dao|din)|"
    r"\b(?:partage|partagez|envoyez|dites|comparte|compartas|envia)\b"
    r".{0,30}\b(?:code|otp)\b|"
    r"\b(?:أرسل|شارك|قل)\b.{0,30}\b(?:الرمز|الكود|رمز)\b)"
)

OTP_SAFE_LOCAL_RE = re.compile(
    r"(?is)(?:share|give|show|tell).{0,50}(?:delivery agent|at your door|"
    r"only after the package|when the package arrives|with the courier)|"
    r"(?:কুরিয়ার|ডেলিভারি এজেন্ট).{0,30}(?:প্যাকেজ|পৌঁছানোর পরে)"
)


def _flag_features(
    text: str,
    lowered: str,
    urls: list[UrlInfo],
    emails: list[str],
    phones: list[str],
    amounts: list[dict],
    otps: list[str],
    crypto: list[str],
    matches: dict[str, list[str]],
    scripts: dict[str, float],
) -> dict[str, float]:
    letters = [ch for ch in text if ch.isalpha()]
    upper_ratio = _safe_div(sum(ch.isupper() for ch in letters), len(letters))
    digit_ratio = _safe_div(sum(ch.isdigit() for ch in text), len(text))
    alpha_ratio = _safe_div(len(letters), len(text))
    punct_ratio = _safe_div(
        sum(1 for ch in text if not ch.isalnum() and not ch.isspace()), len(text)
    )
    tokens = re.findall(r"\w+", text, flags=re.U)
    emoji_count = len(EMOJI_RE.findall(text))
    exclamations = text.count("!")
    questions = text.count("?")
    shorteners = sum(1 for u in urls if u.is_shortener)
    suspicious_tlds = sum(1 for u in urls if u.is_suspicious_tld)
    ip_urls = sum(1 for u in urls if u.is_ip)
    obfuscated_urls = sum(1 for u in urls if u.is_obfuscated)
    punycode = sum(1 for u in urls if u.has_punycode)
    credential_urls = sum(1 for u in urls if u.has_credentials)
    legit_urls = sum(1 for u in urls if is_legitimate_domain(u))
    suspicious_urls = sum(
        1 for u in urls
        if u.is_shortener or u.is_suspicious_tld or u.is_ip or u.is_obfuscated
        or u.has_punycode or u.has_credentials
    )
    crypto_amounts = sum(
        1 for a in amounts if str(a.get("currency", "")).upper()
        in {"BTC", "ETH", "USDT", "USDC", "BNB", "SOL", "XRP", "DOGE", "LTC", "TRX"}
    )

    has_spaced_letters = bool(SPACED_LETTER_RE.search(text))
    has_zero_width = bool(ZERO_WIDTH_RE.search(text))
    has_leet = bool(LEET_RE.search(lowered))
    first_url_pos = min(
        (m.start() for m in URL_RE.finditer(refang(text))), default=len(text) + 1
    )

    f: dict[str, float] = defaultdict(float)
    f.update(
        len_chars=len(text),
        len_tokens=len(tokens),
        digit_ratio=digit_ratio,
        alpha_ratio=alpha_ratio,
        upper_ratio=upper_ratio,
        punct_ratio=punct_ratio,
        emoji_count=emoji_count,
        exclamation_count=exclamations,
        question_count=questions,
        url_count=len(urls),
        shortener_count=shorteners,
        suspicious_tld_count=suspicious_tlds,
        ip_url_count=ip_urls,
        obfuscated_url_count=obfuscated_urls,
        punycode_count=punycode,
        credential_url_count=credential_urls,
        email_count=len(emails),
        phone_count=len(phones),
        amount_count=len(amounts),
        otp_count=len(otps),
        crypto_addr_count=len(crypto),
        crypto_amount_count=crypto_amounts,
        urgency_count=len(matches.get("urgency", [])),
        reward_count=len(matches.get("reward", [])),
        threat_count=len(matches.get("threat", [])),
        authority_count=len(matches.get("authority", [])),
        payment_count=len(matches.get("payment", [])),
        investment_count=len(matches.get("investment", [])),
        job_count=len(matches.get("job", [])),
        romance_count=len(matches.get("romance", [])),
        delivery_count=len(matches.get("delivery", [])),
        otp_word_count=len(matches.get("otp", [])),
        marketplace_count=len(matches.get("marketplace", [])),
        p2p_verb_count=len(matches.get("p2p_verbs", [])),
        click_count=len(matches.get("click", [])),
        guarantee_count=len(matches.get("guarantee", [])),
    )
    f.update(
        whatsapp_flag=float("whatsapp" in lowered),
        telegram_flag=float("telegram" in lowered),
        bank_flag=float(
            any(x in lowered for x in (
                "bank", "bkash", "bikash", "nagad", "paypal", "sbi", "icici",
                "hdfc", "axis", "chase", "bank of america", "بنك", "ব্যাংক", "बैंक",
            ))
        ),
        kyc_flag=float(
            any(x in lowered for x in ("kyc", "know your customer", "update your details"))
        ),
        prize_flag=float(
            any(x in lowered for x in (
                "prize", "lottery", "jackpot", "won", "winner", "পুরস্কার",
                "জিতেছেন", "premio", "ganaste", "lottery", "prix", "gagné",
                "جائزة", "ربحت",
            ))
        ),
        send_money_flag=float(
            any(x in lowered for x in (
                "send money", "send me", "wire transfer", "transfer money",
                "bhej do", "paise bhejo", "taka pathao", "send $", "send ₹",
                "envoie", "envia", "أرسل", "পাঠান",
            ))
        ),
        upi_flag=float(
            any(x in lowered for x in (
                "upi", "phonepe", "google pay", "gpay", "paytm", "@ok", "@ybl",
                "@paytm", "@upi",
            ))
        ),
        bkash_flag=float(any(x in lowered for x in ("bkash", "bikash", "বিকাশ"))),
        nagad_flag=float(any(x in lowered for x in ("nagad", "নগদ"))),
        paypal_flag=float("paypal" in lowered),
        zelle_flag=float("zelle" in lowered),
        venmo_flag=float("venmo" in lowered),
        cashapp_flag=float(any(x in lowered for x in ("cash app", "cashapp"))),
        giftcard_flag=float(
            any(x in lowered for x in (
                "gift card", "giftcard", "itunes card", "steam card",
                "google play card", "amazon card", "voucher code",
            ))
        ),
        rtx_flag=float("rtx" in lowered),
        ps5_flag=float(any(x in lowered for x in ("ps5", "playstation 5"))),
        iphone_flag=float("iphone" in lowered),
        job_offer_flag=float(
            any(x in lowered for x in (
                "work from home", "ghar se kaam", "daily payout", "job offer",
                "hiring", "vacancy", "data entry", "part time job",
            ))
        ),
        romance_flag=float(
            any(x in lowered for x in (
                "single", "dating", "lonely", "my love", "soldier", "widow",
                "marriage", "soulmate", "valobasha", "bhalobasha", "pyaar",
            ))
        ),
        delivery_flag=float(
            any(x in lowered for x in (
                "parcel", "package", "courier", "delivery", "shipment",
                "customs", "tracking", "delivery failed", "redelivery",
                "livraison", "colis", "entrega", "paquete", "توصيل", "ডেলিভারি",
            ))
        ),
        marketplace_flag=float(
            any(x in lowered for x in (
                "selling", "buyer", "seller", "for sale", "pickup", "deposit",
                "zelle only", "cash app only", "concert ticket", "ps5", "rtx",
                "iphone", "macbook", "bechna", "bech raha", "vendo", "للبيع",
            ))
        ),
        all_caps_flag=float(len(letters) >= 12 and upper_ratio >= 0.6),
        many_exclamations_flag=float(exclamations >= 3),
        many_questions_flag=float(questions >= 3),
        mixed_script_flag=float(
            sum(1 for v in scripts.values() if v >= 0.12) >= 2
        ),
        code_mixed_flag=float(
            scripts.get("Latin", 0.0) >= 0.15
            and sum(v for k, v in scripts.items() if k != "Latin") >= 0.15
        ),
        script_latin_share=scripts.get("Latin", 0.0),
        script_devanagari_share=scripts.get("Devanagari", 0.0),
        script_bengali_share=scripts.get("Bengali", 0.0),
        script_arabic_share=scripts.get("Arabic", 0.0),
        script_other_share=sum(
            v for k, v in scripts.items() if k not in {"Latin", "Devanagari", "Bengali", "Arabic"}
        ),
        suspicious_url_flag=float(suspicious_urls > 0),
        legitimate_url_flag=float(legit_urls > 0 and suspicious_urls == 0),
        spacing_obfuscation_flag=float(has_spaced_letters),
        leet_flag=float(has_leet),
        zero_width_removed_flag=float(has_zero_width),
        emoji_flag=float(emoji_count > 0),
        url_early_flag=float(bool(urls) and first_url_pos <= max(20, len(text) * 0.2)),
        short_text_flag=float(len(text) < 40),
        long_text_flag=float(len(text) > 500),
        question_flag=float(questions > 0),
        exclamation_flag=float(exclamations > 0),
        url_present_flag=float(len(urls) > 0),
    )

    f["money_and_contact_flag"] = float(
        len(amounts) >= 1 and (len(urls) >= 1 or len(phones) >= 1 or len(emails) >= 1)
    )
    f["amount_and_url_flag"] = float(len(amounts) >= 1 and len(urls) >= 1)
    f["threat_and_url_flag"] = float(f["threat_count"] >= 1 and len(urls) >= 1)
    f["reward_and_url_flag"] = float(f["reward_count"] >= 1 and len(urls) >= 1)
    f["payment_and_url_flag"] = float(f["payment_count"] >= 1 and len(urls) >= 1)
    f["phone_and_amount_flag"] = float(len(phones) >= 1 and len(amounts) >= 1)
    f["email_and_amount_flag"] = float(len(emails) >= 1 and len(amounts) >= 1)
    f["otp_and_url_flag"] = float(len(otps) >= 1 and len(urls) >= 1)
    f["unrealistic_exchange_flag"] = float(bool(P2P_EXCHANGE_RE.search(text)))
    amount_values = [
        float(a["value"]) for a in amounts
        if a.get("value") is not None and float(a["value"]) > 0
    ]
    if len(amount_values) >= 2:
        ratio = max(amount_values) / min(amount_values)
        send_verbs = (
            "send", "give", "pay", "transfer", "bhej", "debo", "dibo",
            "dunga", "भेज", "दे", "दूंगा", "envía", "envia", "envoie",
            "donne", "أرسل", "أعطني", "পাঠাও", "পাঠান", "ফেরত", "দেব",
            "দিও", "দিন",
        )
        return_words = (
            "back", "return", "wapas", "वापस", "ferot", "double", "keep",
            "rest", "profit", "gain", "te dare", "te donne", "سأعطيك",
            "أعطيك", "ফেরত", "দেব", "দ্বিগুণ", "ওয়াপস",
        )
        if (
            ratio >= 5
            and any(v in lowered for v in send_verbs)
            and any(w in lowered for w in return_words)
        ):
            f["unrealistic_exchange_flag"] = 1.0
    f["account_threat_flag"] = float(
        f["threat_count"] >= 1
        and f["authority_count"] >= 1
        and (
            f["suspicious_url_flag"]
            or f["kyc_flag"]
            or (len(urls) >= 1 and legit_urls == 0)
        )
    )
    f["delivery_fee_flag"] = float(
        f["delivery_count"] >= 1
        and (len(amounts) >= 1 or "fee" in lowered)
        and (len(urls) >= 1 or "pay" in lowered or "পাঠান" in lowered)
    )
    f["otp_theft_flag"] = float(
        (len(otps) >= 1 or f["otp_word_count"] >= 1)
        and not OTP_NEGATION_RE.search(text)
        and not OTP_SAFE_LOCAL_RE.search(text)
        and (
            (len(urls) >= 1 and f["suspicious_url_flag"])
            or bool(OTP_REQUEST_RE.search(text))
        )
    )
    f["crypto_scam_flag"] = float(
        f["investment_count"] >= 1
        and any(x in lowered for x in (
            "double", "guaranteed", "send", "airdrop", "giveaway", "profit",
            "return", "10x", "100x", "300%", "500%",
        ))
    )
    f["job_scam_flag"] = float(
        f["job_count"] >= 1
        and any(x in lowered for x in (
            "whatsapp", "telegram", "daily", "no experience", "earn",
            "income", "roz", "ghar se", "payout",
        ))
    )
    f["romance_scam_flag"] = float(
        f["romance_count"] >= 1
        and any(x in lowered for x in (
            "money", "visa", "ticket", "urgent", "hospital", "soldier",
            "army", "widow", "send", "help",
        ))
    )
    f["marketplace_scam_flag"] = float(
        f["marketplace_count"] >= 1
        and (len(amounts) >= 1 or "deposit" in lowered)
        and any(x in lowered for x in (
            "zelle only", "cash app only", "deposit", "shipping", "ship",
            "pay first", "advance", "serious buyer", "dm me",
        ))
    )
    f["loan_investment_flag"] = float(
        (f["investment_count"] >= 1 or "loan" in lowered)
        and any(x in lowered for x in (
            "guaranteed", "pre-approved", "pre approved", "instant",
            "0% interest", "zero interest", "profit", "returns", "approve",
        ))
    )
    f["impersonation_flag"] = float(
        f["authority_count"] >= 1
        and any(x in lowered for x in (
            "official", "security", "alert", "notice", "verify", "support",
            "team", "department", "customer care",
        ))
    )
    f["pressure_flag"] = float(
        f["urgency_count"] >= 2
        and (f["all_caps_flag"] or exclamations >= 2)
    )
    f["too_good_flag"] = float(
        any(x in lowered for x in (
            "100% free", "100% safe", "risk free", "risk-free", "no risk",
            "double your", "guaranteed return", "guaranteed profit",
            "instant approval", "within minutes", "absolutely free",
        ))
    )
    return dict(f)


FEATURE_NAMES: list[str] = [
    "len_chars", "len_tokens", "digit_ratio", "alpha_ratio", "upper_ratio",
    "punct_ratio", "emoji_count", "exclamation_count", "question_count",
    "url_count", "shortener_count", "suspicious_tld_count", "ip_url_count",
    "obfuscated_url_count", "punycode_count", "credential_url_count",
    "email_count", "phone_count", "amount_count", "otp_count",
    "crypto_addr_count", "crypto_amount_count", "urgency_count",
    "reward_count", "threat_count", "authority_count", "payment_count",
    "investment_count", "job_count", "romance_count", "delivery_count",
    "otp_word_count", "marketplace_count", "p2p_verb_count", "click_count",
    "guarantee_count", "whatsapp_flag", "telegram_flag", "bank_flag",
    "kyc_flag", "prize_flag", "send_money_flag", "upi_flag", "bkash_flag",
    "nagad_flag", "paypal_flag", "zelle_flag", "venmo_flag", "cashapp_flag",
    "giftcard_flag", "rtx_flag", "ps5_flag", "iphone_flag", "job_offer_flag",
    "romance_flag", "delivery_flag", "marketplace_flag", "all_caps_flag",
    "many_exclamations_flag", "many_questions_flag", "mixed_script_flag",
    "code_mixed_flag", "script_latin_share", "script_devanagari_share",
    "script_bengali_share", "script_arabic_share", "script_other_share",
    "heuristic_score", "url_present_flag", "money_and_contact_flag",
    "amount_and_url_flag", "threat_and_url_flag", "reward_and_url_flag",
    "payment_and_url_flag", "phone_and_amount_flag", "email_and_amount_flag",
    "otp_and_url_flag", "unrealistic_exchange_flag", "account_threat_flag",
    "delivery_fee_flag", "otp_theft_flag", "crypto_scam_flag",
    "job_scam_flag", "romance_scam_flag", "marketplace_scam_flag",
    "loan_investment_flag", "impersonation_flag", "pressure_flag",
    "too_good_flag", "suspicious_url_flag", "legitimate_url_flag",
    "spacing_obfuscation_flag", "leet_flag", "zero_width_removed_flag",
    "emoji_flag", "url_early_flag", "short_text_flag", "long_text_flag",
    "question_flag", "exclamation_flag",
]


def _add_signal(
    signals: dict[str, dict],
    name: str,
    title: str,
    weight: float,
    severity: str,
    evidence: Sequence[str],
) -> None:
    if weight <= 0:
        return
    current = signals.get(name)
    payload = {
        "name": name,
        "title": title,
        "weight": round(float(weight), 4),
        "severity": severity,
        "evidence": [str(x)[:220] for x in evidence if str(x).strip()][:5],
    }
    if current is None or payload["weight"] > current["weight"]:
        signals[name] = payload


def build_signals(
    text: str,
    lowered: str,
    urls: list[UrlInfo],
    emails: list[str],
    phones: list[str],
    amounts: list[dict],
    otps: list[str],
    crypto: list[str],
    matches: dict[str, list[str]],
    f: dict[str, float],
) -> list[dict]:
    signals: dict[str, dict] = {}
    suspicious = [
        u for u in urls
        if u.is_shortener or u.is_suspicious_tld or u.is_ip or u.is_obfuscated
        or u.has_punycode or u.has_credentials
    ]
    url_evidence = [u.domain or u.raw for u in suspicious[:4]] or [u.domain or u.raw for u in urls[:4]]

    if suspicious:
        _add_signal(
            signals, "suspicious_url", "Suspicious or obfuscated link",
            0.34 if len(suspicious) > 1 else 0.28,
            "high" if len(suspicious) > 1 else "medium", url_evidence,
        )
    if f.get("spacing_obfuscation_flag") or any(u.is_obfuscated for u in urls):
        _add_signal(
            signals, "obfuscated_link", "Disguised link or spaced-out text",
            0.28, "medium", [u.raw for u in urls if u.is_obfuscated][:3] or ["spaced letters"],
        )
    if not suspicious and urls:
        _add_signal(
            signals, "url_present", "Contains a link", 0.04, "low",
            [u.domain or u.raw for u in urls[:3]],
        )

    urgency = matches.get("urgency", [])
    if urgency:
        _add_signal(
            signals, "urgency_pressure", "Urgency or deadline pressure",
            0.22 if len(urgency) >= 2 else 0.14,
            "high" if len(urgency) >= 2 else "medium", urgency[:4],
        )

    reward = matches.get("reward", [])
    if reward:
        _add_signal(
            signals, "reward_bait", "Prize, lottery, or reward bait",
            0.28 if f.get("prize_flag") else 0.16,
            "high" if f.get("prize_flag") else "medium", reward[:4],
        )

    if f.get("unrealistic_exchange_flag"):
        _add_signal(
            signals, "unrealistic_exchange", "Unrealistic exchange or money-doubling offer",
            0.52, "high",
            [text[:220]],
        )

    if f.get("account_threat_flag"):
        _add_signal(
            signals, "account_threat", "Account suspension or verification threat",
            0.42 if f.get("kyc_flag") else 0.34,
            "high" if f.get("kyc_flag") else "medium",
            (matches.get("threat", []) + matches.get("authority", []))[:5],
        )

    if f.get("kyc_flag") and (
        f.get("suspicious_url_flag")
        or (urls and f.get("legitimate_url_flag") == 0 and f.get("threat_count") >= 1)
    ):
        _add_signal(
            signals, "kyc_phishing", "Fake KYC update request", 0.45, "high",
            (matches.get("threat", []) + url_evidence)[:5],
        )

    if f.get("delivery_fee_flag"):
        _add_signal(
            signals, "delivery_fee", "Delivery fee or customs-fee request",
            0.38, "high",
            ([a["raw"] for a in amounts[:3]] + url_evidence)[:5],
        )

    if f.get("otp_theft_flag"):
        _add_signal(
            signals, "otp_theft", "Attempt to obtain an OTP or verification code",
            0.48, "high",
            [mask_secret(x) for x in otps[:3]] + matches.get("otp", [])[:2],
        )

    if f.get("crypto_scam_flag"):
        _add_signal(
            signals, "crypto_scam", "Crypto doubling, giveaway, or investment bait",
            0.44, "high",
            (matches.get("investment", []) + [a["raw"] for a in amounts[:2]])[:5],
        )

    if f.get("job_scam_flag"):
        _add_signal(
            signals, "job_scam", "Too-easy job or daily-payout offer",
            0.30, "medium",
            (matches.get("job", []) + matches.get("urgency", []))[:5],
        )

    if f.get("romance_scam_flag"):
        _add_signal(
            signals, "romance_scam", "Romance or emergency-money pattern",
            0.26, "medium",
            (matches.get("romance", []) + [a["raw"] for a in amounts[:2]])[:5],
        )

    if f.get("marketplace_scam_flag"):
        _add_signal(
            signals, "marketplace_scam", "Marketplace deposit or shipping trap",
            0.30, "medium",
            (matches.get("marketplace", []) + [a["raw"] for a in amounts[:2]])[:5],
        )

    if f.get("loan_investment_flag"):
        _add_signal(
            signals, "loan_investment", "Loan or guaranteed-return investment bait",
            0.30, "medium",
            (matches.get("investment", []) + matches.get("guarantee", []) + matches.get("urgency", []))[:5],
        )

    if f.get("impersonation_flag") and not f.get("account_threat_flag"):
        _add_signal(
            signals, "impersonation", "Impersonation of a brand or authority",
            0.12, "low",
            matches.get("authority", [])[:4],
        )

    if f.get("pressure_flag"):
        _add_signal(
            signals, "pressure_tactics", "All-caps and punctuation pressure",
            0.12, "low",
            [text[:160]],
        )

    if f.get("too_good_flag"):
        _add_signal(
            signals, "too_good_to_be_true", "Too-good-to-be-true language",
            0.26, "medium",
            matches.get("guarantee", [])[:4],
        )

    if f.get("send_money_flag") and (
        f.get("amount_count") or f.get("upi_flag") or f.get("paypal_flag")
        or f.get("bkash_flag") or f.get("nagad_flag") or f.get("phone_count")
        or f.get("email_count")
    ):
        _add_signal(
            signals, "financial_request", "Direct request to send money or pay a fee",
            0.20, "medium",
            (matches.get("payment", []) + [a["raw"] for a in amounts[:2]])[:5],
        )

    if f.get("many_exclamations_flag") or f.get("all_caps_flag"):
        _add_signal(
            signals, "aggressive_formatting", "Aggressive capitalization or punctuation",
            0.05, "low", [text[:120]],
        )

    return sorted(signals.values(), key=lambda s: s["weight"], reverse=True)[:10]


def heuristic_score_from_signals(signals: Sequence[dict]) -> float:
    if not signals:
        return 0.0
    prod = 1.0
    for signal in signals:
        prod *= max(0.0, 1.0 - float(signal["weight"]))
    return float(min(0.985, 1.0 - prod))


def analyze_text(text: object) -> TextAnalysis:
    raw = str(text or "")
    cleaned = clean_text(raw)
    refanged = refang(cleaned)
    lowered = refanged.casefold()
    urls = extract_urls(cleaned)
    emails = extract_emails(cleaned)
    phones = extract_phones(cleaned)
    amounts = extract_amounts(cleaned)
    otps = extract_otps(cleaned)
    crypto = extract_crypto_addresses(cleaned)
    language, confidence, scripts = detect_language(cleaned)
    matches = {name: _matches(lowered, name) for name in LEXICON_PATTERNS}
    f = _flag_features(
        cleaned, lowered, urls, emails, phones, amounts, otps, crypto, matches, scripts
    )
    f["zero_width_removed_flag"] = float(bool(ZERO_WIDTH_RE.search(raw)))
    signals = build_signals(
        cleaned, lowered, urls, emails, phones, amounts, otps, crypto, matches, f
    )
    f["heuristic_score"] = heuristic_score_from_signals(signals)
    for name in FEATURE_NAMES:
        f.setdefault(name, 0.0)
    return TextAnalysis(
        text=cleaned,
        language=language,
        language_confidence=confidence,
        script_shares=scripts,
        urls=urls,
        emails=emails,
        phones=phones,
        amounts=amounts,
        otps=otps,
        crypto_addresses=crypto,
        features=f,
        signals=signals,
        matched_terms={k: v for k, v in matches.items() if v},
    )


@lru_cache(maxsize=250000)
def feature_vector_cached(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in analyze_text(text).feature_vector)


class SignalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [feature_vector_cached(str(t)) for t in X]
        if not rows:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
