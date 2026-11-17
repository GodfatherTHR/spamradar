# Dataset Card — Multilingual Spam & Scam Detection

Generated with seed **42**.

## Sources

- **ScamShield Multilingual Smishing & Social Engineering** (`scamshield`): Real and synthetic SMS covering English, Hindi, Hinglish, UPI/KYC/courier scams. Original splits were discarded and global leakage-safe splits were rebuilt. License: MIT.
- **UCI SMS Spam Collection** (`uci_sms`): Classic 5,572-message English SMS spam corpus. License: Unknown / research use.
- **Bengali SMS Smishing Dataset** (`bengali_smishing`): Bengali, Banglish, English and code-mixed SMS labelled smish/normal/promo. License: MIT.
- **Hindi SMS Spam-Ham 15K** (`hindi_sms`): Balanced Hindi/Hinglish SMS corpus. License: Unknown.
- **Arabic SMS Spam Dataset** (`arabic_sms`): Arabic ham/spam SMS. License: GPL-3.0.
- **SMS Spam Mexico** (`spanish_sms`): Mexican Spanish SMS phishing and ham examples. License: CC-BY-SA-4.0.
- **French Spam/Ham Detection Free 2K** (`french_reviews`): French promotional/phishing and ham texts. Used as a secondary multilingual source. License: Other.
- **SMS Scam Detection Dataset Merged** (`merged_sms`): Large merged SMS scam/ham corpus with partial language labels. License: Other.
- **MultiLanguage SMS Classification Dataset** (`multilang_sms`): Promotional vs service SMS. Service messages are valuable hard negatives (OTP, transaction alerts, delivery updates). License: Unknown.
- **A2P vs P2P SMS Classifier** (`p2p_sms`): Peer-to-peer and application-to-person SMS with Hinglish examples. Used for hard negatives and promotional positives. License: MIT.
- **Phishing Email & SMS with NLP Categories** (`phishing_email`): English phishing positives grouped by urgency, authority and persuasion. License: MIT.
- **Template-based multilingual augmentation** (`synthetic`): Modern P2P, crypto, marketplace, delivery/OTP, romance, job and bank scams plus hard negatives across English, Bengali, Banglish, Hindi, Hinglish, Spanish, French, Arabic and code-mixed text. License: Generated for this project.

## Leakage controls

- All sources were merged before splitting; the original ScamShield/UCI splits were not reused.
- Exact duplicates and SimHash near-duplicates were removed globally before splitting.
- Splitting is group-aware: synthetic template groups never cross train/val/test.
- Stratification uses the `label|language` key with rare strata merged.

## Split summary

```text
{
  "groups": 144601,
  "groups_with_overlap": 0,
  "split_rows": {
    "train": 120659,
    "test": 15162,
    "val": 15045
  },
  "exact_hash_overlap_train_val": 0,
  "exact_hash_overlap_train_test": 0,
  "exact_hash_overlap_val_test": 0
}
```

## Label distribution

```text
{
  "train": {
    "rows": 120659,
    "spam": 54232,
    "ham": 66427,
    "spam_prior": 0.4495
  },
  "val": {
    "rows": 15045,
    "spam": 6761,
    "ham": 8284,
    "spam_prior": 0.4494
  },
  "test": {
    "rows": 15162,
    "spam": 6848,
    "ham": 8314,
    "spam_prior": 0.4517
  },
  "val_cal": {
    "rows": 5177,
    "spam": 1035,
    "ham": 4142,
    "spam_prior": 0.1999
  },
  "val_tune": {
    "rows": 5178,
    "spam": 1036,
    "ham": 4142,
    "spam_prior": 0.2001
  },
  "test_realistic": {
    "rows": 10392,
    "spam": 2078,
    "ham": 8314,
    "spam_prior": 0.2
  }
}
```

## Language distribution

```text
{
  "train": {
    "English": 99041,
    "Hinglish": 5477,
    "French": 3794,
    "Hindi": 2268,
    "Banglish": 1974,
    "Bengali": 1960,
    "Spanish": 1823,
    "mixed": 1724,
    "German": 1405,
    "Arabic": 1187,
    "Tamil": 2,
    "ms": 1,
    "Malayalam": 1,
    "Telugu": 1,
    "nl": 1
  },
  "val": {
    "English": 12335,
    "Hinglish": 685,
    "French": 495,
    "Banglish": 300,
    "Hindi": 253,
    "Spanish": 237,
    "mixed": 232,
    "Bengali": 201,
    "German": 176,
    "Arabic": 130,
    "Malayalam": 1
  },
  "test": {
    "English": 12423,
    "Hinglish": 698,
    "French": 479,
    "Bengali": 304,
    "Hindi": 304,
    "Banglish": 289,
    "mixed": 209,
    "Spanish": 188,
    "German": 175,
    "Arabic": 92,
    "Telugu": 1
  }
}
```

## Data quality notes

- Some real-world corpora contain label noise. Obvious contradictory noisy-source rows were filtered with the heuristic audit; ambiguous clusters were dropped during deduplication.
- The `Kaggle_Hindi_Merged` subset from ScamShield was excluded because its texts were low quality and heavily skewed.
- Synthetic hard negatives deliberately include money requests, legitimate URLs, OTP notices, bank alerts and urgent family messages.
- This dataset is for research and defensive spam filtering. Check each upstream license before commercial use.
