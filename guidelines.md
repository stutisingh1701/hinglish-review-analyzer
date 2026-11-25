# Annotation guidelines: Hinglish product reviews

**Unit:** one YouTube comment or review. Read it fully before labelling.

## 1. Is it usable?
Mark **spam / not a review** for links, promotions, "first comment", questions with no opinion ("price kya hai?").
Mark **not Hinglish** if it is fully English or fully Devanagari Hindi.

## 2. Aspects
Label only aspects the text **gives an opinion on**. A bare mention ("battery 5000mAh hai") is **neutral**, not positive.

| Aspect | Covers | Example words |
|---|---|---|
| battery | backup, charging speed, charger, drain | battery, backup, charging, mAh |
| camera | photos, video, selfie, zoom, night mode | camera, pic, selfie, video |
| display | screen, brightness, refresh rate, touch | display, screen, AMOLED, 120Hz |
| performance | speed, lag, heating, gaming, RAM | lag, hang, garam, gaming, smooth |
| price | price, value for money | price, paisa vasool, mehnga, sasta |
| design | look, build, weight, colour, buttons | design, look, plastic, heavy |
| audio | speaker, call audio, earphones | speaker, awaaz, sound |
| service | delivery, seller, packaging, warranty, customer care | delivery, seller, replacement |

## 3. Sentiment
- **positive**: praise ("camera mast hai", "battery din bhar chalti hai")
- **negative**: complaint ("phone garam hota hai", "paisa barbaad")
- **neutral**: factual or mixed-but-flat ("theek thaak hai", "average")
- Negation flips polarity: "camera accha **nahi** hai" → negative.
- Comparisons: "Samsung se better camera" → camera positive.
- Sarcasm: label the **intended** meaning ("wah kya battery hai, 2 ghante me khatam" → negative).
- If one aspect gets both praise and complaint, choose the one the writer stresses; if truly equal, neutral.

## 4. Quality check
Two annotators label the same 100 comments independently. Run `python scripts/kappa.py a.csv b.csv`.
Target: kappa ≥ 0.7. Discuss disagreements, update these guidelines, then label the rest.
