"""Classroom demo dataset in the challenge format, with Indic-script names.

Builds on the synthetic generator (typos, abbreviations, DBA names,
landmarks, missing postcodes, hard negatives) and adds what the real data
showed: India records in Sources 2/3 written fully in Hindi, Kannada or
Tamil script, using hand-written spellings of each business word.
Train covers US + India; test adds France, which training never sees.
"""
import json
import random
from pathlib import Path

import base_gen as G

SCRIPT = {
    "hi": {"Lakshmi": "लक्ष्मी", "Shree": "श्री", "Ganesh": "गणेश", "Sai": "साई", "Balaji": "बालाजी",
           "Krishna": "कृष्णा", "Durga": "दुर्गा", "Annapurna": "अन्नपूर्णा", "Bharat": "भारत", "Om": "ओम",
           "Vinayak": "विनायक", "Jai": "जय", "Mahalaxmi": "महालक्ष्मी", "Sharma": "शर्मा", "Patel": "पटेल",
           "Agarwal": "अग्रवाल", "Reddy": "रेड्डी", "Kaveri": "कावेरी", "Ganga": "गंगा", "Surya": "सूर्या",
           "Traders": "ट्रेडर्स", "Sweets": "स्वीट्स", "Textiles": "टेक्सटाइल्स", "Electronics": "इलेक्ट्रॉनिक्स",
           "Jewellers": "ज्वैलर्स", "Hardware": "हार्डवेयर", "Enterprises": "एंटरप्राइजेज", "Medicals": "मेडिकल्स",
           "Agencies": "एजेंसीज", "Motors": "मोटर्स", "Private Limited": "प्राइवेट लिमिटेड",
           "Pvt Ltd": "प्रा. लि.", "Limited": "लिमिटेड", "Ltd": "लिमिटेड"},
    "kn": {"Lakshmi": "ಲಕ್ಷ್ಮೀ", "Shree": "ಶ್ರೀ", "Ganesh": "ಗಣೇಶ್", "Sai": "ಸಾಯಿ", "Balaji": "ಬಾಲಾಜಿ",
           "Krishna": "ಕೃಷ್ಣ", "Durga": "ದುರ್ಗಾ", "Annapurna": "ಅನ್ನಪೂರ್ಣ", "Bharat": "ಭಾರತ್", "Om": "ಓಂ",
           "Vinayak": "ವಿನಾಯಕ", "Jai": "ಜೈ", "Mahalaxmi": "ಮಹಾಲಕ್ಷ್ಮಿ", "Sharma": "ಶರ್ಮಾ", "Patel": "ಪಟೇಲ್",
           "Agarwal": "ಅಗರ್ವಾಲ್", "Reddy": "ರೆಡ್ಡಿ", "Kaveri": "ಕಾವೇರಿ", "Ganga": "ಗಂಗಾ", "Surya": "ಸೂರ್ಯ",
           "Traders": "ಟ್ರೇಡರ್ಸ್", "Sweets": "ಸ್ವೀಟ್ಸ್", "Textiles": "ಟೆಕ್ಸ್‌ಟೈಲ್ಸ್", "Electronics": "ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್",
           "Jewellers": "ಜ್ಯುವೆಲ್ಲರ್ಸ್", "Hardware": "ಹಾರ್ಡ್‌ವೇರ್", "Enterprises": "ಎಂಟರ್‌ಪ್ರೈಸಸ್",
           "Medicals": "ಮೆಡಿಕಲ್ಸ್", "Agencies": "ಏಜೆನ್ಸೀಸ್", "Motors": "ಮೋಟಾರ್ಸ್",
           "Private Limited": "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Pvt Ltd": "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Limited": "ಲಿಮಿಟೆಡ್", "Ltd": "ಲಿಮಿಟೆಡ್"},
    "ta": {"Lakshmi": "லட்சுமி", "Shree": "ஸ்ரீ", "Ganesh": "கணேஷ்", "Sai": "சாய்", "Balaji": "பாலாஜி",
           "Krishna": "கிருஷ்ணா", "Durga": "துர்கா", "Annapurna": "அன்னபூர்ணா", "Bharat": "பாரத்", "Om": "ஓம்",
           "Vinayak": "விநாயக்", "Jai": "ஜெய்", "Mahalaxmi": "மகாலட்சுமி", "Sharma": "சர்மா", "Patel": "படேல்",
           "Agarwal": "அகர்வால்", "Reddy": "ரெட்டி", "Kaveri": "காவேரி", "Ganga": "கங்கா", "Surya": "சூர்யா",
           "Traders": "டிரேடர்ஸ்", "Sweets": "ஸ்வீட்ஸ்", "Textiles": "டெக்ஸ்டைல்ஸ்", "Electronics": "எலக்ட்ரானிக்ஸ்",
           "Jewellers": "ஜுவல்லர்ஸ்", "Hardware": "ஹார்டுவேர்", "Enterprises": "எண்டர்பிரைசஸ்",
           "Medicals": "மெடிக்கல்ஸ்", "Agencies": "ஏஜென்சீஸ்", "Motors": "மோட்டார்ஸ்",
           "Private Limited": "பிரைவேட் லிமிடெட்", "Pvt Ltd": "பிரைவேட் லிமிடெட்", "Limited": "லிமிடெட்", "Ltd": "லிமிடெட்"},
}
R = random.Random(11)
G.R.seed(11)


def indic_name(e):
    script = R.choices(["hi", "kn", "ta"], weights=[5, 3, 2])[0]
    d = SCRIPT[script]
    words = [d[w] for w in e["core"].split()] + [d[e["trade"]]]
    suf = e["suf"][1] if R.random() < 0.6 else e["suf"][0]
    if suf:
        words.append(d.get(suf, ""))
    return " ".join(w for w in words if w), script


def build(countries, n, start=1):
    s1, s2, s3, gt, meta = [], [], [], {}, {}
    c = {"1": start, "2": start * 3, "3": start * 3}

    def nid(src):
        c[src] += 1
        return f"S{src}-{c[src]:06d}"

    for country in countries:
        for _ in range(n):
            e = G.make_entity(country, nid("1"))
            s1.append((e["eid"], *G.render(e, False), country))
            matches = []
            if R.random() < 0.75:                       # 25% singletons
                for src, bucket in (("2", s2), ("3", s3)):
                    for _ in range(R.choice([0, 1, 1, 1, 2])):
                        rid = nid(src)
                        name, addr = G.render(e, True)
                        if country == "India" and R.random() < 0.25:
                            name, sc = indic_name(e)
                            meta[rid] = sc
                        bucket.append((rid, name, addr, country))
                        matches.append(rid)
            gt[e["eid"]] = matches
            if R.random() < 0.3:                        # same brand, other branch
                f = dict(e)
                f["num"], f["pc"] = R.randint(1, 999), str(int(e["pc"]) + R.randint(1, 50))
                f["sname"] = R.choice(G.WORDS[country])
                s2.append((nid("2"), *G.render(f, True), country))
            if R.random() < 0.3:                        # same core name, other trade
                f = G.make_entity(country, "x")
                f["core"] = e["core"]
                s3.append((nid("3"), *G.render(f, True), country))
            if R.random() < 0.3:                        # other business, same building
                f = G.make_entity(country, "x")
                for k in ("street", "sname", "num", "city", "state", "pc"):
                    f[k] = e[k]
                s2.append((nid("2"), *G.render(f, True), country))
    R.shuffle(s2)
    R.shuffle(s3)
    return s1, s2, s3, gt, meta


def as_records(rows):
    return [dict(zip(("entity_id", "business_name", "business_address", "country"), r)) for r in rows]


def main(out="data"):
    out = Path(out)
    out.mkdir(exist_ok=True)
    tr = build(["US", "India"], 700, start=1)
    te = build(["US", "India", "France"], 120, start=500000)
    for name, (s1, s2, s3, gt, meta) in (("train", tr), ("test", te)):
        json.dump({"s1": as_records(s1), "pool": as_records(s2) + as_records(s3),
                   "truth": gt, "script": meta}, open(out / f"{name}.json", "w"), ensure_ascii=False)
        print(name, "S1", len(s1), "pool", len(s2) + len(s3), "indic", len(meta))


if __name__ == "__main__":
    main()
