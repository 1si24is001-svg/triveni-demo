# TRIVENI: live entity resolution demo

TRIVENI finds every record of the same business across three noisy data
sources (Amazon ML Challenge 2026). This folder holds a classroom demo that
runs the trained models live in a web browser, plus the code that builds it.

## Open the demo

Open `dist/triveni_demo.html` in Chrome or Edge. It is one self-contained
file: the engine, the trained LightGBM models and the test data are inside
it, so it also works offline (fonts fall back to system fonts without
internet).

## What the demo shows

| Tab | What happens |
| --- | --- |
| Normalize a name | Type a business name in English, Hindi, Kannada, Tamil, Malayalam or Bengali; watch it become a phonetic key. Three spellings of one shop collapse to the same key. |
| Compare two records | The stage-1 LightGBM model scores two records you type, with its similarity features and shared blocking keys. |
| Run the pipeline | Runs the whole pipeline on 360 businesses and 865 records (US, India, France): blocking, two models, calibration, expected-F0.5 decoding. Shows the score, per-country results and every decision. |
| How it works | The eight stages and what the model learned. |

## Five-minute class script

1. **Normalize (1 min).** Start on "Ganga Surya Sweets": English, Kannada and
   Hindi all give `knk sr svts`. Click "Silver Consultancy" (Malayalam,
   Bengali): close keys, not identical, so the model has to judge.
2. **Compare (1.5 min).** "Hindi vs English" scores 1.00. "Same brand, other
   branch" scores near 0: the names match, but postcode and house number
   conflict. Edit the house number live to show the score move.
3. **Pipeline (2 min).** Click "Run the pipeline". Point at the funnel
   (103,800 possible pairs to 2,880 candidates), the F0.5 score, and France,
   which the model never saw. Open "Mistakes" to discuss a failure honestly.
4. **How it works (0.5 min).** The eight stages and top features.

## Results (demo data)

Test set: 360 businesses, 865 records, including 48 names in Hindi, Kannada
or Tamil script. Macro F0.5 0.977 (precision 98.8%, recall 96.1%); France,
never seen in training, 0.992. Blocking keeps 98.8% of true matches while
removing 97.2% of pairs. The browser reproduces the Python predictions
exactly (identical candidates, probabilities equal to 1e-15).

The demo data is synthetic: it copies the challenge's format and noise so
everything can run in a browser. The full-scale pipeline for the real 24M
records is in `pipeline/`.

## Rebuild from scratch

```bash
pip install lightgbm scikit-learn numpy
python demo_data.py        # generate train/test data (data/*.json)
python train_demo.py       # train models, evaluate, write web/bundle.json
python export_tables.py    # copy normalization tables into the bundle
node parity.js             # check the browser engine matches Python
python build_web.py        # write dist/triveni_demo.html
```

## Files

```
demo_data.py       demo dataset generator (with Indic-script names)
base_gen.py        base noise generator (typos, abbreviations, hard negatives)
normalize.py       normalizer: cleaning, legal forms, phonetic key
translit.py        Indic-script transliteration (8 scripts)
demo_core.py       blocking, 27 pair features, stage-2 context features
decode.py          expected-F0.5 set decoding
train_demo.py      trains stage 1, stage 2, isotonic calibration; evaluates
export_tables.py   shares normalization tables with the browser
web/triveni.js     JavaScript port of the whole pipeline
web/template.html  the user interface
build_web.py       assembles the single-file demo
parity.js          Python vs browser parity test
pipeline/          full TRIVENI pipeline for the real 24M-record data
```
