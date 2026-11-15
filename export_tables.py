"""Copy normalizer and transliteration tables into web/bundle.json so the
browser uses exactly the same dictionaries as Python."""
import json
import normalize as N
import translit as T

n = N.Normalizer()
tables = {
    "cons": T.CONS, "nukta_base": T.NUKTA_BASE, "vowels": T.VOWELS, "matras": T.MATRAS,
    "marks": T.MARKS, "digits": T.DIGITS, "chillu": T.CHILLU, "virama": T.VIRAMA, "nukta": T.NUKTA,
    "loan": [[rx.pattern, rep] for rx, rep in T.LOAN],
    "suffixes": sorted(n.suffixes), "leading_stop": sorted(N.LEADING_STOP),
    "name_abbrev": n.name_abbrev, "addr_abbrev": n.addr_abbrev,
    "us_states": N.US_STATES, "in_states": N.IN_STATES, "web_tokens": sorted(N.WEB_TOKENS),
    "landmark_re": N.LANDMARK_RE.pattern, "dba_re": N.DBA_RE.pattern,
}
b = json.load(open("web/bundle.json"))
b["tables"] = tables
json.dump(b, open("web/bundle.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("tables:", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in tables.items()})
