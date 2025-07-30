#!/usr/bin/env python3
"""
Translate the 'question' and 'answers' fields of a Pew-style JSON file to Hebrew.

Usage:
    python translate_questions.py <input_json> [-o translated_questions.json]

• Always imports ./converted_questions.json as the reference - just for schema
  parity checks; it is **not modified**.
• Relies on `deep_translator` (pip install deep-translator).  You only hit the
  free Google-Translate web endpoint, so no API key is required.

Version: 2
• Added time measurement.
"""

import argparse, json, sys, time
from pathlib import Path
from deep_translator import GoogleTranslator

# --------------------------------------------------------------------------- #
# 1.  CLI & file checks
# --------------------------------------------------------------------------- #

start = time.time()
p = argparse.ArgumentParser()
p.add_argument("input_json", help="file whose questions/answers you want to translate")
p.add_argument("-o", "--output", default="translated_questions.json",
               help="output path (default: translated_questions.json)")
args = p.parse_args()

src_path = Path(args.input_json)
ref_path = Path("converted_questions.json")
if not src_path.exists():
    sys.exit(f"❌ Cannot find {src_path!s}")
if not ref_path.exists():
    sys.exit("❌ converted_questions.json is missing - needed for sanity-checks")

src = json.loads(src_path.read_text(encoding="utf-8"))
ref = {q["id"]: q for q in json.loads(ref_path.read_text(encoding="utf-8"))}

# --------------------------------------------------------------------------- #
# 2.  Build a consistent answer-translation dictionary
# --------------------------------------------------------------------------- #

# collect every unique answer string that appears in *this* input JSON
unique_answers: set[str] = set()
for q in src:
    unique_answers.update(q["answers"].values())

translator = GoogleTranslator(source="en", target="iw")
answer_he: dict[str, str] = {}

print("🔄 Translating answer options …")
for ans in sorted(unique_answers):
    # avoid re-translating if the exact same English answer appeared before
    answer_he[ans] = translator.translate(ans)

# --------------------------------------------------------------------------- #
# 3.  Translate questions & inject Hebrew answers
# --------------------------------------------------------------------------- #

print("🔄 Translating questions …")
for q in src:
    # sanity-check: make sure the metadata lines up with the reference set
    if q["id"] not in ref:
        print(f"⚠️  id {q['id']} not found in converted_questions.json", file=sys.stderr)

    # translate question text
    q["question"] = translator.translate(q["question"])

    # translate each answer using the dictionary we just built
    for code, eng_text in q["answers"].items():
        q["answers"][code] = answer_he[eng_text]

# append the dictionary as the last element
src.append({"answer_translation_dict": answer_he})

# --------------------------------------------------------------------------- #
# 4.  Save
# --------------------------------------------------------------------------- #

out_path = Path(args.output)
out_path.write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
end = time.time()
print(f"✅ Wrote {out_path!s}")
print(f"⏱️  Total time: {end - start:.2f} seconds")