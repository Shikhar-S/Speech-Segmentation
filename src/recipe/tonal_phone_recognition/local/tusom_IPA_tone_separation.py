import yaml
import re
import glob
import os

def split_tone_letters(tone_letters):
    # Chao tone letters: ˥˦˧˨˩ (U+02E5–U+02EA)
    pattern = r'([^\u02E5-\u02EA]+)([\u02E5-\u02EA]+)'
    matches = re.findall(pattern, tone_letters)
    ipa_parts = []
    tone_parts = []
    for ipa, tone in matches:
        ipa_parts.append(ipa.strip())
        tone_parts.append(tone.strip())
    ipa_only = ' '.join(ipa_parts)
    tone_only = ' '.join(tone_parts)
    return ipa_only, tone_only

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    for key, entry in data.items():
        tone_letters = entry.get('tone_letters')
        if tone_letters:
            ipa, tone = split_tone_letters(tone_letters)
            entry['ipa_only'] = ipa
            entry['tone_only'] = tone

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

# Process all .yml files in the current directory
for yml_file in glob.glob("*.yml"):
    out_file = f"separated_{yml_file}"
    process_file(yml_file, out_file)
    print(f"Processed {yml_file} -> {out_file}")