
import os
os.environ["PYTHONUTF8"] = "1"

import yaml
import epitran
import itertools
from tqdm import tqdm
import re

def load_cedict_dict(cedict_path):
    cedict = {}
    with open(cedict_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            trad, simp, rest = parts
            pinyin = rest.split(']')[0].strip('[')
            cedict[simp] = pinyin
    return cedict

def sentence_to_pinyin(sentence, cedict):
    words = sentence.split(" ")
    pinyin_list = []
    for word in words:
        if word in cedict:
            pinyin_list.append(cedict[word])
        else:
            # fallback: map each character
            chars_pinyin = [cedict.get(char, char) for char in word]
            pinyin_list.extend(chars_pinyin)
    return ' '.join(pinyin_list)

def sentence_to_ipa(sentence, epi):
    words = sentence.split(" ") 
    ipa_list = [epi.transliterate(word) for word in words]
    return ' '.join(ipa_list)

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

if __name__ == "__main__":
    cedict_path = 'cedict_ts.u8'
    transcript_path = 'aishell_transcript_v0.8.txt'
    output_path = 'transcript_pinyin.yml'

    cedict = load_cedict_dict(cedict_path)
    epi = epitran.Epitran('cmn-Hans', tones = True, cedict_file='cedict_1_0_ts_utf-8_mdbg.txt')

    with open(transcript_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            utt_id, sentence = parts
            pinyin = sentence_to_pinyin(sentence, cedict)
            ipa_with_tones = sentence_to_ipa(sentence, epi)
            ipa_only, tone_only = split_tone_letters(ipa_with_tones)
            yaml.dump({utt_id: {
                'sentence': sentence, 
                'pinyin': pinyin, 
                'ipa_with_tones': ipa_with_tones, 
                'ipa_only': ipa_only, 
                'tone_only': tone_only}}, 
                fout, allow_unicode=True, sort_keys=False)