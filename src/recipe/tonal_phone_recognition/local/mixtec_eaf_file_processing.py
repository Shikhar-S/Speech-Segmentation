import pympi
import yaml
import epitran

def sentence_to_ipa(sentence, epi):
    words = sentence.split(" ") 
    ipa_list = [epi.transliterate(word) for word in words]
    return ' '.join(ipa_list)

eaf_file_path = 'Agricultura-Finalizados\Yolox_Agric_CTB501_Fases-de-la-luna-para-sembrar_2009-11-26-k_ed-2025-02-12-normalized.eaf'

# Load the .eaf file
eaf = pympi.Elan.Eaf(eaf_file_path)

# Prepare data to dump to YAML
data = {}

for tier_name in eaf.get_tier_names():
    tier_info = []
    annotations = eaf.get_annotation_data_for_tier(tier_name)
    epi = epitran.Epitran('xty-Latn', tones = True)
    for ann in annotations:
        start_time, end_time, value = ann[:3]
        ipa_with_tones = sentence_to_ipa(value, epi)
        tier_info.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': value,
            'ipa_with_tones': ipa_with_tones
        })
    
    data[tier_name] = tier_info

# Save as YAML
with open(f'{eaf_file_path}.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)