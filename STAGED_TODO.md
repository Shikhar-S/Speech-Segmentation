# Staged 변경분 기반 TODO

## 토크나이저 (`src/core/tokenizer`)
- [x] `BaseTokenizer`를 상속하는 실제 tokenizer 구현체 추가 (`encode`, `decode`, label/id 매핑 로직 완성).
  - `CharacterTokenizer` 구현 완료 (`src/core/tokenizer/character_tokenizer.py`).
- [x] Datamodule과 공유할 vocab/label 사양 문서화 후 `src/core/__init__.py`, `src/core/tokenizer/__init__.py`의 export 목록 확정.
  - `docs/tokenization.md` 작성 및 `__init__.py` export 완료.

## 모델 헤드 (`src/model/head*`)
- [x] `RNNHead`에 GRU/LSTM 스택, pooling 전략, 드롭아웃 등 forward 경로 구현.
- [x] `TransformerHead`에 TransformerEncoder, positional encoding, length masking을 포함한 forward 로직 구현.
- [x] 두 head 모두를 위한 config 스키마(Hydra/OmegaConf) 정의 및 `_target_` 업데이트, 필요한 경우 `TaskType` 확장 검토.

## L1 분류 레시피 (`src/recipe/l1_classification/model_module.py`)
- [x] Head forward가 완성된 뒤 end-to-end forward/taining step을 통합 테스트.
- [x] Datamodule/Tokenizer에서 제공하는 label ↔︎ id 매핑을 recipe에 주입해 메트릭(`MulticlassAccuracy/F1`)과 일관성 확보.
- [x] Encoder freeze/unfreeze 조합별 optimizer/scheduler 동작 검증 및 grad logging(`grad_norm`) 범위 조정.

## Hydra 설정
### `configs/model/l1_classification.yaml`
- [x] `num_classes` 기본값을 실제 L1 클래스 수(7)로 채우고 experiment 쪽 중복 정의 제거.
- [x] `id_to_label`을 고정 라벨 리스트로 설정 (`["ar", "en", "es", "hi", "ko", "vi", "zh"]`).
- [x] `input_type` 파라미터 추가 ("audio" | "ipa") 모드 지원.
- [ ] Scheduler/optimizer 하이퍼파라미터를 실험 기반으로 검증하고 필요 시 warmup 정책 반영.

### `configs/experiment/phoneme_ipa_l1_classification.yaml` (신규)
- [x] IPA 텍스트 기반 L1 분류 실험 config 작성.
- [x] `CmuL2ArcticIPADataModule` 연동 및 JSON 경로 설정.
- [x] `IPAEmbedding` 모듈 연동.

### `configs/experiment/phoneme_hidden_l1_classification.yaml` (신규)
- [x] Audio encoder hidden state 기반 L1 분류 실험 config 작성.
- [ ] PR #3 병합 후 `cmu_l2arctic_l1` DataModule 연동 필요.

## 이중 파이프라인 지원 (Audio + IPA)
### DataModule
- [x] `src/data/cmu_l2arctic/ipa_data.py`: IPA 전용 DataModule 구현 (JSON 로딩).
- [x] `configs/data/cmu_l2arctic_ipa.yaml`: IPA DataModule config.

### Model
- [x] `src/model/common/ipa_embedding.py`: IPA Embedding 모듈 구현.
- [x] `configs/model/net/ipa_embedding.yaml`: IPA Embedding config.
- [x] `src/recipe/l1_classification/model_module.py`: `input_type` 분기 로직 추가.
