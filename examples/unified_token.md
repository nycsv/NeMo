좋습니다. 지금 기준으로는 **NeMo 원본 코어 파일을 직접 고치기보다**, **repo 안에 작은 커스텀 모듈을 추가해서 formatter를 오버라이드**하는 방식이 가장 안전합니다. NeMo 예제 config는 `prompt_format: canary2`를 지원하고, dataset 쪽은 `text_field: "text"`, `lang_field: "target_lang"`를 씁니다. tokenizer 생성 스크립트는 `--data_file` 기반 txt 입력과 `--spe_user_defined_symbols`를 지원합니다. SentencePiece 쪽도 `user_defined_symbols`와 `control_symbols`를 tokenizer 학습 명령에 실제로 넘깁니다. ([GitHub][1])

## 어디에 두면 되나

가장 추천하는 위치는 **NeMo repo 바깥 별도 패키지**이지만, repo 안에 두고 싶다면 아래처럼 두는 게 관리하기 좋습니다.

```text
NeMo/
  examples/
    asr/
      custom/
        __init__.py
        canary2_tokens.py
        custom_canary2_formatter.py
```

이유는 간단합니다.

* `nemo/collections/...` 내부를 직접 수정하면 upstream pull/update 때 충돌이 잘 납니다.
* `examples/asr/custom/` 같은 경로는 실험용 커스텀 코드를 두기 좋습니다.
* 학습 스크립트 실행 시 repo root에서 실행하면 `examples.asr.custom...` import가 쉽습니다.

즉, **`custom_canary2_formatter.py`는 `examples/asr/custom/` 아래**에 두는 걸 추천합니다.

---

# 1) `examples/asr/custom/canary2_tokens.py`

아래는 **locale 구조 10개 언어** 기준 예시입니다.
토큰 문자열은 반드시 tokenizer 학습 시 넣는 문자열과 같아야 합니다.

```python id="m99z7n"
# examples/asr/custom/canary2_tokens.py

LANG_TOKENS = {
    "en-US": "<|lang:en-US|>",
    "en-GB": "<|lang:en-GB|>",
    "ko-KR": "<|lang:ko-KR|>",
    "es-ES": "<|lang:es-ES|>",
    "es-MX": "<|lang:es-MX|>",
    "fr-FR": "<|lang:fr-FR|>",
    "de-DE": "<|lang:de-DE|>",
    "ja-JP": "<|lang:ja-JP|>",
    "pt-BR": "<|lang:pt-BR|>",
    "ar-SA": "<|lang:ar-SA|>",
}

TASK_TOKENS = {
    "asr": "<|task:asr|>",
    "ast": "<|task:ast|>",
}

PNC_TOKENS = {
    "yes": "<|pnc|>",
    "no": "<|nopnc|>",
}

ITN_TOKENS = {
    "yes": "<|itn|>",
    "no": "<|noitn|>",
}

TIMESTAMP_TOKENS = {
    "yes": "<|timestamp|>",
    "no": "<|notimestamp|>",
}

DIARIZE_TOKENS = {
    "yes": "<|diarize|>",
    "no": "<|nodiarize|>",
}

# tokenizer 학습 때 같이 넣을 모든 special tokens
ALL_SPECIAL_TOKENS = (
    [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|endoftext|>",
        "<|emo:undefined|>",
    ]
    + list(LANG_TOKENS.values())
    + list(TASK_TOKENS.values())
    + list(PNC_TOKENS.values())
    + list(ITN_TOKENS.values())
    + list(TIMESTAMP_TOKENS.values())
    + list(DIARIZE_TOKENS.values())
)
```

---

# 2) `examples/asr/custom/custom_canary2_formatter.py`

이 formatter는 **semantic 값**을 **prompt token 문자열**로 바꿉니다.

```python id="c1yjlwm"
# examples/asr/custom/custom_canary2_formatter.py

from dataclasses import dataclass
from typing import Any, Dict

from examples.asr.custom.canary2_tokens import (
    LANG_TOKENS,
    TASK_TOKENS,
    PNC_TOKENS,
    ITN_TOKENS,
    TIMESTAMP_TOKENS,
    DIARIZE_TOKENS,
)


def _normalize_yes_no(value: str | None, default: str = "no") -> str:
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"yes", "true", "1"}:
        return "yes"
    if value in {"no", "false", "0"}:
        return "no"
    raise ValueError(f"Unsupported yes/no value: {value}")


@dataclass
class CustomCanary2PromptFormatter:
    """
    Semantic sample fields:
      - source_lang: "en-US"
      - target_lang: "ko-KR"
      - taskname: "asr" | "ast"
      - pnc: "yes" | "no"
      - itn: "yes" | "no"   (optional)
      - timestamp: "yes" | "no" (optional)
      - diarize: "yes" | "no"   (optional)

    Output prompt string:
      <|startofcontext|><|startoftranscript|><|emo:undefined|>...
    """

    emotion_token: str = "<|emo:undefined|>"
    start_context_token: str = "<|startofcontext|>"
    start_transcript_token: str = "<|startoftranscript|>"

    def _lang_token(self, lang: str) -> str:
        if lang not in LANG_TOKENS:
            raise ValueError(
                f"Unsupported language '{lang}'. "
                f"Supported: {sorted(LANG_TOKENS.keys())}"
            )
        return LANG_TOKENS[lang]

    def _task_token(self, task: str) -> str:
        if task not in TASK_TOKENS:
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Supported: {sorted(TASK_TOKENS.keys())}"
            )
        return TASK_TOKENS[task]

    def build_prompt(self, sample: Dict[str, Any]) -> str:
        source_lang = self._lang_token(sample["source_lang"])
        target_lang = self._lang_token(sample["target_lang"])
        task_token = self._task_token(sample["taskname"])

        pnc_key = _normalize_yes_no(sample.get("pnc"), default="yes")
        itn_key = _normalize_yes_no(sample.get("itn"), default="no")
        ts_key = _normalize_yes_no(sample.get("timestamp"), default="no")
        diarize_key = _normalize_yes_no(sample.get("diarize"), default="no")

        pnc_token = PNC_TOKENS[pnc_key]
        itn_token = ITN_TOKENS[itn_key]
        ts_token = TIMESTAMP_TOKENS[ts_key]
        diarize_token = DIARIZE_TOKENS[diarize_key]

        prompt = (
            f"{self.start_context_token}"
            f"{self.start_transcript_token}"
            f"{self.emotion_token}"
            f"{task_token}"
            f"{source_lang}"
            f"{target_lang}"
            f"{pnc_token}"
            f"{itn_token}"
            f"{ts_token}"
            f"{diarize_token}"
        )
        return prompt

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a shallow-copied sample with a prompt field added.
        """
        out = dict(sample)
        out["prompt"] = self.build_prompt(sample)
        return out
```

---

# 3) tokenizer 생성 스크립트

NeMo의 `process_asr_text_tokenizer.py`는 `--data_file`로 여러 txt 파일을 넣을 수 있고, `--spe_user_defined_symbols`를 그대로 전달합니다. ([GitHub][2])

locale 10개 기준 예시는 아래처럼 가면 됩니다.

```bash id="pk5t4m"
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --data_file=/data/txt/en-US.txt,/data/txt/en-GB.txt,/data/txt/ko-KR.txt,/data/txt/es-ES.txt,/data/txt/es-MX.txt,/data/txt/fr-FR.txt,/data/txt/de-DE.txt,/data/txt/ja-JP.txt,/data/txt/pt-BR.txt,/data/txt/ar-SA.txt \
  --data_root=/exp/tokenizers/canary2_unified_locale \
  --vocab_size=16384 \
  --tokenizer=spe \
  --spe_type=bpe \
  --spe_character_coverage=0.9995 \
  --spe_user_defined_symbols \
    "<|startofcontext|>" "<|startoftranscript|>" "<|endoftext|>" "<|emo:undefined|>" \
    "<|lang:en-US|>" "<|lang:en-GB|>" "<|lang:ko-KR|>" "<|lang:es-ES|>" "<|lang:es-MX|>" \
    "<|lang:fr-FR|>" "<|lang:de-DE|>" "<|lang:ja-JP|>" "<|lang:pt-BR|>" "<|lang:ar-SA|>" \
    "<|task:asr|>" "<|task:ast|>" \
    "<|pnc|>" "<|nopnc|>" \
    "<|itn|>" "<|noitn|>" \
    "<|timestamp|>" "<|notimestamp|>" \
    "<|diarize|>" "<|nodiarize|>" \
  --log
```

---

# 4) config 예시

NeMo 예제 config는 `prompt_format: canary2`, `text_field: "text"`, `lang_field: "target_lang"` 구조를 사용합니다. ([GitHub][1])

당신 구조에 맞춰 최소한 이렇게 두면 됩니다.

```yaml id="unujh6"
model:
  prompt_format: canary2

  tokenizer:
    type: bpe
    dir: /exp/tokenizers/canary2_unified_locale/tokenizer_spe_bpe_v16384

  prompt_defaults:
    pnc: "yes"
    itn: "no"
    timestamp: "no"
    diarize: "no"

  train_ds:
    manifest_filepath: /data/manifests/train.jsonl
    text_field: "text"
    lang_field: "target_lang"

  validation_ds:
    manifest_filepath: /data/manifests/val.jsonl
    text_field: "text"
    lang_field: "target_lang"
```

---

# 5) manifest 예시

학습 데이터는 여전히 structured input이 필요합니다. tokenizer 학습만 txt이고, model 학습은 `source_lang`, `target_lang`, `taskname`, `pnc`를 담은 JSONL이 맞습니다. NeMo 예제도 이 구조를 전제로 합니다. ([GitHub][1])

```json id="6qm27r"
{"audio_filepath": "/data/audio/a.wav", "duration": 3.2, "text": "hello world", "source_lang": "en-US", "target_lang": "en-US", "taskname": "asr", "pnc": "yes"}
{"audio_filepath": "/data/audio/b.wav", "duration": 2.4, "text": "안녕하세요", "source_lang": "ko-KR", "target_lang": "ko-KR", "taskname": "asr", "pnc": "yes"}
{"audio_filepath": "/data/audio/c.wav", "duration": 4.7, "text": "hola", "source_lang": "es-ES", "target_lang": "en-US", "taskname": "ast", "pnc": "no"}
```

---

# 6) formatter를 실제로 어디서 호출하나

여기서 가장 중요한 실무 포인트가 하나 있습니다.

**`prompt_format: canary2`만 config에 적는다고 해서, NeMo가 자동으로 당신의 locale token 규칙 `<|lang:en-US|>`를 알아서 만들어주지는 않습니다.**
그래서 아래 두 방식 중 하나가 필요합니다.

### 방식 A. 데이터 전처리 단계에서 `prompt`를 미리 만들어 manifest에 넣기

이게 가장 단순합니다.

예를 들어 manifest 생성 스크립트에서:

```python id="e2evrh"
from examples.asr.custom.custom_canary2_formatter import CustomCanary2PromptFormatter

formatter = CustomCanary2PromptFormatter()

sample = {
    "audio_filepath": "/data/audio/a.wav",
    "duration": 3.2,
    "text": "hello world",
    "source_lang": "en-US",
    "target_lang": "en-US",
    "taskname": "asr",
    "pnc": "yes",
}

sample["prompt"] = formatter.build_prompt(sample)
print(sample["prompt"])
```

이렇게 만들어서 JSONL에 `prompt` 필드를 추가합니다.

그다음 dataset/collate 쪽이 `prompt` 필드를 직접 쓰게 연결하면 됩니다.

### 방식 B. dataloader/collate 단계에서 formatter 호출

이건 더 “정석”이지만, NeMo 내부 dataset builder나 collate path를 조금 손봐야 할 수 있습니다.

당신이 지금 “빨리 돌리는 것”이 목적이면 **A가 더 현실적**입니다.

---

# 7) 지금 단계에서 가장 쉬운 운영 전략

당장은 이렇게 가세요.

1. **txt corpus로 unified tokenizer 생성**
2. **formatter로 prompt 문자열 생성**
3. **train/val JSONL에 `prompt` 필드까지 미리 넣음**
4. NeMo dataset이 `prompt` 필드를 읽게 연결
5. scratch training 시작

이 방식의 장점은, tokenizer와 formatter가 맞는지 **학습 전에 prompt 문자열을 눈으로 검증할 수 있다는 점**입니다.

예상 prompt:

```text id="xv8fbi"
<|startofcontext|><|startoftranscript|><|emo:undefined|><|task:asr|><|lang:en-US|><|lang:ko-KR|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
```

이 문자열이 tokenizer에 들어갔을 때 special token들이 atomic하게 잡혀야 합니다.

---

# 8) sanity check 코드

```python id="ybps9e"
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("/exp/tokenizers/canary2_unified_locale/tokenizer_spe_bpe_v16384/tokenizer.model")

prompt = "<|startofcontext|><|startoftranscript|><|emo:undefined|><|task:asr|><|lang:en-US|><|lang:ko-KR|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
pieces = sp.encode(prompt, out_type=str)

print(pieces)
```

여기서 `<|lang:en-US|>` 같은 토큰이 조각나지 않고 그대로 하나씩 보이면 정상입니다.

---

# 9) 한 줄 답

* `custom_canary2_formatter.py`는 **`examples/asr/custom/` 아래**에 두는 걸 추천합니다.
* tokenizer는 **txt 입력**으로 만들고,
* 학습 데이터는 **JSONL semantic field**를 유지하고,
* formatter는 `"en-US" -> "<|lang:en-US|>"` 매핑을 담당하게 두면 됩니다.


[1]: https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml "NeMo/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml at main · NVIDIA-NeMo/NeMo · GitHub"
[2]: https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py "NeMo/scripts/tokenizers/process_asr_text_tokenizer.py at main · NVIDIA-NeMo/NeMo · GitHub"


당신이 scratch로 **새 Canary2-style 모델 + 새 unified tokenizer**를 만드는 거라면, 중요한 건 **문자열 모양 자체**가 아니라 아래 두 가지가 일치하는 것입니다.

1. **formatter가 넣는 token 문자열**
2. **tokenizer에 special token으로 포함된 문자열**

즉, 당신이 원하면 그냥 이렇게 써도 됩니다.

```text
<|en-US|>
<|ko-KR|>
<|es-ES|>
<|asr|>
<|ast|>
<|pnc|>
<|nopnc|>
<|itn|>
<|noitn|>
<|timestamp|>
<|notimestamp|>
<|diarize|>
<|nodiarize|>
```

이 구조가 더 자연스럽다면 그걸 쓰면 됩니다.

---

## 그러면 어떻게 바꾸면 되나

### 1) token 파일 수정

`canary2_tokens.py`를 이렇게 바꾸면 됩니다.

```python
# examples/asr/custom/canary2_tokens.py

LANG_TOKENS = {
    "en-US": "<|en-US|>",
    "en-GB": "<|en-GB|>",
    "ko-KR": "<|ko-KR|>",
    "es-ES": "<|es-ES|>",
    "es-MX": "<|es-MX|>",
    "fr-FR": "<|fr-FR|>",
    "de-DE": "<|de-DE|>",
    "ja-JP": "<|ja-JP|>",
    "pt-BR": "<|pt-BR|>",
    "ar-SA": "<|ar-SA|>",
}

TASK_TOKENS = {
    "asr": "<|asr|>",
    "ast": "<|ast|>",
}

PNC_TOKENS = {
    "yes": "<|pnc|>",
    "no": "<|nopnc|>",
}

ITN_TOKENS = {
    "yes": "<|itn|>",
    "no": "<|noitn|>",
}

TIMESTAMP_TOKENS = {
    "yes": "<|timestamp|>",
    "no": "<|notimestamp|>",
}

DIARIZE_TOKENS = {
    "yes": "<|diarize|>",
    "no": "<|nodiarize|>",
}

ALL_SPECIAL_TOKENS = (
    [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|endoftext|>",
        "<|emo:undefined|>",
    ]
    + list(LANG_TOKENS.values())
    + list(TASK_TOKENS.values())
    + list(PNC_TOKENS.values())
    + list(ITN_TOKENS.values())
    + list(TIMESTAMP_TOKENS.values())
    + list(DIARIZE_TOKENS.values())
)
```

---

### 2) formatter는 그대로 써도 됨

`custom_canary2_formatter.py`는 사실 거의 안 바꿔도 됩니다.
왜냐하면 formatter는 그냥:

* `"en-US"`를 읽고
* `LANG_TOKENS["en-US"]`
* 즉 `"<|en-US|>"`

를 꺼내 쓰기만 하면 되기 때문입니다.

즉 이 부분은 그대로입니다.

```python
def _lang_token(self, lang: str) -> str:
    if lang not in LANG_TOKENS:
        raise ValueError(f"Unsupported language '{lang}'")
    return LANG_TOKENS[lang]
```

---

### 3) tokenizer 생성 command도 바꾸기

이전의 `<|lang:en-US|>`를 그냥 `<|en-US|>`로 바꾸면 됩니다.

```bash
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --data_file=/data/txt/en-US.txt,/data/txt/en-GB.txt,/data/txt/ko-KR.txt,/data/txt/es-ES.txt,/data/txt/es-MX.txt,/data/txt/fr-FR.txt,/data/txt/de-DE.txt,/data/txt/ja-JP.txt,/data/txt/pt-BR.txt,/data/txt/ar-SA.txt \
  --data_root=/exp/tokenizers/canary2_unified_locale \
  --vocab_size=16384 \
  --tokenizer=spe \
  --spe_type=bpe \
  --spe_character_coverage=0.9995 \
  --spe_user_defined_symbols \
    "<|startofcontext|>" "<|startoftranscript|>" "<|endoftext|>" "<|emo:undefined|>" \
    "<|en-US|>" "<|en-GB|>" "<|ko-KR|>" "<|es-ES|>" "<|es-MX|>" \
    "<|fr-FR|>" "<|de-DE|>" "<|ja-JP|>" "<|pt-BR|>" "<|ar-SA|>" \
    "<|asr|>" "<|ast|>" \
    "<|pnc|>" "<|nopnc|>" \
    "<|itn|>" "<|noitn|>" \
    "<|timestamp|>" "<|notimestamp|>" \
    "<|diarize|>" "<|nodiarize|>" \
  --log
```

---

## 최종 prompt는 이렇게 생김

예를 들어 sample이:

```json
{
  "source_lang": "en-US",
  "target_lang": "ko-KR",
  "taskname": "asr",
  "pnc": "yes"
}
```

이면 formatter 출력은:

```text
<|startofcontext|><|startoftranscript|><|emo:undefined|><|asr|><|en-US|><|ko-KR|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
```

이렇게 됩니다.

이게 훨씬 단순하고, 당신이 원하는 스타일과 맞습니다.

---

## 이 방식이 괜찮은 이유

당신은 scratch로 새 모델을 학습하는 중이므로,
**반드시 기존 Canary2의 literal token naming을 그대로 복제할 필요는 없습니다.**

중요한 건:

* tokenizer에 들어간 special token
* formatter가 만들어내는 prompt 문자열
* training/inference 모두 같은 규칙 사용

이 세 개가 일치하는 것입니다.

---

## 내가 추천하는 방향

당신 상황에서는 이게 제일 깔끔합니다.

* 언어 token: `<|en-US|>`, `<|ko-KR|>`, `<|es-ES|>`
* task token: `<|asr|>`, `<|ast|>`
* control token: `<|pnc|>`, `<|noitn|>`, `<|notimestamp|>`

즉 **최대한 짧고 직관적으로** 가는 게 좋습니다.


