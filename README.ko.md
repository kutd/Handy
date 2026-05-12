# Handy Qwen3 MLX 한국어 ASR 포크

이 저장소는 원본 [cjpais/Handy](https://github.com/cjpais/Handy)를 기반으로, 한국어 받아쓰기 성능과 속도를 Apple Silicon Mac에서 실험하기 위한 포크입니다.

> 현재 이 포크의 Qwen3 MLX 경로는 **macOS Apple Silicon 전용**입니다. 원본 Handy는 Windows, macOS, Linux를 지원하지만, 이 포크에서 추가한 Qwen3 ASR MLX 기능은 Apple MLX 런타임을 사용하므로 M 시리즈 Mac을 대상으로 합니다.

English README: [README.md](README.md)

## 이 포크의 변경점

- **Qwen3 ASR 0.6B 8-bit MLX** 모델 옵션 추가: `qwen3-mlx-0.6b-8bit`
- **Qwen3 ASR 1.7B 4-bit MLX** 모델 옵션 추가: `qwen3-mlx-1.7b-4bit`
- 0.6B는 속도 우선, 1.7B는 정확도 우선 선택지로 사용
- Handy의 기존 VAD, 녹음, 단축키, 붙여넣기 흐름은 유지
- 녹음이 끝난 오디오를 Qwen3 ASR MLX worker로 전달
- Python worker를 상시 실행해 `mlx-qwen3-asr`의 `Session`을 재사용
- 매번 모델을 다시 불러오지 않아 짧은 한국어 받아쓰기 응답 속도를 줄이는 구조
- Qwen3 언어 힌트는 한국어로 지정하고, Handy 사용자 지정 단어만 선택적으로 Qwen3 context로 전달
- 기존 Whisper, Parakeet, SenseVoice, GigaAM, Canary, Cohere 경로는 그대로 유지

## 릴리즈

이 포크의 macOS Apple Silicon 빌드는 [Releases](https://github.com/kutd/Handy/releases)에서 받을 수 있습니다.

주의할 점:

- Qwen3 MLX 모델 파일은 릴리즈에 포함되어 있지 않습니다.
- Python 환경도 포함되어 있지 않습니다.
- 로컬에서 `mlx_qwen3_asr`를 import할 수 있는 Python을 준비해야 합니다.
- 현재 빌드는 실험용이며 Apple 공증을 거친 배포본이 아닙니다.

## Qwen3 MLX 모델 준비

Qwen3 MLX 모델 파일은 Handy 앱 안에서 다운로드할 수 있습니다. 모델 파일은 공개 GitHub Release 자산으로 호스팅됩니다.

- [qwen3-asr-0.6b-mlx-q8-g64.tar.gz](https://github.com/kutd/Handy/releases/download/v0.8.3-qwen3-mlx-ko.4/qwen3-asr-0.6b-mlx-q8-g64.tar.gz)
- [qwen3-asr-1.7b-mlx-q4-g64.tar.gz](https://github.com/kutd/Handy/releases/download/v0.8.3-qwen3-mlx-ko.4/qwen3-asr-1.7b-mlx-q4-g64.tar.gz)

Handy는 `uv`를 함께 포함하며, 첫 사용 시 Qwen3 MLX 전용 Python 런타임을 자동으로 만듭니다. 이 런타임에는 `mlx-qwen3-asr==0.3.3`이 설치되며 위치는 다음과 같습니다.

```text
~/Library/Application Support/com.pais.handy/models/.qwen3-mlx-runtime
```

처음 Qwen3 모델을 사용할 때는 런타임 생성 때문에 시간이 더 걸릴 수 있습니다. 최초 생성에는 인터넷 연결이 필요합니다.

Qwen3-ASR 모델 계열은 Hugging Face에서 Apache License 2.0 메타데이터로 공개되어 있습니다. 이 포크의 모델 tarball에는 출처와 MLX/양자화 재패키징 내용을 기록한 `LICENSE`와 `NOTICE` 파일을 포함했습니다.

직접 준비한 Python 환경을 쓰고 싶다면, `mlx_qwen3_asr`를 import할 수 있는 Python 실행 파일을 `HANDY_QWEN3_MLX_PYTHON` 환경 변수로 지정하면 됩니다.

Handy의 모델 폴더 안에 Qwen3 MLX 모델 디렉터리를 아래 이름으로 배치합니다.

```text
qwen3-asr-0.6b-mlx-q8-g64
qwen3-asr-1.7b-mlx-q4-g64
```

1.7B 4bit 디렉터리는 `Qwen/Qwen3-ASR-1.7B`를 `mlx-qwen3-asr`와 호환되는 4bit, group-size-64 형식으로 변환한 모델을 기준으로 합니다.

macOS에서 일반적인 위치는 다음과 같습니다.

```text
~/Library/Application Support/com.pais.handy/models/qwen3-asr-0.6b-mlx-q8-g64
~/Library/Application Support/com.pais.handy/models/qwen3-asr-1.7b-mlx-q4-g64
```

수동으로 Python 경로를 지정하려면 방법은 둘 중 하나입니다.

환경 변수 사용:

```bash
HANDY_QWEN3_MLX_PYTHON=/path/to/python
```

또는 모델 디렉터리 안에 `.handy-python` 파일을 만들고, 그 안에 Python 실행 파일 경로를 한 줄로 적습니다.

```text
/path/to/python
```

## 사용 방법

1. 이 포크의 [Releases](https://github.com/kutd/Handy/releases)에서 macOS Apple Silicon용 DMG를 받습니다.
2. 앱을 설치하고 실행합니다.
3. macOS에서 마이크와 손쉬운 사용 권한을 허용합니다.
4. Qwen3 MLX 모델 디렉터리와 Python 경로를 준비합니다.
5. Handy 설정에서 `Qwen3 ASR 0.6B 8-bit MLX` 또는 `Qwen3 ASR 1.7B 4-bit MLX` 모델과 한국어를 선택합니다.
6. 단축키를 눌러 한국어 받아쓰기를 테스트합니다.

## 원본 Handy 소개

Handy는 완전히 로컬에서 동작하는 무료 오픈소스 음성 인식 데스크톱 앱입니다. 단축키를 누르고 말하면, 인식된 텍스트가 현재 사용 중인 입력창에 붙여넣어집니다. 오디오는 클라우드로 전송되지 않습니다.

원본 프로젝트:

- 저장소: [cjpais/Handy](https://github.com/cjpais/Handy)
- 웹사이트: [handy.computer](https://handy.computer)

## 개발 및 빌드

자세한 빌드 방법은 원본 문서 [BUILD.md](BUILD.md)를 참고하세요.

이 포크에서 로컬 실행용 macOS 앱 번들을 만들 때는 업데이터 서명 산출물을 끄고 빌드할 수 있습니다.

```bash
bun run tauri build --bundles app --config '{"bundle":{"createUpdaterArtifacts":false}}'
```

## 라이선스

원본 Handy의 라이선스를 따릅니다.
