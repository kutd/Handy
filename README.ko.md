# Handy Qwen3 MLX 한국어 ASR 포크

이 저장소는 원본 [cjpais/Handy](https://github.com/cjpais/Handy)를 기반으로, 한국어 받아쓰기 성능과 속도를 Apple Silicon Mac에서 실험하기 위한 포크입니다.

> 현재 이 포크의 Qwen3 MLX 경로는 **macOS Apple Silicon 전용**입니다. 원본 Handy는 Windows, macOS, Linux를 지원하지만, 이 포크에서 추가한 Qwen3 ASR 0.6B 8-bit MLX 기능은 Apple MLX 런타임을 사용하므로 M 시리즈 Mac을 대상으로 합니다.

English README: [README.md](README.md)

## 이 포크의 변경점

- **Qwen3 ASR 0.6B 8-bit MLX** 모델 옵션 추가: `qwen3-mlx-0.6b-8bit`
- Handy의 기존 VAD, 녹음, 단축키, 붙여넣기 흐름은 유지
- 녹음이 끝난 오디오를 Qwen3 ASR MLX worker로 전달
- Python worker를 상시 실행해 `mlx-qwen3-asr`의 `Session`을 재사용
- 매번 모델을 다시 불러오지 않아 짧은 한국어 받아쓰기 응답 속도를 줄이는 구조
- 한국어 언어 힌트와 Handy 사용자 지정 단어를 Qwen3 context로 전달
- 기존 Whisper, Parakeet, SenseVoice, GigaAM, Canary, Cohere 경로는 그대로 유지

## 릴리즈

이 포크의 macOS Apple Silicon 빌드는 [Releases](https://github.com/kutd/Handy/releases)에서 받을 수 있습니다.

주의할 점:

- Qwen3 MLX 모델 파일은 릴리즈에 포함되어 있지 않습니다.
- Python 환경도 포함되어 있지 않습니다.
- 로컬에서 `mlx_qwen3_asr`를 import할 수 있는 Python을 준비해야 합니다.
- 현재 빌드는 실험용이며 Apple 공증을 거친 배포본이 아닙니다.

## Qwen3 MLX 모델 준비

Handy의 모델 폴더 안에 Qwen3 MLX 모델 디렉터리를 아래 이름으로 배치합니다.

```text
qwen3-asr-0.6b-mlx-q8-g64
```

macOS에서 일반적인 위치는 다음과 같습니다.

```text
~/Library/Application Support/com.pais.handy/models/qwen3-asr-0.6b-mlx-q8-g64
```

그 다음 Handy가 `mlx_qwen3_asr`를 import할 수 있는 Python 실행 파일을 찾을 수 있게 해야 합니다. 방법은 둘 중 하나입니다.

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
5. Handy 설정에서 `Qwen3 ASR 0.6B 8-bit MLX` 모델과 한국어를 선택합니다.
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
