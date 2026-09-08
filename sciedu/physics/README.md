# physics

Isaac Sim 물리 실험 씬. 각 폴더는 **자체 포함(self-contained)** 패키지라 클론 후 바로 열립니다.

| 씬 | 열 파일 | 내용 |
|---|---|---|
| physics01 | `physics01/physics01.usd` | 다리 · 카트 · 스프링 · 추 |
| physics02 | `physics02/physics02.usd` | 병 · 접시 · 테이프 |

두 씬 모두 OpenArm 양팔 로봇(`0/moveit_openarm.usd`)과 화학실험실 환경을 포함합니다.

## 여는 법

Isaac Sim 에서 File ▸ Open 으로 위 `.usd` 파일을 선택하면 됩니다.
숫자 폴더(`0/`, `1/` …)는 참조 에셋이므로 옮기거나 이름을 바꾸지 마세요.

## 참고

- 모든 참조는 패키지 내부 **상대경로**입니다. 다른 PC 로 통째로 복사해도 동작합니다.
- `OmniPBR.mdl` 등 MDL 머티리얼은 Isaac Sim 설치본에서 해석되므로 포함되어 있지 않습니다.
