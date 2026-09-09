# physics

Isaac Sim 물리 실험 씬. 각 씬 폴더는 **자체 포함(self-contained)** 이라 클론 후 바로 열립니다.

## 씬

| 씬 | OpenArm v1.0 (권장) | OpenArm v2.0 (원본) |
|---|---|---|
| physics01 — 다리·카트·스프링·추 | `physics01/physics01_v1.usda` | `physics01/physics01.usd` |
| physics02 — 병·접시·테이프 | `physics02/physics02_v1.usda` | `physics02/physics02.usd` |

Isaac Sim 에서 File ▸ Open 으로 위 파일을 선택하면 됩니다.
숫자 폴더(`0/`, `1/` …)와 `openarm_v1.0/` 은 참조 에셋이므로 옮기거나 이름을 바꾸지 마세요.

## OpenArm v1.0 전환

`*_v1.usda` 는 원본 씬을 참조하는 얇은 합성 레이어입니다. 원본 `.usd` 는 수정하지 않으므로
v2.0 씬도 그대로 남아 있습니다.

```
physics0N_v1.usda
├── references physics0N.usd </World>     배경·테이블·물체·조명·physicsScene 그대로
├── over "openarm_v20" { active = false } 내장 v2.0 articulation 비활성화
└── def "openarm_v1"                      공식 v1.0 bimanual 을 같은 자리에 배치
        references ../openarm_v1.0/openarm_bimanual.usd
        translate (-0.0072, 0, 0.3448)
```

두 모델 모두 어깨가 base 기준 `(0, ±0.0935, 0.698)` 에 있어, 같은 transform 이면
어깨 world 위치가 **오차 0 mm** 로 겹칩니다.

| | v2.0 | v1.0 |
|---|---|---|
| 팔 관절 | 좌우 7 DoF | 좌우 7 DoF |
| 그리퍼 | revolute mimic, `±0.75 rad` | **prismatic, `0.0–0.044 m`** |
| EE body | `openarm_*_ee_base_link` | `openarm_*_hand`, TCP `openarm_*_ee_tcp` |
| base body | `openarm_body_link0` | `openarm_body_link` |
| articulation root | `/World/openarm_v20/root_joint` | `/World/openarm_v1/root_joint` |

그리퍼 명령 범위가 바뀌므로, v2.0 으로 수집한 에피소드와 v1.0 데이터를 섞지 마세요.

## openarm_v1.0/

Isaac Sim 5.1 공식 OpenArm bimanual 에셋입니다. 출처와 파일별 SHA-256 은
`openarm_v1.0/OPENARM_V1_ASSET_MANIFEST.json` 에 있습니다.

## 참고

- 모든 참조는 상대경로입니다. 통째로 복사해도 동작합니다.
- `OmniPBR.mdl` 등 MDL 머티리얼은 Isaac Sim 설치본에서 해석되므로 포함되어 있지 않습니다.
