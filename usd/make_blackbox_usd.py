#!/usr/bin/env python3
# ============================================================================
#  make_blackbox_usd.py
#  검은색 상자 USD 생성 — Isaac Sim 5.1
#
#  현실 크기: 가로 11 × 세로 7.5 × 높이 5 cm
#             = (0.11, 0.075, 0.05) m
#
#  성격: 기본 STATIC (고정 구조물 / 받침대 / 장애물)
#        -> 집어서 옮길 대상으로 쓰려면 아래 DYNAMIC = True 로 변경
#
#  배치: 부스 바닥 위. 바닥면 ≈ z=0.02, 상자 중심 z = 0.02 + 0.05/2
#
#  실행:
#    python make_blackbox_usd.py --out ./blackbox.usd
# ============================================================================

import argparse

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf

# ---- 상자 설정 ----
SIZE = (0.11, 0.075, 0.05)   # 가로 X, 세로 Y, 높이 Z (meter)
COLOR = (0.05, 0.05, 0.05)   # 검정 RGB 0~1
DYNAMIC = False              # True면 집을 수 있는 동적 강체, False면 고정
MASS = 0.10                  # DYNAMIC일 때만 사용 (100g, 실측 보정)

# 배치 위치 (부스 바닥 위, 약간 뒤쪽)
FLOOR_TOP = 0.02
POS = (0.0, -0.05, FLOOR_TOP + SIZE[2] / 2)  # 중심 좌표


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="출력 blackbox.usd 경로")
    args = ap.parse_args()

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    prim_path = "/World/BlackBox"
    box = UsdGeom.Cube.Define(stage, prim_path)
    box.GetSizeAttr().Set(2.0)  # 기본 한 변 2.0 -> scale로 맞춤
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*POS))
    xf.AddScaleOp().Set(Gf.Vec3f(SIZE[0] / 2.0, SIZE[1] / 2.0, SIZE[2] / 2.0))

    # 충돌체는 항상
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    if DYNAMIC:
        UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateMassAttr(MASS)
        kind = f"dynamic (mass={MASS*1000:.0f}g)"
    else:
        kind = "static (고정)"

    box.CreateDisplayColorAttr([Gf.Vec3f(*COLOR)])

    stage.GetRootLayer().Export(args.out)
    print(f"[box] BlackBox  size=(11, 7.5, 5)cm  pos={POS}  {kind}")
    print(f"[done] blackbox USD written: {args.out}")
    print("  - 동적으로 바꾸려면 스크립트 상단 DYNAMIC=True")
    simulation_app.close()


if __name__ == "__main__":
    main()
