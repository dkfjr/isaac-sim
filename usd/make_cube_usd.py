#!/usr/bin/env python3
# ============================================================================
#  make_cube_usd.py
#  빨간 큐브(파지 대상) USD 생성 — Isaac Sim 5.1
#
#  현실 크기: 4 × 4 × 4 cm = 0.04 m 정육면체
#  성격: 동적 강체(RigidBody) — 집어서 옮길 대상
#        부스 벽(static)과 달리 중력 받고 움직임
#
#  배치: 부스 바닥 위 (바닥면 ≈ z=0.02). 큐브 중심 z = 0.02 + 0.02 = 0.04
#        부스 안 적당히 앞쪽(+Y)으로 둠 -> 그리퍼 접근 쉬움
#
#  실행:
#    python make_cube_usd.py --out ./cube.usd
# ============================================================================

import argparse

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf

# ---- 큐브 설정 ----
SIZE = 0.04                 # 한 변 4cm (meter)
COLOR = (0.85, 0.12, 0.12)  # 빨강 RGB 0~1
MASS = 0.05                 # 50g (가벼운 플라스틱/나무 블록 가정, 실측으로 보정)

# 배치 위치 (부스 바닥 위, 약간 앞쪽)
FLOOR_TOP = 0.02            # 부스 바닥면 높이
POS = (0.0, 0.10, FLOOR_TOP + SIZE / 2)  # 중심 좌표


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="출력 cube.usd 경로")
    args = ap.parse_args()

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    prim_path = "/World/RedCube"
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(2.0)  # 기본 한 변 2.0 -> scale로 맞춤
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*POS))
    xf.AddScaleOp().Set(Gf.Vec3f(SIZE / 2.0, SIZE / 2.0, SIZE / 2.0))

    # 동적 강체 (집어서 옮길 대상)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr(MASS)

    # 색
    cube.CreateDisplayColorAttr([Gf.Vec3f(*COLOR)])

    stage.GetRootLayer().Export(args.out)
    print(f"[cube] RedCube  size={SIZE*100:.0f}cm  pos={POS}  mass={MASS*1000:.0f}g")
    print(f"[done] cube USD written: {args.out}")
    print("  - 동적 강체(RigidBody): 중력 받고 그리퍼로 집을 수 있음")
    print("  - 부스 바닥 위에 배치됨 (필요시 POS 변수로 위치 조정)")
    simulation_app.close()


if __name__ == "__main__":
    main()
