#!/usr/bin/env python3
# ============================================================================
#  make_booth_usd.py
#  LeKiwi 작업 부스(ㄷ자) USD 생성 — Isaac Sim 5.1
#
#  형상 (현실 치수 기준):
#    바닥 60(가로 X) × 55(세로 Y) cm
#    벽 높이 40 cm, 벽 두께 2 cm
#    천장 없음 / 정면(+Y) 트임 → 뒷벽(−Y) + 좌벽(−X) + 우벽(+X)
#    원점 = 바닥 중심 (0,0,0), Z-up
#
#  각 구성요소: 시각 메쉬(Cube) + static collider (RigidBody 없음 = 고정)
#
#  실행:
#    ./python.sh make_booth_usd.py --out /path/to/booth.usd
#  (Isaac Sim GUI에서 단독으로 열거나, 로봇 씬에 reference로 add)
# ============================================================================

import argparse

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# ---- 치수 (meter) ----
W = 0.60    # 가로 X
D = 0.55    # 세로 Y
H = 0.40    # 벽 높이 Z
T = 0.02    # 벽/바닥 두께

# (이름, size(x,y,z), center pos(x,y,z))  — 앞서 검증한 좌표
PARTS = [
    ("floor", (W,     D,     T), (0.0,            0.0,          T / 2)),
    ("wall_back",  (W,     T,     H), (0.0,           -D / 2 + T / 2, H / 2)),
    ("wall_left",  (T,     D,     H), (-W / 2 + T / 2, 0.0,          H / 2)),
    ("wall_right", (T,     D,     H), (W / 2 - T / 2,  0.0,          H / 2)),
]

# 색 (RGB 0~1) — 전부 흰색
COLORS = {
    "floor":      (1.0, 1.0, 1.0),
    "wall_back":  (1.0, 1.0, 1.0),
    "wall_left":  (1.0, 1.0, 1.0),
    "wall_right": (1.0, 1.0, 1.0),
}


def add_box(stage, parent_path, name, size, pos, color):
    """Cube prim 생성 + scale + translate + static collider + 색."""
    prim_path = f"{parent_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, prim_path)
    # UsdGeom.Cube 는 기본 한 변 2.0 -> size 맞추려면 scale = size/2
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))

    # static collider (RigidBodyAPI 미적용 => 고정된 충돌체)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # 색상
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube.GetPrim()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="출력 booth.usd 경로")
    args = ap.parse_args()

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 부스 루트 Xform (나중에 통째로 옮기거나 reference 하기 쉬움)
    root_path = "/World/Booth"
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    for name, size, pos in PARTS:
        add_box(stage, root_path, name, size, pos, COLORS[name])
        print(f"[booth] {name:11s} size={size} pos={pos}")

    stage.GetRootLayer().Export(args.out)
    print(f"\n[done] booth USD written: {args.out}")
    print("  - 원점 = 바닥 중심, 정면(+Y) 트임, 천장 없음")
    print("  - 벽/바닥은 static collider (고정). RigidBody 없음.")
    simulation_app.close()


if __name__ == "__main__":
    main()
