#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_data/trueDepth_XXXX -> data/trueDepth_1 NeRF 风格数据集转换脚本

python convert_dataset.py \
  --input_dir source_data/trueDepth_20251127_1 \
  --output_dir data/trueDepth_1 \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

功能：
- 读取 pose_i.txt / intrinsics_i.txt / color_i.png / depth_i.png
- 生成 train / val / test 目录与图片命名
- 估计 normal 图
- 生成 transforms_train.json / transforms_val.json / transforms_test.json
"""

import os
import math
import glob
import json
import argparse

import numpy as np
from PIL import Image


def parse_pose(pose_path):
    """从 pose_i.txt 解析 px,py,pz,qx,qy,qz,qw"""
    vals = {}
    with open(pose_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals[k.strip()] = float(v.strip())
    px = vals.get("px", 0.0)
    py = vals.get("py", 0.0)
    pz = vals.get("pz", 0.0)
    qx = vals.get("qx", 0.0)
    qy = vals.get("qy", 0.0)
    qz = vals.get("qz", 0.0)
    qw = vals.get("qw", 1.0)
    return (px, py, pz, qx, qy, qz, qw)


def quat_to_rot_matrix(qx, qy, qz, qw):
    """四元数 -> 3x3 旋转矩阵，假设 (qx,qy,qz,qw)，右手系"""
    # 归一化
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def build_transform_matrix(pose_path):
    """pose_i.txt -> 4x4 camera-to-world transform matrix"""
    px, py, pz, qx, qy, qz, qw = parse_pose(pose_path)
    R = quat_to_rot_matrix(qx, qy, qz, qw)
    t = np.array([px, py, pz], dtype=np.float32).reshape(3, 1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def parse_intrinsics(intr_path):
    """intrinsics_i.txt -> fx, fy, cx, cy"""
    vals = {}
    with open(intr_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals[k.strip()] = float(v.strip())
    fx = vals.get("fx", 1.0)
    fy = vals.get("fy", 1.0)
    cx = vals.get("cx", 0.0)
    cy = vals.get("cy", 0.0)
    return fx, fy, cx, cy


def compute_camera_angle_x(sample_color_path, sample_intr_path):
    """用一张彩色图和 intrinsics 估计水平视场角 camera_angle_x"""
    fx, _, _, _ = parse_intrinsics(sample_intr_path)
    with Image.open(sample_color_path) as img:
        w, _ = img.size
    # fov_x = 2 * atan( W / (2 * fx) )
    fov_x = 2.0 * math.atan(w / (2.0 * fx))
    return float(fov_x)


def depth_to_normal(depth_path, intr_path, out_path):
    """
    用 depth + intrinsics 估计 normal 图，保存为 RGB PNG
    这是一个简单近似：通过相邻像素的 3D 点做叉积求法线。
    """
    fx, fy, cx, cy = parse_intrinsics(intr_path)

    depth_img = Image.open(depth_path)
    depth = np.array(depth_img).astype(np.float32)

    # 处理单通道 depth
    if depth.ndim == 3:
        depth = depth[..., 0]

    h, w = depth.shape

    # 构建每个像素的 3D 点（相机坐标系）
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth
    # 避免除以 0
    z_safe = np.where(z <= 0, np.nan, z)

    X = (uu - cx) * z_safe / fx
    Y = (vv - cy) * z_safe / fy
    Z = z_safe

    pts = np.stack([X, Y, Z], axis=-1)

    # 邻域向量：右 & 下
    pts_right = np.roll(pts, -1, axis=1)
    pts_down = np.roll(pts, -1, axis=0)

    v1 = pts_right - pts
    v2 = pts_down - pts

    # 叉积得到法线
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-8, norm)
    n = n / norm

    # 处理无效深度（z<=0），强制设为(0,0,1)
    invalid = np.isnan(Z) | (Z <= 0)
    n[invalid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 方向统一：朝向相机或远离相机，任选一个，这里取远离相机
    n = -n

    # [-1,1] -> [0,255]
    n_img = ((n + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    normal_img = Image.fromarray(n_img, mode="RGB")
    normal_img.save(out_path)


def collect_indices(input_dir):
    """根据 pose_*.txt 收集所有帧索引"""
    pose_files = glob.glob(os.path.join(input_dir, "pose_*.txt"))
    indices = []
    for p in pose_files:
        base = os.path.basename(p)
        # 形如 pose_123.txt
        try:
            idx_str = base.split("_")[1].split(".")[0]
            idx = int(idx_str)
            indices.append(idx)
        except Exception:
            continue
    indices = sorted(set(indices))
    return indices


def split_indices(indices, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """按比例切分 train/val/test（自动归一化一下比例）"""
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Ratios must be positive.")
    # 归一化到和为 1
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # 把余数都丢给 test，保证总数不丢

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_split(split_name, indices, input_dir, out_root, camera_angle_x):
    """
    对一个 split(train/val/test)：
    - 拷贝并改名 color/depth -> r_i.* 系列
    - 生成 normal
    - 记录 frames 列表
    """
    out_dir = os.path.join(out_root, split_name)
    ensure_dir(out_dir)

    frames = []
    rot_const = 0.012566370614359171  # 和你给的例子保持一致

    for i in indices:
        pose_path = os.path.join(input_dir, f"pose_{i}.txt")
        intr_path = os.path.join(input_dir, f"intrinsics_{i}.txt")
        color_path = os.path.join(input_dir, f"color_{i}.png")
        depth_path = os.path.join(input_dir, f"depth_{i}.png")

        if not os.path.exists(pose_path):
            print(f"[WARN] pose file not found: {pose_path}, skip")
            continue
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            print(f"[WARN] color/depth missing for index {i}, skip")
            continue
        if not os.path.exists(intr_path):
            print(f"[WARN] intrinsics missing for index {i}, skip")
            continue

        # 生成 transform_matrix
        T = build_transform_matrix(pose_path)
        T_list = T.tolist()

        # 拷贝 & 改名图片
        # 彩图：r_i.png
        out_color = os.path.join(out_dir, f"r_{i}.png")
        Image.open(color_path).save(out_color)

        # 深度：r_i_depth_0001.png
        out_depth = os.path.join(out_dir, f"r_{i}_depth_0001.png")
        Image.open(depth_path).save(out_depth)

        # normal：r_i_normal_0001.png
        out_normal = os.path.join(out_dir, f"r_{i}_normal_0001.png")
        depth_to_normal(depth_path, intr_path, out_normal)

        # frames 条目里的 file_path 相对 transforms_*.json 的位置
        frame = {
            "file_path": f"./{split_name}/r_{i}",
            "rotation": rot_const,
            "transform_matrix": T_list,
        }
        frames.append(frame)

    # 写出 transforms_split.json
    out_json = os.path.join(out_root, f"transforms_{split_name}.json")
    data = {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] wrote {out_json} with {len(frames)} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Convert trueDepth_XXXX dataset to NeRF-style structure"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="原始 trueDepth_XXXX 文件夹路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 trueDepth_1 文件夹路径",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    args = parser.parse_args()

    input_dir = args.input_dir
    out_root = args.output_dir
    ensure_dir(out_root)

    indices = collect_indices(input_dir)
    if not indices:
        print("[ERROR] no pose_*.txt found. Check input_dir.")
        return

    print(f"[INFO] found {len(indices)} frames, indices: {indices[0]} ~ {indices[-1]}")

    # 用第一帧估算 camera_angle_x
    sample_idx = indices[0]
    sample_color = os.path.join(input_dir, f"color_{sample_idx}.png")
    sample_intr = os.path.join(input_dir, f"intrinsics_{sample_idx}.txt")
    if not (os.path.exists(sample_color) and os.path.exists(sample_intr)):
        print("[ERROR] cannot find sample color/intrinsics to compute camera_angle_x")
        return
    camera_angle_x = compute_camera_angle_x(sample_color, sample_intr)
    print(f"[INFO] camera_angle_x = {camera_angle_x:.6f} rad")

    train_idx, val_idx, test_idx = split_indices(
        indices, args.train_ratio, args.val_ratio, args.test_ratio
    )

    print(
        f"[INFO] split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    process_split("train", train_idx, input_dir, out_root, camera_angle_x)
    process_split("val", val_idx, input_dir, out_root, camera_angle_x)
    process_split("test", test_idx, input_dir, out_root, camera_angle_x)

    print("[DONE] dataset conversion finished.")


if __name__ == "__main__":
    main()
