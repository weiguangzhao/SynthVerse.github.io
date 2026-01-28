import os
import glob
import math
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open3d as o3d
import tqdm
from datetime import datetime


# -------------------------
# IO
# -------------------------
def read_sample(path: str):
    return np.load(path, allow_pickle=True).item()

def list_frames(frames_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    ps = []
    for e in exts:
        ps += glob.glob(os.path.join(frames_dir, e))
    # sort out filenames with _uv and _depth
    ps = [p for p in sorted(ps) if not (("_uv." in p) or ("_depth." in p) or ("_sem." in p))]
    print(len(ps), "frames found in", frames_dir)
    if len(ps) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    return ps

def get_hw_from_first_frame(frames_dir: str):
    fp = list_frames(frames_dir)[0]
    w, h = Image.open(fp).size
    return h, w

def linspace_indices(T: int, k: int):
    if k <= 1:
        return np.array([0], dtype=int)
    xs = np.linspace(0, T - 1, k)
    return np.unique(np.round(xs).astype(int))


# -------------------------
# Runtime logging helper
# -------------------------
LOG_CALLS_PATH = os.path.join(os.path.dirname(__file__), "./vis_out/function_calls_log.txt")

def _serialize_value(v):
    try:
        if isinstance(v, np.ndarray):
            return f"ndarray dtype={v.dtype} shape={v.shape}"
        if isinstance(v, (list, tuple)):
            return f"{type(v).__name__} len={len(v)}"
        if isinstance(v, dict):
            return f"dict keys={list(v.keys())}"
        return repr(v)
    except Exception:
        return str(type(v))

def log_function_call(func_name: str, params: dict):
    try:
        with open(LOG_CALLS_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} - {func_name}\n")
            for k, v in params.items():
                f.write(f"  {k}: {_serialize_value(v)}\n")
            f.write("\n")
    except Exception:
        # Don't let logging break the main program
        pass


# -------------------------
# Pose utilities
# -------------------------
def orthogonalize_rotation_matrix(R):
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        R_orth[:, -1] *= -1
    return R_orth

def cam_to_world_T(matrix_world_4x4: np.ndarray, do_orthogonalize=True):
    """matrix_world: Blender camera matrix_world (cam->world). Optionally orthogonalize rotation."""
    T = matrix_world_4x4.copy()
    if do_orthogonalize:
        R = T[:3, :3]
        t = T[:3, 3]
        R = orthogonalize_rotation_matrix(R)
        T[:3, :3] = R
        T[:3, 3] = t
    return T

def apply_T(points_cam: np.ndarray, T_cam2world: np.ndarray):
    if points_cam.size == 0:
        return points_cam
    ones = np.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)
    homog = np.concatenate([points_cam, ones], axis=1)  # [N,4]
    out = (T_cam2world @ homog.T).T
    return out[:, :3]


# -------------------------
# 2D visualization
# -------------------------
def visualize_tracking_2d(
    frames_dir: str,
    sample_npy_path: str,
    out_path: str,
    num_sample_frames: int = 6,
    point_size: float = 8.0,
    line_width: float = 1.5,
    dpi: int = 150,
    first_frame_visible_only: bool = True,
    max_points: int = 16,
    last_frame=None,
    skip_occluded_points: bool = False,  # 是否跳过任何帧中occluded的点（仅traj_lim=None时有效）
    traj_lim: int = None,  # None=连续轨迹, 0=只画点, >0=每个关键帧画前traj_lim帧轨迹
    point_indices: np.ndarray = None,  # 预先计算好的点索引，如果提供则直接使用
    point_colors: np.ndarray = None,  # 预先计算好的点颜色 [N, 3]，确保多帧之间颜色一致
):
    """
    2D：将采样帧的RGB图片从左到右排列，tracking点覆盖在RGB上。
    
    traj_lim 控制绘制模式：
    - None: 绘制关键帧之间的连续轨迹（需要padding，判断全局可见性）
    - 0: 只绘制关键帧处的点（不绘制轨迹，为每个关键帧单独检查occluded）
    - >0: 为每个关键帧绘制其前traj_lim帧的轨迹（只判断局部可见性）
    
    coords: [T,N,2], occluded: [T,N] (True=遮挡)
    first_frame_visible_only: 只显示第一帧未被遮挡的点
    skip_occluded_points: 如果为True且traj_lim=None，跳过任何帧中被遮挡的点
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = read_sample(sample_npy_path)
    coords = data["coords"].astype(np.float32)          # [T,N,2]
    occluded = data["occluded"].astype(bool)            # [T,N]
    T, N, _ = coords.shape

    if last_frame is not None:
        T = last_frame

    # 只选择均匀采样的帧用于显示RGB
    # 特殊情况：num_sample_frames==1时，生成单帧
    if num_sample_frames == 1:
        if last_frame is not None:
            sample_idx = np.array([last_frame - 1])  # last_frame是帧数，索引需要-1
        else:
            sample_idx = np.array([0])
    else:
        sample_idx = linspace_indices(T, num_sample_frames)
    num_frames = len(sample_idx)
    
    # 如果提供了 point_indices，直接使用
    if point_indices is not None:
        valid_point_indices = point_indices
    else:
        # 过滤：只保留第一帧未被遮挡的点
        if first_frame_visible_only:
            first_frame_visible = ~occluded[0]  # [N]
            valid_point_indices = np.where(first_frame_visible)[0]
        else:
            valid_point_indices = np.arange(N)
        
        # 过滤：跳过任何帧中被遮挡的点（只保留始终可见的点）
        if skip_occluded_points:
            always_visible = ~occluded[:T].any(axis=0)  # [N], True表示该点在所有帧都未被遮挡
            always_visible_indices = np.where(always_visible)[0]
            valid_point_indices = np.intersect1d(valid_point_indices, always_visible_indices)
            print(f"After skip_occluded_points filter: {len(valid_point_indices)} points always visible")
        
        # 如果点数超过max_points，随机采样
        if len(valid_point_indices) > max_points:
            valid_point_indices = np.random.choice(valid_point_indices, max_points, replace=False)
            valid_point_indices = np.sort(valid_point_indices)
    
    print(f"2D Visualization: {len(valid_point_indices)}/{N} points")
    print(f"Using {num_frames} RGB frames: {sample_idx.tolist()}")
    print(f"Drawing trajectories using all {T} frames")

    # 读取采样帧的RGB图片
    all_frame_paths = list_frames(frames_dir)
    frame_images = []
    for fidx in sample_idx:
        img = Image.open(all_frame_paths[fidx]).convert("RGB")  # 统一转为RGB
        frame_images.append(np.array(img))
    
    H, W = frame_images[0].shape[:2]
    
    # 水平拼接所有帧
    concat_img = np.concatenate(frame_images, axis=1)  # [H, W*num_frames, 3]
    total_width = W * num_frames

    # 根据 traj_lim 决定是否进行 padding
    # 只有 traj_lim=None（连续轨迹模式）且不跳过occluded点时才需要padding
    if traj_lim is None and not skip_occluded_points:
        # 连续轨迹模式：计算所有valid点在所有帧的y坐标范围，确定是否需要扩展画布
        all_y_coords = []
        for j in valid_point_indices:
            xy_all = coords[:, j, :]  # [T, 2]
            valid_mask = np.isfinite(xy_all).all(axis=1)
            if valid_mask.any():
                all_y_coords.extend(xy_all[valid_mask, 1].tolist())
        
        if len(all_y_coords) > 0:
            y_min = min(all_y_coords)
            y_max = max(all_y_coords)
        else:
            y_min, y_max = 0, H
        
        # 计算需要的padding（上下留白）
        pad_top = max(0, -y_min + 20)
        pad_bottom = max(0, y_max - H + 20)
        
        # 扩展画布
        canvas_height = int(H + pad_top + pad_bottom)
        canvas = np.ones((canvas_height, total_width, 3), dtype=np.uint8) * 255
        
        # 将拼接的RGB图片放到画布中间
        y_offset = int(pad_top)
        canvas[y_offset:y_offset + H, :, :] = concat_img
    else:
        # traj_lim=0 或 traj_lim>0 或 skip_occluded_points=True 时不需要padding
        pad_top = 0
        pad_bottom = 0
        y_offset = 0
        canvas_height = H
        canvas = concat_img.copy()
    
    print(f"Canvas size: {total_width}x{canvas_height} (pad_top={pad_top:.0f}, pad_bottom={pad_bottom:.0f})")

    # 为每个点生成一个随机颜色（用于区分不同的tracking点）
    # 如果外部提供了 point_colors，则使用外部的，确保多帧之间颜色一致
    if point_colors is None:
        point_colors = np.random.rand(N, 3)

    fig = plt.figure(figsize=(total_width / dpi, canvas_height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_xlim(0, total_width)
    ax.set_ylim(canvas_height, 0)
    ax.axis("off")

    tracking_bar = tqdm.tqdm(valid_point_indices, desc="Drawing 2D tracking")
    for j in tracking_bar:
        # 使用所有帧的坐标来画轨迹（形成平滑曲线）
        xy_all = coords[:T, j, :]  # [T, 2]
        occ_all = occluded[:T, j]   # [T]
        
        color = point_colors[j]
        
        # 计算所有帧在拼接图中的全局坐标
        # 帧t对应的x偏移：线性映射到[0, total_width]
        # 只绘制未被遮挡的帧
        global_xy_all = []
        if skip_occluded_points:
            for t in range(T):
                if np.isfinite(xy_all[t]).all() and not occ_all[t]:
                    # x坐标按帧比例线性映射
                    x_offset = (t / (T - 1)) * (total_width - W)
                    gx = xy_all[t, 0] + x_offset
                    gy = xy_all[t, 1] + y_offset  # y坐标加上padding偏移
                    global_xy_all.append((gx, gy, t))
        else:
            for t in range(T):
                if np.isfinite(xy_all[t]).all():
                    # x坐标按帧比例线性映射
                    x_offset = (t / max(T - 1, 1)) * (total_width - W)
                    gx = xy_all[t, 0] + x_offset
                    gy = xy_all[t, 1] + y_offset  # y坐标加上padding偏移
                    global_xy_all.append((gx, gy, t))

        
        if len(global_xy_all) < 1:
            continue
        
        # 根据 traj_lim 决定绘制模式
        if traj_lim is None:
            # 模式1: traj_lim=None - 绘制连续轨迹
            if len(global_xy_all) >= 2:
                xs = [p[0] for p in global_xy_all]
                ys = [p[1] for p in global_xy_all]
                ax.plot(xs, ys, linewidth=line_width, color=color, alpha=0.5, zorder=3)
            
            # 在采样帧位置画点
            # for gx, gy, t in global_xy_all:
            #     if t in sample_idx:
            #         ax.scatter(
            #             [gx], [gy],
            #             s=point_size, c=[color],
            #             edgecolors="none", linewidths=0., alpha=0.8, zorder=5
            #         )

            for ki, kf_t in enumerate(sample_idx):
                if occ_all[kf_t]:
                    continue  # 该点在这一帧被遮挡，跳过
                if not np.isfinite(xy_all[kf_t]).all():
                    continue  # 坐标无效，跳过
                
                kf_x_offset = ki * W
                gx = xy_all[kf_t, 0] + kf_x_offset
                gy = xy_all[kf_t, 1] + y_offset
                
                ax.scatter(
                    [gx], [gy],
                    s=point_size, c=[color],
                    edgecolors="none", linewidths=0., alpha=0.8, zorder=5
                )
        
        elif traj_lim == 0:
            # 模式2: traj_lim=0 - 只绘制关键帧处的点（不绘制轨迹）
            # 为每个关键帧单独检查 occluded
            for ki, kf_t in enumerate(sample_idx):
                if occ_all[kf_t]:
                    continue  # 该点在这一帧被遮挡，跳过
                if not np.isfinite(xy_all[kf_t]).all():
                    continue  # 坐标无效，跳过
                
                kf_x_offset = ki * W
                gx = xy_all[kf_t, 0] + kf_x_offset
                gy = xy_all[kf_t, 1] + y_offset
                
                ax.scatter(
                    [gx], [gy],
                    s=point_size, c=[color],
                    edgecolors="none", linewidths=0., alpha=0.8, zorder=5
                )
        
        else:
            # 模式3: traj_lim>0 - 为每个关键帧绘制其前 traj_lim 帧的轨迹
            for ki, kf_t in enumerate(sample_idx):
                kf_x_offset = ki * W
                
                # 收集从 (kf_t - traj_lim) 到 kf_t 范围内的点
                nearby_points = []
                for t in range(max(0, kf_t - traj_lim), kf_t + 1):
                    if np.isfinite(xy_all[t]).all() and not occ_all[t]:
                        gx = xy_all[t, 0] + kf_x_offset
                        gy = xy_all[t, 1] + y_offset
                        nearby_points.append((gx, gy, t))
                
                # 画轨迹线
                if len(nearby_points) >= 2:
                    nearby_points.sort(key=lambda p: p[2])
                    xs = [p[0] for p in nearby_points]
                    ys = [p[1] for p in nearby_points]
                    ax.plot(xs, ys, linewidth=line_width, color=color, alpha=0.8, zorder=3, solid_capstyle='round')
                
                # 在关键帧位置画点（如果该点在关键帧可见）
                if not occ_all[kf_t] and np.isfinite(xy_all[kf_t]).all():
                    gx = xy_all[kf_t, 0] + kf_x_offset
                    gy = xy_all[kf_t, 1] + y_offset
                    ax.scatter(
                        [gx], [gy],
                        s=point_size, c=[color],
                        edgecolors="none", linewidths=0., alpha=0.8, zorder=5
                    )

    fig.savefig(out_path, dpi=dpi, transparent=False, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved 2D visualization: {out_path}")


# -------------------------
# 3D visualization (using matplotlib, no OpenGL required)
# -------------------------
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_depth_image(depth_path: str, depth_range: tuple):
    """加载16-bit深度图并归一化到实际深度值"""
    depth_img = np.array(Image.open(depth_path))
    depth_min, depth_max = depth_range
    depth = depth_min + (depth_img.astype(np.float32) / 65535.0) * (depth_max - depth_min)
    return depth

def reconstruct_scene_points(
    rgb_path: str,
    depth_path: str,
    intrinsics: np.ndarray,
    matrix_world: np.ndarray,
    depth_range: tuple,
    max_points: int = 10000,
    depth_threshold: float = 100.0,
):
    """从RGB和深度图重建场景点云"""
    # 加载RGB和深度
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = load_depth_image(depth_path, depth_range)
    
    H, W = depth.shape
    
    # 生成像素网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)
    z = depth.flatten()
    
    # 过滤无效深度
    valid = (z > 0) & (z < depth_threshold) & np.isfinite(z)
    u, v, z = u[valid], v[valid], z[valid]
    colors = rgb.reshape(-1, 3)[valid] / 255.0
    
    # 下采样
    if len(u) > max_points:
        idx = np.random.choice(len(u), max_points, replace=False)
        u, v, z, colors = u[idx], v[idx], z[idx], colors[idx]
    
    # 反投影到相机坐标系
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    # 相机坐标系点（与run_bench.py一致，不翻转）
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # 变换到世界坐标系
    Tcw = cam_to_world_T(matrix_world, do_orthogonalize=True)
    points_world = apply_T(points_cam, Tcw)
    
    return points_world, colors

def get_camera_frustum_corners(
    intrinsics: np.ndarray,
    matrix_world: np.ndarray,
    image_width: int,
    image_height: int,
    frustum_scale: float = 0.5,
):
    """
    计算相机视锥体的成像面四边形角点（世界坐标）
    返回: focal_point (焦点位置), corners (四边形四个角点)
    """
    Tcw = cam_to_world_T(matrix_world, do_orthogonalize=True)
    focal_point = Tcw[:3, 3]  # 相机位置（焦点）
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 成像面四个角点的像素坐标
    corners_2d = np.array([
        [0, 0],                        # 左上
        [image_width, 0],              # 右上
        [image_width, image_height],   # 右下
        [0, image_height],             # 左下
    ], dtype=np.float32)
    
    # 反投影到相机坐标系（假设深度=frustum_scale）
    z = frustum_scale
    corners_cam = []
    for u, v in corners_2d:
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        # 相机坐标系（与run_bench.py一致，不翻转）
        corners_cam.append([x_cam, y_cam, z])
    
    corners_cam = np.array(corners_cam, dtype=np.float32)
    
    # 变换到世界坐标系
    corners_world = apply_T(corners_cam, Tcw)
    
    return focal_point, corners_world

def visualize_tracking_3d(
    frames_dir: str,
    sample_npy_path: str,
    out_path: str,
    num_sample_frames: int = 6,
    point_size: float = 2.0,
    line_width: float = 1.0,
    figsize: tuple = (16, 12),
    dpi: int = 150,
    first_frame_visible_only: bool = True,
    max_scene_points: int = 10000,
    frustum_scale: float = 0.1,  # 缩小相机视锥体
    depth_threshold: float = 5.0,  # 只显示深度5米以内的点
    max_points: int = 64,
    skip_occluded_points: bool = False,  # 是否跳过任何帧中occluded的点
):
    """
    3D可视化：使用matplotlib绑制（无需OpenGL/EGL）
    - 绘制场景点云（从第一帧的RGB+Depth重建）
    - 绘制tracking点的3D轨迹（所有帧，形成平滑曲线）
    - 绘制相机位姿（焦点+成像面四边形）
    
    traj_3d: [T,N,3], occluded: [T,N]
    matrix_world: [T,4,4]  (cam->world)
    intrinsics: [T,3,3]
    depth_range: (min, max)
    skip_occluded_points: 如果为True，跳过任何帧中被遮挡的点（只画始终可见的点）
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    data = read_sample(sample_npy_path)
    traj = data["traj_3d"].astype(np.float32)     # [T,N,3]
    occluded = data["occluded"].astype(bool)      # [T,N]
    matrix_world = data.get("matrix_world", None) # [T,4,4]
    intrinsics = data.get("intrinsics", None)     # [T,3,3]
    depth_range = data.get("depth_range", None)   # (min, max)
    T, N, _ = traj.shape
    # T = 231
    # 采样帧用于显示相机位姿和RGB
    sample_idx = linspace_indices(T, num_sample_frames)
    
    # 过滤：只保留第一帧未被遮挡的点
    if first_frame_visible_only:
        first_frame_visible = ~occluded[0]  # [N]
        valid_point_indices = np.where(first_frame_visible)[0]
    else:
        valid_point_indices = np.arange(N)
    
    # 过滤：跳过任何帧中被遮挡的点（只保留始终可见的点）
    if skip_occluded_points:
        always_visible = ~occluded.any(axis=0)  # [N], True表示该点在所有帧都未被遮挡
        always_visible_indices = np.where(always_visible)[0]
        valid_point_indices = np.intersect1d(valid_point_indices, always_visible_indices)
        print(f"After skip_occluded_points filter: {len(valid_point_indices)} points always visible")
    
    # 如果点数超过max_points，随机采样
    if len(valid_point_indices) > max_points:
        # np.random.seed(123)
        valid_point_indices = np.random.choice(valid_point_indices, max_points, replace=False)
        valid_point_indices = np.sort(valid_point_indices)
    
    print(f"3D Visualization: {len(valid_point_indices)}/{N} tracking points")
    print(f"Using all {T} frames for trajectories")
    print(f"Camera poses at frames: {sample_idx.tolist()}")

    # 为每个点生成随机颜色
    # np.random.seed(123)
    point_colors = np.random.rand(N, 3)

    # 创建3D figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 获取图片尺寸和路径
    all_frame_paths = list_frames(frames_dir)
    first_img = Image.open(all_frame_paths[0])
    img_width, img_height = first_img.size

    # 时间颜色映射：越早的帧越透明（alpha从0.05到0.8）
    num_sample = len(sample_idx)
    alpha_values = np.linspace(0.1, 1.0, num_sample)  # 早的帧很透明，晚的帧清晰
    
    # 1. 对每个采样帧重建场景点云
    scene_points_all = []
    if depth_range is not None and intrinsics is not None and matrix_world is not None:
        print(f"Reconstructing scene from RGB+Depth for {num_sample} frames (depth < {depth_threshold}m)...")
        
        for fi, t in enumerate(sample_idx):
            frame_path = all_frame_paths[t]
            depth_path = frame_path.replace(".png", "_depth.png").replace(".jpg", "_depth.png")
            
            if os.path.exists(depth_path):
                scene_pts, scene_cols = reconstruct_scene_points(
                    rgb_path=frame_path,
                    depth_path=depth_path,
                    intrinsics=intrinsics[t],
                    matrix_world=matrix_world[t],
                    depth_range=depth_range,
                    max_points=max_scene_points // num_sample,  # 每帧分配一部分点数
                    depth_threshold=depth_threshold,
                )
                
                # 根据时间调整颜色亮度（越早越浅）
                # 将颜色向白色混合
                blend_factor = 1.0 - alpha_values[fi]  # 越早blend_factor越大（越接近白色）
                scene_cols_blended = scene_cols * (1 - blend_factor) + blend_factor
                
                scene_points_all.append((scene_pts, scene_cols_blended, alpha_values[fi]))
                print(f"  Frame {t}: {len(scene_pts)} points")
            else:
                print(f"  Frame {t}: depth image not found")
        
        # 绘制所有帧的场景点云
        for scene_pts, scene_cols, alpha in scene_points_all:
            ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                      c=scene_cols, s=0.5, alpha=alpha)
    else:
        print("Missing depth_range, intrinsics, or matrix_world for scene reconstruction")

    # 2. 绘制tracking轨迹（使用所有帧，只显示深度阈值内的点）
    # 计算第一帧相机位置用于深度过滤
    if matrix_world is not None:
        cam_pos_0 = cam_to_world_T(matrix_world[0], do_orthogonalize=True)[:3, 3]
    else:
        cam_pos_0 = np.zeros(3)
    
    # 为轨迹创建时间-颜色映射（越早越浅）
    cmap_traj = plt.get_cmap("viridis")
    
    tracking_bar = tqdm.tqdm(valid_point_indices, desc="Drawing 3D tracking")
    for j in tracking_bar:
        xyz_all = traj[:, j, :]  # [T, 3]
        valid_mask = np.isfinite(xyz_all).all(axis=1)
        
        # 根据深度阈值过滤：只保留第一帧位置在深度阈值内的点
        if valid_mask[0]:
            dist_to_cam = np.linalg.norm(xyz_all[0] - cam_pos_0)
            if dist_to_cam > depth_threshold:
                continue  # 跳过距离相机超过阈值的点
        
        if not valid_mask.any():
            continue
        
        base_color = point_colors[j]
        
        # 找出连续的有效段
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 1:
            continue
        
        # 画连续轨迹线（分段处理，颜色随时间变化）
        segments = []
        seg_start = valid_indices[0]
        for i in range(1, len(valid_indices)):
            if valid_indices[i] != valid_indices[i-1] + 1:
                segments.append((seg_start, valid_indices[i-1]))
                seg_start = valid_indices[i]
        segments.append((seg_start, valid_indices[-1]))
        
        for seg_s, seg_e in segments:
            if seg_e > seg_s:
                # 逐段画线，越早越透明
                for t_idx in range(seg_s, seg_e):
                    t_ratio = t_idx / max(T - 1, 1)  # 时间比例 0->1
                    alpha = 0.1 + 0.7 * t_ratio  # alpha从0.1到0.8，早期很透明
                    # 颜色向白色混合（越早越浅）
                    blend = 1.0 - t_ratio
                    line_color = np.clip(base_color * (1 - blend * 0.5) + blend * 0.5, 0, 1)
                    
                    ax.plot([xyz_all[t_idx, 0], xyz_all[t_idx + 1, 0]],
                           [xyz_all[t_idx, 1], xyz_all[t_idx + 1, 1]],
                           [xyz_all[t_idx, 2], xyz_all[t_idx + 1, 2]],
                           linewidth=line_width, color=line_color, alpha=alpha)
        
        # 在采样帧位置画点强调（颜色也随时间变化，越早越透明）
        for fi, t in enumerate(sample_idx):
            if valid_mask[t]:
                t_ratio = t / max(T - 1, 1)
                alpha = 0.15 + 0.7 * t_ratio  # alpha从0.15到0.85，早期很透明
                blend = 1.0 - t_ratio
                pt_color = np.clip(base_color * (1 - blend * 0.5) + blend * 0.5, 0, 1)
                
                ax.scatter([xyz_all[t, 0]], [xyz_all[t, 1]], [xyz_all[t, 2]],
                          c=[pt_color], s=point_size * 10, 
                          edgecolors='none', linewidths=0.3, alpha=alpha)

    # 3. 绘制相机位姿（焦点+成像面四边形）
    if matrix_world is not None and intrinsics is not None:
        cmap = plt.get_cmap("turbo")
        cam_colors = cmap(np.linspace(0.0, 1.0, len(sample_idx)))[:, :3]
        
        print("Drawing camera frustums...")
        for i, t in enumerate(sample_idx):
            focal_point, corners = get_camera_frustum_corners(
                intrinsics=intrinsics[t],
                matrix_world=matrix_world[t],
                image_width=img_width,
                image_height=img_height,
                frustum_scale=frustum_scale,
            )
            
            color = cam_colors[i]
            
            # 画焦点
            ax.scatter([focal_point[0]], [focal_point[1]], [focal_point[2]],
                      c=[color], s=80, marker='o', edgecolors='none', linewidths=0.8)
            
            # 画成像面四边形
            verts = [corners.tolist()]
            poly = Poly3DCollection(verts, alpha=0.3, facecolor=color, edgecolor=color, linewidth=1.5)
            ax.add_collection3d(poly)
            
            # 画焦点到四边形四个角的连线
            for corner in corners:
                ax.plot([focal_point[0], corner[0]], 
                       [focal_point[1], corner[1]], 
                       [focal_point[2], corner[2]],
                       color=color, linewidth=0.8, alpha=0.6)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 自动调整视角 - 收集所有点计算边界
    all_points = []
    for j in valid_point_indices:
        xyz_all = traj[:, j, :]
        valid_mask = np.isfinite(xyz_all).all(axis=1)
        if valid_mask.any():
            all_points.append(xyz_all[valid_mask])
    
    # 优先使用场景点云来计算视野范围（让场景更大更清晰）
    scene_only_points = []
    for scene_pts, _, _ in scene_points_all:
        scene_only_points.append(scene_pts)
    
    if len(scene_only_points) > 0:
        # 基于场景点云计算中心和范围
        scene_all = np.vstack(scene_only_points)
        center = scene_all.mean(axis=0)
        max_range = (scene_all.max(axis=0) - scene_all.min(axis=0)).max() / 2
        
        # 缩小范围让场景更大（0.5 = 放大约2倍）
        max_range *= 0.5
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
    elif len(all_points) > 0:
        # 如果没有场景点云，退回使用所有点
        if matrix_world is not None:
            for t in sample_idx:
                Tcw = cam_to_world_T(matrix_world[t], do_orthogonalize=True)
                all_points.append(Tcw[:3, 3].reshape(1, 3))
        
        all_points = np.vstack(all_points)
        center = all_points.mean(axis=0)
        max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2
        
        max_range *= 0.6
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # 设置好看的视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved 3D visualization: {out_path}")


def visualize_tracking_3d_keyframes(
    frames_dir: str,
    sample_npy_path: str,
    out_path: str,
    num_keyframes: int = 6,
    point_size: float = 2.0,
    scene_point_size: float = 5.0,
    line_width: float = 1.5,
    figsize: tuple = (24, 8),
    dpi: int = 150,
    first_frame_visible_only: bool = True,
    frustum_scale: float = 0.5,
    max_scene_points: int = 15000,
    depth_threshold: float = 8.0,
    keyframe_spacing: float = 5.0,
    max_points: int = 64,
    view_elev: float = 25,    # 仰角（垂直角度）：0=水平看，90=信视
    view_azim: float = 45,   # 方位角（水平旋转）：0=正前方，-90=左侧
    last_frame=None,
    skip_occluded_points: bool = False,  # 是否跳过任何帧中occluded的点
    point_indices: np.ndarray = None,  # 预先计算好的点索引，如果提供则直接使用
):
    """
    3D关键帧拼接可视化：类似2D，将关键帧点云水平排开，轨迹连续贯穿
    直接生成静态图片
    
    观察角度说明：
    - view_elev: 仰角，0=水平看，90=从正上方信视，负值=从下方看
    - view_azim: 方位角，0=正前方(+Y)，-90=左侧(+X)，90=右侧(-X)
    skip_occluded_points: 如果为True，跳过任何帧中被遮挡的点（只画始终可见的点）
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- Log function call parameters ---
    try:
        _params = dict(
            frames_dir=frames_dir,
            sample_npy_path=sample_npy_path,
            out_path=out_path,
            num_keyframes=num_keyframes,
            point_size=point_size,
            scene_point_size=scene_point_size,
            line_width=line_width,
            figsize=figsize,
            dpi=dpi,
            first_frame_visible_only=first_frame_visible_only,
            frustum_scale=frustum_scale,
            max_scene_points=max_scene_points,
            depth_threshold=depth_threshold,
            keyframe_spacing=keyframe_spacing,
            max_points=max_points,
            view_elev=view_elev,
            view_azim=view_azim,
            last_frame=last_frame,
            skip_occluded_points=skip_occluded_points,
            point_indices=point_indices,
        )
        log_function_call("visualize_tracking_3d_keyframes", _params)
    except Exception:
        pass

    data = read_sample(sample_npy_path)
    traj = data["traj_3d"].astype(np.float32)     # [T,N,3]
    occluded = data["occluded"].astype(bool)      # [T,N]
    matrix_world = data.get("matrix_world", None) # [T,4,4]
    intrinsics = data.get("intrinsics", None)     # [T,3,3]
    depth_range = data.get("depth_range", None)   # (min, max)
    T, N, _ = traj.shape

    if last_frame is not None:
        T = last_frame

    # 关键帧索引
    keyframe_idx = linspace_indices(T, num_keyframes)
    num_kf = len(keyframe_idx)
    
    print(f"3D Keyframe Visualization: {num_kf} keyframes at {keyframe_idx.tolist()}")
    
    # 如果提供了 point_indices，直接使用
    if point_indices is not None:
        valid_point_indices = point_indices
    else:
        # 过滤：只保留第一帧未被遮挡的点
        if first_frame_visible_only:
            first_frame_visible = ~occluded[0]
            valid_point_indices = np.where(first_frame_visible)[0]
        else:
            valid_point_indices = np.arange(N)
        
        # 过滤：跳过任何帧中被遮挡的点（只保留始终可见的点）
        if skip_occluded_points:
            always_visible = ~occluded.any(axis=0)  # [N], True表示该点在所有帧都未被遮挡
            always_visible_indices = np.where(always_visible)[0]
            valid_point_indices = np.intersect1d(valid_point_indices, always_visible_indices)
            print(f"After skip_occluded_points filter: {len(valid_point_indices)} points always visible")
        
        if len(valid_point_indices) > max_points:
            valid_point_indices = np.random.choice(valid_point_indices, max_points, replace=False)
            valid_point_indices = np.sort(valid_point_indices)
    
    print(f"Using {len(valid_point_indices)}/{N} tracking points")

    # 为每个点生成随机颜色
    # np.random.seed(123)
    point_colors = np.random.rand(N, 3)

    # 创建3D figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 获取图片路径
    all_frame_paths = list_frames(frames_dir)

    # 1. 对每个关键帧重建场景点云，并水平偏移
    scene_points_all = []
    if depth_range is not None and intrinsics is not None and matrix_world is not None:
        print(f"Reconstructing scene from RGB+Depth for {num_kf} keyframes...")
        
        for fi, t in enumerate(keyframe_idx):
            frame_path = all_frame_paths[t]
            depth_path = frame_path.replace(".png", "_depth.png").replace(".jpg", "_depth.png")
            
            if os.path.exists(depth_path):
                scene_pts, scene_cols = reconstruct_scene_points(
                    rgb_path=frame_path,
                    depth_path=depth_path,
                    intrinsics=intrinsics[t],
                    matrix_world=matrix_world[t],
                    depth_range=depth_range,
                    max_points=max_scene_points,
                    depth_threshold=depth_threshold,
                )
                
                # 水平偏移：每个关键帧在X方向偏移
                x_offset = fi * keyframe_spacing
                scene_pts_offset = scene_pts.copy()
                scene_pts_offset[:, 0] -= x_offset
                
                scene_points_all.append((scene_pts_offset, scene_cols, fi))
                print(f"  Keyframe {fi} (frame {t}): {len(scene_pts)} points, x_offset={x_offset:.1f}")
            else:
                print(f"  Keyframe {fi} (frame {t}): depth image not found")
        
        # 绘制所有关键帧的场景点云（使用方形marker）
        for scene_pts, scene_cols, fi in scene_points_all:
            ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                      c=scene_cols, s=scene_point_size, alpha=0.8, marker='s')  # 's' = 方形

    # 2. 绘制tracking轨迹（连续贯穿所有关键帧）
    # 轨迹点根据时间比例进行水平偏移
    if matrix_world is not None:
        cam_pos_0 = cam_to_world_T(matrix_world[0], do_orthogonalize=True)[:3, 3]
    else:
        cam_pos_0 = np.zeros(3)
    
    tracking_bar = tqdm.tqdm(valid_point_indices, desc="Drawing 3D tracking")
    for j in tracking_bar:
        xyz_all = traj[:T, j, :].copy()  # [T, 3]
        valid_mask = np.isfinite(xyz_all).all(axis=1)
        
        # 深度过滤
        if valid_mask[0]:
            dist_to_cam = np.linalg.norm(xyz_all[0] - cam_pos_0)
            if dist_to_cam > depth_threshold:
                continue
        
        if not valid_mask.any():
            continue
        
        base_color = point_colors[j]
        
        # 对轨迹点进行水平偏移
        xyz_offset = xyz_all.copy()
        for t in range(T):
            if valid_mask[t]:
                # 时间比例映射到关键帧间距
                t_ratio = t / max(T - 1, 1) * (num_kf - 1)
                x_offset = t_ratio * keyframe_spacing
                xyz_offset[t, 0] -= x_offset
        
        # 画连续轨迹线
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) >= 2:
            # 分段处理连续段
            segments = []
            seg_start = valid_indices[0]
            for i in range(1, len(valid_indices)):
                if valid_indices[i] != valid_indices[i-1] + 1:
                    segments.append((seg_start, valid_indices[i-1]))
                    seg_start = valid_indices[i]
            segments.append((seg_start, valid_indices[-1]))
            
            for seg_s, seg_e in segments:
                if seg_e > seg_s:
                    xs = xyz_offset[seg_s:seg_e+1, 0]
                    ys = xyz_offset[seg_s:seg_e+1, 1]
                    zs = xyz_offset[seg_s:seg_e+1, 2]
                    ax.plot(xs, ys, zs, linewidth=line_width, color=base_color, alpha=0.5)
        
        # 在关键帧位置画点强调
        for fi, t in enumerate(keyframe_idx):
            if valid_mask[t]:
                ax.scatter([xyz_offset[t, 0]], [xyz_offset[t, 1]], [xyz_offset[t, 2]],
                          c=[base_color], s=point_size * 5, 
                          edgecolors='none', linewidths=0., alpha=0., zorder=10)

    # 3. 绘制相机位姿（每个关键帧）
    if matrix_world is not None and intrinsics is not None:
        first_img = Image.open(all_frame_paths[0])
        img_width, img_height = first_img.size
        
        cmap = plt.get_cmap("turbo")
        cam_colors = cmap(np.linspace(0.0, 1.0, num_kf))[:, :3]
        
        print("Drawing camera frustums...")
        for fi, t in enumerate(keyframe_idx):
            focal_point, corners = get_camera_frustum_corners(
                intrinsics=intrinsics[t],
                matrix_world=matrix_world[t],
                image_width=img_width,
                image_height=img_height,
                frustum_scale=frustum_scale,
            )
            

            # 水平偏移
            x_offset = fi * keyframe_spacing
            focal_point_offset = focal_point.copy()
            focal_point_offset[0] -= x_offset
            corners_offset = corners.copy()
            corners_offset[:, 0] -= x_offset


            color = cam_colors[fi]
            
            # 画焦点
            ax.scatter([focal_point_offset[0]], [focal_point_offset[1]], [focal_point_offset[2]],
                      c=[color], s=5, marker='o', edgecolors='none', linewidths=0.)
            
            # 画成像面四边形
            verts = [corners_offset.tolist()]
            poly = Poly3DCollection(verts, alpha=0.25, facecolor=color, edgecolor=color, linewidth=1)
            ax.add_collection3d(poly)
            
            # 画焦点到四边形四个角的连线
            for corner in corners_offset:
                ax.plot([focal_point_offset[0], corner[0]], 
                       [focal_point_offset[1], corner[1]], 
                       [focal_point_offset[2], corner[2]],
                       color=color, linewidth=0.6, alpha=0.5)

    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 计算视野范围（只基于场景点云，让点云更大更清晰）
    if len(scene_points_all) > 0:
        # 收集所有场景点（过滤空数组）
        valid_scene_pts = [pts for pts, _, _ in scene_points_all if len(pts) > 0]
        if len(valid_scene_pts) > 0:
            all_scene_pts = np.vstack(valid_scene_pts)
            
            # 计算各轴范围
            x_min, x_max = all_scene_pts[:, 0].min(), all_scene_pts[:, 0].max()
            y_min, y_max = all_scene_pts[:, 1].min(), all_scene_pts[:, 1].max()
            z_min, z_max = all_scene_pts[:, 2].min(), all_scene_pts[:, 2].max()
            
            # 计算中心
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            z_center = (z_max + z_min) / 2
            
            # 计算各轴的实际数据范围
            x_data_range = max(x_max - x_min, 0.1)
            y_data_range = max(y_max - y_min, 0.1)
            z_data_range = max(z_max - z_min, 0.1)
            
            # Y和Z使用相同范围（保持深度方向比例），稍微留一点边距
            yz_max_range = max(y_data_range, z_data_range) / 2 * 1.1
            
            # X方向使用自己的范围（因为关键帧横向排列）
            x_half_range = x_data_range / 2 * 1.05
            
            ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
            ax.set_ylim(y_center - yz_max_range, y_center + yz_max_range)
            ax.set_zlim(z_center - yz_max_range, z_center + yz_max_range)
            
            # 设置 box aspect 根据实际显示范围比例
            ax.set_box_aspect([x_half_range * 2, yz_max_range * 2, yz_max_range * 2])
        else:
            print("Warning: No valid scene points found")
    
    # 设置观察角度
    ax.view_init(elev=view_elev, azim=view_azim)
    
    # 关闭网格让图更清晰
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 隐藏坐标轴（移除刻度和标签）
    try:
        ax.set_axis_off()
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved 3D keyframe visualization: {out_path}")


# -------------------------
# VIZ 交互式可视化 (WebGL)
# -------------------------
import json
import struct
import zlib
import cv2
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
import socket
import sys

VIZ_HTML_PATH = Path(__file__).parent / "utils" / "viz.html"

def compress_and_write(filename, header, blob):
    """将数据压缩并写入文件"""
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def prepare_viz_data(
    frames_dir: str,
    sample_npy_path: str,
    output_file: str,
    width: int = 256,
    height: int = 192,
    fps: int = 4,
    max_points: int = 512,
    first_frame_visible_only: bool = True,
    keyframe_mode: bool = False,  # 关键帧分离模式
    num_keyframes: int = 6,       # 关键帧数量
    keyframe_spacing: float = 3.0,  # 关键帧间距（相对于场景尺度）
    depth_threshold: float = None,  # 深度阈值，超过此值的点不显示（None表示不过滤）
    random_pick: bool = True,  # True=随机选择tracking points, False=按顺序选择前max_points个
):
    """
    将数据转换为 viz 可视化所需的格式
    
    keyframe_mode: 如果为True，则将关键帧点云水平分离排列，轨迹连续贯穿
    depth_threshold: 深度阈值，超过此值的点将被过滤掉（设为超大值）
    """
    fixed_size = (width, height)
    
    # 读取数据
    data = read_sample(sample_npy_path)
    traj_3d = data["traj_3d"].astype(np.float32)  # [T, N, 3]
    occluded = data["occluded"].astype(bool)       # [T, N]
    matrix_world = data.get("matrix_world", None)  # [T, 4, 4] cam->world
    intrinsics_orig = data.get("intrinsics", None) # [T, 3, 3]
    depth_range = data.get("depth_range", None)    # (min, max)
    
    T, N, _ = traj_3d.shape
    
    # 过滤tracking点
    if first_frame_visible_only:
        first_frame_visible = ~occluded[0]
        valid_point_indices = np.where(first_frame_visible)[0]
    else:
        valid_point_indices = np.arange(N)
    
    if len(valid_point_indices) > max_points:
        # np.random.seed(42)
        if random_pick:
            valid_point_indices = np.random.choice(valid_point_indices, max_points, replace=False)
        else:
            valid_point_indices = valid_point_indices[:max_points]
        valid_point_indices = np.sort(valid_point_indices)
    
    # 只保留有效的tracking点
    traj_3d = traj_3d[:, valid_point_indices, :]
    print(f"Using {len(valid_point_indices)} tracking points")
    
    # 读取RGB帧
    all_frame_paths = list_frames(frames_dir)
    T = min(T, len(all_frame_paths))
    
    first_img = np.array(Image.open(all_frame_paths[0]).convert("RGB"))
    orig_H, orig_W = first_img.shape[:2]
    
    # 关键帧索引
    if keyframe_mode:
        keyframe_idx = linspace_indices(T, num_keyframes)
        print(f"Keyframe mode: {len(keyframe_idx)} keyframes at {keyframe_idx.tolist()}")
    else:
        keyframe_idx = None
    
    # 加载并resize所有帧
    rgb_video = []
    for t in tqdm.tqdm(range(T), desc="Loading RGB frames"):
        img = np.array(Image.open(all_frame_paths[t]).convert("RGB"))
        img_resized = cv2.resize(img, fixed_size, interpolation=cv2.INTER_AREA)
        rgb_video.append(img_resized)
    rgb_video = np.stack(rgb_video)  # [T, H, W, C]
    
    # 加载深度图
    depth_video = []
    for t in tqdm.tqdm(range(T), desc="Loading depth frames"):
        depth_path = all_frame_paths[t].replace(".png", "_depth.png").replace(".jpg", "_depth.png")
        if os.path.exists(depth_path):
            depth_img = np.array(Image.open(depth_path))
            # 归一化到实际深度值
            depth_min, depth_max = depth_range
            depth = depth_min + (depth_img.astype(np.float32) / 65535.0) * (depth_max - depth_min)
            # 应用深度阈值过滤：超过阈值的点设为超大值（在viz中不显示）
            if depth_threshold is not None:
                depth[depth > depth_threshold] = 1e6
            depth_resized = cv2.resize(depth, fixed_size, interpolation=cv2.INTER_NEAREST)
        else:
            depth_resized = np.ones(fixed_size[::-1], dtype=np.float32)
        depth_video.append(depth_resized)
    depth_video = np.stack(depth_video)  # [T, H, W]
    
    if depth_threshold is not None:
        print(f"Applied depth threshold: {depth_threshold}m")
    
    # 深度范围计算 - 忽略被设为1e6的无效深度点
    valid_depths = depth_video[depth_video < 1e5]  # 过滤掉超大值
    if len(valid_depths) > 0:
        min_depth = float(valid_depths.min()) * 1
        max_depth = float(valid_depths.max()) * 1
    else:
        # 如果没有有效深度点，使用原始深度范围
        min_depth = float(depth_range[0]) * 1
        max_depth = float(depth_range[1]) * 1
    
    print(f"Effective depth range for viz: [{min_depth:.2f}, {max_depth:.2f}]m")
    
    # 缩放内参
    scale_x = fixed_size[0] / orig_W
    scale_y = fixed_size[1] / orig_H
    intrinsics = intrinsics_orig[:T].copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    # 计算FOV
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(fixed_size[1] / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(fixed_size[0] / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (orig_W / intrinsics_orig[0, 0, 0]) / (orig_H / intrinsics_orig[0, 1, 1])
    
    # 计算extrinsics (world->cam，即matrix_world的逆)
    extrinsics = np.zeros((T, 4, 4), dtype=np.float32)
    cam_to_world_all = np.zeros((T, 4, 4), dtype=np.float32)
    for t in range(T):
        Tcw = cam_to_world_T(matrix_world[t], do_orthogonalize=True)  # cam -> world
        cam_to_world_all[t] = Tcw
        extrinsics[t] = np.linalg.inv(Tcw)  # world -> cam

    # 深度编码为16位RGB（使用之前计算的有效深度范围）
    # 将超过阈值的深度点clip到max_depth，这样在viz中会被正确处理
    depth_video_clipped = np.clip(depth_video, min_depth, max_depth)
    depth_normalized = (depth_video_clipped - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    # 为 viz 保持点云和轨迹在世界坐标系：
    # - 我们输出的 "extrinsics" 仍为 world->cam（与之前一致），
    # - 而 "inv_extrinsics" 会被写为 cam->world（用于从相机坐标重建场景点云），
    # - trajectories 直接使用原始世界坐标 traj_3d（不再额外归一化）。
    normalized_extrinsics = extrinsics.copy()  # world->cam

    # 直接使用世界坐标系的轨迹（T, N, 3）
    normalized_trajs = traj_3d[:T].copy()
    
    # 关键帧模式：水平偏移轨迹和相机
    if keyframe_mode:
        print("Applying keyframe spacing...")
        num_kf = len(keyframe_idx)

        # 轨迹点按时间比例水平偏移（在世界坐标系下偏移 X）
        for t in range(T):
            t_ratio = t / max(T - 1, 1) * (num_kf - 1)
            x_offset = t_ratio * keyframe_spacing
            normalized_trajs[t, :, 0] += x_offset
        
        # 相机位姿也相应偏移
        for t in range(T):
            t_ratio = t / max(T - 1, 1) * (num_kf - 1)
            x_offset = t_ratio * keyframe_spacing
            # 修改平移部分
            normalized_extrinsics[t, 0, 3] -= x_offset  # extrinsics是world->cam，所以减去
    
    # 打包数据
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics.astype(np.float32),
        "extrinsics": normalized_extrinsics.astype(np.float32),
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics).astype(np.float32),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }
    
    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape if hasattr(arr, 'shape') else (),
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)
    
    print(f"viz_depth_range: {min_depth}, {max_depth}")
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": list(fixed_size),
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }
    
    compress_and_write(output_file, header, compressed_blob)
    print(f"Saved viz data: {output_file}")

def visualize_with_viz(
    frames_dir: str,
    sample_npy_path: str,
    port: int = 8000,
    width: int = 256,
    height: int = 192,
    fps: int = 4,
    max_points: int = 512,
    keyframe_mode: bool = False,
    num_keyframes: int = 6,
    keyframe_spacing: float = 3.0,
    depth_threshold: float = None,  # 深度阈值，超过此值的点不显示
    random_pick: bool = True,  # True=随机选择tracking points, False=按顺序选择前max_points个
):
    """
    使用 viz 工具进行交互式3D可视化
    启动本地服务器，在浏览器中查看
    
    keyframe_mode: 关键帧分离模式，点云水平排开，轨迹连续
    num_keyframes: 关键帧数量
    keyframe_spacing: 关键帧间距
    depth_threshold: 深度阈值，超过此值的点将被过滤掉
    """
    if not VIZ_HTML_PATH.exists():
        print(f"Error: viz.html not found at {VIZ_HTML_PATH}")
        print("Please make sure utils/viz.html exists")
        return
    
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 准备数据
        prepare_viz_data(
            frames_dir=frames_dir,
            sample_npy_path=sample_npy_path,
            output_file=str(temp_path / "data.bin"),
            width=width,
            height=height,
            fps=fps,
            max_points=max_points,
            keyframe_mode=keyframe_mode,
            num_keyframes=num_keyframes,
            keyframe_spacing=keyframe_spacing,
            depth_threshold=depth_threshold,
            random_pick=random_pick,
        )
        
        # 复制 viz.html
        shutil.copy(VIZ_HTML_PATH, temp_path / "index.html")
        
        # 切换到临时目录
        os.chdir(temp_path)
        
        host = "0.0.0.0"  # 允许外部访问
        
        Handler = SimpleHTTPRequestHandler
        httpd = None
        
        try:
            httpd = ThreadingTCPServer((host, port), Handler)
        except OSError as e:
            if e.errno == socket.errno.EADDRINUSE:
                print(f"Port {port} is already in use, trying a random port...")
                try:
                    httpd = ThreadingTCPServer((host, 0), Handler)
                    port = httpd.server_address[1]
                except OSError as e2:
                    print(f"Failed to bind to a random port: {e2}", file=sys.stderr)
                    return
            else:
                print(f"Failed to start server: {e}", file=sys.stderr)
                return
        
        if httpd:
            print(f"\n{'='*50}")
            print(f"VIZ Server running at:")
            print(f"  http://localhost:{port}")
            print(f"  http://0.0.0.0:{port}")
            print(f"{'='*50}")
            print("Press Ctrl+C to stop the server\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
            finally:
                httpd.server_close()


# -------------------------
# Batch over a folder
# -------------------------
def visualize_folder(
    episode_dir: str,
    out_dir: str,
    sample_npy_name: str = None,
    num_sample_frames: int = 4,
    max_points: int = 64,
    last_frame=None,
    eval: str = None,
    skip_occluded_points: bool = False,  # 是否跳过任何帧中occluded的点
    random_pick: bool = True,  # True=随机选择tracking points, False=按顺序选择前max_points个
):
    """
    episode_dir:
      - frames/xxxx.png, xxxx_depth.png
      - xxxx.npy  (或你自定义 sample_npy_name)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Log function call parameters ---
    try:
        _params = dict(
            episode_dir=episode_dir,
            out_dir=out_dir,
            sample_npy_name=sample_npy_name,
            num_sample_frames=num_sample_frames,
            max_points=max_points,
            last_frame=last_frame,
            skip_occluded_points=skip_occluded_points,
        )
        log_function_call("visualize_folder", _params)
    except Exception:
        pass

    frames_dir = os.path.join(episode_dir, "frames")
    if sample_npy_name is None:
        # 自动找一个 .npy（默认取 episode_dir 下第一个）
        # cands = sorted(glob.glob(os.path.join(episode_dir, "*.npy")))
        if eval is None:
            cands = sorted(glob.glob(os.path.join(episode_dir, "[0-9][0-9][0-9][0-9].npy")))
        else:
            cands = sorted(glob.glob(os.path.join(episode_dir, f"*_{eval}_pred.npy")))
        if len(cands) == 0:
            raise FileNotFoundError(f"No npy found in {episode_dir}")
        sample_npy_path = cands[0]
    else:
        sample_npy_path = os.path.join(episode_dir, sample_npy_name)

    print(f"Using .npy file: {os.path.basename(sample_npy_path)}")

    out2d_basename = "tracking_2d"

    if eval is not None:
        out2d_basename = f'{out2d_basename}_{eval}'
    
    if last_frame is not None and num_sample_frames == 1:
        out2d_basename = out2d_basename + f'_{last_frame - 1}'

    out2d_basename += ".png"

    out2d = os.path.join(out_dir, out2d_basename)
    out3d = os.path.join(out_dir, "tracking_3d.png")
    out3d_kf = os.path.join(out_dir, "tracking_3d_keyframes.png")

    # 预先计算 tracking point 索引，确保 2D 和 3D 可视化使用相同的点
    data = read_sample(sample_npy_path)
    occluded = data["occluded"].astype(bool)  # [T, N]
    T_data, N = occluded.shape
    T = last_frame if last_frame is not None else T_data
    
    # 过滤：只保留第一帧未被遮挡的点
    first_frame_visible = ~occluded[0]  # [N]
    valid_point_indices = np.where(first_frame_visible)[0]
    
    # 过滤：跳过任何帧中被遮挡的点（只保留始终可见的点）
    if skip_occluded_points:
        always_visible = ~occluded[:T].any(axis=0)  # [N]
        always_visible_indices = np.where(always_visible)[0]
        valid_point_indices = np.intersect1d(valid_point_indices, always_visible_indices)
        print(f"After skip_occluded_points filter: {len(valid_point_indices)} points always visible")
    
    # 如果点数超过 max_points，根据 random_pick 选择
    if len(valid_point_indices) > max_points:
        if random_pick:
            valid_point_indices = np.random.choice(valid_point_indices, max_points, replace=False)
        else:
            valid_point_indices = valid_point_indices[:max_points]
        valid_point_indices = np.sort(valid_point_indices)
    
    print(f"Using {len(valid_point_indices)}/{N} tracking points for both 2D and 3D visualization")

    visualize_tracking_2d(
        frames_dir=frames_dir,
        sample_npy_path=sample_npy_path,
        out_path=out2d,
        num_sample_frames=num_sample_frames,
        first_frame_visible_only=True,
        max_points=max_points,
        last_frame=last_frame,
        skip_occluded_points=skip_occluded_points,
        point_size=10,
        line_width=2,
        traj_lim=15,  # 0=只画关键帧处的点, None=连续轨迹, >0=每帧画前N帧轨迹
        point_indices=valid_point_indices,  # 使用预先计算的索引
    )

    # visualize_tracking_3d(
    #     frames_dir=frames_dir,
    #     sample_npy_path=sample_npy_path,
    #     out_path=out3d,
    #     num_sample_frames=num_sample_frames,
    #     first_frame_visible_only=True,
    #     max_points=max_points,
    #     skip_occluded_points=skip_occluded_points,
    # )
    
    # 3D关键帧拼接可视化
    visualize_tracking_3d_keyframes(
        frames_dir=frames_dir,
        sample_npy_path=sample_npy_path,
        out_path=out3d_kf,
        num_keyframes=num_sample_frames,
        first_frame_visible_only=True,
        frustum_scale=1,
        keyframe_spacing=10,
        depth_threshold=20,
        max_points=max_points,
        max_scene_points=30000,         # 场景点云最大数量
        view_elev=20,
        view_azim=10,
        last_frame=last_frame,
        scene_point_size=0.1,
        skip_occluded_points=skip_occluded_points,
        point_indices=valid_point_indices,  # 使用预先计算的索引
    )

    # print("Saved:", out2d, out3d, out3d_kf)
    print("Saved:", out2d)


def visualize_folder_viz(
    episode_dir: str,
    sample_npy_name: str = None,
    port: int = 8000,
    width: int = 256,
    height: int = 192,
    fps: int = 2,
    max_points: int = 512,
    keyframe_mode: bool = False,
    num_keyframes: int = 6,
    keyframe_spacing: float = 3.0,
    eval: str = None,   # co2, co3, sp_off, sp_on
    depth_threshold: float = None,  # 深度阈值，超过此值的点不显示
    random_pick: bool = True,  # True=随机选择tracking points, False=按顺序选择前max_points个
):
    """
    使用 viz 工具进行交互式可视化
    
    keyframe_mode: 关键帧分离模式，点云水平排开，轨迹连续
    depth_threshold: 深度阈值，超过此值的点将被过滤掉
    """
    frames_dir = os.path.join(episode_dir, "frames")
    if sample_npy_name is None:
        # cands = sorted(glob.glob(os.path.join(episode_dir, "*.npy")))
        if eval is None:
            cands = sorted(glob.glob(os.path.join(episode_dir, "[0-9][0-9][0-9][0-9].npy")))
        else:
            cands = sorted(glob.glob(os.path.join(episode_dir, f"*_{eval}_pred.npy")))
        if len(cands) == 0:
            raise FileNotFoundError(f"No npy found in {episode_dir}")
        sample_npy_path = cands[0]
    else:
        sample_npy_path = os.path.join(episode_dir, sample_npy_name)

    print(f"Using .npy file: {os.path.basename(sample_npy_path)}")

    visualize_with_viz(
        frames_dir=frames_dir,
        sample_npy_path=sample_npy_path,
        port=port,
        width=width,
        height=height,
        fps=fps,
        max_points=max_points,
        keyframe_mode=keyframe_mode,
        num_keyframes=num_keyframes,
        keyframe_spacing=keyframe_spacing,
        depth_threshold=depth_threshold,
        random_pick=random_pick,
    )


if __name__ == "__main__":
    # ep = "part1_190_usd_0084_navigation_trajectory_0_0_0" # 使用你测试的episode
    # ep = "part1_122_usd_0028_navigation_trajectory_1_1_0"
    ep = 1208
    # ep = 6101
    # ep = 1385
    np.random.seed(42)
    
    # 'sp_off' 'sp_on' 'co3' 'co2' None
    eval = None # 1385, 2528, 2537, 2579, 3700, 2537
    use_viz = True

    if not use_viz:
        # 使用静态图片可视化（包含3D关键帧拼接图）
        visualize_folder(
            episode_dir=f"/mnt/oss/zhaoweiguang/datasets/4dtracking/making/enhance2_flower/{ep:04d}/",
            out_dir="./vis_out",
            num_sample_frames=4,
            max_points=1024,
            skip_occluded_points=False,
            last_frame=None,
            eval=eval,
            random_pick=True
        )
    else:
        # 使用 viz 交互式可视化
        visualize_folder_viz(
            episode_dir=f"/mnt/oss/zhaoweiguang/datasets/4dtracking/making/enhance2_flower/{ep:04d}/",
            port=7812,
            width=512,
            height=512,
            keyframe_mode=False,
            max_points=512,
            depth_threshold=80,
            eval=eval,
            random_pick=True,
            fps=5
        )
