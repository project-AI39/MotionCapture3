import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def determine_coordinate_system(points_3d):
    # x, y, z の平均を取得
    mean_x = np.mean(points_3d[:, 0])
    mean_y = np.mean(points_3d[:, 1])
    mean_z = np.mean(points_3d[:, 2])

    # 各座標の平均が正ならば、それが向いている方向が正となる
    orientation_x = "右" if mean_x > 0 else "左"
    orientation_y = "下" if mean_y > 0 else "上"
    orientation_z = "奥" if mean_z > 0 else "手前"

    result = f"x軸は{orientation_x}方向、y軸は{orientation_y}方向、z軸は{orientation_z}方向を向いています。"

    # x, y, z の符号から右手座標系か左手座標系かを判定
    handedness = "右手座標系" if np.all(np.array([mean_x, mean_y, mean_z]) > 0) else "左手座標系"
    result += f" また、座標系は{handedness}です。"

    return result


# 例として関数を呼び出す部分を追加
# points_3d_example = np.array([points[0], points[1], points[2]])


def plot_3d_points_with_arrows(
    points, cam1_position, cam1_rotation, cam2_position, cam2_rotation
):
    n = len(points) - 2
    for i in range(n):
        points_3d = np.array([points[i], points[i + 1], points[i + 2]])
        system_result = determine_coordinate_system(points_3d)
        print(system_result)

    # # データから最も大きな値を取得して軸の範囲を指定
    max_val = max(
        max(max(point) for point in points),
        float(max(cam1_position)),
        float(max(cam2_position)),
    )
    min_val = min(
        min(min(point) for point in points),
        float(min(cam1_position)),
        float(min(cam2_position)),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 原点からの座標軸を示す向きベクトルを描写
    ax.quiver(0, 0, 0, 1 * max_val, 0, 0, color="r", label="X")
    ax.quiver(0, 0, 0, 0, 1 * max_val, 0, color="g", label="Y")
    ax.quiver(0, 0, 0, 0, 0, 1 * max_val, color="b", label="Z")

    # ポイントを描写
    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        [point[2] for point in points],
        c="black",
        marker="o",
        label="Points",
    )

    # カメラ1の位置を赤い点で描写
    ax.scatter(
        cam1_position[0],
        cam1_position[1],
        cam1_position[2],
        color="red",
        marker="o",
        s=100,
        label="Camera 1 Position",
    )

    # カメラ2の位置を赤い点で描写
    ax.scatter(
        cam2_position[0],
        cam2_position[1],
        cam2_position[2],
        color="blue",
        marker="o",
        s=100,
        label="Camera 2 Position",
    )

    # カメラ1の向きを黄色い矢印で描写
    ax.quiver(
        cam1_position[0],
        cam1_position[1],
        cam1_position[2],
        cam1_rotation[0] * max_val,
        cam1_rotation[1] * max_val,
        cam1_rotation[2] * max_val,
        color="red",
        label="Camera 1 Orientation",
    )

    # カメラ2の向きを黄色い矢印で描写
    ax.quiver(
        cam2_position[0],
        cam2_position[1],
        cam2_position[2],
        cam2_rotation[0] * max_val,
        cam2_rotation[1] * max_val,
        cam2_rotation[2] * max_val,
        color="blue",
        label="Camera 2 Orientation",
    )

    # 赤、青、緑の軸の設定
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    plt.legend()
    plt.show()


# # 入力データ
# points = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [2, 4, 6],
#     [1, 8, 4],
#     [2, 8, 4],
#     [9, 1, 7],
#     [1, 1, 0],
#     [-1, -4, -6],
#     [-1, -2, -3],
#     [-4, -5, -6],
#     [-7, -8, -9],
#     [-10, -11, -12],
# ]

# cam1_position = np.array([1, 2, 3])
# cam1_rotation = np.array([1, 1, 1])
# cam2_position = np.array([4, 5, 6])
# cam2_rotation = np.array([0, 0, 1])

# # プロットの実行
# plot_3d_points_with_arrows(
#     points, cam1_position, cam1_rotation, cam2_position, cam2_rotation
# )


def rotation_matrix_to_rotation_vector(R):
    """回転行列を回転ベクトルに変換する."""
    theta = np.arccos((np.trace(R) - 1) / 2)

    # 特殊な場合: 回転がない場合
    if np.isclose(theta, 0):
        return np.zeros((3, 1))

    r = np.array(
        [
            [R[2, 1] - R[1, 2]],
            [R[0, 2] - R[2, 0]],
            [R[1, 0] - R[0, 1]],
        ]
    )
    r = r / (2 * np.sin(theta))
    r = np.multiply(r, theta)  # 要素ごとの積を取る
    return r
