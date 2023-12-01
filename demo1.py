import numpy as np
import cv2
from scipy.linalg import svd
import dd

# カメラ1内部パラメータ
A1 = cam1_camera_matrix = np.array(
    [
        [2.67126877e03, 0.00000000e00, 9.57917092e02],
        [0.00000000e00, 2.67168557e03, 5.19849867e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# カメラ2内部パラメータ
A2 = cam2_camera_matrix = np.array(
    [
        [2.67126877e03, 0.00000000e00, 9.57917092e02],
        [0.00000000e00, 2.67168557e03, 5.19849867e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# 対応点2D座標(x,y)リスト
# 画像1の特徴点
keypoints1 = np.array(
    [
        [547.1563275434244, 129.1439205955335],
        [602.1808873720137, 183.5551763367463],
        [547.1986809563067, 949.5375103050288],
        [602.1073446327683, 895.425988700565],
        [1373.1917577796467, 129.72918418839362],
        [1318.5394144144145, 184.00225225225225],
        [1373.235841081995, 949.2831783601015],
        [1318.4018161180477, 894.9704880817253],
        [960.1367690782953, 158.16352824578792],
        [960.3856722276741, 348.6182531894014],
        [960.3470588235294, 539.1539215686274],
        [1152.3682132280355, 539.3899308983218],
        [1344.000975609756, 539.3990243902439],
    ]
)
# 画像2の特徴点
keypoints2 = np.array(
    [
        [682.0972222222222, 278.9375],
        [873.8262032085562, 295.7486631016043],
        [682.1353211009174, 800.0206422018349],
        [873.9312169312169, 783.2222222222222],
        [1067.6677631578948, 237.3371710526316],
        [1258.7410881801127, 259.72232645403375],
        [1067.7107438016528, 841.5619834710744],
        [1258.813909774436, 819.2669172932331],
        [960.1551362683438, 269.6687631027254],
        [960.2348008385744, 404.50314465408803],
        [960.1721991701245, 539.1597510373444],
        [1059.8174757281554, 539.1825242718446],
        [1166.9876543209878, 539.2257495590829],
    ]
)


def normalize_points(homo_points):
    """画像座標の1,2次元成分を正規化する."""
    mean = homo_points[:2].mean(axis=1)
    scale = np.sqrt(2) / np.mean(
        np.linalg.norm(homo_points[:2] - mean.reshape(2, 1), axis=0)
    )

    mean = np.append(mean, 0)
    T = np.array(
        [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
    )

    homo_points = T @ homo_points
    return homo_points, T


def find_fundamental_matrix(points1, points2, verbose=0):
    """正規化8点アルゴリズムでF行列を推定する."""

    assert points1.shape[1] == points2.shape[1] == 2
    assert points1.shape[0] >= 8

    points1 = np.array([(kpt[0], kpt[1], 1) for kpt in points1]).T.astype(np.float64)
    points2 = np.array([(kpt[0], kpt[1], 1) for kpt in points2]).T.astype(np.float64)

    # 正規化
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    # エピポーラ拘束式
    A = np.zeros((points1.shape[1], 9), dtype=np.float64)
    for i in range(points1.shape[1]):
        A[i] = [
            points1_norm[0, i] * points2_norm[0, i],
            points1_norm[1, i] * points2_norm[0, i],
            points2_norm[0, i],
            points1_norm[0, i] * points2_norm[1, i],
            points1_norm[1, i] * points2_norm[1, i],
            points2_norm[1, i],
            points1_norm[0, i],
            points1_norm[1, i],
            1.0,
        ]

    # 特異値分解で連立方程式を解く
    U, S, Vt = svd(A, lapack_driver="gesvd")
    # S, U, Vt = cv2.SVDecomp(A.T @ A)  # OpenCVを使う場合

    if verbose >= 1:
        print("SVD decomposition eigen values:", S)
    F = Vt[-1].reshape(3, 3)

    # 最小特異値を厳密に0とし、F行列をランク2にする
    U, S, Vt = svd(F, lapack_driver="gesvd")
    # S, U, Vt = cv2.SVDecomp(F)  # OpenCVを使う場合

    if verbose >= 1:
        print("Rank SVD decomposition eigen values:", S)

    S = S.reshape(-1)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # 正規化を戻す
    F = T2.T @ F @ T1
    # f_33を1にする
    F = F / F[2, 2]

    return F


def _triangulate_one_point(P1, P2, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    X = np.vstack(
        [
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :],
        ]
    )
    _, _, Vt = svd(X)
    x_w = Vt[-1]
    x_w = x_w / x_w[-1]
    return x_w


def triangulate(P1, P2, points1, points2):
    assert points1.shape == points2.shape
    assert points1.ndim <= 2

    if points1.ndim == 1:
        # 2次元にする
        points1 = points1.reshape(1, -1)
        points2 = points2.reshape(1, -1)

    if points1.shape[1] == 3:
        # 同次座標の場合
        points1 = points1[:, :2]
        points2 = points2[:, :2]

    X_w = []
    for pt1, pt2 in zip(points1, points2):
        x_w = _triangulate_one_point(P1, P2, pt1, pt2)
        X_w.append(x_w)
    X_w = np.vstack(X_w)

    return x_w


def recover_pose(E, pts1, pts2):
    assert E.shape == (3, 3)
    assert len(pts1) == len(pts2)

    # 同次座標にする
    if pts1.shape[1] == 2:
        pts1 = np.column_stack([pts1, np.ones(len(pts1))])
    if pts2.shape[1] == 2:
        pts2 = np.column_stack([pts2, np.ones(len(pts2))])

    # SVDを実行
    U, _, Vt = svd(E)

    # 右手系にする
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt = -Vt

    # 回転行列は２つの候補がある
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # 並進ベクトルはE.Tの固有値0の固有ベクトルとして求められる(up to scale)
    # 正負の不定性で二つの候補がある
    t1 = U[:, 2]
    t2 = -U[:, 2]

    # 4つの姿勢候補がある
    pose_candidates = [[R1, t1], [R1, t2], [R2, t1], [R2, t2]]

    # 正しい姿勢を選択する
    # 各姿勢候補について、両カメラの前に再構成された3次元点の数をカウント
    # 一番カウントが多いのが正しい姿勢
    # 自然画像からの3次元最高性は誤差も大きいため、カウントで判断する
    front_count = []
    for _R, _t in pose_candidates:
        count = 0
        for pt1, pt2 in zip(pts1, pts2):
            P1 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            P2 = np.column_stack([_R, _t])
            x_w = triangulate(P1, P2, pt1[:2], pt2[:2])
            x_c_1 = x_w[:3]
            x_c_2 = np.dot(_R, x_c_1) + _t
            if (x_c_1[-1] > 0) and (x_c_2[-1] > 0):
                count += 1
        front_count.append(count)
    R, t = pose_candidates[int(np.argmax(front_count))]
    return R, t


def recover_pose_opencv(E, points1, points2):
    n_points, R, t, mask = cv2.recoverPose(E, points1, points2)
    return R, t


# 正しいカメラパラメータを使って三角測量を行う
def triangulate2(pts1, pts2, A1, A2, R, t):
    # カメラ座標系をワールド座標系に変換する行列
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = np.hstack((R1, t1))

    R2 = R
    t2 = t.reshape((3, 1))
    P2 = np.hstack((R2, t2))

    # カメラ行列をプロジェクション行列に変換
    proj_mat1 = A1 @ P1
    proj_mat2 = A2 @ P2

    # 三角測量
    points_4d_homogeneous = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1.T, pts2.T)
    points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3, :]

    # 3D座標を取得
    points_3d = points_3d_homogeneous[:3, :].T

    return points_3d


#################################################################################################
# Parameters ---------------
seed = 0
verbose = 0

# F行列を推定する
keypoints1 = keypoints1.astype(np.float64)
keypoints2 = keypoints2.astype(np.float64)
F = find_fundamental_matrix(keypoints1, keypoints2, verbose=verbose)
# E行列への変換
E = A2.T @ F @ A1
R, t = recover_pose(E, keypoints1, keypoints2)

# R1 = np.eye(3)
# t1 = np.zeros((3, 1))
# P1 = np.hstack((R1, t1))

# R2 = R
# t2 = t.reshape((3, 1))
# P2 = np.hstack((R2, t2))
points_3d = triangulate2(keypoints1, keypoints2, A1, A2, R, t)
# points_3d = triangulate(P1, P2, keypoints1, keypoints2)

print("計算された3D座標:")
print(points_3d)

print(t)
print(R)

dd.plot_3d_points_with_arrows(
    points_3d,
    np.zeros((3, 1)),
    [0, 0, 0],
    t,
    dd.rotation_matrix_to_rotation_vector(R),
)
