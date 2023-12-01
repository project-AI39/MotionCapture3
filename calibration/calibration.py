import cv2
import glob
import numpy as np


def calculate_camera_parameters(image_paths, pattern_size):
    # オブジェクト座標と画像座標の対応付けを格納するリスト
    obj_points = []  # オブジェクト座標
    img_points = []  # 画像座標

    # チェスボードのコーナーを検出して対応付ける
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
                -1, 2
            )
            obj_points.append(objp)
            img_points.append(corners)

            # チェスボードのコーナーを描写
            # show1(img, pattern_size, corners, ret)

    # カメラ内部パラメータを算出
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # カメラ内部パラメータを表示
    show2(mtx, dist)

    return mtx, dist


def show1(img, pattern_size, corners, ret):
    # チェスボードのコーナーを描写
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    cv2.imshow("Chessboard Corners", img)
    cv2.waitKey(1000)


def show2(camera_matrix, distortion_coefficients):
    # カメラ内部パラメータを表示
    print("カメラ内部パラメータ:")
    print("カメラ行列:")
    print(camera_matrix)
    print("歪み係数:")
    print(distortion_coefficients)


def calculate_camera_parameters_list(calibration_image_paths_list, pattern_size):
    # カメラ行列と歪み係数のリストを格納する変数
    camera_matrix_list = []
    distortion_coefficients_list = []

    # 各キャリブレーション画像パスに対してカメラパラメータを計算し、リストに追加する
    for calibration_image_paths in calibration_image_paths_list:
        camera_matrix, distortion_coefficients = calculate_camera_parameters(
            calibration_image_paths, pattern_size
        )
        camera_matrix_list.append(camera_matrix)
        distortion_coefficients_list.append(distortion_coefficients)

    return camera_matrix_list, distortion_coefficients_list


calibration_image_paths_list = np.array(
    [
        [
            "./bord\\0\\0.png",
            "./bord\\0\\1.png",
            "./bord\\0\\2.png",
            "./bord\\0\\3.png",
            "./bord\\0\\4.png",
            "./bord\\0\\5.png",
            "./bord\\0\\6.png",
            "./bord\\0\\7.png",
            "./bord\\0\\8.png",
        ],
        [
            "./bord\\1\\0.png",
            "./bord\\1\\1.png",
            "./bord\\1\\2.png",
            "./bord\\1\\3.png",
            "./bord\\1\\4.png",
            "./bord\\1\\5.png",
            "./bord\\1\\6.png",
            "./bord\\1\\7.png",
            "./bord\\1\\8.png",
        ],
        [
            "./bord\\2\\0.png",
            "./bord\\2\\1.png",
            "./bord\\2\\2.png",
            "./bord\\2\\3.png",
            "./bord\\2\\4.png",
            "./bord\\2\\5.png",
            "./bord\\2\\6.png",
            "./bord\\2\\7.png",
            "./bord\\2\\8.png",
        ],
        [
            "./bord\\3\\0.png",
            "./bord\\3\\1.png",
            "./bord\\3\\2.png",
            "./bord\\3\\3.png",
            "./bord\\3\\4.png",
            "./bord\\3\\5.png",
            "./bord\\3\\6.png",
            "./bord\\3\\7.png",
            "./bord\\3\\8.png",
        ],
        [
            "./bord\\4\\0.png",
            "./bord\\4\\1.png",
            "./bord\\4\\2.png",
            "./bord\\4\\3.png",
            "./bord\\4\\4.png",
            "./bord\\4\\5.png",
            "./bord\\4\\6.png",
            "./bord\\4\\7.png",
            "./bord\\4\\8.png",
        ],
        [
            "./bord\\5\\0.png",
            "./bord\\5\\1.png",
            "./bord\\5\\2.png",
            "./bord\\5\\3.png",
            "./bord\\5\\4.png",
            "./bord\\5\\5.png",
            "./bord\\5\\6.png",
            "./bord\\5\\7.png",
            "./bord\\5\\8.png",
        ],
        [
            "./bord\\6\\0.png",
            "./bord\\6\\1.png",
            "./bord\\6\\2.png",
            "./bord\\6\\3.png",
            "./bord\\6\\4.png",
            "./bord\\6\\5.png",
            "./bord\\6\\6.png",
            "./bord\\6\\7.png",
            "./bord\\6\\8.png",
        ],
        [
            "./bord\\7\\0.png",
            "./bord\\7\\1.png",
            "./bord\\7\\2.png",
            "./bord\\7\\3.png",
            "./bord\\7\\4.png",
            "./bord\\7\\5.png",
            "./bord\\7\\6.png",
            "./bord\\7\\7.png",
            "./bord\\7\\8.png",
        ],
        [
            "./bord\\8\\0.png",
            "./bord\\8\\1.png",
            "./bord\\8\\2.png",
            "./bord\\8\\3.png",
            "./bord\\8\\4.png",
            "./bord\\8\\5.png",
            "./bord\\8\\6.png",
            "./bord\\8\\7.png",
            "./bord\\8\\8.png",
        ],
    ]
)
pattern_size = (9, 8)

camera_matrix_list, distortion_coefficients_list = calculate_camera_parameters_list(
    calibration_image_paths_list, pattern_size
)

print("カメラ行列リスト")
print(camera_matrix_list)
print("歪み係数リスト")
print(distortion_coefficients_list)
