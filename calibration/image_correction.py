import numpy as np
import glob
import cv2


def undistort_image(image, mtx, dist):
    # 画像の歪みを補正する関数
    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_mtx)
    return undistorted_image


def undistort_images(image_paths, camera_matrix_list, distortion_coefficients_list):
    # 複数の画像の歪みを補正する関数
    undistorted_images = []
    for image_path, mtx, dist in zip(
        image_paths,
        camera_matrix_list,
        distortion_coefficients_list,
    ):
        image = cv2.imread(image_path)
        undistorted_image = undistort_image(image, mtx, dist)
        undistorted_images.append(undistorted_image)

    show(undistorted_images)

    return undistorted_images


def show(undistorted_images):
    # 表示
    for undistorted_image in undistorted_images:
        cv2.imshow("Undistorted Image", undistorted_image)
        cv2.waitKey(1000)


image_paths = np.array(
    [
        "./point_images\\0.png",
        "./point_images\\1.png",
        "./point_images\\2.png",
        "./point_images\\3.png",
        "./point_images\\4.png",
        "./point_images\\5.png",
        "./point_images\\6.png",
        "./point_images\\7.png",
        "./point_images\\8.png",
    ]
)
camera_matrix_list = np.array(
    [
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
        (
            [
                [2.67126877e03, 0.00000000e00, 9.57917092e02],
                [0.00000000e00, 2.67168557e03, 5.19849867e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        ),
    ]
)
distortion_coefficients_list = np.array(
    [
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
        (
            [
                [
                    -3.77843904e-02,
                    1.25103313e00,
                    -3.16470322e-04,
                    8.48658535e-04,
                    -3.96901306e00,
                ]
            ]
        ),
    ]
)
undistort_images = undistort_images(
    image_paths, camera_matrix_list, distortion_coefficients_list
)

# save images
for i, undistort_image in enumerate(undistort_images):
    cv2.imwrite(f"./undistorted_point_images/{i}.png", undistort_image)
