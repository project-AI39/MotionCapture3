import os
from PIL import Image
import numpy as np
import tkinter as tk
from PIL import ImageTk
import time
import cv2
import glob


def get_marker_center_coordinates(undistorted_images):
    # 色の範囲を指定する
    colors = [
        ([30, 252, 254], [31, 255, 255]),  #
        ([16, 252, 254], [18, 255, 255]),
        ([23, 252, 254], [25, 255, 255]),
        ([28, 252, 254], [29, 255, 255]),  #
        ([149, 252, 254], [151, 255, 255]),
        ([119, 252, 254], [121, 255, 255]),
        ([89, 252, 254], [91, 255, 255]),
        ([59, 252, 254], [61, 255, 255]),
        ([32, 252, 254], [34, 255, 255]),
        ([37, 252, 254], [38, 255, 255]),
        ([0, 252, 254], [1, 255, 255]),
        ([69, 252, 254], [71, 255, 255]),
        ([45, 252, 254], [47, 255, 255]),
    ]

    # 画像を読み込む
    image_files = undistorted_images

    center_coordinates_list = []
    for image in image_files:
        # 画像をHSVに変換する
        image = cv2.imread(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 各色の範囲でマスクを作成する
        masks = []
        for lower, upper in colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_image, lower, upper)
            mask = np.all((hsv_image >= lower) & (hsv_image <= upper), axis=-1)
            masks.append(mask)

        # マスクされた白黒画像を作成する
        masked_image = np.zeros_like(hsv_image)
        for mask in masks:
            masked_image[mask] = [255, 255, 255]

        # show_masked_image(masked_image)

        # 各色の範囲の中心座標を取得してリストに格納する
        center_coordinates = []
        for mask in masks:
            nonzero_pixels = np.nonzero(mask)
            center_x = np.mean(nonzero_pixels[1])
            center_y = np.mean(nonzero_pixels[0])
            center_coordinates.append([center_x, center_y])

        center_coordinates_list.append(center_coordinates)

    show(center_coordinates_list)

    return center_coordinates_list


def show_masked_image(masked_image):
    # 白黒画像を表示する
    root = tk.Tk()
    image_tk = ImageTk.PhotoImage(Image.fromarray(masked_image))
    label = tk.Label(root, image=image_tk)
    label.pack()
    root.after(2000, root.destroy)  # 2秒後にウィンドウを閉じる
    root.mainloop()


def show(center_coordinates_list):
    print("中心座標のリスト:")
    print(center_coordinates_list)
    for i, center_coordinates in enumerate(center_coordinates_list):
        # i番目のカメラの中心座標のリストを取得
        for j, center_coordinate in enumerate(center_coordinates):
            # j番目のマーカーの中心座標を取得
            print(f"{i}番目のカメラの{j}番目のマーカーの中心座標: {center_coordinate}")


# show points on image
def show_points_on_image(image, points):
    for point in points:
        center = (
            int(point[0]),
            int(point[1]),
        )  # Convert center coordinates to integers
        cv2.circle(image, center, 5, (255, 255, 255), -1)
    cv2.imshow("points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


undistorted_images = glob.glob("undistorted_point_images/*.png")
print(undistorted_images)

center_coordinates_list = get_marker_center_coordinates(undistorted_images)

for image, center_coordinates in zip(undistorted_images, center_coordinates_list):
    # 画像を読み込む
    image = cv2.imread(image)

    # 中心座標を表示する
    show_points_on_image(image, center_coordinates)
