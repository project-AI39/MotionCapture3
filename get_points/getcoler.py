import cv2
import numpy as np
import tkinter as tk
import PIL
import os
from PIL import ImageTk
from PIL import Image, ImageTk

image_path = "point_images/0.png"

# 画像の読み込み
image = cv2.imread(image_path)

# 色空間をBGRからHSVに変換
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 画像の表示
# cv2.imshow("HSV Image", hsv_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# クリックした部分のHSV色を取得する関数
def get_hsv_color(event):
    x, y = event.x, event.y
    pixel = hsv_image[y, x]
    print(f"クリックした部分のHSV色: {pixel}")


# 画像ウィンドウを作成
window = tk.Tk()

window.title("HSV Image")
window.geometry(f"{image.shape[1]}x{image.shape[0]}")

# 画像を表示するキャンバスを作成
canvas = tk.Canvas(window, width=image.shape[1], height=image.shape[0])
canvas.pack()

# 画像をキャンバスに描画
image_pil = Image.open(image_path)
image_tk = ImageTk.PhotoImage(image=image_pil)
canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

# クリックイベントをバインドする
canvas.bind("<Button-1>", get_hsv_color)

# ウィンドウを表示
window.mainloop()
