import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Button
from tkinter import filedialog
import tkinter as tk

class Quadtree:
    def __init__(self, x, y, width, height, max_splits, level=0):
        self.boundary = (x, y, width, height)
        self.max_splits = max_splits
        self.level = level
        self.divided = False
        self.children = []

    def should_split(self, image):
        """ตรวจสอบว่าควรแบ่งพื้นที่นี้หรือไม่"""
        x, y, w, h = map(int, self.boundary)
        region = image[y:y + h, x:x + w]
        variance = np.var(region)  # คำนวณความแปรปรวนของค่าสีในพื้นที่
        return variance > 10  # หากความแปรปรวนสูงกว่าเกณฑ์ ให้แบ่งพื้นที่

    def split(self, image):
        if self.level >= self.max_splits:
            return

        if not self.should_split(image):
            return

        x, y, w, h = self.boundary
        half_w, half_h = w / 2, h / 2

        self.children = [
            Quadtree(x, y, half_w, half_h, self.max_splits, self.level + 1),
            Quadtree(x + half_w, y, half_w, half_h, self.max_splits, self.level + 1),
            Quadtree(x, y + half_h, half_w, half_h, self.max_splits, self.level + 1),
            Quadtree(x + half_w, y + half_h, half_w, half_h, self.max_splits, self.level + 1)
        ]

        for child in self.children:
            child.split(image)

        self.divided = True

    def draw(self, ax, height):
        """วาด Quadtree และแสดงสถานะ leaf node"""
        x, y, w, h = self.boundary
        y = height - y - h  # กลับพิกัด Y
        ax.plot([x, x + w], [y, y], color="blue")
        ax.plot([x, x + w], [y + h, y + h], color="blue")
        ax.plot([x, x], [y, y + h], color="blue")
        ax.plot([x + w, x + w], [y, y + h], color="blue")

        # ตรวจสอบว่าโหนดนี้เป็น leaf node หรือไม่
        if not self.divided:
            center_x = x + w / 2
            center_y = y + h / 2
            ax.text(center_x, center_y, "", color="red", fontsize=8, ha="center", va="center")

        if self.divided:
            for child in self.children:
                child.draw(ax, height)

# --- ตั้งค่าภาพสำหรับวาด ---
width, height = 256, 256
image = np.ones((height, width), dtype=np.uint8) * 255  # ตั้งค่าพื้นหลังเป็นสีขาว
drawing = False
prev_point = None

# --- ฟังก์ชันจับอีเวนต์เมาส์ ---
def on_mouse_press(event):
    global drawing, prev_point
    if event.xdata is not None and event.ydata is not None:
        drawing = True
        prev_point = (int(event.xdata), int(event.ydata))

def on_mouse_release(event):
    global drawing, image
    drawing = False
    update_quadtree()

def on_mouse_move(event):
    global prev_point, image
    if drawing and prev_point and event.xdata is not None and event.ydata is not None:
        curr_point = (int(event.xdata), int(event.ydata))

        # แปลง y พิกัดให้ตรงกับภาพ
        prev_point_inverted = (prev_point[0], height - prev_point[1])
        curr_point_inverted = (curr_point[0], height - curr_point[1])

        # วาดเส้นในภาพ (ใช้สีดำ)
        cv2.line(image, prev_point_inverted, curr_point_inverted, 0, thickness=2)  # 0 คือสีดำ
        prev_point = curr_point
        redraw_canvas()

# --- ฟังก์ชันอัพเดท Quadtree และ Redraw ---
def update_quadtree():
    """สร้าง Quadtree ใหม่และอัพเดทหน้าจอ"""
    global image, width, height
    ax.clear()
    ax.imshow(image, cmap='gray', extent=(0, width, 0, height), origin="upper", vmin=0, vmax=255)

    quadtree = Quadtree(0, 0, width, height, max_splits=6)
    quadtree.split(image)
    quadtree.draw(ax, height)

    plt.draw()

def redraw_canvas():
    """อัพเดทภาพที่กำลังวาด"""
    ax.clear()
    ax.imshow(image, cmap='gray', extent=(0, width, 0, height), origin="upper", vmin=0, vmax=255)
    plt.draw()

# --- ฟังก์ชันล้างภาพ (Clear) ---
def clear(event):
    """ล้างภาพทั้งหมด"""
    global image
    image = np.full((height, width), 255, dtype=np.uint8)  # รีเซ็ตภาพเป็นพื้นหลังขาว
    redraw_canvas()

# --- ฟังก์ชันสำหรับการโหลดรูปภาพ ---
def load_image(event):
    """ให้ผู้ใช้เลือกและโหลดรูปภาพ"""
    global image
    root = tk.Tk()
    root.withdraw()  # ซ่อนหน้าต่างหลักของ tkinter
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # อ่านภาพเป็น grayscale
        image = cv2.resize(image, (width, height))  # เปลี่ยนขนาดให้ตรงกับขนาดที่ตั้งไว้
        
        # แบ่ง Quadtree และวาดภาพ
        update_quadtree()

# --- สร้างหน้าต่างวาดภาพ ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_title("Drag to draw, release to update Quadtree")

fig.canvas.mpl_connect("button_press_event", on_mouse_press)
fig.canvas.mpl_connect("button_release_event", on_mouse_release)
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

# --- เพิ่มปุ่ม Clear และ Load Image ---
clear_ax = fig.add_axes([0.2, 0.0001, 0.25, 0.075])  # ขยายปุ่ม Clear และย้ายไปข้างล่าง
clear_button = Button(clear_ax, "Clear")
clear_button.on_clicked(clear)

load_ax = fig.add_axes([0.5, 0.0001, 0.25, 0.075])  # ขยายปุ่ม Load Image และย้ายไปข้างล่าง
load_button = Button(load_ax, "Load Image")
load_button.on_clicked(load_image)

plt.show()