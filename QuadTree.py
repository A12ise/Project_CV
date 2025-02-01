import matplotlib.pyplot as plt
import cv2
import numpy as np

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
        variance = np.var(region)  # คำนวณความแปรปรวน (variance) ของค่าสีในพื้นที่
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
        is_leaf = not self.divided
        if is_leaf:
            # แสดงสถานะ leaf node ตรงกลางของพื้นที่
            center_x = x + w / 2
            center_y = y + h / 2
            ax.text(center_x, center_y, "Leaf", color="red", fontsize=8, ha="center", va="center")

        if self.divided:
            for child in self.children:
                child.draw(ax, height)

# โหลดรูปภาพ
img_path = "D:\Study\P3T2\CV\Circle.png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image_color = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# สร้าง Quadtree
height, width = image.shape
quadtree = Quadtree(0, 0, width, height, max_splits=6)

# เริ่มการแบ่ง Quadtree
quadtree.split(image)

# วาด Quadtree และแสดงสถานะ leaf node
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap='gray', extent=(0, width, 0, height), origin="upper")
quadtree.draw(ax, height)
plt.show()

#Test Jiw