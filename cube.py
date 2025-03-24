import cv2 as cv
import numpy as np
import math

from numpy.ma.core import angle


class ImgGeneration:

    def __init__(self):
        self.width, self.height = 640, 480
        self.angle_x, self.angle_y, self.angle_z = 0, 0, 0
        self.img = self.fill_bg(cv.imread("inter.png"))
        self.frame = self.generate_cube()

    def run(self):
        while True:
            self.frame = self.generate_cube()
            cv.imshow('Spinning Cube - Red Edges', self.frame)
            angle_chg = 0.02
            self.angle_x += angle_chg
            self.angle_y += angle_chg
            self.angle_z += angle_chg
            if cv.waitKey(10) == 27:
                break

        cv.destroyAllWindows()

    def fill_bg(self, img):
        mask = (img[:, :, :3] == 0).all(axis=2).astype(np.uint8) * 255
        bg = np.zeros_like(img, dtype=np.uint8)
        bg[:, :] = [255, 255, 255]
        bg = cv.bitwise_and(bg, bg, mask=mask)
        return cv.add(img, bg)

    def generate_cube(self):
        cube_vertices = np.array([
            [-1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [-1,  1,  1],
            [ 1, -1, -1],
            [ 1, -1,  1],
            [ 1,  1, -1],
            [ 1,  1,  1]
        ])

        faces = [
            (3, 2, 0, 1),  # -X face
            (6, 7, 5, 4),  # +X face
            (0, 4, 5, 1),  # -Y face
            (2, 3, 7, 6),  # +Y face
            (2, 6, 4, 0),  # -Z face
            (7, 3, 1, 5)  # +Z face
        ]

        face_normals = [
            [-1,  0,  0],
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  1,  0],
            [ 0,  0, -1],
            [ 0,  0,  1]
        ]

        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5),
            (4, 6), (5, 7), (6, 7)
        ]

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        rotation_matrix = self.rotate()
        rotated_vertices = np.dot(cube_vertices, rotation_matrix.T)

        projected_points = [self.project(v, self.width, self.height) for v in rotated_vertices]

        for edge in edges:
            pt1 = projected_points[edge[0]]
            pt2 = projected_points[edge[1]]
            cv.line(frame, pt1, pt2, (120, 120, 120), 2)

        for face, normal in zip(faces, face_normals):
            rotated_normal = rotation_matrix @ np.array(normal)
            view_dir = np.array([0, 0, -1])
            if np.dot(rotated_normal, view_dir) <= 0.2:
                continue

            pts_src = np.float32([
                [0, 0],
                [self.img.shape[1] - 1, 0],
                [self.img.shape[1] - 1, self.img.shape[0] - 1],
                [0, self.img.shape[0] - 1]
            ])

            pts_dst = np.array([projected_points[i] for i in face], dtype=np.float32)

            matrix, _ = cv.findHomography(pts_src, pts_dst)
            warped = self.project_image(self.img, matrix, (self.height, self.width))

            gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
            ret, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(mask)

            bg = cv.bitwise_and(frame, frame, mask=mask_inv)
            fg = cv.bitwise_and(warped, warped, mask=mask)
            frame = cv.add(bg, fg)

        return frame

    def project_image(self, src, H, dst_size):
        h_out, w_out = dst_size
        dst = np.zeros((h_out, w_out, 3), dtype=np.uint8)

        H_inv = np.linalg.inv(H)

        x, y = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(x)
        dst_coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T

        src_coords = H_inv @ dst_coords
        src_coords /= src_coords[2]

        sx = np.round(src_coords[0]).astype(int)
        sy = np.round(src_coords[1]).astype(int)

        valid = (
                (0 <= sx) & (sx < src.shape[1]) &
                (0 <= sy) & (sy < src.shape[0])
        )

        dst_x = x.flatten()[valid]
        dst_y = y.flatten()[valid]
        src_x = sx[valid]
        src_y = sy[valid]

        dst[dst_y, dst_x] = src[src_y, src_x]

        return dst

    def rotate(self):
        cx, cy, cz = math.cos(self.angle_x), math.cos(self.angle_y), math.cos(self.angle_z)
        sx, sy, sz = math.sin(self.angle_x), math.sin(self.angle_y), math.sin(self.angle_z)

        rot_x = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])

        rot_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        rot_z = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])

        return rot_z @ rot_y @ rot_x

    def project(self, point, width, height, scale=800):
        x, y, z = point
        z += 5
        f = scale / z
        x, y = int(x * f + width / 2), int(-y * f + height / 2)
        return (x, y)

sim = ImgGeneration()
sim.run()