import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import io
from matplotlib.collections import LineCollection
import math 


class DrawPoseImage:
    CATEGORY = "FullersTool"
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "string_data":("STRING",{"forceinput":True}),
            "body_scale_factor":("FLOAT",{"forceinput":True, "default":1,"min":0, "max":10, "step":0.01}),
            "head_scale_factor":("FLOAT",{"forceinput":True, "default":1,"min":0, "max":10, "step":0.01}),
            "eye_scale_factor":("FLOAT",{"forceinput":True, "default":1,"min":0, "max":10, "step":0.01}),
            "dot_radius" :("FLOAT",{"forceinput":True, "default":0.8,"min":0, "max":10, "step":0.01}),
            "line_width" :("FLOAT",{"forceinput":True, "default":1,"min":0, "max":10, "step":0.01}),
            "image_alpha" :("FLOAT",{"forceinput":True, "default":0.7,"min":0, "max":1, "step":0.01}),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_pose"

    def draw_pose(self, string_data, body_scale_factor, head_scale_factor, eye_scale_factor, dot_radius, line_width, image_alpha):

        json_strings = json.loads(string_data)
        json_objects = [json.loads(s) for s in json_strings]

        tensor_images = []
        for index,json_file in enumerate(json_objects):
            img_file_path = self.draw_single_pose(index,json_file, body_scale_factor, head_scale_factor, eye_scale_factor, dot_radius, line_width, image_alpha)
            i = Image.open(img_file_path)
            #i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)
            print(f"Image shape: {image.shape}")  # 打印形状以检查
        
            tensor_images.append(image)

        batch_tensor = torch.stack(tensor_images, dim=0)
        print(batch_tensor.size())

        return (batch_tensor,)


    def draw_single_pose(self, index, keypoints,  body_scale_factor, head_scale_factor, eye_scale_factor,dot_radius, line_width, image_alpha):
        BODY_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), 
                            (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (0, 15), (14, 16), (15, 17)]
        HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), 
                            (0, 17), (17, 18), (18, 19), (19, 20)] 
        
        if len(keypoints['people'][0]) >0:
            people = keypoints['people'][0] 
            body_keypoints = people.get('pose_keypoints_2d', [])
            face_keypoints = people.get('face_keypoints_2d', [])
            left_hand_keypoints = people.get('hand_left_keypoints_2d', [])
            right_hand_keypoints = people.get('hand_right_keypoints_2d', [])

        #图像比例缩放
        #整体缩放
        body_fixed_point = self.calculate_fixed_point(body_keypoints)
        body_keypoints = self.scale_points(body_fixed_point, body_scale_factor,body_keypoints)
        face_keypoints = self.scale_points(body_fixed_point, body_scale_factor,face_keypoints)
        left_hand_keypoints = self.scale_points(body_fixed_point, body_scale_factor,left_hand_keypoints)
        right_hand_keypoints = self.scale_points(body_fixed_point, body_scale_factor,right_hand_keypoints)

        #头部缩放
        #面部点缩放 -1
        head_fixed_point = [body_keypoints[3], body_keypoints[4], body_keypoints[5]]
        face_keypoints = self.scale_points(head_fixed_point, math.sqrt(head_scale_factor), face_keypoints)
        body_keypoints = self.scale_points_by_index(head_fixed_point, math.sqrt(head_scale_factor), body_keypoints, [0,14,15,16,17])
        #面部点缩放 -2
        head_fixed_point = [body_keypoints[0], body_keypoints[1], body_keypoints[2]]
        face_keypoints = self.scale_points(head_fixed_point, math.sqrt(head_scale_factor), face_keypoints)
        body_keypoints = self.scale_points_by_index(head_fixed_point, math.sqrt(head_scale_factor), body_keypoints, [14,15,16,17])
        #眼睛大小调整
        left_eye_fixed_point = [face_keypoints[68*3],face_keypoints[68*3+1],face_keypoints[68*3+2]]
        face_keypoints = self.scale_points_by_index(left_eye_fixed_point, eye_scale_factor, face_keypoints, [36,37,38,39,40,41])
        right_eye_fixed_point = [face_keypoints[69*3],face_keypoints[69*3+1],face_keypoints[69*3+2]]
        face_keypoints = self.scale_points_by_index(right_eye_fixed_point, eye_scale_factor, face_keypoints, [42,43,44,45,46,47])


        img_width = keypoints['canvas_width']
        img_height = keypoints['canvas_height']
        print(f"image size:{img_width},{img_height}")
        # 创建一个与原图相同尺寸的图像，透明背景
        fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
        # ax.imshow(original_image, extent=[0, img_width, img_height,0])  # 绘制原图作为底图
        fig.patch.set_facecolor('black')  # 设置图像背景为黑色
        fig.patch.set_alpha(1)            # 设置图像背景不透明
        ax.set_facecolor('black')         # 设置轴背景为黑色
        ax.patch.set_alpha(1)             # 设置轴背景不透明
        ax.set_xlim(0, img_width)
        ax.set_ylim(0, img_height)
        ax.invert_yaxis()  # 翻转 y 轴以适应图像坐标系
        ax.axis('off')  # 不显示坐标轴

        self.draw_keypoints_and_connections_on_image(body_keypoints, BODY_CONNECTIONS, 'red', ax, img_width, img_height, dot_radius, line_width, image_alpha)
        self.draw_keypoints_and_connections_on_image(face_keypoints, [], 'white', ax, img_width, img_height, dot_radius, line_width, image_alpha)
        self.draw_keypoints_and_connections_on_image(left_hand_keypoints, HAND_CONNECTIONS, 'red', ax, img_width, img_height, dot_radius, line_width, image_alpha)
        self.draw_keypoints_and_connections_on_image(right_hand_keypoints, HAND_CONNECTIONS, 'red', ax, img_width, img_height, dot_radius, line_width, image_alpha)

        index = 1000+index
        save_path = f'custom_nodes/Fuller_tools/image/tem_img_{index}.jpg'
        plt.savefig(save_path, dpi=100, transparent=False)
        print(f"tem_img saved at{save_path}")

        # pil_image = self.fig_to_pil(fig)
        # tensor = self.pil_to_tensor(pil_image)
        plt.close(fig)

        return save_path

        #绘制图像
    def draw_keypoints_and_connections_on_image(self, keypoints, connections, color, ax, img_width, img_height, dot_radius, line_width, image_alpha):
        #确保有效
        if keypoints is not None:
            # 绘制关键点
            for i in range(0, len(keypoints), 3):
                # 缩放关键点坐标到原图尺寸
                x = keypoints[i]# * img_width / 512  # 将坐标缩放到原图尺寸
                y = keypoints[i + 1]# * img_height / 512  # 将坐标缩放到原图尺寸
                z = keypoints[i + 2]
                if z != 0: #置信度不为0
                    ax.scatter(x, y, color=color, s=  dot_radius*img_width/80, alpha=image_alpha)
            
            # 绘制连接
            i = 0
            for (start, end) in connections:
                if start * 3 < len(keypoints) and end * 3 < len(keypoints):
                    start_confidence = keypoints[start * 3 + 2] 
                    end_confidence =  keypoints[end * 3 + 2] 
                    if start_confidence*end_confidence != 0: #置信度不为0
                        x_start = keypoints[start * 3]# * img_width / 512
                        y_start = keypoints[start * 3 + 1]# * img_height / 512
                        x_end = keypoints[end * 3] #* img_width / 512
                        y_end = keypoints[end * 3 + 1] #* img_height / 512
                        #ax.plot([x_start, x_end], [y_start, y_end], color=plt.cm.rainbow(i / (len(connections) - 1)), linewidth = 1.5, alpha=0.5)
                        
                        # 创建渐变线段
                        num_segments = 20  # 增加分段数量
                        max_wide = line_width*4 #最宽宽度
                        min_wide = line_width*1 #最细宽度

                        x_vals = np.linspace(x_start, x_end, num_segments)
                        y_vals = np.linspace(y_start, y_end, num_segments)
                        line_segments = [
                            [(x_vals[j], y_vals[j]), (x_vals[j+1], y_vals[j+1])]
                            for j in range(num_segments - 1)
                        ]
                        # 设置渐变线宽，从两头细到中间粗
                        widths = np.linspace(min_wide, max_wide, num_segments//2).tolist() + np.linspace(max_wide, min_wide, num_segments//2).tolist()
                        
                        # 使用 LineCollection 绘制线段
                        line_collection = LineCollection(
                            line_segments,
                            linewidths=widths,
                            colors=plt.cm.rainbow(i / (len(connections) - 1)),
                            alpha=image_alpha
                        )
                        ax.add_collection(line_collection)
                        i+= 1


    #坐标转换，xy为图片的像素位置，z为置信度
    def conver_to_coor(self, keypoints):
        points_coor = []
        if keypoints is not None:
            for i in range(0, len(keypoints), 3):
                x = keypoints[i] #* img_width / 512  # 将坐标缩放到原图尺寸
                y = keypoints[i + 1] #* img_height / 512  # 将坐标缩放到原图尺寸
                z =  keypoints[i + 2]
                points_coor.append([x,y,z])
        return points_coor
    
    #找到固定点（基于最低的两个有效点的中点）
    def calculate_fixed_point(self, keypoints):
        #把数值转换为xyz坐标
        points_coor = self.conver_to_coor(keypoints)
        #提取出有效的特征点（置信度不为0）
        valid_points_coor = []
        for point in points_coor:
            if point[2] != 0:
                valid_points_coor.append(point)

        # 找到y最高的两个有效特征点
        sorted_points = sorted(valid_points_coor , key=lambda point: point[1], reverse=True)
        
        # Get the two points with the highest y values
        top_point1 = sorted_points[0]
        top_point2 = sorted_points[1]
        #print(top_point1 , top_point2)
        
        # Calculate the midpoint between these two points
        fixed_point = [(top_point1[0] + top_point2[0]) / 2, (top_point1[1] + top_point2[1]) / 2]
        # if fixed_point[1] > 512:
        #     fixed_point[1] = 512

        return fixed_point
    
    def caclulate_points_distance(self, x1,y1,x2,y2):
        len_x = x1-x2
        len_y = y1 - y2
        distance = math.sqrt( len_x * len_x + len_y * len_y )
        return distance
    
    #指定点缩放
    def scale_points_by_index(self, fixed_point, scale_factor, keypoints, points_index):
        #把数值转换为xyz坐标
        points= self.conver_to_coor(keypoints)
        scaled_points = []
        index_count = 0
        for point in points:
            scaled_x = point[0]
            scaled_y = point[1]
            for i in points_index:
                if index_count == i:
                    scaled_x = fixed_point[0] + ( point[0] - fixed_point[0] ) * scale_factor
                    scaled_y = fixed_point[1] + ( point[1] - fixed_point[1] ) * scale_factor
            scaled_points.append([scaled_x, scaled_y, point[2]])

            index_count += 1

        #将坐标格式转回list格式
        return_keypoints = []
        for point in scaled_points:
            return_keypoints.append(point[0])
            return_keypoints.append(point[1])
            return_keypoints.append(point[2])

        return return_keypoints
    
    def scale_points(self, fixed_point, scale_factor, keypoints):

        points_index = []
        if keypoints is not None:
            for i in range(len(keypoints)):
                points_index.append(i)

        return self.scale_points_by_index(fixed_point, scale_factor, keypoints, points_index)