import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import hashlib
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))
folder_paths.add_model_folder_path("videos", os.path.join(folder_paths.models_dir, "videos"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new, draw_aaface_by_meta
# from .retarget_pose_test import get_retarget_pose
from .retarget_pose import get_retarget_pose

class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "These models are loaded from the 'ComfyUI/models/detection' -folder",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Device to run the ONNX models on"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads ONNX models for pose and face detection. ViTPose for pose estimation and YOLO for object detection."

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):

        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }

        return (model, )

# class PoseAndFaceDetection:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "model": ("POSEMODEL",),
#                 "images": ("IMAGE",),
#                 "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
#                 "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
#             },
#             "optional": {
#                 "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optional reference image for pose retargeting"}),
#             },
#         }

#     RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", )
#     RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", )
#     FUNCTION = "process"
#     CATEGORY = "WanAnimatePreprocess"
#     DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

#     def process(self, model, images, width, height, retarget_image=None):
#         detector = model["yolo"]
#         pose_model = model["vitpose"]
#         B, H, W, C = images.shape

#         shape = np.array([H, W])[None]
#         images_np = images.numpy()

#         IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
#         IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
#         input_resolution=(256, 192)
#         rescale = 1.25

#         detector.reinit()
#         pose_model.reinit()
#         if retarget_image is not None:
#             refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
#             ref_bbox = (detector(
#                 cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
#                 shape
#                 )[0][0]["bbox"])

#             if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
#                 ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

#             center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
#             refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

#             img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
#             img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

#             ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
#             refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

#         comfy_pbar = ProgressBar(B*2)
#         progress = 0
#         bboxes = []
#         for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
#             bboxes.append(detector(
#                 cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
#                 shape
#                 )[0][0]["bbox"])
#             progress += 1
#             if progress % 10 == 0:
#                 comfy_pbar.update_absolute(progress)

#         detector.cleanup()

#         kp2ds = []
#         for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
#             if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
#                 bbox = np.array([0, 0, img.shape[1], img.shape[0]])

#             bbox_xywh = bbox
#             center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
#             img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

#             img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
#             img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

#             keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
#             kp2ds.append(keypoints)
#             progress += 1
#             if progress % 10 == 0:
#                 comfy_pbar.update_absolute(progress)

#         pose_model.cleanup()

#         kp2ds = np.concatenate(kp2ds, 0)
#         pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

#         face_images = []
#         for idx, meta in enumerate(pose_metas):
#             face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))

#             x1, x2, y1, y2 = face_bbox_for_image
#             face_image = images_np[idx][y1:y2, x1:x2]
#             face_image = cv2.resize(face_image, (512, 512))
#             face_images.append(face_image)

#         face_images_np = np.stack(face_images, 0)
#         face_images_tensor = torch.from_numpy(face_images_np)

#         if retarget_image is not None and refer_pose_meta is not None:
#             retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
#         else:
#             retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

#         bbox = np.array(bboxes[0]).flatten()
#         if bbox.shape[0] >= 4:
#             bbox_ints = tuple(int(v) for v in bbox[:4])
#         else:
#             bbox_ints = (0, 0, 0, 0)

#         key_frame_num = 4 if B >= 4 else 1
#         key_frame_step = len(pose_metas) // key_frame_num
#         key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

#         key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]

#         for key_frame_index in key_frame_index_list:
#             keypoints_body_list = []
#             body_key_points = pose_metas[key_frame_index]['keypoints_body']
#             for each_index in key_points_index:
#                 each_keypoint = body_key_points[each_index]
#                 if None is each_keypoint:
#                     continue
#                 keypoints_body_list.append(each_keypoint)

#             keypoints_body = np.array(keypoints_body_list)[:, :2]
#             wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
#             points = (keypoints_body * wh).astype(np.int32)
#             points_dict_list = []
#             for point in points:
#                 points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

#         pose_data = {
#             "retarget_image": refer_img if retarget_image is not None else None,
#             "pose_metas": retarget_pose_metas,
#             "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
#             "pose_metas_original": pose_metas,
#         }

#         return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints],)

class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "Optional reference image for pose retargeting"}),
                "t_pose_image": ("IMAGE", {"default": None, "tooltip": "Optional T-pose image for pose template"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", )
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

    def process(self, model, images, width, height, retarget_image=None, t_pose_image=None):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        tpose_pose_meta = None
        if t_pose_image is not None:
            tpose_img = resize_by_area(t_pose_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            tpose_bbox = (detector(
                cv2.resize(tpose_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if tpose_bbox is None or tpose_bbox[-1] <= 0 or (tpose_bbox[2] - tpose_bbox[0]) < 10 or (tpose_bbox[3] - tpose_bbox[1]) < 10:
                tpose_bbox = np.array([0, 0, tpose_img.shape[1], tpose_img.shape[0]])

            center, scale = bbox_from_detector(tpose_bbox, input_resolution, rescale=rescale)
            tpose_img = crop(tpose_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (tpose_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            tpose_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            tpose_pose_meta = load_pose_metas_from_kp2ds_seq(tpose_keypoints, width=t_pose_image.shape[2], height=t_pose_image.shape[1])[0]

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        face_images = []
        for idx, meta in enumerate(pose_metas):
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))

            x1, x2, y1, y2 = face_bbox_for_image
            face_image = images_np[idx][y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        if retarget_image is not None and refer_pose_meta is not None:
            # 如果有 t_pose_image，使用 tpose_pose_meta 作为模板，否则使用 pose_metas[0]
            template_pose_meta = tpose_pose_meta if tpose_pose_meta is not None else pose_metas[0]
            print(tpose_pose_meta is not None)
            retarget_pose_metas = get_retarget_pose(template_pose_meta, refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]

        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                each_keypoint = body_key_points[each_index]
                if None is each_keypoint:
                    continue
                keypoints_body_list.append(each_keypoint)

            keypoints_body = np.array(keypoints_body_list)[:, :2]
            wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
            points = (keypoints_body * wh).astype(np.int32)
            points_dict_list = []
            for point in points:
                points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints],)

class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "retarget_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the retargeted pose image is padded and resized to the target size"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "Whether to draw head keypoints"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", )
    RETURN_NAMES = ("pose_images", "drawn_pose_data", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data."

    def _convert_aapose_to_dict(self, aapose_meta):
        """将AAPoseMeta对象转换为字典格式，与load_pose_metas_from_kp2ds_seq的输出格式一致
        
        Args:
            aapose_meta: AAPoseMeta对象，包含像素坐标
        """
        # 将像素坐标转换为归一化坐标（0-1范围）
        # 注意：AAPoseMeta中坐标是像素坐标，基于其自身的width和height
        
        # 身体关键点：像素坐标 -> 归一化坐标
        kps_body_normalized = aapose_meta.kps_body.copy().astype(np.float64)
        kps_body_normalized[:, 0] /= aapose_meta.width
        kps_body_normalized[:, 1] /= aapose_meta.height
        keypoints_body = np.concatenate([kps_body_normalized, aapose_meta.kps_body_p[:, None]], axis=1)
        
        # 左手关键点：像素坐标 -> 归一化坐标
        kps_lhand_normalized = aapose_meta.kps_lhand.copy().astype(np.float64)
        kps_lhand_normalized[:, 0] /= aapose_meta.width
        kps_lhand_normalized[:, 1] /= aapose_meta.height
        keypoints_left_hand = np.concatenate([kps_lhand_normalized, aapose_meta.kps_lhand_p[:, None]], axis=1)
        
        # 右手关键点：像素坐标 -> 归一化坐标
        kps_rhand_normalized = aapose_meta.kps_rhand.copy().astype(np.float64)
        kps_rhand_normalized[:, 0] /= aapose_meta.width
        kps_rhand_normalized[:, 1] /= aapose_meta.height
        keypoints_right_hand = np.concatenate([kps_rhand_normalized, aapose_meta.kps_rhand_p[:, None]], axis=1)
        
        # 面部关键点：像素坐标 -> 归一化坐标（如果存在）
        if hasattr(aapose_meta, 'kps_face') and aapose_meta.kps_face is not None:
            kps_face_normalized = aapose_meta.kps_face.copy().astype(np.float64)
            kps_face_normalized[:, 0] /= aapose_meta.width
            kps_face_normalized[:, 1] /= aapose_meta.height
            keypoints_face = np.concatenate([kps_face_normalized, aapose_meta.kps_face_p[:, None]], axis=1)
        else:
            # 如果没有面部关键点，创建一个空数组
            keypoints_face = np.zeros((68, 3), dtype=np.float64)
        
        meta_dict = {
            "width": aapose_meta.width,
            "height": aapose_meta.height,
            "keypoints_body": keypoints_body,
            "keypoints_left_hand": keypoints_left_hand,
            "keypoints_right_hand": keypoints_right_hand,
            "keypoints_face": keypoints_face,
        }
        
        return meta_dict

    def _transform_keypoints_for_final_image(self, meta, original_width, original_height, final_width, final_height, 
                                              use_retarget_resize, lowestY, crop_target_image, retarget_padding):
        """
        将关键点坐标从原始尺寸变换到最终绘制后图像的坐标系统
        
        Args:
            meta: AAPoseMeta对象，包含原始关键点
            original_width, original_height: 绘制画布的尺寸
            final_width, final_height: 最终输出图像的尺寸
            use_retarget_resize: 是否使用resize_to_bounds
            lowestY: padding_resize的lowestY参数
            crop_target_image: resize_to_bounds的参考图像
            retarget_padding: extra_padding参数
        
        Returns:
            变换后的pose meta字典
        """
        # 首先将AAPoseMeta转换为字典格式（归一化坐标）
        meta_dict = self._convert_aapose_to_dict(meta)
        
        # 关键点当前是基于meta.width x meta.height的归一化坐标
        # 需要变换到最终图像的归一化坐标
        
        # 计算变换参数
        if use_retarget_resize:
            # 模拟resize_to_bounds的变换逻辑
            # 这里简化处理：假设居中放置并保持宽高比
            # 实际的resize_to_bounds会先裁剪再缩放
            ori_aspect = original_width / original_height
            target_aspect = final_width / final_height
            
            if ori_aspect > target_aspect:
                # 宽度填满
                new_width = final_width
                new_height = int(final_width / ori_aspect)
            else:
                # 高度填满
                new_height = final_height
                new_width = int(final_height * ori_aspect)
            
            x_offset = (final_width - new_width) // 2
            y_offset = (final_height - new_height) // 2
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            
        else:
            # padding_resize的变换逻辑
            if lowestY > 0:
                # 使用裁切和贴底模式
                if lowestY <= 1.0:
                    lowestY_int = int(lowestY * original_height)
                else:
                    lowestY_int = int(lowestY)
                
                cropped_height = lowestY_int
                cropped_width = original_width
                
                aspect_ratio = cropped_width / cropped_height
                target_aspect = final_width / final_height
                
                if aspect_ratio > target_aspect:
                    new_width = final_width
                    new_height = int(final_width / aspect_ratio)
                else:
                    new_height = final_height
                    new_width = int(final_height * aspect_ratio)
                
                # 贴底并水平居中
                y_offset = final_height - new_height
                x_offset = (final_width - new_width) // 2
                
                # 缩放因子（相对于裁切后的图像）
                scale_x = new_width / cropped_width
                scale_y = new_height / cropped_height
                
            else:
                # 原始padding_resize逻辑
                ori_aspect = original_width / original_height
                target_aspect = final_width / final_height
                
                if ori_aspect > target_aspect:
                    new_width = final_width
                    new_height = int(final_width / ori_aspect)
                    x_offset = 0
                    y_offset = (final_height - new_height) // 2
                else:
                    new_height = final_height
                    new_width = int(final_height * ori_aspect)
                    x_offset = (final_width - new_width) // 2
                    y_offset = 0
                
                scale_x = new_width / original_width
                scale_y = new_height / original_height
        
        # 应用变换到所有关键点类型
        def transform_keypoints(kps, conf):
            """变换关键点坐标"""
            transformed = []
            for kp, c in zip(kps, conf):
                # kp是归一化坐标(0-1)，基于meta.width和meta.height
                # 转换为像素坐标（基于原始绘制尺寸）
                x_pixel = kp[0] * meta.width
                y_pixel = kp[1] * meta.height
                
                # 应用变换到最终图像坐标
                new_x_pixel = x_pixel * scale_x + x_offset
                new_y_pixel = y_pixel * scale_y + y_offset
                
                # 转换回归一化坐标（基于最终图像尺寸）
                new_x_norm = new_x_pixel / final_width
                new_y_norm = new_y_pixel / final_height
                
                transformed.append([new_x_norm, new_y_norm, c])
            
            return np.array(transformed, dtype=np.float64)
        
        # 变换身体关键点
        body_kps_norm = meta_dict['keypoints_body'][:, :2]
        body_conf = meta_dict['keypoints_body'][:, 2]
        transformed_body = transform_keypoints(body_kps_norm, body_conf)
        
        # 变换左手关键点
        lhand_kps_norm = meta_dict['keypoints_left_hand'][:, :2]
        lhand_conf = meta_dict['keypoints_left_hand'][:, 2]
        transformed_lhand = transform_keypoints(lhand_kps_norm, lhand_conf)
        
        # 变换右手关键点
        rhand_kps_norm = meta_dict['keypoints_right_hand'][:, :2]
        rhand_conf = meta_dict['keypoints_right_hand'][:, 2]
        transformed_rhand = transform_keypoints(rhand_kps_norm, rhand_conf)
        
        # 变换面部关键点
        face_kps_norm = meta_dict['keypoints_face'][:, :2]
        face_conf = meta_dict['keypoints_face'][:, 2]
        transformed_face = transform_keypoints(face_kps_norm, face_conf)
        
        # 创建变换后的meta字典
        transformed_meta = {
            "width": final_width,
            "height": final_height,
            "keypoints_body": transformed_body,
            "keypoints_left_hand": transformed_lhand,
            "keypoints_right_hand": transformed_rhand,
            "keypoints_face": transformed_face,
        }
        
        return transformed_meta

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(canvas, meta, draw_hand=draw_hand, draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width, lowestY=-1.0)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        # 将AAPoseMeta对象转换为字典格式，并应用变换使坐标对应最终输出图像
        pose_metas_dict_final = []
        for idx, meta in enumerate(pose_metas):
            if isinstance(meta, AAPoseMeta):
                # 变换关键点坐标到最终图像坐标系统
                transformed_meta = self._transform_keypoints_for_final_image(
                    meta, width, height, width, height,
                    use_retarget_resize, -1.0, crop_target_image, retarget_padding
                )
                
                if idx == 0:
                    # 打印第一帧的调试信息
                    print(f"[DrawViTPose] Frame 0 - 原始AAPoseMeta尺寸: {meta.width}x{meta.height}")
                    print(f"[DrawViTPose] Frame 0 - 绘制画布尺寸: {width}x{height}")
                    print(f"[DrawViTPose] Frame 0 - 最终输出尺寸: {width}x{height}")
                    print(f"[DrawViTPose] Frame 0 - 变换后身体关键点示例（归一化）: {transformed_meta['keypoints_body'][0][:2]}")
                
                pose_metas_dict_final.append(transformed_meta)
            else:
                # 如果已经是字典，也需要变换
                # 这里假设字典格式也需要变换（如果来自之前的处理）
                pose_metas_dict_final.append(meta)
        
        # 创建绘制后的pose data
        # pose_metas_original现在包含的是变换后的坐标，对应最终输出图像
        drawn_pose_data = {
            "retarget_image": retarget_image,
            "pose_metas": pose_metas,  # 保留AAPoseMeta格式，以便其他可能需要的节点使用
            "refer_pose_meta": pose_data.get("refer_pose_meta", None),
            "pose_metas_original": pose_metas_dict_final,  # 变换后的字典格式，坐标对应最终输出图像
        }

        return (pose_images_tensor, drawn_pose_data, )

class PoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("prompt", "retarget_prompt", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Generates text prompts for pose retargeting based on visibility of arms and legs in the template pose. Originally used for Flux Kontext"

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the person to face forward.", "Change the person to face forward.", )
        tpl_pose_metas = pose_data["pose_metas_original"]
        arm_visible = False
        leg_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)
            if np.any(tpl_keypoints[3]) != 0 or np.any(tpl_keypoints[4]) != 0 or np.any(tpl_keypoints[6]) != 0 or np.any(tpl_keypoints[7]) != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if np.any(tpl_keypoints[9]) != 0 or np.any(tpl_keypoints[12]) != 0 or np.any(tpl_keypoints[10]) != 0 or np.any(tpl_keypoints[13]) != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return (tpl_prompt, refer_prompt, )

class GetLowestKeypointY:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "frame_index": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "Which frame to extract the lowest point from. Use -1 to find the global lowest point across all frames"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum confidence threshold for keypoints"}),
                "use_normalized": ("BOOLEAN", {"default": False, "tooltip": "Return normalized coordinate (0-1) instead of pixel coordinate"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks for compensation. Set to -1 for auto (same as DrawViTPose)"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("lowest_y", "info",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Extracts the Y coordinate of the lowest body keypoint from pose data (excluding hands, face, and arms - only torso and legs)."

    def process(self, pose_data, frame_index, confidence_threshold, use_normalized, body_stick_width):
        # 使用原始的pose_metas
        pose_metas_original = pose_data.get("pose_metas_original", None)
        
        if pose_metas_original is None or len(pose_metas_original) == 0:
            return (0.0, "Error: No pose data available",)
        
        # 如果frame_index为负数，则遍历所有帧找全局最低点
        if frame_index < 0:
            return self._process_all_frames(pose_metas_original, confidence_threshold, use_normalized, body_stick_width)
        
        # 确保frame_index不越界
        frame_index = min(frame_index, len(pose_metas_original) - 1)
        
        meta = pose_metas_original[frame_index]
        
        lowest_y = -1
        lowest_point_info = None
        
        # 只检查身体关键点
        keypoints_to_check = {
            'body': meta['keypoints_body'],
        }
        
        width = meta['width']
        height = meta['height']
        
        # 遍历所有关键点找到最低点
        # 关键点索引：0-Nose, 1-Neck, 2-RShoulder, 3-RElbow, 4-RWrist, 5-LShoulder, 6-LElbow, 7-LWrist,
        #           8-RHip, 9-RKnee, 10-RAnkle, 11-LHip, 12-LKnee, 13-LAnkle, 14-REye, 15-LEye, 16-REar, 17-LEar, 18-LToe, 19-RToe
        # 排除手臂关键点：3-RElbow, 4-RWrist, 6-LElbow, 7-LWrist
        # 保留躯干、腿部和脚部：2,5(肩), 8,11(髋), 9,12(膝), 10,13(踝), 18,19(脚趾)
        excluded_body_indices = {3, 4, 6, 7}  # 排除肘部和腕部
        
        for kp_type, keypoints in keypoints_to_check.items():
            keypoints_array = np.array(keypoints)
            for i, kp in enumerate(keypoints_array):
                if len(kp) < 3:
                    continue
                
                # 排除手臂关键点
                if kp_type == 'body' and i in excluded_body_indices:
                    continue
                    
                x, y, confidence = float(kp[0]), float(kp[1]), float(kp[2])
                
                # 过滤低置信度的点
                if confidence < confidence_threshold:
                    continue
                
                # 验证坐标的有效性：必须在0-1范围内（归一化坐标）
                # 排除无效点，如(0,0)填充值、负数或超出范围的坐标
                if x < 0.001 or y < 0.001 or x > 1.0 or y > 1.0:
                    continue
                
                # 找最大y值（图像坐标系中y越大越低）
                if y > lowest_y:
                    lowest_y = y
                    lowest_point_info = {
                        'x_normalized': x,
                        'y_normalized': y,
                        'x_pixel': int(x * width),
                        'y_pixel': int(y * height),
                        'type': kp_type,
                        'index': i,
                        'confidence': confidence
                    }
        
        if lowest_point_info is None:
            return (0.0, f"Error: No valid keypoints found with confidence >= {confidence_threshold}",)
        
        # 计算stickwidth补偿（与DrawViTPose保持一致）
        H, W = height, width
        if body_stick_width == -1:
            # 自动计算（默认v2模式）
            stickwidth = max(int(min(H, W) / 200), 1)
        else:
            stickwidth = body_stick_width
        
        # 返回y坐标（加上stickwidth补偿以实现精确贴底）
        if use_normalized:
            # 归一化坐标补偿
            stick_compensation = stickwidth / H
            result_y = lowest_point_info['y_normalized'] + stick_compensation
            info_str = (f"Frame {frame_index}: Lowest point at normalized y={result_y:.4f} "
                       f"(keypoint={lowest_point_info['y_normalized']:.4f} + stick_compensation={stick_compensation:.4f}, "
                       f"image_size={W}x{H}, "
                       f"type={lowest_point_info['type']}, index={lowest_point_info['index']}, "
                       f"confidence={lowest_point_info['confidence']:.3f})")
        else:
            # 像素坐标补偿
            result_y = float(lowest_point_info['y_pixel']) + stickwidth
            info_str = (f"Frame {frame_index}: Lowest point at pixel y={result_y:.1f} "
                       f"(keypoint={lowest_point_info['y_pixel']} + stick_width={stickwidth}, "
                       f"image_size={W}x{H}, "
                       f"type={lowest_point_info['type']}, index={lowest_point_info['index']}, "
                       f"confidence={lowest_point_info['confidence']:.3f})")
        
        return (result_y, info_str,)
    
    def _process_all_frames(self, pose_metas_original, confidence_threshold, use_normalized, body_stick_width):
        """遍历所有帧，找到全局最低点（仅身体关键点）"""
        global_lowest_y = -1
        global_lowest_info = None
        
        # 遍历所有帧
        for frame_idx, meta in enumerate(pose_metas_original):
            # 只检查身体关键点
            keypoints_to_check = {
                'body': meta['keypoints_body'],
            }
            
            width = meta['width']
            height = meta['height']
            
            # 遍历当前帧的所有关键点
            # 关键点索引：0-Nose, 1-Neck, 2-RShoulder, 3-RElbow, 4-RWrist, 5-LShoulder, 6-LElbow, 7-LWrist,
            #           8-RHip, 9-RKnee, 10-RAnkle, 11-LHip, 12-LKnee, 13-LAnkle, 14-REye, 15-LEye, 16-REar, 17-LEar, 18-LToe, 19-RToe
            # 排除手臂关键点：3-RElbow, 4-RWrist, 6-LElbow, 7-LWrist
            excluded_body_indices = {3, 4, 6, 7}  # 排除肘部和腕部
            
            for kp_type, keypoints in keypoints_to_check.items():
                keypoints_array = np.array(keypoints)
                for i, kp in enumerate(keypoints_array):
                    if len(kp) < 3:
                        continue
                    
                    # 排除手臂关键点
                    if kp_type == 'body' and i in excluded_body_indices:
                        continue
                        
                    x, y, confidence = float(kp[0]), float(kp[1]), float(kp[2])
                    
                    # 过滤低置信度的点
                    if confidence < confidence_threshold:
                        continue
                    
                    # 验证坐标的有效性：必须在0-1范围内（归一化坐标）
                    # 排除无效点，如(0,0)填充值、负数或超出范围的坐标
                    if x < 0.001 or y < 0.001 or x > 1.0 or y > 1.0:
                        continue
                    
                    # 找最大y值（图像坐标系中y越大越低）
                    if y > global_lowest_y:
                        global_lowest_y = y
                        global_lowest_info = {
                            'frame': frame_idx,
                            'x_normalized': x,
                            'y_normalized': y,
                            'x_pixel': int(x * width),
                            'y_pixel': int(y * height),
                            'type': kp_type,
                            'index': i,
                            'confidence': confidence
                        }
        
        if global_lowest_info is None:
            return (0.0, f"Error: No valid keypoints found with confidence >= {confidence_threshold} in all frames",)
        
        # 计算stickwidth补偿（使用找到最低点的那一帧的尺寸）
        frame_with_lowest = pose_metas_original[global_lowest_info['frame']]
        H, W = frame_with_lowest['height'], frame_with_lowest['width']
        if body_stick_width == -1:
            # 自动计算（默认v2模式）
            stickwidth = max(int(min(H, W) / 200), 1)
        else:
            stickwidth = body_stick_width
        
        # 返回全局最低点的y坐标（加上stickwidth补偿）
        if use_normalized:
            stick_compensation = stickwidth / H
            result_y = global_lowest_info['y_normalized'] + stick_compensation
            info_str = (f"Global lowest point across all {len(pose_metas_original)} frames: "
                       f"Frame {global_lowest_info['frame']}, normalized y={result_y:.4f} "
                       f"(keypoint={global_lowest_info['y_normalized']:.4f} + stick_compensation={stick_compensation:.4f}, "
                       f"image_size={W}x{H}, "
                       f"type={global_lowest_info['type']}, index={global_lowest_info['index']}, "
                       f"confidence={global_lowest_info['confidence']:.3f})")
        else:
            result_y = float(global_lowest_info['y_pixel']) + stickwidth
            info_str = (f"Global lowest point across all {len(pose_metas_original)} frames: "
                       f"Frame {global_lowest_info['frame']}, pixel y={result_y:.1f} "
                       f"(keypoint={global_lowest_info['y_pixel']} + stick_width={stickwidth}, "
                       f"image_size={W}x{H}, "
                       f"type={global_lowest_info['type']}, index={global_lowest_info['index']}, "
                       f"confidence={global_lowest_info['confidence']:.3f})")
        
        return (result_y, info_str,)

class CropCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_images": ("IMAGE",),
                "lowestY": ("FLOAT", {"default": 0.0, "min": 0.0, "tooltip": "Y coordinate of the lowest point for cropping"}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Target height after padding"}),
                "buffer": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Buffer value as percentage of target height"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Crops pose images at lowestY and pads upward to reach target height."

    def process(self, pose_images, lowestY, target_height, buffer):
        # pose_images shape: (B, H, W, C)
        B, H, W, C = pose_images.shape
        
        # 将lowestY转换为整数像素值
        # 如果lowestY <= 1.0，视为归一化坐标；否则视为像素坐标
        if lowestY <= 1.0:
            # 归一化坐标，转换为像素
            crop_y = int(lowestY * H)
        else:
            # 像素坐标
            crop_y = int(lowestY)
        
        # 计算缓冲区高度（目标高度的百分比）
        buffer_height = int(buffer * target_height)
        
        # 实际裁剪位置：crop_y + 缓冲值
        actual_crop_y = crop_y + buffer_height
        
        # 确保actual_crop_y在有效范围内
        actual_crop_y = max(1, min(actual_crop_y, H))
        
        # 裁剪：保留从0到actual_crop_y的部分
        cropped_images = pose_images[:, :actual_crop_y, :, :]
        
        # 计算需要padding的高度
        cropped_height = actual_crop_y
        if cropped_height < target_height:
            # 需要向上padding
            pad_height = target_height - cropped_height
            
            # 创建padding（黑色，值为0）
            # shape: (B, pad_height, W, C)
            padding = torch.zeros((B, pad_height, W, C), dtype=pose_images.dtype, device=pose_images.device)
            
            # 向上padding：将padding放在上方，原图放在下方
            result_images = torch.cat([padding, cropped_images], dim=1)
        elif cropped_height > target_height:
            # 如果裁剪后高度仍大于目标高度，从底部截取target_height的高度
            result_images = cropped_images[:, -target_height:, :, :]
        else:
            # 高度正好相等
            result_images = cropped_images
        
        return (result_images,)

class LoadVideoFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "输入图像序列"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "要提取的帧索引（从0开始）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING",)
    RETURN_NAMES = ("image", "total_frames", "info",)
    FUNCTION = "load_frame"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "从图像序列中选择并输出指定索引的帧。输入为图像序列，输出为单张图像。"

    def load_frame(self, images, frame_index):
        # images shape: (B, H, W, C) 其中B是总帧数
        B, H, W, C = images.shape
        total_frames = B
        
        # 检查帧索引是否有效
        if frame_index >= total_frames:
            frame_index = total_frames - 1
            print(f"警告: 帧索引超出范围，已调整为最后一帧 (索引: {frame_index})")
        
        if frame_index < 0:
            frame_index = 0
            print(f"警告: 帧索引为负数，已调整为第一帧 (索引: {frame_index})")
        
        # 提取指定索引的帧
        selected_frame = images[frame_index:frame_index+1]  # 保持batch维度
        
        # 生成信息字符串
        info = (f"图像序列信息:\n"
               f"总帧数: {total_frames}\n"
               f"分辨率: {W}x{H}\n"
               f"当前帧: {frame_index}\n"
               f"输出图像形状: {selected_frame.shape}")
        
        print(info)
        
        return (selected_frame, total_frames, info)

class RepeatImageToFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入单张图像"}),
                "frame_count": ("INT", {"default": 24, "min": 1, "max": 9999, "step": 1, "tooltip": "重复生成的帧数"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING",)
    RETURN_NAMES = ("frames", "total_frames", "info",)
    FUNCTION = "repeat_image"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "将单张图像重复指定帧数，生成视频序列帧。输入为单张图像，输出为图像序列。"

    def repeat_image(self, image, frame_count):
        # image shape: (B, H, W, C) 通常B=1
        B, H, W, C = image.shape
        
        # 如果输入已经是多帧，只取第一帧
        if B > 1:
            image = image[0:1]
            print(f"警告: 输入图像包含多帧，仅使用第一帧")
        
        # 重复图像
        repeated_frames = image.repeat(frame_count, 1, 1, 1)
        
        # 生成信息字符串
        info = (f"重复图像信息:\n"
               f"输入图像分辨率: {W}x{H}\n"
               f"重复帧数: {frame_count}\n"
               f"输出序列形状: {repeated_frames.shape}")
        
        print(info)
        
        return (repeated_frames, frame_count, info)

NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "DrawViTPose": DrawViTPose,
    "PoseRetargetPromptHelper": PoseRetargetPromptHelper,
    "GetLowestKeypointY": GetLowestKeypointY,
    "CropCorrection": CropCorrection,
    "LoadVideoFrame": LoadVideoFrame,
    "RepeatImageToFrames": RepeatImageToFrames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "DrawViTPose": "Draw ViT Pose",
    "PoseRetargetPromptHelper": "Pose Retarget Prompt Helper",
    "GetLowestKeypointY": "Get Lowest Keypoint Y",
    "CropCorrection": "Crop Correction",
    "LoadVideoFrame": "Load Video Frame",
    "RepeatImageToFrames": "Repeat Image to Frames",
}

