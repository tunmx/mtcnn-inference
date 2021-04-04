import cv2
import numpy as np
import os
from os.path import join as path_join
import time

# model file
PNET_PROTO_NAME = 'det1.prototxt'
PNET_MODEL_NAME = 'det1.caffemodel'
RNET_PROTO_NAME = 'det2.prototxt'
RNET_MODEL_NAME = 'det2.caffemodel'
ONET_PROTO_NAME = 'det3-half.prototxt'
ONET_MODEL_NAME = 'det3-half.caffemodel'
# p-net param
PNET_STRIDE = 2
PNET_CELL_SIZE = 12
PNET_MAX_DETECT_NUM = 5000
# r-net param
RNET_INPUT_SIZE = (24, 24)
# o-net param
ONET_INPUT_SIZE = (48, 48)
# mean & std
MEAN_VAL = 127.5
STD_VAL = 0.0078125
# mini batch size
STEP_SIZE = 128


class FaceInfo:
    def __init__(self, bboxes, reg):
        self.x1 = bboxes[1]
        self.y1 = bboxes[2]
        self.x2 = bboxes[3]
        self.y2 = bboxes[4]
        self.score = bboxes[0]
        self.reg_box = reg


class MtcnnFaceDetector:
    def __init__(self, models_folder):
        self.pnet_ = cv2.dnn.readNetFromCaffe(path_join(models_folder, PNET_PROTO_NAME),
                                              path_join(models_folder, PNET_MODEL_NAME))
        self.rnet_ = cv2.dnn.readNetFromCaffe(path_join(models_folder, RNET_PROTO_NAME),
                                              path_join(models_folder, RNET_MODEL_NAME))
        self.onet_ = cv2.dnn.readNetFromCaffe(path_join(models_folder, ONET_PROTO_NAME),
                                              path_join(models_folder, ONET_MODEL_NAME))

    def _bbox_pad_square(self, bboxes, height, width):
        h, w = bboxes[:, 4] - bboxes[:, 2] + 1, bboxes[:, 3] - bboxes[:, 1] - 1
        side = np.maximum(h, w)
        bboxes[:, 1] = np.round(np.maximum(bboxes[:, 1] + (w - side) * 0.5, 0))
        bboxes[:, 2] = np.round(np.maximum(bboxes[:, 2] + (h - side) * 0.5, 0))
        bboxes[:, 3] = np.round(np.minimum(bboxes[:, 1] + side - 1, width - 1))
        bboxes[:, 4] = np.round(np.minimum(bboxes[:, 2] + side - 1, height - 1))

    def _bbox_pad(self, bboxes, height, width):
        bboxes[:, 1] = np.round(np.maximum(bboxes[:, 1], 0))
        bboxes[:, 2] = np.round(np.maximum(bboxes[:, 2], 0))
        bboxes[:, 3] = np.round(np.minimum(bboxes[:, 3], width - 1))
        bboxes[:, 4] = np.round(np.minimum(bboxes[:, 4], height - 1))

    def _landmark_regression(self, landmark, bboxes):
        h, w = bboxes[:, 4] - bboxes[:, 2] + 1, bboxes[:, 3] - bboxes[:, 1] - 1
        x1 = bboxes[:, 1]
        y1 = bboxes[:, 2]
        points = landmark.reshape(-1, 5, 2)
        # 这边有问题！！！！
        for idx in range(0, points.shape[0]):
            points[idx][:, 0] = points[idx][:, 0] * w[idx] + x1[idx]
            points[idx][:, 1] = points[idx][:, 1] * h[idx] + y1[idx]

        return points.reshape(-1, 10)

    def _bbox_regression(self, bboxes):
        """
        Fixed face boxes to add offset to anchor boxes to form new boxes.
        :param bboxes: (9, batch-size) [score, x1, y1, x2, y2, o_x1, o_y2, o_x2, o_y2]
        :param landmark: O-Net need landmark(batch-size, 10)
        :return: only the fixed face boxes or boxes and landmark is returned.
            [score, x1, y1, x2, y2] or [score, x1, y1, x2, y2, ldx1, ldy1, ldx2, ldy2......]
        """
        _, x1, y1, x2, y2, reg_x1, reg_y1, reg_x2, reg_y2 = bboxes.T
        w = x2 - x1
        h = y2 - y1
        x1 += reg_x1 * w
        y1 += reg_y1 * h
        x2 += reg_x2 * w
        y2 += reg_y2 * h

        return np.array([bboxes[:, 0], x1, y1, x2, y2])

    def _nms(self, bboxes, thresh, method_type='u'):
        x1 = bboxes[:, 1]
        y1 = bboxes[:, 2]
        x2 = bboxes[:, 3]
        y2 = bboxes[:, 4]
        scores = bboxes[:, 0]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # for i , x in enumerate(xx2):
            #     print(x, yy2[i])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            # print(w, '，', h)
            inter = w * h
            if method_type == 'u':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif method_type == 'm':
                areas_order = areas[order[1:]]
                min_v = np.minimum(areas[i], areas_order)
                ovr = inter / min_v
            else:
                print("Method type error: {}".format(method_type))
                return
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _generate_bbox(self, confidence, reg_box, scale, thresh):
        """
        Generate the face anchor boxes of MtCNN and select some invalid boxes according to the threshold value.
        :param confidence: The Detector Confidence, shape(2, height, width).Record the confidence of each anchor boxes
        :param reg_box: The Detector Offsets, shape(4, height, width).Record the offsets of each anchor boxes.
        :param scale: The scale of the image resize.
        :param thresh: Threshold of classified Confidence.
        :return: Generated faces boxes (9, batch-size) [score, x1, y1, x2, y2, o_x1, o_y2, o_x2, o_y2]
        """
        feature_map_h_, feature_map_w_ = confidence.shape[1:3]
        probability = confidence[1]
        # Find the classification confidence that is greater than the threshold
        index = np.where(probability >= thresh)
        new_probability = probability[index]
        final_candidate_info = list()
        final_offset_boxes = list()
        if new_probability.size != 0:
            new_reg_box = list()
            for idx in range(0, 4):
                new_reg_box.append(reg_box[idx][index])
            new_reg_box = np.asarray(new_reg_box).T
            feature_map_points = np.asarray(index).T
            # Generate a box based on the feature map
            # Pnet has done the pooling with the size of 2, so the need * PNET_CELL_SIZE
            x_min_map = (feature_map_points[:, 1] * PNET_STRIDE) / scale
            y_min_map = (feature_map_points[:, 0] * PNET_STRIDE) / scale
            x_max_map = (feature_map_points[:, 1] * PNET_STRIDE + PNET_CELL_SIZE - 1) / scale
            y_max_map = (feature_map_points[:, 0] * PNET_STRIDE + PNET_CELL_SIZE - 1) / scale
            final_candidate_info = np.asarray([new_probability, x_min_map, y_min_map, x_max_map, y_max_map]).T
            final_candidate_info = np.concatenate([final_candidate_info, new_reg_box], axis=1)

        return np.asarray(final_candidate_info)

    def _proposal_net(self, image, mini_size=20, threshold=0.7, factor=0.709):
        height, width, _ = image.shape
        scale = 12 / mini_size
        min_hw = min(height, width) * scale
        scales = list()
        while min_hw >= 12:
            scales.append(scale)
            min_hw *= factor
            scale *= factor
        processed_bbox = list()
        for idx, scale in enumerate(scales):
            ws = int(np.ceil(width * scale))
            hs = int(np.ceil(height * scale))
            resize_img = cv2.resize(image, (ws, hs))
            input_blob = cv2.dnn.blobFromImage(resize_img, scalefactor=STD_VAL, mean=(MEAN_VAL, MEAN_VAL, MEAN_VAL),
                                               swapRB=False)
            self.pnet_.setInput(input_blob, "data")
            prob, reg = self.pnet_.forward(["prob1", "conv4-2"])
            # prob, reg = prob[0], reg[0]
            candidate_info = self._generate_bbox(prob[0], reg[0], scale, threshold)
            if candidate_info.size != 0:
                keep = self._nms(candidate_info, 0.5)
                nms_candidate_info = candidate_info[keep]
                processed_bbox.append(nms_candidate_info)
        processed_bbox = np.concatenate(processed_bbox, axis=0)
        if processed_bbox.size != 0:
            keep = self._nms(processed_bbox, 0.5)
            res_boxes = processed_bbox[keep]
            processed_bbox = self._bbox_regression(res_boxes)
            self._bbox_pad_square(processed_bbox.T, height, width)

        return processed_bbox

    def _refine_net(self, image, previous_tensor, threshold=0.6):
        height, width, _ = image.shape
        input_images = list()
        for _, x1, y1, x2, y2 in previous_tensor.T:
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, RNET_INPUT_SIZE)
            input_images.append(roi)
        input_blob = cv2.dnn.blobFromImages(input_images, scalefactor=STD_VAL, mean=(MEAN_VAL, MEAN_VAL, MEAN_VAL),
                                            swapRB=False)
        self.rnet_.setInput(input_blob)
        # prob(batch_size, 2) reg(batch_size, 4)
        prob, reg = self.rnet_.forward(["prob1", "conv5-2"])
        # select a confidence of dimension index 1
        probability = prob[:, 1]
        index = np.where(probability >= threshold)
        confidences = probability[index][:, np.newaxis]
        offsets = reg[index]
        # pre-stage P-NET predicts out of the boxes, Confidence(dim 1) needs to be removed
        previous_bboxes = previous_tensor[1:, ].T[index]
        bboxes = np.concatenate([confidences, previous_bboxes, offsets], axis=1)
        # print("pnet out: ", bboxes.shape)
        keep = self._nms(bboxes, 0.4, 'm')
        keep_bboxes = bboxes[keep]
        # print("pnet nms out: ", keep_bboxes.shape)
        processed_bbox = self._bbox_regression(keep_bboxes)
        self._bbox_pad_square(processed_bbox.T, height, width)

        return processed_bbox

    def _output_net(self, image, previous_tensor, threshold=0.6):
        height, width, _ = image.shape
        input_images = list()
        for _, x1, y1, x2, y2 in previous_tensor.T:
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, ONET_INPUT_SIZE)
            input_images.append(roi)
        input_blob = cv2.dnn.blobFromImages(input_images, scalefactor=STD_VAL, mean=(MEAN_VAL, MEAN_VAL, MEAN_VAL),
                                            swapRB=False)
        self.onet_.setInput(input_blob)
        prob, reg, ldmark = self.onet_.forward(["prob1", "conv6-2", "conv6-3"])
        probability = prob[:, 1]
        # print(input_blob.shape)
        # print(probability.shape)
        # for idx, img in enumerate(input_images):
        #     print(probability[idx])
        #     cv2.imshow("s", img)
        #     cv2.waitKey(0)
        #     cv2.imwrite("../imgs/2.jpg", img)
        index = np.where(probability >= threshold)
        confidences = probability[index][:, np.newaxis]
        # print('shape: ', confidences)
        offsets = reg[index]
        ldmark = ldmark[index]
        # pre-stage O-NET predicts out of the boxes, Confidence(dim 1) needs to be removed
        previous_bboxes = previous_tensor[1:, ].T[index]
        bboxes = np.concatenate([confidences, previous_bboxes, offsets], axis=1)
        keep = self._nms(bboxes, 0.7, 'u')
        keep_bboxes = bboxes[keep]
        landmark = ldmark[keep]
        landmark = self._landmark_regression(landmark, keep_bboxes)
        processed_bbox = self._bbox_regression(keep_bboxes)
        self._bbox_pad(processed_bbox.T, height, width)
        # Connect the BBox to Landmark to form a final FACE location information
        # shape = (bath-size, 15)
        final_face_location = np.concatenate([processed_bbox.T, landmark], axis=1)

        return final_face_location

    def detection(self, image, mini_size=50, threshold=(0.7, 0.6, 0.6), factor=0.709, stage=3):
        result = list()
        pnet_bboxes = self._proposal_net(image, mini_size=mini_size, threshold=threshold[0], factor=factor)
        if pnet_bboxes.size > 0:
            rnet_bboxes = self._refine_net(image, previous_tensor=pnet_bboxes, threshold=threshold[1])
            if rnet_bboxes.size > 0:
                result = self._output_net(image, previous_tensor=rnet_bboxes, threshold=threshold[2])

        return result


if __name__ == '__main__':
    mtcnn = MtcnnFaceDetector('/Users/yh-mac/Desktop/face/')
    img = cv2.imread("/Users/yh-mac/Desktop/mz.jpg")
    print(img.shape)
    c = time.time()
    # pnet_bboxes = mtcnn._proposal_net(img, mini_size=50)
    # # print('pnet: ', pnet_bboxes.T)
    # rnet_bboxes = mtcnn._refine_net(img, pnet_bboxes)
    # onet_bboxes = mtcnn._output_net(img, rnet_bboxes)
    out_bboxes = mtcnn.detection(img)
    for idx, bbox in enumerate(out_bboxes):
        print(bbox)
        _, x1, y1, x2, y2 = bbox[:5]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        print(idx, _)
        # cv2.imshow("s", img)
        # cv2.waitKey(0)
        # landmark = bbox[5:]
        # (int(x), int(y)), (255, 0, 0), 2)

    print("time: ", time.time() - c)
    cv2.imshow("s", img)
    cv2.waitKey(0)
    # print(mtcnn.candidate_boxes_)
