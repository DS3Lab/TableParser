import logging
import logging.config
import os

#import mrcnn.model as modellib

#from docparser.utils.data_utils import DocsDataset
#from docparser.utils.experiment_utils import DocparserDefaultConfig, TimeHistory


import argparse
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor


#from detectron2.data.datasets import register_coco_instances
#register_coco_instances("yearbooks-weak-tables-weaktrain3", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_v3_weak_train_coco/yearbooks_AUTOv1_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_v3_weak_train")
#
#register_coco_instances("yearbooks-target-tables-train1", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_manual_train_coco_v1/yearbooks_GTJpprfinal_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/yearbooks_manual_train_v1")
#
#register_coco_instances("yearbooks-target-tables-train-v2", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_train_v2_with_L_multicells_coco/yearbooks_GTJ2postpr_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_train_v2_with_L_multicells")
#register_coco_instances("yearbooks-target-tables-val-v2", {}, "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_val_v2_with_L_multicells_coco/yearbooks_GTJ2postpr_objects.json", "/mnt/ds3lab-scratch/jrausch/git/detectron2/datasets/manual_val_v2_with_L_multicells")


#logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'logging.conf'))
#logger = logging.getLogger(__name__)
logger = setup_logger()


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #for CPU inference
    cfg.merge_from_list(["MODEL.DEVICE", 'cpu'])

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



class EntityDetector(object):




    def __init__(self):#, nms_threshold=None, resize_mode='square', detection_max_instances=100):

        #TODO: adapt args to be static
        self.args = get_parser().parse_args()
        setup_logger(name="fvcore")
        logger.info("Arguments: " + str(self.args))
        self.cpu_device = torch.device("cpu")


        #modify this to get right class name mapping


#        class_list = []
#        self.classes = DocsDataset.ALL_CLASSES
#
#        self.dataset_num_all_classes = len(DocsDataset.ALL_CLASSES) + 1  # one extra background class
#
#        class InferenceConfig(DocparserDefaultConfig):
#            NAME = 'docparser_inference'
#            DETECTION_MAX_INSTANCES = detection_max_instances
#            IMAGE_RESIZE_MODE = resize_mode
#            if nms_threshold:
#                DETECTION_NMS_THRESHOLD = nms_threshold
#            NUM_CLASSES = self.dataset_num_all_classes
#            GPU_COUNT = 1
#            IMAGES_PER_GPU = 1
#
#        self.update_train_config()
#        self.inference_config = InferenceConfig()




    def init_model(self, model_log_dir=None, default_weights=None, custom_weights=None, train_mode=False):

#        if train_mode:
#            self.model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_log_dir)
#        else:
#            self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_log_dir)

        exclude = []
        model_weights_path = None

        if default_weights is not None:
            assert custom_weights is None
            if default_weights == 'highlevel_wsft':
                raise NotImplementedError("Not yet implemented")
            elif default_weights == 'highlevel_ws':
                model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'ws', 'model_0539999.pth')

            elif default_weights == 'highlevel_wsft_medical':
                model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'ws_medical', 'model_0539999.pth')
                self.args.config_file = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'ws_medical', 'meddocs_finetune_layout_4gpu.yaml')

            elif default_weights == 'lowlevel_wsft_yearbooks':
                raise NotImplementedError("Not yet implemented")
            elif default_weights == 'yearbooks_ws':
                model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext', 'model_0539999.pth')
                self.args.config_file = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext', 'config.yaml')
            #elif default_weights == 'yearbooks_wsft1':
                #model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_from_yearbooks', 'model_0029999.pth')
                #self.args.config_file = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_from_yearbooks', 'config.yaml')
            elif default_weights == 'yearbooks_wsft2':
                model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_yearbooks', 'model_0029999.pth')
                self.args.config_file = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_yearbooks_ws_tables_4gpu_thesis_v1_swisstext_finetune_v2_from_yearbooks', 'config.yaml')
            elif default_weights == 'Austrian_tables_ws1':
                print('working on this')
                model_weights_path = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_Au_tables_4gpu_finetune_v1', 'model_0029999.pth')
                self.args.config_file = os.path.join(os.path.dirname(__file__), 'default_models', 'detectron2', 'docparser_Au_tables_4gpu_finetune_v1', 'config.yaml')
            else:
                raise NotImplementedError("Could not find matching default default_weights")
        elif custom_weights is not None:
            model_weights_path = custom_weights
        if model_weights_path is None:
            raise NotImplementedError("Not yet implemented")
        else:
            print("loading model weights from {}".format(model_weights_path))
            self.args.opts += ['MODEL.WEIGHTS', model_weights_path]


        if train_mode:
            raise NotImplementedError
        else:
            self.cfg = setup_cfg(self.args)
            #takes the naming scheme etc. from the train dataset in config
            self.metadata = MetadataCatalog.get(
                self.cfg.DATASETS.TRAIN[0] if len(self.cfg.DATASETS.TRAIN) else "__unused"
            )
            #TODO: check for alternative ways to set this in future (e.g. directly in file)
            self.metadata.bbox_mode = 0  #    
#XYXY_ABS = 0
#(x0, y0, x1, y1) in absolute floating points coordinates.
#XYWH_ABS = 1
#(x0, y0, w, h) in absolute floating points coordinates.
            if default_weights is not None:
                if not 'yearbooks' in default_weights:
                    self.metadata.thing_classes = [
                                                            "content_block",
                                                            "table",
                                                            "tabular",
                                                            "figure",
                                                            "heading",
                                                            "abstract",
                                                            "equation",
                                                            "itemize",
                                                            "item",
                                                            "bib_block",
                                                            "table_caption",
                                                            "figure_graphic",
                                                            "figure_caption",
                                                            "head",
                                                            "foot",
                                                            "page_nr",
                                                            "date",
                                                            "subject",
                                                            "author",
                                                            "affiliation",
                                                            ]
                elif 'yearbooks' in default_weights and 'wsft' in default_weights:
                    self.metadata.thing_classes = ["table", "tabular", "table_row", "table_col", "table_cell", "table_caption", "table_footnote"]
                elif 'yearbooks' in default_weights and 'wsft' not in default_weights:
                    self.metadata.thing_classes = ["table", "tabular", "table_row", "table_col", "table_caption", "table_footnote"]
                

        self.predictor = DefaultPredictor(self.cfg)

#    def load_model_weights(self, model_weights_path):
#        print("loading model weights from {}".format(model_weights_path))
#        self.model.load_weights(model_weights_path, by_name=True)

#    def get_config(self):
#        return self.model.config

    def predict(self, image, use_original_img_coords=True):

        predictions = self.predictor(image)
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

       
        #print(instances) 
        height, width = instances.image_size
        #TODO: get check for img or grayscale
        orig_shape = (height, width, 3)

        print('height/width: {}/{}'.format(height, width))
        pred_boxes = list(instances.pred_boxes)
        pred_scores = list(instances.scores)
        pred_class_ids = list(instances.pred_classes)
#        print(boxes)
#        print(scores)
        #print('class ids: {}'.format(pred_class_ids))
        
        num_preds = len(pred_boxes)

        prediction_list = []
        for pred_nr in range(num_preds):
            class_name = self.metadata.thing_classes[pred_class_ids[pred_nr]]
            pred_bbox = pred_boxes[pred_nr]
            pred_score = pred_scores[pred_nr]
            prediction_list.append({'pred_nr': pred_nr, 'class_name': class_name, 'pred_score': pred_score,
                                    'bbox_orig_coords': pred_bbox, 'orig_img_shape': orig_shape})

        predictions = {'prediction_list': prediction_list, 'orig_img_shape': orig_shape}

        return predictions


#        results = self.model.detect([image])
#        result_dict = results[0]
#        r = result_dict
#        pred_bboxes = r['rois']
#        pred_class_ids = r['class_ids']
#        pred_scores = r['scores']
#        classes_with_background = ['background'] + self.classes

#        orig_shape = list(image.shape)
#
##        num_preds = len(pred_bboxes)
#        prediction_list = []
#        for pred_nr in range(num_preds):
#            class_name = classes_with_background[pred_class_ids[pred_nr]]
#            pred_bbox = pred_bboxes[pred_nr]
#            pred_score = pred_scores[pred_nr]
#            prediction_list.append({'pred_nr': pred_nr, 'class_name': class_name, 'pred_score': pred_score,
#                                    'bbox_orig_coords': pred_bbox, 'orig_img_shape': orig_shape})
#
#        predictions = {'prediction_list': prediction_list, 'orig_img_shape': orig_shape}
#
#        return predictions

    def get_original_bbox_coords(self, bboxes, image_meta):
        #	meta_image_id = image_meta[0]
        #	meta_original_image_shape = image_meta[1:4]
        #	meta_image_shape = image_meta[4:7]
        meta_window = image_meta[7:11]
        meta_scale = image_meta[11]
        #	meta_active_class_ids = image_meta[12:]

        meta_inverse_scale = 1.0 / meta_scale

        x_offset = meta_window[1]
        y_offset = meta_window[0]
        # bbox format: y1, x1, y2, x2
        bboxes_with_offset = [[bbox[0] - y_offset, bbox[1] - x_offset, bbox[2] - y_offset, bbox[3] - x_offset] for bbox
                              in bboxes]
        bboxes_with_scaling = [[x * meta_inverse_scale for x in bbox] for bbox in bboxes_with_offset]
        return bboxes_with_scaling

    @staticmethod
    #NOTE: default bbox format changed
    def write_predictions_to_file(predictions, target_dir, file_name, use_original_img_coords=True, bbox_format='x1y1x2y2'):
        detections_textfile = os.path.join(target_dir, file_name)
        detections_output_lines = []
        logger.debug("saving predictions to {}".format(detections_textfile))

        if len(predictions['orig_img_shape']) == 2:
            logger.debug('Expanding dimension for image size')
            predictions['orig_img_shape'].append(1)
        detections_output_lines.append(
            'orig_height:{};orig_width:{};orig_depth:{}'.format(*predictions['orig_img_shape']))
        for pred in predictions['prediction_list']:
            pred_nr = pred['pred_nr']
            class_name = pred['class_name']
            pred_score = pred['pred_score']
            if use_original_img_coords:
                pred_bbox = pred['bbox_orig_coords']
            else:
                pred_bbox = pred['pred_bbox']
            if bbox_format == 'y1x1y2x2':
                y1, x1, y2, x2 = pred_bbox
            elif bbox_format == 'x1y1x2y2': #detectron2 default
                x1, y1, x2, y2 = pred_bbox
            else:
                raise NotImplementedError
            pred_output_line = '{} {} {} {} {} {} {}'.format(pred_nr, class_name, pred_score, x1, y1, x2, y2)
            detections_output_lines.append(pred_output_line)

        with open(detections_textfile, 'w') as out_file:
            for line in detections_output_lines:
                out_file.write("{}\n".format(line))

    def save_predictions_to_file(self, predictions, target_dir, file_name, use_original_img_coords=True):
        EntityDetector.write_predictions_to_file(predictions, target_dir, file_name,
                                                 use_original_img_coords=use_original_img_coords)

#    def update_train_config(self, gpu_count=1, nms_threshold=None, train_rois_per_image=None, max_gt_instances=None,
#                            detection_max_instances=None, steps_per_epoch=None, validation_steps=None,
#                            learning_rate=None,
#                            resize_mode=None, name='docparser_default'):
#        class TrainConfig(DocparserDefaultConfig):
#            if train_rois_per_image is not None:
#                TRAIN_ROIS_PER_IMAGE = train_rois_per_image
#            if max_gt_instances is not None:
#                MAX_GT_INSTANCES = max_gt_instances
#            if detection_max_instances is not None:
#                DETECTION_MAX_INSTANCES = detection_max_instances
#            if steps_per_epoch is not None:
#                STEPS_PER_EPOCH = steps_per_epoch
#            if validation_steps is not None:
#                VALIDATION_STEPS = validation_steps
#            if learning_rate is not None:
#                LEARNING_RATE = learning_rate
#            if gpu_count is not None:
#                GPU_COUNT = gpu_count
#            if resize_mode is not None:
#                IMAGE_RESIZE_MODE = resize_mode
#            if nms_threshold:
#                DETECTION_NMS_THRESHOLD = nms_threshold
#            NAME = name
#            print('setting num classes to {}'.format(self.dataset_num_all_classes))
#            NUM_CLASSES = self.dataset_num_all_classes
#
#        self.train_config = TrainConfig()

#    def train(self, dataset_train, dataset_val, custom_callbacks=[], augmentation=None, epochs1=20, epochs2=60,
#              epochs3=80):
#
#        custom_callbacks = [TimeHistory()] + custom_callbacks
#
#        if epochs1 > 0:
#            # Training - Stage 1
#            logger.info("Training network heads")
#            self.model.train(dataset_train, dataset_val,
#                             learning_rate=self.train_config.LEARNING_RATE,
#                             epochs=epochs1,  # 40
#                             layers='heads',
#                             augmentation=augmentation,
#                             custom_callbacks=custom_callbacks)
#
#        if epochs2 > 0:
#            # Training - Stage 2
#            logger.info("Fine tune Resnet stage 4 and up")
#            self.model.train(dataset_train, dataset_val,
#                             learning_rate=self.train_config.LEARNING_RATE,
#                             epochs=epochs2,  # 120
#                             layers='4+',
#                             augmentation=augmentation,
#                             custom_callbacks=custom_callbacks)
#
#        if epochs3 > 0:
#            # Training - Stage 3
#            logger.info("Fine tune all layers")
#            self.model.train(dataset_train, dataset_val,
#                             learning_rate=self.train_config.LEARNING_RATE / 10,
#                             epochs=epochs3,  # 160
#                             layers='all',
#                             augmentation=augmentation,
#                             custom_callbacks=custom_callbacks)