import json
import logging.config
import os
from collections import defaultdict
from multiprocessing import Pool
from collections import Counter


import networkx as nx
import matplotlib.pyplot as plt

import skimage.io
from tqdm import tqdm
import copy

from queue import Queue
from docparser.utils.data_utils import create_dir_if_not_exists, create_annotations_to_add, \
    gather_and_sanity_check_anns, find_available_documents


logging.config.fileConfig('docparser/logging.conf')
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)



def create_coco_annotations(dataset_dir, dataset_version, output_dataset_dir, entity_classes, coco_dataset_descriptor, page=0, only_multicells=False,
                       only_labelcells=False, create_relations=False, structure_classes=[], is_weak_dataset=False):

    entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=dataset_version,
                                            subset_sample=False, subset_numimgs=None,
                                            manualseed=None)

    global_ann_id_counter = 0
    image_id_to_name_mapping = dict()
    #class_mapping = dict()
    ann_id_to_old_id_mapping = dict()
    gt_img_infos = dict()

    global_relation_id_counter = 0
    global_attribute_id_counter = 0

    classes = entity_classes + structure_classes


    #all_classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation table_row table_col table_cell'.split()
    class_mapping = dict()
    for i, class_name in enumerate(classes): #NOTE: for detectron2, do not use ALL classes, but only those used for this dataset type
        class_mapping[class_name] = i + 1 #DO NOT reserve 0 for background. detectron2 uses the id NUM_CLASSES for background automatically


    coco_info = {"description": "arxivdocs_target_layout_train"}
    coco_categories = [] 
    for class_name, class_id in class_mapping.items():
        coco_categories.append({'supercategory':class_name, 'name':class_name, 'id': class_id})

    coco_annotation_dict = {'categories': coco_categories, 'annotations': [], 'images': []}
    vg_relation_dict = {'relations': []}
    vg_attribute_dict = {'attributes': []}
    global_dataset_info = {'dataset_dir': dataset_dir, 'dataset_version': dataset_version,
            'entity_classes': entity_classes, 'structure_classes':structure_classes, 'only_multicells': only_multicells, 'only_labelcells': only_labelcells}

    all_process_inputs = [(i, example_id, dataset_dir, version, output_dataset_dir, only_multicells, only_labelcells, entity_classes, page, class_mapping, create_relations, structure_classes, is_weak_dataset) for (i, (example_id, dataset_dir, version)) in enumerate(entry_tuples)]

   
    #TODO: remove after debug
    #all_process_inputs = [x for x in all_process_inputs if x[1] == '1506.06961_16']

    #NOTE Parallel version
    all_occurring_predicates = set()
    pool_size = 20
    with Pool(processes=pool_size) as p:
        max_ = len(all_process_inputs)
        with tqdm(total=max_) as pbar:
            for i, output_tuple in enumerate(p.imap_unordered(process_single_entry, all_process_inputs)):
    #NOTE: use this for sequential processing
    #for entry_nr, process_input in tqdm(enumerate(all_process_inputs), total=len(entry_tuples)):
                #output_tuple = process_single_entry(process_input)


                [img_dict, new_annotations, img_info, example_id, image_id, coco_filename, orig_ann_ids, new_relations, new_attributes] = output_tuple
                for ann, orig_ann_id in zip(new_annotations, orig_ann_ids):
                    ann['id'] += global_ann_id_counter
                    ann_id_to_old_id_mapping[ann['id']] = (example_id, orig_ann_id)

                if create_relations:
                    for rel in new_relations:
                        #also adapt the annotation ids in the relations to the global id
                        rel['relationship_id'] += global_relation_id_counter
                        rel['subject_id'] += global_ann_id_counter
                        rel['object_id'] += global_ann_id_counter
                        all_occurring_predicates.add(rel['predicate'])

                    num_new_relations = len(new_relations)
                    global_relation_id_counter += num_new_relations
                    vg_relation_dict['relations'] += new_relations

                    for attribute in new_attributes:
                        #also adapt the annotation ids in the relations to the global id
                        attribute['attribute_id'] += global_attribute_id_counter
                        attribute['object_id'] += global_ann_id_counter

                    num_new_attributes = len(new_attributes)
                    global_attribute_id_counter += num_new_attributes
                    vg_attribute_dict['attributes'] += new_attributes


                num_new_anns = len(new_annotations)
                global_ann_id_counter += num_new_anns 

                #print('update global ann id offset to {}'.format(global_ann_id_counter))
                coco_annotation_dict['annotations'] += new_annotations
                coco_annotation_dict['images'].append(img_dict)
                gt_img_infos[example_id] = img_info
                image_id_to_name_mapping[image_id] = coco_filename
    

                #TODO: comment for sequential
                pbar.update() 


    aux_info = dict()
    aux_info['coco_id_to_orig_example_id_and_ann_id'] = ann_id_to_old_id_mapping 
    aux_info['dataset_infos_for_imgs'] = gt_img_infos  
    aux_info['dataset_infos_global'] = global_dataset_info  
    aux_info['class_mapping'] = class_mapping 
    aux_info['image_id_to_name_mapping'] = image_id_to_name_mapping
    #class_mapping = dict()
    

    
    coco_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_objects.json')
    coco_aux_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_datasetinfo.json')
    logger.info('saving annotations {}'.format(output_dataset_dir))
    with open(coco_json_filename, 'w') as out_file:
        json.dump(coco_annotation_dict, out_file, indent=1)
    with open(coco_aux_json_filename, 'w') as out_file:
        json.dump(aux_info, out_file, indent=1)

    if create_relations:
        vg_relation_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_relations.json')
        with open(vg_relation_json_filename, 'w') as out_file:
            json.dump(vg_relation_dict, out_file, indent=1)

        vg_attribute_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_attributes.json')
        with open(vg_attribute_json_filename, 'w') as out_file:
            json.dump(vg_attribute_dict, out_file, indent=1)


        #TODO: add scene_graph_attributes in the same format as relationships
        scene_graph_image_metadata, scene_graph_objects, scene_graph_relationships, scene_graph_attributes = generate_scene_graph_formatted_data(coco_annotation_dict, vg_relation_dict, vg_attribute_dict)

        print('saving scene graph jsons..')

        scene_graph_image_metadata_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_image_data.json')
        with open(scene_graph_image_metadata_json_filename, 'w') as out_file:
            json.dump(scene_graph_image_metadata, out_file, indent=1)

        scene_graph_objects_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_objects.json')
        with open(scene_graph_objects_json_filename, 'w') as out_file:
            json.dump(scene_graph_objects, out_file, indent=1)


        scene_graph_relationships_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_relationships.json')
        with open(scene_graph_relationships_json_filename, 'w') as out_file:
            json.dump(scene_graph_relationships, out_file, indent=1)

        scene_graph_attributes_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_attributes.json')
        with open(scene_graph_attributes_json_filename, 'w') as out_file:
            json.dump(scene_graph_attributes, out_file, indent=1)

        #create dummy attribute synsets
        scene_graph_attributes_synsets_json_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_attributes_synsets.json')
        with open(scene_graph_attributes_synsets_json_filename, 'w') as out_file:
            json.dump(dict(), out_file, indent=1)



        object_list = [category['name'] for category in coco_annotation_dict['categories']]
        object_alias = []
        predicate_list = list(all_occurring_predicates)
        predicate_alias = []
        scene_graph_object_txt_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_object_list.txt')
        scene_graph_object_alias_txt_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_object_alias.txt')
        scene_graph_predicate_txt_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_predicate_list.txt')
        scene_graph_predicate_alias_txt_filename = os.path.join(output_dataset_dir, coco_dataset_descriptor + '_scene_graph_predicate_alias.txt')

        with open(scene_graph_object_txt_filename, "w") as out_file:
            out_file.write("\n".join(object_list))
        with open(scene_graph_object_alias_txt_filename, "w") as out_file:
            out_file.write("\n".join(object_alias))
        with open(scene_graph_predicate_txt_filename, "w") as out_file:
            out_file.write("\n".join(predicate_list))
        with open(scene_graph_predicate_alias_txt_filename, "w") as out_file:
            out_file.write("\n".join(predicate_alias))


def generate_scene_graph_formatted_data(coco_annotation_dict, vg_relation_dict, vg_attribute_dict):

    reverse_class_mapping = dict()
    for coco_category in coco_annotation_dict['categories']:
        reverse_class_mapping[coco_category['id']] = coco_category['name']

    object_id_to_name_mapping = dict()
    #image metadata list of dicts image_data.json, containing e.g. [{"width": 800, "url": "https://cs.stanford.edu/people/rak248/vg_100k_2/1.jpg", "height": 600, "image_id": 1, "coco_id": null, "flickr_id": null}, ...
    scene_graph_image_metadata = []
    vg_format_all_image_dicts = copy.deepcopy(coco_annotation_dict['images'])
    for img_dict in vg_format_all_image_dicts:
        img_dict['url'] =  None
        img_dict['coco_id'] =None
        img_dict['flickr_id'] = None
        img_dict['image_id'] = img_dict.pop('id')
        scene_graph_image_metadata.append(img_dict)


    #scene graph, objects.json
    #objects per image, e.g. [{"image_id": 1, "objects": [{"synsets": ["tree.n.01"], "h": 557, "object_id": 1058549, "merged_object_ids": [], "names": ["trees"], "w": 799, "y": 0, "x": 0}, ...

    anns_per_image = defaultdict(list)
    for annotation in coco_annotation_dict['annotations']:
        anns_per_image[annotation['image_id']].append(annotation)

    scene_graph_objects = []
    object_by_id = dict()
    rels_per_image = dict() 

    all_img_ids = set(anns_per_image.keys())

    for img_id, annotations in anns_per_image.items():
        scene_graph_object_dict = {'image_id': img_id}
        objects_list = []
        for ann in annotations:
            assert 'bbox' in ann #NOTE root objects such as document or meta currently get a  full-image-sized bbox such that they exist within the image
            class_name = reverse_class_mapping[ann['category_id']]
            [x,y,w,h] = ann['bbox']
            object_item = {"synsets": [], 'x':x, 'y':y, 'w':w, 'h':h, 'object_id': ann['id'], 'merged_object_ids': [], 'names':[class_name]}
            object_by_id[object_item['object_id']] = object_item
            objects_list.append(object_item)
            object_id_to_name_mapping[object_item['object_id']] = class_name


        scene_graph_object_dict['objects'] = objects_list
        scene_graph_objects.append(scene_graph_object_dict)

        #NOTE: the scene graph data preprocessing tools later require there to be relations for every single image. As such, we have to make sure that each img_id that also exists for the object dictionary also exists in rels_per_image. It is possible that there are no relations at all (i.e. if there are no annotations except for the root nodes ('document' and 'meta'). these root nodes have no edges to each other
        rels_per_image[img_id] = []

    #scene graph, relationships.json
    #list of dict(s), one key: 'relationships'

    #rels_per_image = defaultdict(list)
    all_img_ids_in_rels = set()
    for rel in vg_relation_dict['relations']:
        img_id = rel['image_id']
        rels_per_image[img_id].append(rel)
        all_img_ids_in_rels.add(img_id)
    missing_img_ids = all_img_ids - all_img_ids_in_rels
    if len(missing_img_ids) > 0:
        logger.warning("no relations at all for some img ids: {}. Added empty lists for relations.".format(missing_img_ids))

    scene_graph_relationships = []
    for img_id, relations in rels_per_image.items():
        scene_graph_relationship_dict = {'image_id': img_id}
        #scene_graph_relation_dict
        relationships_list = []

        for rel in relations:
            obj = object_by_id[rel['object_id']]
            subj = object_by_id[rel['subject_id']]
            relation_object = {'name': obj['names'][0], 'object_id': obj['object_id'], 'synsets': obj['synsets'], 'x':obj['x'], 'y':obj['y'], 'w':obj['w'], 'h':obj['h']}
            relation_subject = {'name': subj['names'][0], 'object_id': subj['object_id'], 'synsets': subj['synsets'], 'x':subj['x'], 'y':subj['y'], 'w':subj['w'], 'h':subj['h']}
            relation_dict = {'relationship_id': rel['relationship_id'], 'predicate': rel['predicate'], 'synsets': [], 'object': relation_object, 'subject': relation_subject}
            relationships_list.append(relation_dict)
        scene_graph_relationship_dict['relationships'] = relationships_list
        scene_graph_relationships.append(scene_graph_relationship_dict)


    attrs_per_image = defaultdict(list)
    for attr in vg_attribute_dict['attributes']:
        img_id = attr['image_id']
        attrs_per_image[img_id].append(attr)

    scene_graph_attributes = []
    for img_id, attributes in attrs_per_image.items():
        scene_graph_attributes_dict = {'image_id': img_id}
        attributes_list = []

        for attr in attributes:
            obj = object_by_id[attr['object_id']]
            #attribute_object = {'name': obj['names'][0], 'object_id': obj['object_id'], 'synsets': obj['synsets'], 'x':obj['x'], 'y':obj['y'], 'w':obj['w'], 'h':obj['h']}
            #class_name = reverse_class_mapping[ann['category_id']]
            attribute_dict = {'attributes': [attr['attribute']], 'synsets': [], 'object_id': obj['object_id'], 'names':[object_id_to_name_mapping[obj['object_id']]],'x':obj['x'], 'y':obj['y'], 'w':obj['w'], 'h':obj['h']}
            #attribute_dict = {'attribute_id': attr['attribute_id'], 'attribute': attr['attribute'], 'synsets': [], 'object': attribute_object}
            attributes_list.append(attribute_dict)
        scene_graph_attributes_dict['attributes'] = attributes_list
        scene_graph_attributes.append(scene_graph_attributes_dict)


    return scene_graph_image_metadata, scene_graph_objects, scene_graph_relationships, scene_graph_attributes

def create_VG_annotations(dataset_dir, dataset_version, output_dataset_dir, classes, dataset_descriptor, page=0, only_multicells=False, only_labelcells=False, structure_classes_VG=['document', 'meta'], is_weak_dataset=False):

    create_coco_annotations(dataset_dir, dataset_version, output_dataset_dir, classes, dataset_descriptor, page=page, only_multicells=only_multicells, only_labelcells=only_labelcells, create_relations=True, structure_classes=structure_classes_VG, is_weak_dataset=is_weak_dataset)


def process_single_entry(input_tuple):
    
    (entry_nr, example_id, dataset_dir, version, output_dataset_dir, only_multicells, only_labelcells, entity_classes, page, class_mapping, create_relations, structure_classes, is_weak_dataset) = input_tuple 

    
    image_id = entry_nr
    annotations_path = os.path.join(dataset_dir, example_id, example_id + '-' + version + '.json')
    image_path = os.path.join(dataset_dir, example_id, example_id + '-' + str(page) + '.png')

    
    generated_annotations, root_annotations, all_unmodified_annotations = create_annotations_to_add(annotations_path, only_multicells, only_labelcells, entity_classes,
                                                      page, return_root_nodes_for_classes=structure_classes)
    

    logger.debug("current doc: {}, annotations path: {}".format(example_id, annotations_path))
    annotations, ann_by_id = gather_and_sanity_check_anns(generated_annotations)

    img_info = {'annotations_path': annotations_path, 'path': image_path, 'page': page}
    #gt_img_infos[example_id] = img_info

    image_name = os.path.basename(image_path)
    create_dir_if_not_exists(output_dataset_dir)

    #new_img_dict = img_dic
    new_annotations = []
    new_ann_id_to_old_id_mapping = dict()

    #get mscoco img information
    img = skimage.io.imread(image_path)
    img_shape = list(img.shape)
    [img_height, img_width, depth] = img_shape
    coco_filename = os.path.join(example_id, image_name) #NOTE: currently we don't have one folder with all images, but images are all residing in subdirectories
    #image_id_to_name_mapping[image_id] = coco_filename
    img_dict = {
        "id":image_id,
        "width": img_width,
        "height": img_height,
        "file_name": coco_filename 
        }
    #coco_annotation_dict['images'].append(img_dict)

    local_ann_id_counter = 0
    orig_ann_ids = []
    orig_ann_id_to_new_ann_id = dict()


    for ann in root_annotations:
        ann_id = local_ann_id_counter
        local_ann_id_counter += 1 
        orig_ann_ids.append(ann['id'])
        orig_ann_id_to_new_ann_id[ann['id']] = ann_id
        class_name = ann['category']
        class_id = class_mapping[class_name]    
        coco_ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": class_id,
            "document": example_id 
         }
        #NOTE: (experimental) we add full-image-sized bboxes for root annotations
        [x1,y1,w,h] = [0,0, img_width, img_height]
        polygon = [[x1,y1, x1+w,y1, x1+w,y1+h, x1,y1+h]]
        area = w * h
        coco_ann["segmentation"] = polygon
        coco_ann["area"] =  area
        coco_ann["bbox"] =  [x1, y1, w, h]
        coco_ann["iscrowd"] = 0
        coco_ann["bbox_mode"] = 1  # XYWH_ABS = 1


        new_annotations.append(coco_ann)




    for ann in annotations:
        ann_id = local_ann_id_counter
        local_ann_id_counter += 1 
        [x1,y1,w,h] = ann['bbox']
        orig_ann_ids.append(ann['id'])
        orig_ann_id_to_new_ann_id[ann['id']] = ann_id
        class_name = ann['category']
        class_id = class_mapping[class_name]    
        polygon = [[x1,y1, x1+w,y1, x1+w,y1+h, x1,y1+h]]
        area = w * h
        coco_ann = {
            "id": ann_id,
            "image_id": image_id,
            "segmentation": polygon,
            "category_id": class_id,
            "area": area,
            "bbox": [x1,y1,w,h],
            "iscrowd": 0,
            "bbox_mode": 1,# XYWH_ABS = 1            
            "document": example_id 
         }
        new_annotations.append(coco_ann)


    new_relations = None
    new_attributes = None
    if create_relations:
        new_relations = []
        valid_classes = entity_classes + structure_classes 

        #TODO: the graph could also be modified properly without handling two sets of annotations all the time (complete and pruned)
        pruned_document_graph, disconnected_ann_ids = generate_relations_from_annotations(annotations, valid_classes, root_annotations,
                                            root_annotation_classes=structure_classes,
                                            all_unmodified_annotations=all_unmodified_annotations)
        all_pruned_document_graph_nodes = set(pruned_document_graph.nodes())
        all_filtered_orig_ann_ids = set([x['id'] for x in root_annotations]) | set([x['id'] for x in annotations])
        #logger.debug(all_pruned_document_graph_nodes - all_filtered_orig_ann_ids)
        if all_pruned_document_graph_nodes != all_filtered_orig_ann_ids:
            missing_from_pruned = all_filtered_orig_ann_ids - all_pruned_document_graph_nodes

            missing_from_document = all_pruned_document_graph_nodes - all_filtered_orig_ann_ids
            logger.warning('missing annotations from pruned graph: {}, \nmissing from document anns: {}'.format(missing_from_pruned, missing_from_document))
            assert len(missing_from_document) == 0
            if missing_from_pruned.issubset(disconnected_ann_ids):
                if is_weak_dataset is True:
                    logger.warning("Mismatch between pruned nodes and filtered original annotations in document {}!\nWeak annotations are being modified: Removing disconnected annotations from annotation list, based on document graph".format(example_id))
                    new_anns_to_remove = set()
                    for disconnected_ann_id in missing_from_pruned:
                        new_ann_id = orig_ann_id_to_new_ann_id[disconnected_ann_id]
                        new_anns_to_remove.add(new_ann_id)
                        new_annotations = [x for x in new_annotations if x['id'] not in new_anns_to_remove]
                else:
                    raise AssertionError("Mismatch between pruned nodes and filtered original annotations in document {}! Disconnected annotations exist in document, based on graph generation. Annotations are not marked as weakly supervised, so no automatic fixes are applied.".format(example_id))
            else:
                raise AssertionError("Mismatch between pruned nodes and filtered original annotations in document {}!".format(example_id))

        #TODO: have another one version without meta and document children, and without meta sequences
        #complete relation dictionary

        all_edges = pruned_document_graph.edges(data=True)
        relation_id_counter = 0
        for edge_tuple in all_edges:
            (subj, obj, data_dict) = edge_tuple
            predicate = data_dict['label']
            new_rel = {'relationship_id': relation_id_counter, 'predicate': predicate, 'synsets': [], 'subject_id': orig_ann_id_to_new_ann_id[subj], 'object_id': orig_ann_id_to_new_ann_id[obj], "image_id": image_id}
            new_relations.append(new_rel)
            relation_id_counter += 1



        new_attributes = create_attributes_from_relations(pruned_document_graph, root_annotations)
        for attr in new_attributes:
            attr['object_id'] = orig_ann_id_to_new_ann_id[attr['object_id']] #update subject ID
            attr['image_id'] = image_id #add image ID

    return [img_dict, new_annotations, img_info, example_id, image_id, coco_filename, orig_ann_ids, new_relations, new_attributes]


def create_attributes_from_relations(pruned_document_graph, root_annotations):
    new_attributes = []
    attribute_id_counter = 0
    for root_ann in root_annotations:
        root_ann_id = root_ann['id']
        root_ann_category = root_ann['category']
        bfs_tree_from_root = nx.bfs_tree(pruned_document_graph, root_ann_id)
        for subtree_node_id in bfs_tree_from_root.nodes():
            new_attribute = {'attribute_id': attribute_id_counter, 'attribute': root_ann_category, 'object_id': subtree_node_id, 'attribute_canon': []}
            attribute_id_counter += 1
            new_attributes.append(new_attribute)


    return new_attributes



def create_gt_relations(gt_img_infos, gt_output_folder, is_gt=True):
    if is_gt is True:
        gt_relations_output_json = os.path.join(gt_output_folder, 'groundtruths_origimg_relations.json')
    else: 
        gt_relations_output_json = os.path.join(gt_output_folder, 'detections_origimg_relations.json')
    relations_by_image = create_gt_structure(gt_img_infos)

    logger.debug('saving relations ground truth to {}'.format(gt_relations_output_json))
    with open(gt_relations_output_json, 'w') as out_file:
        json.dump(relations_by_image, out_file, sort_keys=True, indent=1)


def generate_relations(parent, current_children, full_graph, ann_by_id):
    parent_relations = []
    comes_before_relations = []
    for i, child in enumerate(current_children):
        if parent is not None:
            is_parent_of_relation = (parent, child, "IS_PARENT_OF")
            parent_relations.append(is_parent_of_relation)

        if i < len(current_children) - 1:
            next_child = current_children[i + 1]
            comes_before_relation = (child, next_child, "COMES_BEFORE")
            comes_before_relations.append(comes_before_relation)

        childs_children = list(nx.dfs_preorder_nodes(full_graph, source=child, depth_limit=1))[1:]
        new_parent_relations, new_comes_before_relations = generate_relations(child, childs_children, full_graph,
                                                                              ann_by_id)
        parent_relations += new_parent_relations
        comes_before_relations += new_comes_before_relations
    return parent_relations, comes_before_relations


def create_gt_structure(gt_img_infos, root_nodes = ['document']):
    relations_by_image = dict()
    for doc_id, gt_img_info in gt_img_infos.items():
        annotations_path = gt_img_info['annotations_path']
        img_filename = os.path.basename(gt_img_info['path'])
        valid_classes = gt_img_info['classes'] + root_nodes 
        with open(annotations_path, 'r') as in_file:
            annotations = json.load(in_file)

        #TODO: refactor for rework
        #comes_before_relations, is_parent_of_relations, category_by_id = generate_relations_from_annotations(annotations)

        # NOTE: consider not including 'comes_before' relations for children of floats (i.e.: table + table_caption, figure_graphic + figure_caption), as they can be ambiguous
    
        relations_by_image[img_filename] = {'relations': comes_before_relations + is_parent_of_relations,
                                            'category_by_id': category_by_id}
    return relations_by_image


def generate_relations_from_annotations(pruned_annotations, valid_classes, root_annotations, root_annotation_classes=['document'], all_unmodified_annotations=None, draw_debug_graph=False):


    pruned_ann_by_id = dict()
    #parent_for_ann_id = dict()
    pruned_anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    #pruned_full_graph = nx.OrderedDiGraph()

    root_ann_ids = [ann['id'] for ann in root_annotations]

    for ann in pruned_annotations + root_annotations:
        pruned_ann_by_id[ann['id']] = ann
        pruned_anns_by_cat[ann['category']].append(ann)
#        if ann['parent'] is not None:
#            ann_children_ids[ann['parent']].append(ann['id'])


    unpruned_full_graph = nx.OrderedDiGraph()
    #unpruned_children_by_parent_id = defaultdict(list)
    #unpruned_existing_ids = set(unpruned_child_to_parent_mapping.keys()) | set(unpruned_child_to_parent_mapping.values())
    #input("unpruned existing ids: {}".format(unpruned_existing_ids))

    disconnected_ann_ids = set()
    unpruned_siblings_by_parent = defaultdict(list)
    unpruned_child_id_to_parent_id_mapping = dict()
    unpruned_ann_by_id = dict() 
    for ann in all_unmodified_annotations:
        if ann['parent'] is not None:
            unpruned_siblings_by_parent[ann['parent']].append(ann) #ordered lists of siblings, per parent ID
            unpruned_child_id_to_parent_id_mapping[ann['id']] = ann['parent']
        unpruned_ann_by_id[ann['id']] = ann
        unpruned_full_graph.add_node(ann['id'], label=ann['category'])
        #parent-child relationships
        #unpruned_children_by_parent_id[parent_id].append(child_id)


    for parent_id, ordered_children in unpruned_siblings_by_parent.items():
        for i in range(len(ordered_children)):
            child = ordered_children[i]
            unpruned_full_graph.add_edge(parent_id, child['id'], label='parent_of')
            #NOTE: we do this after we pruned the tree later. As the tree is ordered, we can still generate the followed_by generations (before we remove annotations that are also parents)
#            if i > 0:
#                previous_child = ordered_children[i-1]
#                unpruned_full_graph.add_edge(previous_child['id'], child['id'], label='followed_by')




    nodes_to_remove = set(unpruned_ann_by_id.keys()) - set(pruned_ann_by_id.keys())

    pruned_full_graph = unpruned_full_graph.copy()


    #NOTE: remove any nodes not connected to root nodes
    all_nodes_connected_to_root = set()
    for ann in root_annotations:
        bfs_tree_from_root = nx.bfs_tree(pruned_full_graph, ann['id']).nodes()
        #all_nodes_connected_to_root = list(nx.bfs_successors(pruned_full_graph, ann['id']))
        #for subgraph_tuple in all
        all_nodes_connected_to_root |= bfs_tree_from_root
        all_nodes_connected_to_root.add(ann['id'])

    all_nodes_disconnected_from_root = (pruned_full_graph.nodes) - all_nodes_connected_to_root
    if len(all_nodes_disconnected_from_root) > 0:
        logger.warning("Removing {} nodes that are not connected to root nodes".format(len(all_nodes_disconnected_from_root)))
        disconnected_ann_ids |= all_nodes_disconnected_from_root 

    #print('disconnected nodes to be removed: {}'.format(all_nodes_disconnected_from_root))
    #TODO: check if all nodes in graph and in general annotation list appear in both
    pruned_full_graph.remove_nodes_from(all_nodes_disconnected_from_root)

    #NOTE: remove leaves for faster processing.
    has_removed_leaves = True
    while has_removed_leaves:
        all_leaf_nodes =  [x for x in pruned_full_graph.nodes() if pruned_full_graph.out_degree(x) == 0 and pruned_full_graph.in_degree(x) == 1]

 
        all_leaf_nodes_to_prune = [x for x in all_leaf_nodes if x in nodes_to_remove]
        if len(all_leaf_nodes_to_prune) > 0:
            has_removed_leaves = True
            logger.debug("Removing {} leaf nodes from pruned tree".format(len(all_leaf_nodes_to_prune)))
            pruned_full_graph.remove_nodes_from(all_leaf_nodes_to_prune)
        else:
            has_removed_leaves = False

    #refresh list of remainign nodes to remove
    remaining_nodes_in_graph = set(pruned_full_graph.nodes())
    remaining_nodes_to_remove = remaining_nodes_in_graph - set(pruned_ann_by_id.keys())

    #all 'followed_by' edges, so we can reconstruct the order relations, if we move children up to the hierarchy, when their parents might be removed
    for parent_id, ordered_children in unpruned_siblings_by_parent.items():
        if parent_id not in remaining_nodes_in_graph:
            continue
        remaining_ordered_children = [x for x in ordered_children if x['id'] in remaining_nodes_in_graph]
        for i in range(len(remaining_ordered_children)):
            child = remaining_ordered_children[i]
            #unpruned_full_graph.add_edge(parent_id, child['id'], label='parent_of')
            if i > 0:
                previous_child = remaining_ordered_children[i-1]
                pruned_full_graph.add_edge(previous_child['id'], child['id'], label='followed_by')

    #prune top-down
    parent_nodes_to_prune_from = Queue()
    nodes_to_remove_next = Queue()
    _ = [parent_nodes_to_prune_from.put(ann) for ann in root_annotations]


    assert all(isinstance(node, int) for node in pruned_full_graph.nodes())
    assert all(isinstance(node, int) for node in unpruned_full_graph.nodes())

    #debug: check if all nodes that should be removed are also traversed
    all_inspected_nodes = set()
    while not parent_nodes_to_prune_from.empty():
        #NOTE: this assumes that the root nodes (e.g. document/meta) are never removed. we only ever remove children from the perspective of the currently traversed node
        current_parent_node = parent_nodes_to_prune_from.get()
        all_inspected_nodes.add(current_parent_node['id'])

        #print('current node: {}'.format(current_parent_node))
        all_parent_edges =  list(pruned_full_graph.out_edges(current_parent_node['id'], data=True))
        children_ids = [edge_tuple[1] for edge_tuple in all_parent_edges if edge_tuple[2]['label'] == 'parent_of']
        #all_children_edges = list(pruned_full_graph.in_edges(children_ids, data=True)) + list(pruned_full_graph.out_edges(children_ids, data=True))
        #children_sequence_edges = [edge_tuple for edge_tuple in all_children_edges if edge_tuple[2]['label'] == 'followed_by']
        #input("sequence: {}".format(children_sequence_edges))
        for child_id in children_ids:
            all_inspected_nodes.add(child_id)
            if child_id in remaining_nodes_to_remove:
                nodes_to_remove_next.put(unpruned_ann_by_id[child_id])
        #print('removing next node, queue size: {}'.format(nodes_to_remove_next.qsize()))
       
        #for current_node_to_remove in iter(nodes_to_remove_next.get, None):
        while not nodes_to_remove_next.empty():
            current_node_to_remove = nodes_to_remove_next.get()
            logger.debug('removing node: {}'.format(current_node_to_remove))
            children_that_moved_upwards = remove_node_and_rearrange_graph(pruned_full_graph, current_node_to_remove['id'])
            #queue any children that have been moved up to their old parents' position, in case they should also be pruned
            for child_id in children_that_moved_upwards:
                all_inspected_nodes.add(child_id)
                if child_id in remaining_nodes_to_remove:
                    nodes_to_remove_next.put(unpruned_ann_by_id[child_id])


        #if all children have been processed, add the children to the queue
        all_parent_edges =  list(pruned_full_graph.out_edges(current_parent_node['id'], data=True))
        fresh_children_ids = [edge_tuple[1] for edge_tuple in all_parent_edges if edge_tuple[2]['label'] == 'parent_of']
        #print("fresh child ids: {}".format(fresh_children_ids))
        for child_id in fresh_children_ids:
            parent_nodes_to_prune_from.put(unpruned_ann_by_id[child_id])
    assert remaining_nodes_to_remove.issubset(all_inspected_nodes)

    assert all(isinstance(node, int) for node in pruned_full_graph.nodes())

    #debug


    if draw_debug_graph :
        pos = nx.spring_layout(pruned_full_graph)
        nx.draw(pruned_full_graph, pos)
        node_labels = nx.get_node_attributes(pruned_full_graph, 'label')
        nx.draw_networkx_labels(pruned_full_graph, pos, node_labels)
        edge_labels = nx.get_edge_attributes(pruned_full_graph, 'label')
        nx.draw_networkx_edge_labels(pruned_full_graph, pos, edge_labels)
        plt.show()

    return pruned_full_graph, disconnected_ann_ids




def remove_node_and_rearrange_graph(pruned_full_graph, current_id_to_remove):
    all_edges = list(pruned_full_graph.in_edges(current_id_to_remove, data=True)) + list(pruned_full_graph.out_edges(current_id_to_remove, data=True))
    all_edges_as_tuples = set([(x[0], x[1], x[2]['label']) for x in all_edges])
    #input('all edges for node {}: {}'.format(current_id_to_remove, all_edges))
    hierarchy_edges = [edge_tuple for edge_tuple in all_edges_as_tuples if edge_tuple[2] == 'parent_of']
    sequence_edges = [edge_tuple for edge_tuple in all_edges_as_tuples if edge_tuple[2] == 'followed_by']
    
    parents = [edge_tuple[0] for edge_tuple in hierarchy_edges if edge_tuple[1] == current_id_to_remove]
    children = [edge_tuple[1] for edge_tuple in hierarchy_edges if edge_tuple[0] == current_id_to_remove]
    predecessors = [edge_tuple[0] for edge_tuple in sequence_edges if edge_tuple[1] == current_id_to_remove]
    successors = [edge_tuple[1] for edge_tuple in sequence_edges if edge_tuple[1] == current_id_to_remove]
    assert len(parents) == 1
    assert 0 <= len(predecessors) <= 1
    assert 0 <= len(successors) <= 1


    #close gap with predecessor and successor
    children_that_moved_upwards = []
    if len(children) == 0:
        if len(predecessors) > 0 and len(successors) > 0:
            pruned_full_graph.add_edge(predecessors[0], successors[0], label='followed_by')
    elif len(children) == 1:
        child = children[0]
        if len(predecessors) == 1:
            pruned_full_graph.add_edge(predecessors[0], child, label='followed_by')
        if len(successors) == 1:
            pruned_full_graph.add_edge(child, successors[0], label='followed_by')
        pruned_full_graph.add_edge(parents[0], child, label='parent_of')
        children_that_moved_upwards.append(child)
    elif len(children) > 1:
        children_with_out_edges = set()
        children_with_in_edges = set()
        for child in children:
            pruned_full_graph.add_edge(parents[0], child, label='parent_of')
            children_that_moved_upwards.append(child)

        all_children_edges = list(pruned_full_graph.in_edges(children, data=True)) + list(pruned_full_graph.out_edges(children, data=True))
        all_children_edges_as_tuples =  set((x[0],x[1],x[2]['label']) for x in all_children_edges)
        children_sequence_edges = [edge_tuple for edge_tuple in all_children_edges_as_tuples if edge_tuple[2] == 'followed_by']
        for edge_tuple in children_sequence_edges:
            subj_id, obj_id, _ = edge_tuple
            children_with_out_edges.add(subj_id)
            children_with_in_edges.add(obj_id)

        all_children_ids = children_with_in_edges | children_with_out_edges
        first_child_list = list(all_children_ids - children_with_in_edges)
        last_child_list = list(all_children_ids - children_with_out_edges)
        assert len(first_child_list) == len(last_child_list) == 1


        #                                                                 children_sequence_edges))
        if len(predecessors) == 1:
            pruned_full_graph.add_edge(predecessors[0], first_child_list[0], label='followed_by')
        if len(successors) == 1:
            pruned_full_graph.add_edge(last_child_list[0], successors[0], label='followed_by')
    else:
        raise NotImplementedError

    pruned_full_graph.remove_node(current_id_to_remove)
    return children_that_moved_upwards


def save_eval_gt_for_image(annotations, image_name, gt_dir, image_path):
    img = skimage.io.imread(image_path)
    img_shape = list(img.shape)

    groundtruths_textfile = os.path.join(gt_dir, image_name + '.txt')

    groundtruths_output_lines = []
    groundtruths_output_lines.append('orig_height:{};orig_width:{};orig_depth:{}'.format(*img_shape))

    for ann in annotations:
        class_name = ann['category']
        gt_annotation_id = ann['id']
        x1, y1, w, h = ann['bbox']
        y2 = y1 + h
        x2 = x1 + w
        gt_output_line = '{} {} {} {} {} {}'.format(gt_annotation_id, class_name, x1, y1, x2, y2)
        groundtruths_output_lines.append(gt_output_line)

    with open(groundtruths_textfile, 'w') as out_file:
        for line in groundtruths_output_lines:
            out_file.write("{}\n".format(line))

def save_eval_detections_for_image(annotations, image_name, det_dir, image_path):
    img = skimage.io.imread(image_path)
    img_shape = list(img.shape)

    groundtruths_textfile = os.path.join(det_dir, image_name + '.txt')

    groundtruths_output_lines = []
    groundtruths_output_lines.append('orig_height:{};orig_width:{};orig_depth:{}'.format(*img_shape))

    confidence = 1
    for ann in annotations:
        class_name = ann['category']
        gt_annotation_id = ann['id']
        x1, y1, w, h = ann['bbox']
        y2 = y1 + h
        x2 = x1 + w
        det_output_line = '{} {} {} {} {} {} {}'.format(gt_annotation_id, class_name, confidence, x1, y1, x2, y2)
        groundtruths_output_lines.append(det_output_line)

    with open(groundtruths_textfile, 'w') as out_file:
        for line in groundtruths_output_lines:
            out_file.write("{}\n".format(line))



def create_highlevel_weak(dataset_root, coco_mode=True):
    classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/train')
    coco_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/mscoco/train')
    dataset_version = 'mx15'


    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='arxivdocs_weak_layout_train', create_relations=False)


def create_yearbook_tables_weak(dataset_root, coco_mode=True):
    classes = 'table tabular table_row table_col table_caption table_footnote'.split()
    train_dataset = os.path.join(dataset_root, 'yearbooks_v3_with_splits', 'weak_train')
    coco_output_train_dataset = os.path.join(dataset_root, 'yearbooks_v3_with_splits', 'weak_train_coco')
    dataset_version = 'AUTOv1'

    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='yearbooks_{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version

def create_austrian_tables_weak(dataset_root, coco_mode=True):
    classes = 'table tabular table_row table_col'.split()
    train_dataset = os.path.join(dataset_root, 'Au_images_finetune_v1', 'weak_train')
    coco_output_train_dataset = os.path.join(dataset_root, 'Au_images_finetune_v1', 'weak_train_coco')
    dataset_version = 'automated'

    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='Au_tables_{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version

def create_austrian_tables_weak_ada(dataset_root, coco_mode=True):
    classes = 'table tabular table_row table_col table_caption table_footnote'.split()
    train_dataset = os.path.join(dataset_root, 'Au_images_finetune_v2', 'weak_train')
    coco_output_train_dataset = os.path.join(dataset_root, 'Au_images_finetune_v2', 'weak_train_coco')
    dataset_version = 'Ada-post'
    # it could be costly to learn from the table_cells ... eta: 6 days, and GPU OOM (spill to CPU)
    # removed tuning on table_cell (3.3hrs)

    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='Au_tables_Ada_{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version

def create_zh_OCR_tables_weak(dataset_root, coco_mode=True):
    classes = 'table tabular table_row table_col table_cell table_caption table_footnote'.split()
    train_dataset = os.path.join(dataset_root, 'Zh_images_test', 'weak_train')
    coco_output_train_dataset = os.path.join(dataset_root, 'Zh_images_test', 'weak_train_coco')
    # dataset_version = 'Ada-post-auto'
    dataset_version = 'Ada-post'

    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='Zh_images_test{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version

def create_AuHu_OCR_tables_weak(dataset_root, coco_mode=True):
    classes = 'table tabular table_row table_col table_cell table_caption table_footnote'.split()
    train_dataset = os.path.join(dataset_root, 'AuHu_images_test', 'weak_train')
    coco_output_train_dataset = os.path.join(dataset_root, 'AuHu_images_test', 'weak_train_coco')
    # dataset_version = 'Ada-post-auto'
    dataset_version = 'Ada-post'
    
    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='AuHu_images_test{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version

def create_yearbook_tables_target(dataset_root, coco_mode=True):
    #classes = 'table tabular table_row table_col table_caption table_footnote'.split()
    classes = 'table tabular table_row table_col table_cell table_caption table_footnote'.split()
    train_dataset = os.path.join(dataset_root, 'yearbooks_manual', 'manual_train_v2_with_L_multicells')
    val_dataset = os.path.join(dataset_root, 'yearbooks_manual', 'manual_val_v2_with_L_multicells')
    coco_output_train_dataset = os.path.join(dataset_root, 'yearbooks_manual', 'manual_train_v2_with_L_multicells_coco')
    coco_output_val_dataset = os.path.join(dataset_root, 'yearbooks_manual', 'manual_val_v2_with_L_multicells_coco')
    dataset_version = 'GTJ2postpr'
    
    #TODO: adjust data generation such that "L" shaped (i.e. non-rectangular) entities are given a proper polygon mask (instead of just a filled rectangle)
    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='yearbooks_{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version
    create_coco_annotations(val_dataset, dataset_version, coco_output_val_dataset, classes, coco_dataset_descriptor='yearbooks_{}'.format(dataset_version), only_multicells=False, only_labelcells=False, create_relations=False) #NOTE: labelcells is False because annotation bugs in first version



def create_lowlevel_weak(dataset_root, coco_mode=True):
    classes = 'table_row table_col table_cell'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_tabular/train')
    coco_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_tabular/mscoco/train')
    dataset_version = 'mx17'

    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='arxivdocs_weak_tablestruct_cells_multi_label_train', only_multicells=True, only_labelcells=True, create_relations=False)



def create_highlevel_groundtruths(dataset_root, val_and_test_version = 'cleanGT', mode='default'):
    # highlevel
    classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/train')
    val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/dev')
    test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/test')


    if mode == 'default':
        output_train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists/train')
        output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists/dev')
        output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists/test')
        create_annotations(train_dataset, val_and_test_version, output_train_dataset, classes, create_relations=True)
        create_annotations(val_dataset, val_and_test_version, output_val_dataset, classes, create_relations=True)
        create_annotations(test_dataset, val_and_test_version, output_test_dataset, classes, create_relations=True)

    elif mode == 'coco':
        coco_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/mscoco/train')
        coco_output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/mscoco/dev')
        coco_output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/mscoco/test')


        create_coco_annotations(train_dataset, val_and_test_version, coco_output_train_dataset, classes, coco_dataset_descriptor='arxivdocs_target_layout_train', create_relations=False)
        create_coco_annotations(val_dataset, val_and_test_version, coco_output_val_dataset, classes, coco_dataset_descriptor='arxivdocs_target_layout_dev', create_relations=False)
        create_coco_annotations(test_dataset, val_and_test_version, coco_output_test_dataset, classes, coco_dataset_descriptor='arxivdocs_target_layout_test', create_relations=False)

    elif mode == 'VG':
        VG_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/VG/train')
        VG_output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/VG/dev')
        VG_output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/VG/test')
        structure_classes_VG = ['document', 'meta']
        create_VG_annotations(train_dataset, val_and_test_version, VG_output_train_dataset, classes, dataset_descriptor='arxivdocs_target_layout_train', structure_classes_VG=structure_classes_VG)
        create_VG_annotations(val_dataset, val_and_test_version, VG_output_val_dataset, classes, dataset_descriptor='arxivdocs_target_layout_dev', structure_classes_VG=structure_classes_VG)
        create_VG_annotations(test_dataset, val_and_test_version, VG_output_test_dataset, classes, dataset_descriptor='arxivdocs_target_layout_test', structure_classes_VG=structure_classes_VG)
    else:
        logger.warning("Unknown mode: '{}'".format(mode))



def create_highlevel_weak_vg(dataset_root, val_and_test_version = 'mx15', mode='VG'):
    # highlevel
    classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/train')
    val_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/dev')
    test_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/test')

    if mode == 'VG':
        VG_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/VG/train')
        VG_output_val_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/VG/dev')
        VG_output_test_dataset = os.path.join(dataset_root, 'arxivdocs_weak/splits/by_page/VG/test')
        structure_classes_VG = ['document', 'meta']
        create_VG_annotations(train_dataset, val_and_test_version, VG_output_train_dataset, classes, dataset_descriptor='arxivdocs_weak_layout_train', structure_classes_VG=structure_classes_VG, is_weak_dataset=True)
        create_VG_annotations(val_dataset, val_and_test_version, VG_output_val_dataset, classes, dataset_descriptor='arxivdocs_weak_layout_dev', structure_classes_VG=structure_classes_VG, is_weak_dataset=True)
        create_VG_annotations(test_dataset, val_and_test_version, VG_output_test_dataset, classes, dataset_descriptor='arxivdocs_weak_layout_test', structure_classes_VG=structure_classes_VG, is_weak_dataset=True)
    else:
        logger.warning("Mode not supported: '{}'".format(mode))





def create_highlevel_weak_reference_for_targetset(dataset_root, val_and_test_version = 'mx15'):
    # highlevel
    classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/train')
    val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/dev')
    #test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/test')
    output_train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists_referenceweak/train')
    output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists_referenceweak/dev')
    #output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_page/flat_lists_referenceweak/test')

    create_annotations(train_dataset, val_and_test_version, output_train_dataset, classes, create_relations=True, is_gt=False)
    create_annotations(val_dataset, val_and_test_version, output_val_dataset, classes, create_relations=True, is_gt=False)
    #create_annotations(test_dataset, val_and_test_version, output_test_dataset, classes, create_relations=True)



def create_lowlevel_groundtruths_new(dataset_root, coco_mode=False):
    # lowlevel
    classes = 'table_row table_col table_cell'.split()
    train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/train')
    val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/dev')
    test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/test')
    output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/flat_lists/dev')
    output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/flat_lists/test')
    val_and_test_version = 'cleanGT'



    coco_output_train_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/mscoco/train')
    coco_output_val_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/mscoco/dev')
    coco_output_test_dataset = os.path.join(dataset_root, 'arxivdocs_target/splits/by_tabular/mscoco/test')

    if coco_mode == False:

        create_annotations(val_dataset, val_and_test_version, output_val_dataset, classes, only_multicells=True,
                           only_labelcells=True, create_relations=False)
        create_annotations(test_dataset, val_and_test_version, output_test_dataset, classes, only_multicells=True,
                           only_labelcells=True, create_relations=False)
    elif coco_mode == True:
        create_coco_annotations(train_dataset, val_and_test_version, coco_output_train_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_cells_multi_label_train', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(val_dataset, val_and_test_version, coco_output_val_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_cells_multi_label_val', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(test_dataset, val_and_test_version, coco_output_test_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_cells_multi_label_test', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(train_dataset, val_and_test_version, coco_output_train_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_allcells_train', create_relations=False)
        create_coco_annotations(val_dataset, val_and_test_version, coco_output_val_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_allcells_val', create_relations=False)
        create_coco_annotations(test_dataset, val_and_test_version, coco_output_test_dataset, classes, coco_dataset_descriptor='arxivdocs_tablestruct_allcells_test', create_relations=False)



def create_lowlevel_icdar_groundtruths(dataset_root, coco_mode=False):
    # icdar lowlevel
    classes = 'table_row table_col table_cell'.split()
    dataset_root = os.path.join('..', 'datasets')
    train_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/icdar_train')
    val_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/icdar_dev')
    test_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/icdar_test')
    output_val_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/flat_lists/icdar_dev')
    output_test_dataset = os.path.join(dataset_root,
                                       'icdar2013_docparser/mixed_icdar_crop_splits/flat_lists/icdar_test')

    coco_output_train_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/mscoco/icdar_train')
    coco_output_val_dataset = os.path.join(dataset_root, 'icdar2013_docparser/mixed_icdar_crop_splits/mscoco/icdar_dev')
    coco_output_test_dataset = os.path.join(dataset_root,
                                       'icdar2013_docparser/mixed_icdar_crop_splits/mscoco/icdar_test')


    val_and_test_version = 'cleanGT'
    

    if coco_mode == False:
        create_annotations(val_dataset, val_and_test_version, output_val_dataset, classes, only_multicells=True,
                           only_labelcells=True, create_relations=False)
        create_annotations(test_dataset, val_and_test_version, output_test_dataset, classes, only_multicells=True,
                           only_labelcells=True, create_relations=False)

    elif coco_mode == True:
        create_coco_annotations(train_dataset, val_and_test_version, coco_output_train_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_cells_multi_label_train', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(val_dataset, val_and_test_version, coco_output_val_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_cells_multi_label_val', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(test_dataset, val_and_test_version, coco_output_test_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_cells_multi_label_test', only_multicells=True, only_labelcells=True, create_relations=False)
        create_coco_annotations(train_dataset, val_and_test_version, coco_output_train_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_allcells_train', create_relations=False)
        create_coco_annotations(val_dataset, val_and_test_version, coco_output_val_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_allcells_val', create_relations=False)
        create_coco_annotations(test_dataset, val_and_test_version, coco_output_test_dataset, classes, coco_dataset_descriptor='icdar2013_tablestruct_allcells_test', create_relations=False)


def create_annotations(dataset_dir, dataset_version, output_dataset_dir, classes, page=0, only_multicells=False,
                       only_labelcells=False, create_relations=False, is_gt=True):
    logger.debug('gt option is set to {}'.format(is_gt))
    gt_img_infos = dict()

    entry_tuples = find_available_documents(dataset_dir=dataset_dir, version=dataset_version,
                                            subset_sample=False, subset_numimgs=None,
                                            manualseed=None)

#    #TODO remove debug: 
#    entry_tuples [('1506.06961_16', dataset_dir, dataset_version)]

    for entry_nr, entry_tuple in enumerate(entry_tuples):
        (example_id, dataset_dir, version) = entry_tuple
        annotations_path = os.path.join(dataset_dir, example_id, example_id + '-' + version + '.json')
        image_path = os.path.join(dataset_dir, example_id, example_id + '-' + str(page) + '.png')

        generated_annotations = create_annotations_to_add(annotations_path, only_multicells, only_labelcells, classes,
                                                          page)
        annotations, ann_by_id = gather_and_sanity_check_anns(generated_annotations)

        img_info = {'dataset_dir': dataset_dir, 'dataset_version': dataset_version,
                    'annotations_path': annotations_path, 'path': image_path, 'classes': classes, 'page': page,
                    'only_multicells': only_multicells, 'only_labelcells': only_labelcells}
        gt_img_infos[example_id] = img_info

        image_name = os.path.basename(image_path)
        create_dir_if_not_exists(output_dataset_dir)
        if is_gt is True:
            gt_output_dir = os.path.join(output_dataset_dir, 'groundtruths_origimg')
            create_dir_if_not_exists(gt_output_dir)
            save_eval_gt_for_image(annotations, image_name, gt_output_dir, image_path)
        else:
            det_output_dir = os.path.join(output_dataset_dir, 'detections_origimg')
            create_dir_if_not_exists(det_output_dir)
            save_eval_detections_for_image(annotations, image_name, det_output_dir, image_path)

    subset_dict_keys = ['classes', 'page', 'only_multicells', 'only_labelcells', 'dataset_version']
    if is_gt is True:
        gt_img_infos_path = os.path.join(output_dataset_dir, 'gt_img_infos.json')
    else:
        gt_img_infos_path = os.path.join(output_dataset_dir, 'det_img_infos.json')
    gt_img_infos_subset = dict()
    #don't save all infos to json file
    for k, v in gt_img_infos.items():
        gt_img_infos_subset[k] = {subset_key: v[subset_key] for subset_key in subset_dict_keys}
    with open(gt_img_infos_path, 'w') as out_file:
        json.dump(gt_img_infos_subset, out_file, sort_keys=True, indent=1)

    # create relations
    if create_relations:
        create_gt_relations(gt_img_infos, output_dataset_dir, is_gt)





def create_meddocs(dataset_root, coco_mode=True):
    classes = 'content_block table tabular figure heading abstract equation itemize item bib_block table_caption figure_graphic figure_caption head foot page_nr date subject author affiliation'.split()
    train_dataset = os.path.join(dataset_root, 'med_docs/singlepage/splits/train')
    coco_output_train_dataset = os.path.join(dataset_root, 'med_docs/singlepage/mscoco/train')
    val_dataset = os.path.join(dataset_root, 'med_docs/singlepage/splits/val')
    coco_output_val_dataset = os.path.join(dataset_root, 'med_docs/singlepage/mscoco/val')
    dataset_version = 'man'


    assert coco_mode == True #NOTE: no non-coco version yet
    create_coco_annotations(train_dataset, dataset_version, coco_output_train_dataset, classes, coco_dataset_descriptor='med_docs_layout_train', create_relations=False)
    create_coco_annotations(val_dataset, dataset_version, coco_output_val_dataset, classes, coco_dataset_descriptor='med_docs_layout_val', create_relations=False)


def main():
    dataset_root = os.path.join('./', 'datasets')
#    create_highlevel_groundtruths(dataset_root)
#    create_highlevel_weak_reference_for_targetset(dataset_root) #to evaluate weak annotation quality
#    create_lowlevel_groundtruths_new(dataset_root)
#    create_lowlevel_icdar_groundtruths() #TODO: update

#    create_highlevel_groundtruths(dataset_root, coco_mode=True)
#    create_lowlevel_groundtruths_new(dataset_root, coco_mode=True)
#    create_lowlevel_icdar_groundtruths(dataset_root, coco_mode=True) #TODO: update


#vg data

    #create_highlevel_groundtruths(dataset_root, mode='VG')
    #create_highlevel_weak_vg(dataset_root, mode='VG')

    #create_highlevel_weak(dataset_root, coco_mode=True)
    #create_lowlevel_weak(dataset_root, coco_mode=True)

    
    create_yearbook_tables_weak(dataset_root, coco_mode=True)
    # create_yearbook_tables_target(dataset_root, coco_mode=True)
    #create_lowlevel_weak(dataset_root, coco_mode=True)
    # create_austrian_tables_weak(dataset_root, coco_mode=True)

    # create_austrian_tables_weak_ada(dataset_root, coco_mode=True)

    # create_austrian_tables_weak_ada(dataset_root, coco_mode=True)

    # create_zh_OCR_tables_weak(dataset_root, coco_mode=True)

    # create_AuHu_OCR_tables_weak(dataset_root, coco_mode=True)

    #create_meddocs(dataset_root, coco_mode=True)
if __name__ == "__main__":
    main()
