import os, copy
from shutil import copyfile
import json
import networkx as nx
from docparser.utils.data_utils import create_dir_if_not_exists
from distutils.dir_util import copy_tree
from datetime import datetime


def create_graph_from_annotations(all_annotations, doc_root=-1):
    doc_graph = nx.DiGraph()
    doc_graph.add_node(doc_root) #doc_root
    bbox_leaf_ids = set()
    all_ids_in_doc = set()
    for ann in all_annotations:
        all_ids_in_doc.add(ann['id'])
        doc_graph.add_node(ann['id'])
        if 'bbox' in ann:
            bbox_leaf_ids.add(ann['id'])
    for ann in all_annotations:
        if ann['parent'] is None: #applies to 'meta' and 'document'
            doc_graph.add_edge(-1, ann['id'])
        else:
            doc_graph.add_edge(ann['parent'], ann['id'])

    return doc_graph, all_ids_in_doc, bbox_leaf_ids

def split_pages_for_single_doc(doc_name, src_dir, target_dir, src_tag, target_tag):
    doc_dir = os.path.join(src_dir, doc_name)

    all_files_for_doc = os.listdir(doc_dir)
    all_imgs = [x for x in all_files_for_doc if x.endswith('.png')]
    annotation_file = doc_name + '-{}.json'.format(src_tag)
    meta_file = doc_name + '.json'
    assert annotation_file in all_files_for_doc
    assert meta_file in all_files_for_doc
    num_pages = len(all_imgs)
    meta_data_path = os.path.join(doc_dir, meta_file)
    with open(meta_data_path, 'r') as in_file:
        meta_data = json.load(in_file)
    #print(meta_data)
    date_string =  datetime.today().strftime('%d.%m.%Y')
    annotation_file_path = os.path.join(doc_dir, annotation_file)
    try:
        with open(annotation_file_path, 'r') as in_file:
            lines = [x.strip() for x in in_file.readlines()]
            in_file.seek(0)
            orig_annotations = json.load(in_file) 
    except json.decoder.JSONDecodeError as e:
        logger.error("could not read annotation file and will skip generation: {}\nfile contents: {}".format(annotation_file_path, lines))
        return 
    for page_nr in range(num_pages):
        new_meta_data = copy.deepcopy(meta_data)
        new_id = meta_data['id'] + '_{}'.format(page_nr)


        new_meta_data['pages'] = 1
        new_meta_data['id'] = new_id
        new_meta_data['title'] = new_id
        new_meta_data['date'] = date_string




        all_annotations = copy.deepcopy(orig_annotations)
        
        ids_to_preserve = set() # meta, document
        for ann in all_annotations:
            if ann['category'] == 'meta':
                ids_to_preserve.add(ann['id'])
            elif ann['category'] == 'document':
                ids_to_preserve.add(ann['id'])
            if 'page' in ann:
                if ann['page'] != page_nr:
                    ann['remove'] = True 
                else:
                    ann['page'] = 0
        all_annotations = [x for x in all_annotations if not ('remove' in x and x['remove'] == True)]
        #filter out all structure anns without a child that leads to a bbox 
        parent_id_to_ann = dict()
        doc_root = -1
        doc_graph, all_ids_in_doc, bbox_leaf_ids = create_graph_from_annotations(all_annotations, doc_root=doc_root)
        #preserve all annotations that are part of a path to a bbox leaf
        all_valid_ids = set()
        #TODO: it seems there are some (<5) documents where this takes extremely long - add a check on how many nodes there are in the graph (there might be buggy documents with large amount of empty nodes)
        if len(all_ids_in_doc) > 15000:
            logger.warning('document with extremely large amount of nodes in {}: {}'.format(doc_name, len(all_ids_in_doc)))
            #return 
        for leaf_id in bbox_leaf_ids:
            for path in nx.all_simple_paths(doc_graph, source=doc_root, target=leaf_id):
                all_valid_ids.update(set(path))
        all_invalid_ids = all_ids_in_doc - all_valid_ids

        all_valid_ids |= (ids_to_preserve)
        all_annotations = [x for x in all_annotations if x['id'] in all_valid_ids]


        #create new files 
        output_doc_dir = os.path.join(target_dir, new_id)
        create_dir_if_not_exists(output_doc_dir)

        new_meta_path = os.path.join(output_doc_dir, new_id + '.json')
        new_ann_path = os.path.join(output_doc_dir, new_id + '-{}.json'.format(target_tag))

        with open(new_ann_path, 'w')  as out_file:
            json.dump(all_annotations, out_file, indent=1)
        
        with open(new_meta_path, 'w')  as out_file:
            json.dump(new_meta_data, out_file, indent=1)


        orig_img_path = os.path.join(doc_dir, doc_name + '-{}.png'.format(page_nr))
        dest_img_path = os.path.join(output_doc_dir, new_id + '-{}.png'.format(0))

        #input('copy from {} to {}'.format(orig_img_path, dest_img_path))
        copyfile(orig_img_path, dest_img_path)

def make_med_doc_splits(all_singlepage_docs_dir, train_dir, val_dir, val_string):
    create_dir_if_not_exists(train_dir)
    create_dir_if_not_exists(val_dir)
    all_docs = [x for x in os.listdir(all_singlepage_docs_dir) if os.path.isdir(os.path.join(all_singlepage_docs_dir, x)) if not x.endswith('splits')]
    for doc in all_docs:
        src_dir = os.path.join(all_singlepage_docs_dir, doc)
        if val_string in doc:
            target_dir = os.path.join(val_dir, doc)
        else:
            target_dir = os.path.join(train_dir, doc)
        print('copy from {} to {}'.format(src_dir, target_dir))
        copy_tree(src_dir, target_dir)

    #all_dirs_

if __name__ == '__main__':
    med_docs_src = '/mnt/ds3lab-scratch/jrausch/git/docparser_public/datasets/med_docs/multipage'
    med_docs_target = '/mnt/ds3lab-scratch/jrausch/git/docparser_public/datasets/med_docs/singlepage'
    all_docs = [x for x in os.listdir(med_docs_src) if os.path.isdir(os.path.join(med_docs_src, x))]
    print(all_docs)
    src_tag = 'man'
    target_tag = 'man'
#    for doc in all_docs:
#        split_pages_for_single_doc(doc, med_docs_src, med_docs_target, src_tag, target_tag)

    all_singlepage_docs_dir =  '/mnt/ds3lab-scratch/jrausch/git/docparser_public/datasets/med_docs/singlepage'
    val_string = 'HCC22_Known_HCC18_Test_Medical_Record__29__Joy_J_Williams_DOB07061939_Provider_David_Nesler'
    train_dir = '/mnt/ds3lab-scratch/jrausch/git/docparser_public/datasets/med_docs/singlepage/splits/train'
    val_dir = '/mnt/ds3lab-scratch/jrausch/git/docparser_public/datasets/med_docs/singlepage/splits/val'

    make_med_doc_splits(all_singlepage_docs_dir, train_dir, val_dir, val_string) 
