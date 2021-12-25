import os
import json
from datetime import datetime
import copy
import networkx as nx
from docparser.utils.data_utils import create_dir_if_not_exists
from shutil import copyfile
from shutil import copytree
import logging
#import logging.config
#logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from collections import Counter

def get_all_yearbooks(yearbooks_root):
    all_dirs = os.listdir(yearbooks_root)
    print('found {} docs in total'.format(len(all_dirs)))
    all_valid_docs = []
    for doc_name in all_dirs:
        doc_id, sheet_nr = doc_name.split('-')
#        if int(sheet_nr) > 0: #NOTE: currently, sheets of nr > 0 contain duplicate annotations
#            continue
        all_valid_docs.append(doc_name) 
    print('{} valid docs (multiple sheets possible) in total'.format(len(all_valid_docs)))
    return all_valid_docs



def get_all_bboxes_for_ann(doc_graph, ann, ann_by_id):


    all_children_of_ann = nx.descendants(doc_graph, ann['id'])
    all_bbox_children_of_ann = [ann_by_id[x] for x in all_children_of_ann if 'bbox' in ann_by_id[x]]
    x_min, y_min = 100000, 100000
    x_max, y_max = -1, -1 
    for bbox_ann in all_bbox_children_of_ann:
        x0,y0,w,h = bbox_ann['bbox']
        x1 = x0+w
        y1 = y0+h
        if x0 < x_min:
            x_min = x0
        if x1 > x_max:
            x_max = x1
        if y0 < y_min:
            y_min = y0
        if y1 > y_max:
            y_max = y1
    
    return all_bbox_children_of_ann, [x_min, y_min, x_max-x_min, y_max-y_min]


def get_union_bbox(bbox1, bbox2):
    if bbox1 is None and bbox2 is not None:
        return bbox2
    elif bbox1 is not None and bbox2 is None:
        return bbox1 
    elif bbox1 is None and bbox2 is None:
        raise NotImplementedError

    bbox1_x0,bbox1_y0,bbox1_w,bbox1_h = bbox1
    bbox1_x1 = bbox1_x0+bbox1_w
    bbox1_y1 = bbox1_y0+bbox1_h


    bbox2_x0,bbox2_y0,bbox2_w,bbox2_h = bbox2
    bbox2_x1 = bbox2_x0+bbox2_w
    bbox2_y1 = bbox2_y0+bbox2_h

    union_x0 = min(bbox1_x0, bbox2_x0)
    union_y0 = min(bbox1_y0, bbox2_y0)
    union_x1 = max(bbox1_x1, bbox2_x1)
    union_y1 = max(bbox1_y1, bbox2_y1)

    return [union_x0, union_y0, union_x1-union_x0, union_y1-union_y0] 
   
#TODO: adapt for pages != 0 
def create_new_ann(parent, category, new_bbox, max_id, extra_properties, page=0):
    parent_id = parent['id']
    new_id = max_id+1
    max_id += 1
    new_ann = {'id': new_id, 'category': category, 'parent': parent['id']}
    for k,v in extra_properties.items():
        new_ann[k] = v
    new_bbox_id = max_id+1
    max_id +=1 
    new_bbox_ann = {'id': new_bbox_id, 'category': 'box', 'parent': new_id, 'bbox':new_bbox, 'page':page}
    new_anns = [new_ann, new_bbox_ann]
    return new_anns, max_id


def read_out_range_from_string(raw_ranges):
    raw_rows, raw_cols = raw_ranges.split(',')
    num_of_dashes = sum(char == '-' for char in raw_rows) 
    #print(raw_ranges)
    try:
        assert num_of_dashes >= 1 and num_of_dashes <= 3
    except AssertionError as e:
        print("bad ranges: {}, {}".format(raw_ranges, e))
        raise
    if num_of_dashes == 3:
        groups = raw_rows.split('-')
        row_start, row_end = '-'.join(groups[:2]), '-'.join(groups[2:])
    elif num_of_dashes == 2:
        if raw_rows.startswith('-'): 
            row_start, row_end = raw_rows.rsplit('-', 1) 
        else:
            row_start, row_end = raw_cols.split('-', 1)
    else:
        row_start, row_end = raw_rows.split('-') 
    row_start = int(row_start)
    row_end = int(row_end)
    num_of_dashes = sum(char == '-' for char in raw_cols) 
    try:
        assert num_of_dashes >= 1 and num_of_dashes <= 3
    except AssertionError as e:
        print("bad ranges: {}, {}".format(raw_ranges, e))
        raise
    if num_of_dashes == 3:
        groups = raw_cols.split('-')
        col_start, col_end = '-'.join(groups[:2]), '-'.join(groups[2:])
    elif num_of_dashes == 2:
        if raw_cols.startswith('-'): 
            col_start, col_end = raw_cols.rsplit('-', 1) 
        else:
            col_start, col_end  = raw_cols.split('-', 1)
    else:
        col_start, col_end = raw_cols.split('-') 
    col_start = int(col_start)
    col_end = int(col_end)
    return [row_start, row_end], [col_start, col_end]

def generate_rows_and_columns_from_cells(all_annotations):

    doc_root = -1
    doc_graph, all_ids_in_doc, bbox_leaf_ids = create_graph_from_annotations(all_annotations, doc_root=doc_root)

    anns_by_parent = defaultdict(list)
    anns_by_category = defaultdict(list)
    ann_by_id = dict() 
    max_id = -1
    for ann in all_annotations:
        ann_by_id[ann['id']] = ann
        if ann['parent'] is not None:
            anns_by_parent[ann['parent']].append(ann)
        anns_by_category[ann['category']].append(ann)
        if ann['id'] > max_id:
            max_id = ann['id']

    all_tabular_anns = anns_by_category['tabular']

    all_new_anns = []

    for tabular in all_tabular_anns:

        
        all_bbox_children_of_tabular, _ = get_all_bboxes_for_ann(doc_graph, tabular, ann_by_id)
        

        all_cells_in_tabular = [x for x in anns_by_parent[tabular['id']] if x['category'] == 'table_cell']

       
        tabular_bbox = None
 
        col_x_start_values = defaultdict(list)
        col_x_end_values = defaultdict(list)
        row_y_start_values = defaultdict(list)
        row_y_end_values = defaultdict(list)
     
 
        if len(all_cells_in_tabular ) == 0:
            #print('no cells in tabular!')
            continue

        all_valid_cells_in_tabular = []

        orig_min_row = 1000000 
        orig_min_col = 1000000
        for cell in all_cells_in_tabular:
            row_range, col_range = read_out_range_from_string(cell['properties'])

            #TODO: these cells could be removed from the 'existing cells' set
            _, cell_union_bbox = get_all_bboxes_for_ann(doc_graph, cell, ann_by_id)
            [x0,y0,w,h] = cell_union_bbox
            if w < 5 or h < 5:
                #print('skipping very small box at {}, {}: {}'.format(row_range, col_range, cell_union_bbox))
                continue
            else:
                all_valid_cells_in_tabular.append(cell)


            orig_min_row = min(orig_min_row, row_range[0])
            orig_min_col = min(orig_min_col, col_range[0])




        col_indices = set()
        row_indices = set()
        all_existing_cell_positions = set()
        #make sure all row/columns start at zero index
        for cell in all_valid_cells_in_tabular:
            orig_row_range, orig_col_range = read_out_range_from_string(cell['properties'])
            cell['row_range'] = [orig_row_range[0] - orig_min_row, orig_row_range[1] - orig_min_row]
            cell['col_range'] = [orig_col_range[0] - orig_min_col, orig_col_range[1] - orig_min_col]
            cell['properties'] = "{}-{},{}-{}".format(cell['row_range'][0], cell['row_range'][1], cell['col_range'][0], cell['col_range'][1])
            row_range = cell['row_range']
            col_range = cell['col_range']


            for row_nr in row_range:
                for col_nr in col_range:
                    all_existing_cell_positions.add((row_nr, col_nr))
            row_indices.add(row_range[0])
            row_indices.add(row_range[1])
            col_indices.add(col_range[0])
            col_indices.add(col_range[1])

        if len(row_indices) == 0 or len(col_indices) == 0:
            print('either no valid rows or valid cols in table!')
            continue

        #print('all rows: {}, all cols: {}'.format(sorted(list(row_indices)), sorted(list(col_indices))))

        #NOTE: used as sanity check here (not expected to always pass for automatically generated data)
        num_rows_expected = max(row_indices) + 1
        num_cols_expected = max(col_indices) + 1
        all_expected_cell_positions = set()
        for row_nr in range(num_rows_expected):
            for col_nr in range(num_cols_expected):
                all_expected_cell_positions.add((row_nr, col_nr))
       
        missing_cells = all_expected_cell_positions - all_existing_cell_positions
        #if len(missing_cells) > 0:
            #print('missing cells: {}'.format(missing_cells))
 

        for cell in all_valid_cells_in_tabular:
            row_range = cell['row_range']
            col_range = cell['col_range']
            all_cell_bboxes, cell_union_bbox = get_all_bboxes_for_ann(doc_graph, cell, ann_by_id)
            [x0,y0,w,h] = cell_union_bbox
            x1 = x0+w
            y1 = y0+h 
            tabular_bbox = get_union_bbox(tabular_bbox, cell_union_bbox)
            #TODO: this assumes row/column border locations are consistent across cels
            if col_range[0] in col_indices:
                col_x_start_values[col_range[0]].append(x0)
            if col_range[1] in col_indices:
                col_x_end_values[col_range[1]].append(x1)
            if col_range[0]-1 in col_indices:
                col_x_end_values[col_range[0]-1].append(x0)
            if col_range[1]+1 in col_indices:
                col_x_start_values[col_range[1]+1].append(x1)

            if row_range[0] in row_indices:
                row_y_start_values[row_range[0]].append(y0)
            if row_range[1] in row_indices:
                row_y_end_values[row_range[1]].append(y1)
            if row_range[0]-1 in row_indices:
                row_y_end_values[row_range[0]-1].append(y0)
            if row_range[1]+1 in row_indices:
                row_y_start_values[row_range[1]+1].append(y1)
          

        #print('determined tabular bbox as: {}'.format(tabular_bbox)) 
        if tabular_bbox is None:
            logger.warning("No cell bboxes found as children of tabular! not adding any rows/cols") 
            continue
        [x0,y0,w,h] = tabular_bbox
        x1 = x0+w
        y1 = y0+h
        
        for row_id in row_indices: 
            try:
                row_bbox_y0 = Counter(row_y_start_values[row_id]).most_common(1)[0]
                #print('counter output: {}'.format(row_bbox_y0))
                row_bbox_y0 = Counter(row_y_start_values[row_id]).most_common(1)[0][0]
                row_bbox_y1 = Counter(row_y_end_values[row_id]).most_common(1)[0][0]
                row_bbox = [x0, row_bbox_y0, w, row_bbox_y1-row_bbox_y0]
                new_anns, max_id = create_new_ann(tabular, 'table_row', row_bbox, max_id, {'row_nr':row_id})
                #print('adding new row: {}'.format(new_anns))
                all_new_anns += new_anns
            except IndexError as e:
                continue
                #print("could not generate row {}: {}".format(row_id, e))

        for col_id in col_indices: 
            try:
                col_bbox_x0 = Counter(col_x_start_values[col_id]).most_common(1)[0][0]
                col_bbox_x1 = Counter(col_x_end_values[col_id]).most_common(1)[0][0]
                col_bbox = [col_bbox_x0, y0, col_bbox_x1-col_bbox_x0, h]
                new_anns, max_id = create_new_ann(tabular, 'table_col', col_bbox, max_id, {'col_nr':col_id})
                #print('adding new col: {}'.format(new_anns))
                all_new_anns += new_anns

            except IndexError as e:
                continue
                #print("could not generate col {}: {}".format(col_id, e))

        #print("total of {} cells and {} bboxes in tabular, column ids: {}, row ids: {}".format(len(all_cells_in_tabular), len(all_bbox_children_of_tabular), col_indices, row_indices))
    all_annotations += all_new_anns
    return all_annotations

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

def process_single_doc(process_inputs):
    (doc_name, yearbooks_root, preprocessed_root) = process_inputs
    doc_dir = os.path.join(yearbooks_root, doc_name)

    all_files_for_doc = os.listdir(doc_dir)
    all_imgs = [x for x in all_files_for_doc if x.endswith('.png')]
    annotation_file = doc_name + '-automated.json'
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
        for ann in all_annotations:
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
            logger.warning('skipping document due to extremely large amount of nodes in {}: {}'.format(doc_name, len(all_ids_in_doc)))
            return 
        for leaf_id in bbox_leaf_ids:
            for path in nx.all_simple_paths(doc_graph, source=doc_root, target=leaf_id):
                all_valid_ids.update(set(path))
        #print(all_valid_ids)
        all_invalid_ids = all_ids_in_doc - all_valid_ids
        #print('invalid ids: {}'.format(all_invalid_ids))

        ids_to_preserve = {1,2,3} #unk, meta, document
        all_valid_ids |= (ids_to_preserve)
        #print(all_valid_ids)
        all_annotations = [x for x in all_annotations if x['id'] in all_valid_ids]
        #print(all_annotations)

        all_annotations = generate_rows_and_columns_from_cells(all_annotations)

        #create new files 
        output_doc_dir = os.path.join(preprocessed_root, new_id)
        create_dir_if_not_exists(output_doc_dir)

        new_meta_path = os.path.join(output_doc_dir, new_id + '.json')
        new_ann_path = os.path.join(output_doc_dir, new_id + '-AUTOv1.json')

        with open(new_ann_path, 'w')  as out_file:
            json.dump(all_annotations, out_file, indent=1)
        
        with open(new_meta_path, 'w')  as out_file:
            json.dump(new_meta_data, out_file, indent=1)


        orig_img_path = os.path.join(doc_dir, doc_name + '-{}.png'.format(page_nr))
        dest_img_path = os.path.join(output_doc_dir, new_id + '-{}.png'.format(0))

        #input('copy from {} to {}'.format(orig_img_path, dest_img_path))
        copyfile(orig_img_path, dest_img_path)


def generate_singlepage_docs(yearbooks_root, all_valid_docs, preprocessed_root):

    all_process_inputs = [(doc_name, yearbooks_root, preprocessed_root) for doc_name in all_valid_docs]
    pool_size = 30 #TODO: change
    with Pool(processes=pool_size) as p: 
        with tqdm(total=len(all_valid_docs)) as pbar:
            #for i, _ in enumerate(map(process_single_doc, all_process_inputs)):
            for i, _ in enumerate(p.imap_unordered(process_single_doc, all_process_inputs)):
                pbar.update() 
    
def make_weaktrain_and_reserve_split(preprocessed_root, target_dir_root):
    all_valid_docs = get_all_yearbooks(preprocessed_root)
    docs_by_first_number = defaultdict(list)
    for doc in all_valid_docs:
        first_number = doc.split('-')[0]
        docs_by_first_number[first_number].append(doc)
         
    num_reserve = 500
    first_numbers_sorted= sorted(int(x) for x in  docs_by_first_number.keys())
    reserve_numbers = first_numbers_sorted[:500]
    weaktrain_numbers = first_numbers_sorted[500:]
    print('reserve numbers: {}'.format(reserve_numbers)) 
    weak_train_dir_name  = 'weak_train'
    manual_reserve_dir_name  = 'manual_train_dev_test'
    weak_train_dir_path_full = os.path.join(target_dir_root, weak_train_dir_name)
    manual_reserve_dir_path_full = os.path.join(target_dir_root, manual_reserve_dir_name)
    create_dir_if_not_exists(weak_train_dir_path_full)
    create_dir_if_not_exists(manual_reserve_dir_path_full)
#    for first_number in tqdm(reserve_numbers):
#        doc_numbers = docs_by_first_number[str(first_number)]
#        #print('doc numbers for {}: {}'.format(first_number, doc_numbers))
#        for doc_number in doc_numbers:
#            source_dir = os.path.join(preprocessed_root, doc_number)
#            target_dir = os.path.join(manual_reserve_dir_path_full, doc_number)
#            #print('copy dir from {} to {}'.format(source_dir, target_dir))
#            #create_dir_if_not_exists(target_dir)
#            copytree(source_dir, target_dir)

    for first_number in tqdm(weaktrain_numbers):
        doc_numbers = docs_by_first_number[str(first_number)]
        for doc_number in doc_numbers:
            source_dir = os.path.join(preprocessed_root, doc_number)
            target_dir = os.path.join(weak_train_dir_path_full, doc_number)
            copytree(source_dir, target_dir)



if __name__ == "__main__":
    yearbooks_root = '/mnt/ds3lab-scratch/jrausch/docparser/datasets/yearbooks_v3/valid-annotations'
    all_valid_docs = get_all_yearbooks(yearbooks_root)
    preprocessed_root = '/mnt/ds3lab-scratch/jrausch/docparser/datasets/yearbooks_v3_with_splits/preprocessed'
    preprocessed_and_split_dataset_root = '/mnt/ds3lab-scratch/jrausch/docparser/datasets/yearbooks_v3_with_splits/'
    #generate_singlepage_docs(yearbooks_root, all_valid_docs, preprocessed_root)
    make_weaktrain_and_reserve_split(preprocessed_root, preprocessed_and_split_dataset_root)
