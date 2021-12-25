import random
import os
import json
from docparser.utils import arxiv_heuristics
import shutil
from collections import defaultdict
from docparser.utils import postprocess_table_structure


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except FileExistsError as e: # python >2.5
        print("removing directory, followed by copytree: {}".format(dst))
        shutil.rmtree(dst)
        shutil.copytree(src, dst)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def generate_cells_from_cols_and_rows(annotations):
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        if ann['id'] in ann_by_id:
            print('duplicates in list!')
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id


    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    for tabular_ann in anns_by_cat['tabular']:
        #cell_anns = []
        row_anns = []
        col_anns = []
        table_cell_anns = []
        col_x_centers = dict()
        row_y_centers = dict() 
        existing_grid_cells = set()
        new_cell_annotations = []
        #bbox_to_ann
        tabular_child_ids = ann_children_ids[tabular_ann['id']]
        for child_id in tabular_child_ids:
            child_ann = ann_by_id[child_id]
            child_ann_bboxes = [ann_by_id[bbox_ann_id] for bbox_ann_id in ann_children_ids[child_ann['id']] if ann_by_id[bbox_ann_id]['category'] == 'box']
            if len(child_ann_bboxes) > 1:
                print('more than one bbox for table child of type {}: \n{}'.format(child_ann['category'], child_ann))
                continue
            if len(child_ann_bboxes) == 0:
                print('table child of type {} has no bbox. unexpected: \n{}'.format(child_ann['category'], child_ann_bboxes))
                continue
            child_bbox_ann = child_ann_bboxes[0]

            x_center = child_bbox_ann['bbox'][0] + child_bbox_ann['bbox'][2] / 2
            y_center = child_bbox_ann['bbox'][1] + child_bbox_ann['bbox'][3] / 2

            if child_ann['category'] == 'table_cell':
                table_cell_anns.append(child_ann)
            elif child_ann['category'] == 'table_row':
                #if y_center in row_y_centers:
                    #print('y center already exists in dictionary for {}'.format(doc))
                assert y_center not in row_y_centers
                #row_y_centers[y_center].append([child_ann, child_bbox_ann])
                row_y_centers[y_center] = tuple([child_ann, child_bbox_ann])
                row_anns.append(child_ann)

            elif child_ann['category'] == 'table_col':
                #if x_center in col_x_centers:
                    #print('x center already exists in dictionary for {}'.format(doc))

                assert x_center not in col_x_centers
                col_anns.append(child_ann)
                col_x_centers[x_center] = tuple([child_ann, child_bbox_ann])
                #col_x_centers[x_center].append([child_ann, child_bbox_ann])

        #print('found and sorted {} rows and {} cols '.format(len(row_y_centers), len(col_x_centers)))
        x_center_values = sorted(list(col_x_centers.keys()))
        y_center_values = sorted(list(row_y_centers.keys()))
      

        #expected_grid = [-1] * len(row_anns) * len(col_anns) 
        all_table_cells = [ann for ann in annotations if ann['category'] == 'table_cell']
        pre_labeled_multicells = [ann for ann in table_cell_anns if ann['category'] == 'table_cell']
#        if len(pre_labeled_multicells) != len(all_table_cells):
#            print('difference in table cell nrs')
#        if len(pre_labeled_multicells) > 0:
#            print('{} pre-labeled cells'.format(len(pre_labeled_multicells)))
        for ann in pre_labeled_multicells:
            row_range, col_range = ann['properties'].split(',')
            #print('row range raw: {}, col range raw: {}'.format(row_range, col_range))
            if '-' not in row_range:
                if len(row_range) > 0:
                    row_start = int(row_range)
                    row_end = int(row_range)
                else:
                    raise AttributeError('bad cell')
            else:
                row_start, row_end = row_range.split('-')

            if '-' not in col_range:
                if len(col_range) > 0:
                    col_start = int(col_range)
                    col_end = int(col_range)
                else:
                    raise AttributeError('bad cell')
            else:
                col_start, col_end = col_range.split('-')
            new_row_range = [int(row_start), int(row_end)]
            new_col_range = [int(col_start), int(col_end)]
            assert new_row_range[1] - new_row_range[0] >= 1 or new_col_range[1] - new_col_range[0] >= 1
            for grid_row_nr in range(new_row_range[0], new_row_range[1]+1):
                for grid_col_nr in range(new_col_range[0], new_col_range[1]+1):
                    print('adding ({}, {}) to grid for row range {} and col range {}'.format(grid_row_nr, grid_col_nr, new_row_range, new_col_range))
                    existing_grid_cells.add((grid_row_nr, grid_col_nr))

            ann['row_range'] = [row_start, row_end]
            ann['col_range'] = [col_start, col_end]
            ann['properties'] = "{}-{},{}-{}".format(row_start, row_end, col_start, col_end)

        for col_nr, x_center_value in enumerate(x_center_values):
            #for col_ann, _ in col_x_centers[x_center_value]:
                col_ann, _ = col_x_centers[x_center_value]
                col_ann['col_nr'] = col_nr  
                col_ann['properties'] = col_nr  
        for row_nr, y_center_value in enumerate(y_center_values):
            #for row_ann, _ in row_y_centers[y_center_value]:
                row_ann, _ = row_y_centers[y_center_value]
                row_ann['row_nr'] = row_nr  
                row_ann['properties'] = row_nr  
        for col_nr, x_center_value in enumerate(x_center_values):
            #for col_ann, col_bbox in col_x_centers[x_center_value]:
            for row_nr, y_center_value in enumerate(y_center_values):
                col_ann, col_bbox = col_x_centers[x_center_value]
                row_ann, row_bbox = row_y_centers[y_center_value]
            #for row_ann, row_bbox in row_y_centers[y_center_value]:
                intsct_x0 = col_bbox['bbox'][0] 
                intsct_x1 = col_bbox['bbox'][0] + col_bbox['bbox'][2]
                intsct_y0 = row_bbox['bbox'][1] 
                intsct_y1 = row_bbox['bbox'][1] + row_bbox['bbox'][3]
                page = row_bbox['page']
                bbox_from_intersection = [intsct_x0, intsct_y0, intsct_x1-intsct_x0, intsct_y1-intsct_y0]
                row_start = row_nr
                row_end = row_nr
                col_start = col_nr
                col_end = col_nr


                grid_coord = (row_nr, col_nr)
                if grid_coord not in existing_grid_cells:             
                    new_cell_id = get_new_ann_id()
                    new_cell_ann = {'category': 'table_cell', 'id': new_cell_id, 'parent':tabular_ann['id']}
                    new_cell_ann['row_range'] = [row_start, row_end]
                    new_cell_ann['col_range'] = [col_start, col_end]
                    new_cell_ann['properties'] = "{}-{},{}-{}".format(row_start, row_end, col_start, col_end)


                    new_bbox_ann = {'category': 'box', 'id': get_new_ann_id(), 'parent':new_cell_id, 'bbox':bbox_from_intersection, 'page':page}
                    new_cell_annotations.append(new_cell_ann)
                    new_cell_annotations.append(new_bbox_ann)


                    existing_grid_cells.add(grid_coord)
         
        #sanity check:
#        print('filled {} grid cells, expected: {}'.format(len(existing_grid_cells), len(row_anns) * len(col_anns)))
        expected_grid_cells = set()
        for row_nr in range(len(row_anns)):
           for col_nr in range(len(col_anns)):
                expected_grid_cells.add((row_nr,col_nr)) 
        print('{} rows, {} cols; expected cells {}, existing cells: {}'.format(len(row_anns), len(col_anns), len(expected_grid_cells), len(existing_grid_cells)))
#        print('generated grid cells: {}'.format(existing_grid_cells))        
#        print('extra grid cells: {}'.format(existing_grid_cells - expected_grid_cells))

        assert len(x_center_values) * len(y_center_values)  == len(row_anns) * len(col_anns)
        assert len(existing_grid_cells) == len(row_anns) * len(col_anns)
        annotations += new_cell_annotations

            #print('col range: {}, row range: {}'.format(new_col_range, new_row_range))
        #annotations = [ann for ann in annotations if ann.get('delete', False) != True]
    return annotations



def clean_up_tabular_annotations(annotations, doc):

    annotations = arxiv_heuristics.delete_structure_annotations_without_children(annotations)
    

    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        if ann['id'] in ann_by_id:
            print('duplicates in list!')
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])


    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id


    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)


    num_cells = len(anns_by_cat['table_cells'])
    num_rows = len(anns_by_cat['table_row'])
    num_cols = len(anns_by_cat['table_col'])
   
    if not (num_rows == num_cols == 0):
        raise NotImplementedError
        #normal_cells = find_non_multicells(annotations)
        num_normal_cells = len(normal_cells)
        if num_normal_cells == 0:
            generate_cells_from_cols_and_rows()
    elif num_rows == num_cols == 0:
        #generate rows and columns from cells 
        annotations, multi_tabular_cell_ids = generate_rows_and_cols_from_cells(annotations, doc)

    #debug: only preserve rows and cols
    annotations_without_regular_cells = []
    for ann in annotations:
        if ann['category'] == 'table_cell':
            if ann['id'] in multi_tabular_cell_ids:
                annotations_without_regular_cells.append(ann)
                if '31-0' in doc:
                    print('multicell ann: {}'.format(ann))
        else:
            annotations_without_regular_cells.append(ann)
    if '31-0' in doc:
        input('added multicell cell anns')
#    annotations_without_non_multi_cells = [x for x in annotations if not (x['category'] == 'table_cell' and x['id'] not in multi_tabular_cell_ids)]
    
    annotations_without_regular_cells = arxiv_heuristics.delete_structure_annotations_without_children(annotations_without_regular_cells)




    return annotations_without_regular_cells

def generate_rows_and_cols_from_cells(annotation_list, doc, page=0):
    #print(annotation_list)
    def create_fn_get_new_ann_id(max_id):
        current_max_id = max_id
        def get_new_ann_id():
            nonlocal current_max_id
            current_max_id += 1
            return current_max_id
        return get_new_ann_id
    num_anns_start = len(annotation_list)

    
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    #max_id = 0
    for ann in annotation_list:
        ann_children_ids[ann['parent']].append(ann['id'])
    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)


    if len(ann_by_id) == 0:
        return #no annotations found
    max_id = max(set(ann_by_id.keys()))
    get_new_ann_id = create_fn_get_new_ann_id(max_id)

    multi_tabular_cell_ids = set()

    for tabular_ann in anns_by_cat['tabular']:

        all_tabular_children_ids = arxiv_heuristics.get_all_children_recursive(annotation_list, set([tabular_ann['id']]))
        tabular_cell_anns = []
        tabular_row_anns = []
        #tabular_row_indeces_to_delete = set()
        for ann_id in all_tabular_children_ids:
            try:
                current_ann = ann_by_id[ann_id]
            except KeyError as e:
                print('Could not find an annotation by its id in doc {}: {}'.format(doc, e))
                return 
            if current_ann['category'] == 'table_cell':
                tabular_cell_anns.append(current_ann)
            elif current_ann['category'] == 'table_row':
                tabular_row_anns.append(current_ann)

        
        all_cell_bbox_anns = []
        regular_tabular_cell_anns = []
        combined_multi_tabular_cell_anns = []


        # Determine rows/columns bounding boxes for cells
        x_centers = defaultdict(list)
        y_centers = defaultdict(list)
        tabular_cell_id_to_merged_bbox = dict()
        for tabular_cell_ann in tabular_cell_anns:

            if not 'properties' in tabular_cell_ann: 
                print('cell has no properties: {}. adding empty properties'.format(tabular_cell_ann))
                tabular_cell_ann['properties'] = ''
            properties = tabular_cell_ann['properties']

            #cell_children_ids = ann_children_ids[tabular_cell_ann['id']]

            all_cell_children_ids = arxiv_heuristics.get_all_children_recursive(annotation_list, set([tabular_cell_ann['id']]))

            cell_children_bbox_anns = [ann_by_id[child_id] for child_id in all_cell_children_ids if ann_by_id[child_id]['category'] == 'box']
            merged_cell_bbox = None
            all_bboxes_on_page = [x['bbox'] for x in cell_children_bbox_anns if x['page'] == page]
            #print(tabular_cell_ann)
            if len(all_bboxes_on_page) == 0:
                #print('skipping page..')
                continue
            merged_cell_bbox = arxiv_heuristics.union(all_bboxes_on_page)
            tabular_cell_id_to_merged_bbox[tabular_cell_ann['id']] = merged_cell_bbox 
         
            if ';' in properties:
                #combined_multi_tabular_cell_anns.append(tabular_cell_ann)
                children_of_multi_tabular_cell = ann_children_ids[tabular_cell_ann['id']]
                multi_tabular_cell_ids.add(tabular_cell_ann['id'])
                #print("adding 'L' multicell: {}".format(tabular_cell_ann))
                for child_ann_id in children_of_multi_tabular_cell:
                    if ann_by_id[child_ann_id]['category'] == 'table_cell':
                        #print("adding child of 'L' multicell: {}".format(ann_by_id[child_ann_id]))
                        multi_tabular_cell_ids.add(child_ann_id)
                
                        
            else:
                if '-' in properties:
                    row_str, col_str = properties.split(',')
                    row_start,row_end = row_str.split('-')
                    col_start,col_end = col_str.split('-')
                    if row_start == row_end:
                        y_centers[merged_cell_bbox[1] + merged_cell_bbox[3] / 2].append(tabular_cell_ann)
                    if col_start == col_end:
                        x_centers[merged_cell_bbox[0] + merged_cell_bbox[2] / 2].append(tabular_cell_ann)
                    if row_start != row_end or col_start != col_end:
                        multi_tabular_cell_ids.add(tabular_cell_ann['id'])
                        #print("adding multicell: {}".format(tabular_cell_ann))
                else:
                    regular_tabular_cell_anns.append(tabular_cell_ann)
                    x_centers[merged_cell_bbox[0] + merged_cell_bbox[2] / 2].append(tabular_cell_ann)
                    y_centers[merged_cell_bbox[1] + merged_cell_bbox[3] / 2].append(tabular_cell_ann)


                 
        for tabular_cell_ann in regular_tabular_cell_anns:
            new_cell_children_ids = ann_children_ids[tabular_cell_ann['id']]
            all_cell_bbox_anns += [ann_by_id[child_id] for child_id in new_cell_children_ids if ann_by_id[child_id]['category'] == 'box']



#        for cell_bbox_ann in all_cell_bbox_anns:
#            x_centers[cell_bbox_ann['bbox'][0] + cell_bbox_ann['bbox'][2] / 2].append(cell_bbox_ann)
#            y_centers[cell_bbox_ann['bbox'][1] + cell_bbox_ann['bbox'][3] / 2].append(cell_bbox_ann)

        
        x_center_values = sorted(list(x_centers.keys()))
        x_center_values_grouped = dict(enumerate(arxiv_heuristics.grouper(x_center_values, pixels_distance=5)))
        y_center_values = sorted(list(y_centers.keys()))
        y_center_values_grouped = dict(enumerate(arxiv_heuristics.grouper(y_center_values, pixels_distance=5)))

        #TODO: also check whether cells within one group are similarly wide/high (exclude multi-row/col)
        new_col_annotations = []
        for col_nr, x_center_value_group in x_center_values_grouped.items():
#            all_cell_bbox_anns = []
#            for x_center_value in x_center_value_group:
#                if len(x_centers[x_center_value]) > 0:
#                    all_cell_bbox_anns += x_centers[x_center_value]
#            if len(all_cell_bbox_anns) <= 2:
#                continue
            bboxes = []
            for x_center_value in x_center_value_group:
                if len(x_centers[x_center_value]) > 0:
                    for cell_ann in x_centers[x_center_value]:
                        cell_ann_union_bbox = tabular_cell_id_to_merged_bbox[cell_ann['id']]
                        bboxes.append(cell_ann_union_bbox)
#            for cell_bbox_ann in all_cell_bbox_anns:
#                page = cell_bbox_ann['page']
#                bboxes.append(cell_bbox_ann['bbox'])
#                cell_ann = ann_by_id[cell_bbox_ann['parent']]
#                cell_ann['col_span'] = [col_nr,col_nr]
            merged_bbox = arxiv_heuristics.union(bboxes)
            new_col_id = get_new_ann_id()
            new_col_ann = {'category': 'table_col', 'id': new_col_id, 'parent':tabular_ann['id'], 'properties':col_nr, 'col_nr':col_nr}
            new_bbox_ann = {'category': 'box', 'id': get_new_ann_id(), 'parent':new_col_id, 'bbox':merged_bbox, 'page':page}
            new_col_annotations.append(new_col_ann)
            new_col_annotations.append(new_bbox_ann)
            #print('added new col with nr {}'.format(col_nr))

        new_row_annotations = []
        for row_nr, y_center_value_group in y_center_values_grouped.items():
            all_cell_bbox_anns = []
#            for y_center_value in y_center_value_group:
#                if len(y_centers[y_center_value]) > 0:
#                    all_cell_bbox_anns += y_centers[y_center_value]
#            if len(all_cell_bbox_anns) <= 1:
#                continue
            #page = -1
            bboxes = []


            for y_center_value in y_center_value_group:

                if len(y_centers[y_center_value]) > 0:
                    for cell_ann in y_centers[y_center_value]:
                        cell_ann_union_bbox = tabular_cell_id_to_merged_bbox[cell_ann['id']]
                        #bboxes += cell_ann_union_bbox 
                        bboxes.append(cell_ann_union_bbox)

#            for cell_bbox_ann in all_cell_bbox_anns:
#                page = cell_bbox_ann['page']
#                bboxes.append(cell_bbox_ann['bbox'])
#                cell_ann = ann_by_id[cell_bbox_ann['parent']]
#                cell_ann['row_span'] = [row_nr, row_nr]
#                #cell['row'] = i + 1
            merged_bbox = arxiv_heuristics.union(bboxes)
            new_row_id = get_new_ann_id()
            new_row_ann = {'category': 'table_row', 'id': new_row_id, 'parent':tabular_ann['id'], 'properties':row_nr, 'row_nr': row_nr}
            new_bbox_ann = {'category': 'box', 'id': get_new_ann_id(), 'parent':new_row_id, 'bbox':merged_bbox, 'page':page}
            for i, existing_row in enumerate(tabular_row_anns):
                existing_row_bbox = [ann_by_id[child_id] for child_id in ann_children_ids[existing_row['id']] if ann_by_id[child_id]['category'] == 'box'][0]
                if second_bbox_contained_in_first_bbox(new_bbox_ann['bbox'], existing_row_bbox['bbox'], tolerance=5) or second_bbox_contained_in_first_bbox(existing_row_bbox['bbox'], new_bbox_ann['bbox'], tolerance=5):
                    #tabular_row_indeces_to_delete.append(i) 
                    existing_row['delete'] = True

            new_row_annotations.append(new_row_ann)
            new_row_annotations.append(new_bbox_ann)
            #print('added new row with nr {}'.format(row_nr))

          
        #print('adding {} new columns and {} new rows'.format(len(new_col_annotations), len(new_row_annotations)) )
        annotation_list += new_col_annotations
        annotation_list += new_row_annotations
        annotation_list = [ann for ann in annotation_list if ann.get('delete', False) != True]
 


    #temporarliy add 'bbox' into col/row annotations for this posprocessing step
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    #max_id = 0
    for ann in annotation_list:
        ann_children_ids[ann['parent']].append(ann['id'])
    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)

    for ann in anns_by_cat['table_row'] + anns_by_cat['table_col']:
        bbox_child_ann_ids = ann_children_ids[ann['id']]
        assert len(bbox_child_ann_ids) == 1
        bbox_child_ann = ann_by_id[bbox_child_ann_ids[0]]
        ann['bbox'] = bbox_child_ann['bbox']

    #propcess rows and columns per tabular
    for tabular_ann in anns_by_cat['tabular']:
        all_tabular_children_ids = arxiv_heuristics.get_all_children_recursive(annotation_list, set([tabular_ann['id']]))
        
        tabular_subtree_anns = [ann_by_id[x] for x in all_tabular_children_ids]
        postprocess_table_structure.adjust_and_move_row_and_column_borders(tabular_subtree_anns)

    #print(annotation_list)
    #now update bboxes and remove 'bbox' key/val from col/row
    for ann in anns_by_cat['table_row'] + anns_by_cat['table_col']:
        bbox_child_ann_ids = ann_children_ids[ann['id']]
        assert len(bbox_child_ann_ids) == 1
        bbox_child_ann = ann_by_id[bbox_child_ann_ids[0]]
        bbox_child_ann['bbox'] = ann.pop('bbox')

         
    num_anns_end = len(annotation_list)
    #print('start anns: {}, end anns: {}'.format(num_anns_start, num_anns_end))
    return annotation_list, multi_tabular_cell_ids

def create_splits_for_manual(yearbooks_manual_root, yearbooks_labeled_gui_subdir, manual_train_id_file, manual_val_id_file):
    full_gui_path = os.path.join(yearbooks_manual_root, yearbooks_labeled_gui_subdir)
    all_docs = os.listdir(full_gui_path)
    all_docs_by_first_nr = {x.split('-')[0] : x for x in all_docs}
#    doc_numbers_l = list(range(31)) + list(range(68,84))
#    #doc_numbers_j1 = list(range(31,68)) + list(range(83,101))
#    doc_numbers_j2 = list(range(83,101))
#    doc_numbers_s = list(range(101,160))
    doc_numbers_j2_train = list(range(83,101))
    doc_numbers_j2_val = list(range(30,50))
    #print(all_docs)  
    #print('doc numbers J: {}'.format(doc_numbers_j))
    all_docs_with_ann = [x for x in all_docs if any('GTJ2' in y and y.endswith('.json') for y in os.listdir(os.path.join(full_gui_path, x)))]
    all_annotated_docs_j2_train = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_j2_train]
    all_annotated_docs_j2_val = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_j2_val]
#    all_annotated_docs_l = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_l]
    #all_annotated_docs_j = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_j]
#    all_annotated_docs_j2 = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_j2]
#    all_annotated_docs_s = [x for x in all_docs_with_ann if int(x.split('-')[0]) in doc_numbers_s]
#    print('all docs with annotation: {}'.format(all_docs_with_ann))
#    print('annotated l docs ({}): {}'.format(len(all_annotated_docs_l), all_annotated_docs_l))
#    #print('annotated j docs ({}): {}'.format(len(all_annotated_docs_j), all_annotated_docs_j))
#    print('annotated s docs ({}): {}'.format(len(all_annotated_docs_s), all_annotated_docs_s))
    print('annotated j2 train docs ({}): {}'.format(len(all_annotated_docs_j2_train), all_annotated_docs_j2_train))
    print('annotated j2 val docs ({}): {}'.format(len(all_annotated_docs_j2_val), all_annotated_docs_j2_val))

    #just take j2 files for manual training set
    manual_train_ids_path = os.path.join(yearbooks_manual_root, manual_train_id_file)
    
    with open(manual_train_ids_path, 'w') as out_file:
        out_file.write('\n'.join(all_annotated_docs_j2_train))


    manual_val_ids_path = os.path.join(yearbooks_manual_root, manual_val_id_file)
    
    with open(manual_val_ids_path, 'w') as out_file:
        out_file.write('\n'.join(all_annotated_docs_j2_val))

def copy_files_into_postpr_dir(yearbooks_manual_root, yearbooks_labeled_gui_subdir, yearbooks_postprocessed_gui_subdir, manual_train_id_file, manual_val_id_file):
    full_gui_path = os.path.join(yearbooks_manual_root, yearbooks_labeled_gui_subdir)
    full_gui_path_postprocessed = os.path.join(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir)
    manual_train_ids_path = os.path.join(yearbooks_manual_root, manual_train_id_file)
    with open(manual_train_ids_path, 'r') as in_file:
        train_docs = [x.strip() for x in in_file.readlines()]

    manual_val_ids_path = os.path.join(yearbooks_manual_root, manual_val_id_file)
    with open(manual_val_ids_path, 'r') as in_file:
        val_docs = [x.strip() for x in in_file.readlines()]

    for doc in train_docs + val_docs:
        doc_dir_src = os.path.join(full_gui_path, doc)
        doc_dir_target = os.path.join(full_gui_path_postprocessed, doc)
        print('copy dir from {} to {}'.format(doc_dir_src, doc_dir_target))
        copyanything(doc_dir_src, doc_dir_target)

#    for doc in train_docs + val_docs:
#        doc_dir_postpr = os.path.join(full_gui_path_postprocessed, doc)
#        gt_files = os.listdir 


def correct_row_and_column_numbers(annotation_list, doc):
    #temporarliy add 'bbox' into col/row annotations for this posprocessing step
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    #max_id = 0
    for ann in annotation_list:
        ann_children_ids[ann['parent']].append(ann['id'])
    for ann in annotation_list:
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)

    for ann in anns_by_cat['table_row'] + anns_by_cat['table_col']:
        bbox_child_ann_ids = ann_children_ids[ann['id']]
        assert len(bbox_child_ann_ids) == 1
        bbox_child_ann = ann_by_id[bbox_child_ann_ids[0]]
        ann['bbox'] = bbox_child_ann['bbox']


    rows_sorted = sorted(anns_by_cat['table_row'], key=lambda x: x['bbox'][1] + (x['bbox'][3] / 2.0))
    cols_sorted = sorted(anns_by_cat['table_col'], key=lambda x: x['bbox'][0] + (x['bbox'][2] / 2.0))
    for row_nr, row in enumerate(rows_sorted):
        ann_by_id[row['id']]['row_nr'] = row_nr
    for col_nr, col in enumerate(cols_sorted):
        ann_by_id[col['id']]['col_nr'] = col_nr

    num_rows = len(rows_sorted)
    num_cols = len(cols_sorted)

    annotation_list = postprocess_table_structure.adjust_and_move_row_and_column_borders(annotation_list)
    #now update bboxes and remove 'bbox' key/val from col/row
    for ann in anns_by_cat['table_row'] + anns_by_cat['table_col']:
        bbox_child_ann_ids = ann_children_ids[ann['id']]
        assert len(bbox_child_ann_ids) == 1
        bbox_child_ann = ann_by_id[bbox_child_ann_ids[0]]
        bbox_child_ann['bbox'] = ann.pop('bbox')

    return annotation_list
    
    

def postprocess_files(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, manual_train_id_file, manual_val_id_file):
    full_gui_path_postprocessed = os.path.join(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir)
    manual_train_ids_path = os.path.join(yearbooks_manual_root, manual_train_id_file)
    with open(manual_train_ids_path, 'r') as in_file:
        train_docs = [x.strip() for x in in_file.readlines()]


    manual_val_ids_path = os.path.join(yearbooks_manual_root, manual_val_id_file)
    with open(manual_val_ids_path, 'r') as in_file:
        val_docs = [x.strip() for x in in_file.readlines()]

    for doc in train_docs + val_docs:
        doc_dir_postpr = os.path.join(full_gui_path_postprocessed, doc)
        
        gt_files = [x for x in os.listdir(doc_dir_postpr) if 'GTJ2' in x]
        gt_files_filtered = [x for x in gt_files if not 'postpr' in x and not 'ppr' in x]
        #print(gt_files_filtered)
        assert len(gt_files_filtered) == 1
       
        gt_file = gt_files_filtered[0] 
        gt_file_path = os.path.join(doc_dir_postpr, gt_file)
        with open(gt_file_path, 'r') as in_file:
            annotations = json.load(in_file)
        
        postpr_annotations= clean_up_tabular_annotations(annotations, doc)

        postpr_gt_clean_file = gt_file.replace('.json', 'postpr.json')
        postpr_gt_clean_path = os.path.join(doc_dir_postpr, postpr_gt_clean_file)
        with open(postpr_gt_clean_path, 'w') as out_file:
            json.dump(postpr_annotations, out_file, indent=1)


#        final_gt_clean_file = gt_file.replace('.json', 'pprfinal.json')
#        final_gt_clean_path = os.path.join(doc_dir_postpr, final_gt_clean_file)
#
#        gt_files_ppr2 = [x for x in gt_files if 'ppr2' in x]
#        if len(gt_files_ppr2) == 1:
#            postpr_file = gt_files_ppr2[0] 
#            postpr_file_path = os.path.join(doc_dir_postpr, postpr_file)
#            with open(postpr_file_path, 'r') as in_file:
#                postpr_annotations = json.load(in_file)
#
#            adjusted_postprocessed_anns = correct_row_and_column_numbers(postpr_annotations, doc)
#            print('saving to {}'.format(final_gt_clean_path))
#            with open(final_gt_clean_path, 'w') as out_file:
#                json.dump(adjusted_postprocessed_anns, out_file, indent=1)
#        elif len(gt_files_ppr2) == 0:
#            print('saving to {}'.format(final_gt_clean_path))
#            with open(final_gt_clean_path, 'w') as out_file:
#                json.dump(debug_annotations, out_file, indent=1)
#

            #no corrections were needed, save postpr as ppr final

#        for ann in annotations:
#            ann_by_id[ann['id']] = ann
#        for ann in annotations:
#            if ann['category'] == 'table_cell':
#                if not 'col_range' in ann:
#                    raise AttributeError('no col range in {}: {}'.format(doc, ann))
#                if not 'row_range' in ann:
#                    raise AttributeError('no row range in {}: {}, parent: {}'.format(doc, ann, ann_by_id[ann['parent']]))
#


def copy_files_to_split_dirs(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, train_split_dir, val_split_dir, manual_train_id_file, manual_val_id_file):
    manual_train_ids_path = os.path.join(yearbooks_manual_root, manual_train_id_file)
    with open(manual_train_ids_path, 'r') as in_file:
        train_docs = [x.strip() for x in in_file.readlines()]

    manual_val_ids_path = os.path.join(yearbooks_manual_root, manual_val_id_file)
    with open(manual_val_ids_path, 'r') as in_file:
        val_docs = [x.strip() for x in in_file.readlines()]

    print('copying docs to train split')
    for doc in train_docs:
        src_path = os.path.join(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, doc)
        target_path = os.path.join(yearbooks_manual_root, train_split_dir, doc)
        copyanything(src_path, target_path)
        #print('copy from {} to {}'.format(src_path, target_path))
    print('copying docs to val split')
    for doc in val_docs:
        src_path = os.path.join(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, doc)
        target_path = os.path.join(yearbooks_manual_root, val_split_dir, doc)
        copyanything(src_path, target_path)


if __name__ == "__main__":
    random.seed(123)
    yearbooks_manual_root = 'datasets/yearbooks_manual'
    yearbooks_labeled_gui_subdir = 'results-20201015'
    yearbooks_postprocessed_gui_subdir = 'results-20201015-postpr-v2'
    manual_train_id_file = 'manual_train_ids_v2.txt'
    manual_val_id_file = 'manual_val_ids_v2.txt'
    create_splits_for_manual(yearbooks_manual_root, yearbooks_labeled_gui_subdir, manual_train_id_file, manual_val_id_file)
    copy_files_into_postpr_dir(yearbooks_manual_root, yearbooks_labeled_gui_subdir, yearbooks_postprocessed_gui_subdir, manual_train_id_file, manual_val_id_file)
#
    postprocess_files(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, manual_train_id_file, manual_val_id_file)
    
    train_split_dir = 'manual_train_v2_with_L_multicells'
    val_split_dir = 'manual_val_v2_with_L_multicells'
    copy_files_to_split_dirs(yearbooks_manual_root, yearbooks_postprocessed_gui_subdir, train_split_dir, val_split_dir, manual_train_id_file, manual_val_id_file)
