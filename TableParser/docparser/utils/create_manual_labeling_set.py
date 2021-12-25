import os
import random
from collections import defaultdict
from shutil import copyfile
import json
from filter_docs_and_apply_heuristics import copy_subset_to_new_folder, copy_folder, generate_per_tabular_files
import arxiv_heuristics 
from arxiv_heuristics import use_heuristics_to_generate_rows_and_cols, cluster_content_blocks_under_categories


dest_dir_splits = dict()
dest_dir_splits['random_subset'] = {'tables': '/mnt/ds3lab-scratch/arxiv/mixedset/annotations_random_subset_v1_splits/', 'mixed': '/mnt/ds3lab-scratch/arxiv/mixedset/mixed_v2_random_subset_with_imgs_splits/'}

in_tag_per_page = 'mx14'
out_tag_per_page = 'mx14manual'

test_folder = os.path.join(dest_dir_splits['random_subset']['mixed'], 'by_page', 'test')
test_id_list = os.path.join(dest_dir_splits['random_subset']['mixed'], 'by_page', 'test_ids.txt')

dir_manual_labeling_subset_from_automatic_test_set =  os.path.join(dest_dir_splits['random_subset']['mixed'], 'by_page', 'manual_labeling')


manual_sample_ids_file = 'random_samples_for_manual_labeling_v2.txt'
manual_sample_ids_path = os.path.join(dest_dir_splits['random_subset']['mixed'], 'by_page', manual_sample_ids_file)


annotation_gui_manual_labeling_dir = '/home/jrausch/git/doc_annotation/doc-anno-server/public/documents/manual_arxivmixed'
#annotation_gui_manual_labeling_clean_dir = '/home/jrausch/git/doc_annotation/doc-anno-server/public/documents/manual_arxivmixed_clean_v2'
annotation_gui_manual_labeling_clean_dir = '/home/jrausch/git/doc_annotation/doc-anno-server/public/documents/manual_arxivmixed_clean_v3'


cleaned_manual_sample_ids_file = 'cleaned_samples_for_manual_labeling_v2.txt'
#cleaned_manual_sample_ids_file_train = 'cleaned_random_samples_for_manual_labeling_v2_train.txt'
#cleaned_manual_sample_ids_file_dev = 'cleaned_random_samples_for_manual_labeling_v2_dev.txt'
#cleaned_manual_sample_ids_file_test = 'cleaned_random_samples_for_manual_labeling_v2_test.txt'
cleaned_manual_sample_ids_path = os.path.join(annotation_gui_manual_labeling_clean_dir, cleaned_manual_sample_ids_file)


cleaned_annotations_split_root = os.path.join(annotation_gui_manual_labeling_clean_dir, 'splits')

cleaned_manual_sample_ids_path_train = os.path.join(cleaned_annotations_split_root, 'by_page', 'train_ids.txt')
cleaned_manual_sample_ids_path_dev = os.path.join(cleaned_annotations_split_root, 'by_page', 'dev_ids.txt')
cleaned_manual_sample_ids_path_test = os.path.join(cleaned_annotations_split_root, 'by_page', 'test_ids.txt')

def decision(probability):
    return random.random() < probability

def create_manual_labeling_set():
  
    pages_with_table = set()
    pages_with_figure = set()
    pages_with_list = set()
    pages_with_abstract = set()
    pages_with_equation = set()
     
    with open(test_id_list, 'r') as in_file:
        docs_subset = list(set(x.strip() for x in in_file.readlines()))

    per_page_docs_by_docname = defaultdict(list)
    docs_subset = sorted(docs_subset)
    random.shuffle(docs_subset)
    all_page_docs = []


    page_docs_limited_to_one_per_doc = set()
    docs_in_random_sample = set()



    for per_page_doc in docs_subset:
        doc_name = per_page_doc.split('_')[0]
        per_page_docs_by_docname[doc_name].append(per_page_doc)



    all_source_docs = sorted(list(per_page_docs_by_docname.keys()))
    random.shuffle(all_source_docs)
    random_labeling_selection = all_source_docs[:1000]

    random_labeling_selection_single_pages = []
    total_docs_that_have_title_page = 0
    for random_doc in random_labeling_selection:
        current_doc_pages = per_page_docs_by_docname[random_doc]
        doc_page_by_page_nr = {int(doc_page.split('_')[-1]):doc_page for doc_page in current_doc_pages}
        all_page_nrs = list(doc_page_by_page_nr.keys())
        random_page_nr = random.choice(all_page_nrs)
        random_doc_page = doc_page_by_page_nr[random_page_nr]
        random_labeling_selection_single_pages.append(random_doc_page)
#
    page_distribution = defaultdict(int)
    for random_page in random_labeling_selection_single_pages:
        page_nr = int(random_page.split('_')[-1])
        page_distribution[page_nr] += 1
    print('total source docs that have a first page: {}'.format(total_docs_that_have_title_page))
    print(page_distribution)


    with open(manual_sample_ids_path, 'w') as out_file:
        out_file.write('\n'.join(doc_id for doc_id in random_labeling_selection_single_pages))



def copy_files_to_manual_set_folder():

    with open(manual_sample_ids_path, 'r') as in_file:
        docs_subset = list(set(x.strip() for x in in_file.readlines()))
#   
    #src_folder = test_folder

    copy_subset_to_new_folder(None, manual_sample_ids_path, src_dir=test_folder, dest_dir=dir_manual_labeling_subset_from_automatic_test_set) 
    #

def copy_files_to_gui_folder():

    copy_subset_to_new_folder(None, manual_sample_ids_path, src_dir=dir_manual_labeling_subset_from_automatic_test_set, dest_dir=annotation_gui_manual_labeling_dir) 

def make_manual_table_annotations_easier_to_label():
    with open(manual_sample_ids_path, 'r') as in_file:
        docs_subset = list(set(x.strip() for x in in_file.readlines()))

    src_dir = dir_manual_labeling_subset_from_automatic_test_set
    dest_dir = dir_manual_labeling_subset_from_automatic_test_set
    for doc in docs_subset:
        src_annotations_file = os.path.join(src_dir, doc, doc + '-{}.json'.format(in_tag_per_page))
        with open(src_annotations_file) as f:
            annotations = json.load(f)

        remove_ann_ids = set()
        multicell_count = 0
        for ann in annotations:
            if ann['category'] == 'table_cell':
                col_range = ann['col_range']
                row_range = ann['row_range']
                if any(x is None for x in col_range + row_range):
                    #print('none value in col range or row range of table cell ann')
                    remove_ann_ids.add(ann['id'])
                elif col_range[1] > col_range[0]:
                    multicell_count += 1
                    #print('found multicell with col range: {}'.format(col_range))
                elif row_range[1] > row_range[0]:
                    multicell_count += 1
                    #print('found multicell with row range: {}'.format(col_range))
                else:
                    remove_ann_ids.add(ann['id'])
        annotations = [ann for ann in annotations if ann['id'] not in remove_ann_ids]
        annotations = arxiv_heuristics.delete_structure_annotations_without_children(annotations)
        dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag_per_page))
        dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
        with open(dest_annotations_fullpath, 'w') as out_file:
            json.dump(annotations, out_file, indent=1, sort_keys=True)

def keep_only_highlevel_content_blocks(annotations):
    ann_by_id = dict()
    anns_by_cat = defaultdict(list)
    ann_children_ids = defaultdict(list)
    for ann in annotations:
        if ann['id'] in ann_by_id:
            print('duplicates in list!')
        ann_by_id[ann['id']] =  ann 
        anns_by_cat[ann['category']].append(ann)
        ann_children_ids[ann['parent']].append(ann['id'])

    for contentblock_ann in anns_by_cat['content_block']:
        if ann_by_id[contentblock_ann['parent']]['category'] not in {'section', 'document'}:
            contentblock_ann['category'] = 'content_lines'

    #look for 'immediate' box annotations under content blocks and convert them into content_lines type


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


    new_anns = []
    for contentblock_ann in anns_by_cat['content_block']:
        for contentblock_child_ann_id in ann_children_ids[contentblock_ann['id']]:
            contentblock_child_ann = ann_by_id[contentblock_child_ann_id]
            if contentblock_child_ann['category'] == 'box':
                new_content_lines_ann_id = get_new_ann_id()
                new_block_ann = {'category':'content_lines', 'parent':contentblock_ann['id'], 'id':new_content_lines_ann_id}
                #print('converting "box" child of content_block into "content_lines"')
                new_anns.append(new_block_ann)
                contentblock_child_ann['parent'] = new_content_lines_ann_id 
    annotations += new_anns
    return annotations


#def split_bibliography_for_double_column_docs(annotations):
#    ann_by_id = dict()
#    anns_by_cat = defaultdict(list)
#    ann_children_ids = defaultdict(list)
#    for ann in annotations:
#        if ann['id'] in ann_by_id:
#            print('duplicates in list!')
#        ann_by_id[ann['id']] =  ann 
#        anns_by_cat[ann['category']].append(ann)
#        ann_children_ids[ann['parent']].append(ann['id'])
#
#
#    def create_fn_get_new_ann_id(max_id):
#        current_max_id = max_id
#        def get_new_ann_id():
#            nonlocal current_max_id
#            current_max_id += 1
#            return current_max_id
#        return get_new_ann_id
#
#    max_id = max(set(ann_by_id.keys()))
#    get_new_ann_id = create_fn_get_new_ann_id(max_id)
#
#
#    for bib_ann in anns_by_cat['bibliography']:
#        tabular_child_ids = ann_children_ids[bib_ann['id']]
#        for child_id in tabular_child_ids:
#            child_ann = ann_by_id[child_id]
#            child_ann_bboxes = [ann_by_id[bbox_ann_id] for bbox_ann_id in ann_children_ids[child_ann['id']] if ann_by_id[bbox_ann_id]['category'] == 'box']
#
#    return annotations




def clean_up_tabular_annotations(annotations):

    #len_0 = len(annotations)
    annotations = arxiv_heuristics.delete_structure_annotations_without_children(annotations)
    #len_1 = len(annotations)
    #print('removed {} invalid anns'.format(len_0-len_1))
    

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
                print('more than one bbox for table child of type {} unexpected: \n{}'.format(child_ann['category'], child_ann_bboxes))
            if len(child_ann_bboxes) == 0:
                #child_ann['delete'] = True
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

def find_and_clean_labeled_files_in_gui_folder(out_tag='gtclean'):
    all_docs_in_folder = os.listdir(annotation_gui_manual_labeling_dir)
    #print(all_docs_in_folder)
    all_labeled_j = []
    all_labeled_j2 = []
    all_labeled_j3 = []
    all_labeled_o = []
    unlabeled_docs = []
    src_dir =     annotation_gui_manual_labeling_dir
    dest_dir =     annotation_gui_manual_labeling_dir

    for doc in all_docs_in_folder:
        subdir = os.path.join(src_dir, doc)
        gt_file_cands = [x for x in os.listdir(subdir) if '-gt' in x]
        gtj2_cand = [x for x in gt_file_cands if '-gtj2' in x]
        gtj3_cand = [x for x in gt_file_cands if '-gtj3' in x]
        gto_cand = [x for x in gt_file_cands if '-gto' in x]
        gtj_cand = [x for x in gt_file_cands if '-gtj' in x]
        if len(gtj3_cand) == 1:
            all_labeled_j3 += gtj3_cand
        elif len(gtj2_cand) == 1:
            all_labeled_j2 += gtj2_cand
        elif len(gto_cand) ==1 and not len(gtj_cand) == 1:
            all_labeled_o += gto_cand
        elif len(gtj_cand) == 1 and not len(gto_cand) == 1:
            all_labeled_j += gtj_cand
        else:
            #print("WARNING: Could not find definite GT label in {}: {}".format(subdir, gt_file_cands))
            unlabeled_docs.append(doc)

    
    print('total unlabeled: {}, total labeled: {} gtj and {} gto, gtj2: {}, gtj3: {}'.format(len(unlabeled_docs), len(all_labeled_j), len(all_labeled_o), len(all_labeled_j2), len(all_labeled_j3)))
    docs_with_manual_label = {x.split('-gt')[0] : x for x in all_labeled_j + all_labeled_o + all_labeled_j2 + all_labeled_j3}

    #cleanup steps:
    for doc, annotations_in_name in docs_with_manual_label.items():

        print('doing doc {}'.format(doc))
        src_annotations_file = os.path.join(src_dir, doc, annotations_in_name)
        with open(src_annotations_file) as f:
            annotations = json.load(f)
            

            annotations= clean_up_tabular_annotations(annotations)
            annotations= keep_only_highlevel_content_blocks(annotations)
            annotations = cluster_content_blocks_under_categories(annotations, root_types_to_consider = ['bibliography'], new_block_category='bib_block')
            #annotations= split_bibliography_for_double_column_docs(annotations)

            ann_by_id = dict()

            for ann in annotations:
                ann_by_id[ann['id']] = ann
            for ann in annotations:
                if ann['category'] == 'table_cell':
                    if not 'col_range' in ann:
                        raise AttributeError('no col range in {}: {}'.format(doc, ann))
                    if not 'row_range' in ann:
                        raise AttributeError('no row range in {}: {}, parent: {}'.format(doc, ann, ann_by_id[ann['parent']]))

#        annotations = [ann for ann in annotations if ann['id'] not in remove_ann_ids]
#        annotations = arxiv_heuristics.delete_structure_annotations_without_children(annotations)
        dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
        dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)
        with open(dest_annotations_fullpath, 'w') as out_file:
            json.dump(annotations, out_file, indent=1, sort_keys=True)
    return docs_with_manual_label
    #print(docs_with_manual_label)

def copy_clean_files_to_new_folder(docs_with_manual_label):

    if not os.path.exists(annotation_gui_manual_labeling_clean_dir):
        os.mkdir(annotation_gui_manual_labeling_clean_dir)
    with open(cleaned_manual_sample_ids_path, 'w') as out_file:
        out_file.write('\n'.join(doc_id for doc_id in docs_with_manual_label.keys()))


    copy_subset_to_new_folder(None, cleaned_manual_sample_ids_path, src_dir=annotation_gui_manual_labeling_dir, dest_dir=annotation_gui_manual_labeling_clean_dir) 
    



def create_splits_for_clean_files(previous_train_doc_ids, previous_dev_doc_ids, previous_test_doc_ids):
     
    with open(cleaned_manual_sample_ids_path, 'r') as in_file:
        all_clean_docs = [x.strip() for x in in_file.readlines()]

    
    remaining_clean_docs = set(all_clean_docs) - set(previous_train_doc_ids) - set(previous_dev_doc_ids) - set(previous_test_doc_ids)
    print('{} docs remaining for dev/test split after subtracing {} train docs and {} dev docs and {} test docs'.format(len(remaining_clean_docs), len(previous_train_doc_ids), len(previous_dev_doc_ids), len(previous_test_doc_ids)))

    clean_docs = sorted(remaining_clean_docs)

    #random.shuffle(clean_docs)
#
#    #train_docs = clean_docs[:160]
    train_docs = previous_train_doc_ids
    dev_docs = previous_dev_doc_ids 
    test_docs = previous_test_doc_ids
    #dev_docs = clean_docs[:80]
    #test_docs = set(clean_docs) - set(train_docs) - set(dev_docs)
    #test_docs =remaining_clean_docs 
#
#
#
    by_page_subdir = os.path.join(cleaned_annotations_split_root, 'by_page')
    train_dir = os.path.join(by_page_subdir, 'train')
    dev_dir = os.path.join(by_page_subdir, 'dev')
    test_dir = os.path.join(by_page_subdir, 'test')
#
    if not os.path.exists(cleaned_annotations_split_root):
        os.mkdir(cleaned_annotations_split_root)
    if not os.path.exists(by_page_subdir):
        os.mkdir(by_page_subdir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(dev_dir):
        os.mkdir(dev_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)


    with open(cleaned_manual_sample_ids_path_train, 'w') as out_file:
        out_file.write('\n'.join(doc_id for doc_id in train_docs))
#
    with open(cleaned_manual_sample_ids_path_dev, 'w') as out_file:
        out_file.write('\n'.join(doc_id for doc_id in dev_docs))

    with open(cleaned_manual_sample_ids_path_test, 'w') as out_file:
        out_file.write('\n'.join(doc_id for doc_id in test_docs))

#
    for doc in train_docs: 
        src_doc_folder = os.path.join(annotation_gui_manual_labeling_clean_dir, doc)
        dest_doc_folder = os.path.join(train_dir, doc)
        copy_folder(src_doc_folder, dest_doc_folder)
#
#
    for doc in dev_docs: 
        src_doc_folder = os.path.join(annotation_gui_manual_labeling_clean_dir, doc)
        dest_doc_folder = os.path.join(dev_dir, doc)
        copy_folder(src_doc_folder, dest_doc_folder)


    for doc in test_docs: 
        src_doc_folder = os.path.join(annotation_gui_manual_labeling_clean_dir, doc)
        dest_doc_folder = os.path.join(test_dir, doc)
        copy_folder(src_doc_folder, dest_doc_folder)


    generate_per_tabular_files(None, splits_root_dir=cleaned_annotations_split_root, in_tag='gtclean', out_tag='gtcleantabul')


def complete_previous_manual_cells_data():
    manual_cells_data_root = '/mnt/ds3lab-scratch/arxiv/mixedset/big_tables_subset_manual/manual/manual_splits'
    cleaned_manual_cells_data_root = '/mnt/ds3lab-scratch/arxiv/mixedset/big_tables_subset_manual/manual/manual_cleaned/manual_splits'
    original_dirs_root = '/mnt/ds3lab-scratch/arxiv/mixedset/big_tables_subset_manual/manual/grouped_by_document'
#create images again
    by_page_root =os.path.join(manual_cells_data_root, 'by_page') 
    cleaned_by_page_root =os.path.join(cleaned_manual_cells_data_root, 'by_page') 
    #splits = ['train', 'dev', 'test']
    splits = ['train']

    src_dir = manual_cells_data_root 

    for split in splits:
        split_dir = os.path.join(by_page_root, split)
        cleaned_split_dir = os.path.join(cleaned_by_page_root, split)
        if not os.path.exists(cleaned_split_dir):
            os.mkdir(cleaned_split_dir)
        valid_ids = []
        docs_in_dir = os.listdir(split_dir)
        #print(docs_in_dir)
        for doc in docs_in_dir:
            doc_id, page = doc.rsplit('_')
            dest_subdir = os.path.join(cleaned_split_dir, doc)

            dest_meta_ann_name = os.path.join(doc + '.json')
            dest_meta_ann_path = os.path.join(dest_subdir, dest_meta_ann_name)
            new_meta_contents = {'id': doc, 'title':doc, 'pages':1}
            #print('new meta json: {}'.format(dest_meta_ann_path))
            with open(dest_meta_ann_path, 'w') as out_file:
                json.dump(new_meta_contents, out_file, indent=1, sort_keys=True)

            dpi72_img_path = os.path.join(original_dirs_root, split, doc_id, doc_id + '-{}.png'.format(page))
            dest_image_path = os.path.join(dest_subdir, doc + '-0.png')
            copyfile(dpi72_img_path, dest_image_path)

            if not os.path.exists(dest_subdir ):
                os.mkdir(dest_subdir )
            in_tag = 'mn'
            out_tag = 'mnclean'
            src_dir = split_dir
            dest_dir = cleaned_split_dir
            input_tuple = (doc, in_tag, out_tag, src_dir, dest_dir)
            success = use_heuristics_to_generate_rows_and_cols(input_tuple)


        
            #fix page number to zero

            if success is None:
                print('error for: {}'.format(doc))
            else:

                dest_annotations_relpath = os.path.join(doc, doc + '-{}.json'.format(out_tag))
                dest_annotations_fullpath = os.path.join(dest_dir, dest_annotations_relpath)

                with open(dest_annotations_fullpath) as f:
                    annotation_list = json.load(f)
                for ann in annotation_list:
                    if 'page' in ann:
                        ann['page'] = 0
                with open(dest_annotations_fullpath, 'w') as out_file:
                    json.dump(annotation_list, out_file, indent=1, sort_keys=True)

                second_stage_input_tuple = (doc, out_tag, out_tag, dest_dir, dest_dir)
                #success2 = arxiv_heuristics.assign_rowcol_numbers_to_cells_and_rowscols(second_stage_input_tuple)
                success2 = arxiv_heuristics.extend_cells_to_row_and_col_borders(second_stage_input_tuple)
                if success2 is not None:
                    valid_ids.append(success)


        cleaned_split_ids_path = os.path.join(cleaned_by_page_root, split+'_ids.txt')
        with open(cleaned_split_ids_path, 'w') as out_file:
            out_file.write('\n'.join(doc_id for doc_id in valid_ids))

        annotation_gui_manual_labeling_dir = os.path.join('/home/jrausch/git/doc_annotation/doc-anno-server/public/documents/arxivtables_manualcleaned_'+ split)
    
        copy_subset_to_new_folder(None, cleaned_split_ids_path, src_dir=cleaned_split_dir, dest_dir=annotation_gui_manual_labeling_dir) 

def verify_splits_are_consistent():
    with open(test_id_list, 'r') as in_file:
        docs_subset = list(set(x.strip() for x in in_file.readlines()))
    
    test_docs = docs_subset

    manual_ids = '/mnt/ds3lab-scratch/arxiv/mixedset/manual_arxivmixed_bak_23_05_19/manual_file_ids.txt'
    with open(manual_ids, 'r') as in_file:
        manual_subset = list(set(x.strip() for x in in_file.readlines()))

    print(test_docs[:10])
    print(manual_subset[:10])
    print('test ids: {}'.format(len(test_docs)))
    print('manual ids: {}'.format(len(manual_subset)) )
    intersct = set.intersection(set(test_docs), set(manual_subset))
    print('intersection ids: {}'.format(len(intersct)))
    print(set(manual_subset) - set(set(test_docs)))

def fetch_previous_train_set_ids(split_name):
    split_set_ids_path = 'manual_split_ids/by_page/{}_ids.txt'.format(split_name)
    with open(split_set_ids_path, 'r') as in_file:
        split_docs = list(set(x.strip() for x in in_file.readlines()))

    per_page_docs_by_docname = dict()
    for split_doc in split_docs:
        doc_name = split_doc.rsplit('_')[0]
        if doc_name in per_page_docs_by_docname:
            print('error: more than one page per doc for {}'.format(doc))
            return
        per_page_docs_by_docname[doc_name] = split_doc

    return per_page_docs_by_docname

if __name__ == "__main__":
    random.seed(123)
    #create_manual_labeling_set()
    #copy_files_to_manual_set_folder()
    #make_manual_table_annotations_easier_to_label()
    #copy_files_to_gui_folder()
    train_set_ids = fetch_previous_train_set_ids('train')
    dev_set_ids = fetch_previous_train_set_ids('dev')
    test_set_ids = fetch_previous_train_set_ids('test')
    
    previous_train_doc_ids = list(train_set_ids.values())
    previous_dev_doc_ids = list(dev_set_ids.values())
    previous_test_doc_ids = list(test_set_ids.values())
    print('previous train docs: {}, dev: {}, test: {}'.format(len(previous_train_doc_ids), len(previous_dev_doc_ids), len(previous_test_doc_ids)))
    previous_ids_all = set(previous_train_doc_ids + previous_dev_doc_ids + previous_test_doc_ids)

    docs_with_manual_label = find_and_clean_labeled_files_in_gui_folder(out_tag='gtclean')
    matched_files = set.intersection(previous_ids_all, docs_with_manual_label)
    print('files from id lists without labels: {}'.format(previous_ids_all - matched_files))
    unmatched_files = set(docs_with_manual_label) - matched_files 
#    
#    #print('total docs: {}, previous train set: {}, docs different than train set: {}'.format(len(docs_with_manual_label), len(previous_train_doc_ids), len(set(docs_with_manual_label) - set(previous_train_doc_ids))))
    print('total docs: {}, docs that dont appear in any of the splits: {}'.format(len(docs_with_manual_label), unmatched_files))
    #remove files not accounted for in splits:
    valid_doc_ids = set(previous_train_doc_ids + previous_dev_doc_ids + previous_test_doc_ids)
    docs_with_manual_label = {k:v for k,v in  docs_with_manual_label.items() if k in valid_doc_ids}
    print('total docs after matching with split ids: {}'.format(len(docs_with_manual_label)))


    copy_clean_files_to_new_folder(docs_with_manual_label)
    create_splits_for_clean_files(previous_train_doc_ids, previous_dev_doc_ids, previous_test_doc_ids)
#
#    #from old master thesis
#    #complete_previous_manual_cells_data()

