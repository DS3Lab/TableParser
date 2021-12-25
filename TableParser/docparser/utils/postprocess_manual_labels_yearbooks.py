def create_splits_for_manual_docs():

#    clean_docs = sorted(remaining_clean_docs)
    #random.shuffle(clean_docs)
#
#    #train_docs = clean_docs[:160]

if __name__ == "__main__":
    random.seed(123)
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

