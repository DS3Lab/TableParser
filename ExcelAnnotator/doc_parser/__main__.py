import json
import os
import sys
import traceback
import time
import random
from pathlib import Path

from excel_extraction.excel_extraction import ExcelExtraction
from pywintypes import com_error


def run():
    """
    Runs the doc_parser with given inputs.
    :return: None
    """

    load_shuffled_list = True
    # Load intermediate data, first variable in this order matching True will be executed.
    load_extract_annotations = False
    load_calculate_indices = False
    load_calculate_bboxes = False
    load_calculate_annotations = False

    store_intermediate = True
    debug_output = False
    verbose_debug_output = False
    very_verbose_debug_output = False
    output_file_ending = '-automated.json'
    date_time = time.strftime("%Y%m%d-%H%M%S")

    num_file_chunks = 20
    machine_index = 0

    poppler_path = "C:\\Users\\kaise\\source\\repos\\DocParser\\doc_parser\\Util\\poppler\\poppler-0.68.0\\bin"
    metadata_directory = "C:\\Users\\kaise\\source\\repos\\DeExcelerator\\output\\"
    intermediate_directory = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\output\\intermediate\\VM{}\\".format(machine_index)

    # All files
    source_directory = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\data\\"
    shuffled_files_list = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\shuffled_file_list.json"
    output_directory = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\output\\VM{}\\{}\\".format(machine_index, date_time)

    # Example files - Uncomment if only example files should be processed.
    # load_shuffled_list = False
    # source_directory = "C:\\Users\\kaise\\source\\repos\\DocParser\\doc_parser\\example_files\\"
    # shuffled_files_list = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\shuffled_file_list-example-files.json"

    # Variables used for testing. - Uncomment this and block in main for loop as well.
    # output_directory = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\output\\TESTDATA\\VM{}\\{}\\".format(machine_index, date_time)
    # intermediate_directory = "C:\\Users\\kaise\\OneDrive\\Documents\\ETH\\2020_FS\\Master Thesis\\Data\\output\\TESTDATA\\intermediate\\VM{}\\".format(machine_index)
    # files_skip_initially = 0
    # process_max_files = 50
    # file_count = 0

    root_directory = Path(source_directory)

    # Check whether output and intermediate directory exists and create if not.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    if not os.path.exists(intermediate_directory):
        os.makedirs(intermediate_directory, exist_ok=True)

    # Handling of timing measurements.
    timing = []
    timing_file = output_directory + "timing" + date_time + ".json"
    with open(timing_file, 'w', encoding='utf8') as time_file:
        time_file.write("[")

    skipped_files = 0
    com_error_files = 0
    file_name_mapping = {}

    # Handling of file mapping.
    mapping_file = output_directory + "file_mapping" + date_time + ".json"
    with open(mapping_file, 'w', encoding='utf8') as name_mapping_file:
        name_mapping_file.write("[")

    # Load or generate shuffled list of files.
    if load_shuffled_list:
        with open(shuffled_files_list, 'r', encoding='utf8') as shuffled_file:
            file_names = json.loads(shuffled_file.read())
    else:
        # Get relative path to root_directory
        file_names = [os.path.relpath(f, root_directory) for f in root_directory.resolve().glob('**/*') if f.is_file() and (f.name.endswith('.xlsx') or f.name.endswith('.xls'))]

        # Shuffle data, first 500 are for perfect manual annotations.
        random.shuffle(file_names)

        # Store shuffled list for other executions.
        with open(shuffled_files_list, 'w', encoding='utf8') as shuffled_file:
            json.dump(file_names, shuffled_file, indent=2, ensure_ascii=False)

    # Divide all file_names into num_file_chunks and get part for current machine index.
    file_index = machine_index * int(len(file_names) / num_file_chunks)
    if machine_index == num_file_chunks - 1:
        process_file_names = file_names[file_index:]
    else:
        process_file_names = file_names[file_index:file_index + int(len(file_names) / num_file_chunks)]

    # Process all files.
    for file_path_name in process_file_names:
        # Used for testing - Uncomment this and the block above.
        # if files_skip_initially > 0:
        #     files_skip_initially -= 1
        #     file_index += 1
        #     continue
        # if file_count >= process_max_files:
        #     break
        # file_count += 1

        rel_file_path, full_file_name = os.path.split(file_path_name)
        file_name, file_extension = os.path.splitext(full_file_name)
        rel_file_path += "\\"

        # Print filename
        print("[ {} ] {} - {}".format(file_index, file_name, rel_file_path))

        # Timer
        start = time.process_time()
        load_times = [0] * 5
        calculate_times = [-1] * 6
        times_strings = ["metadata: ", "new_table: ", "annotations: ", "bboxes: ", "indices: "]

        # Extract deExcelerator data.
        excel_extraction = ExcelExtraction(source_directory, rel_file_path, file_name, file_extension, file_index, output_directory, output_file_ending, store_intermediate,
                                           intermediate_directory, debug_output, verbose_debug_output, very_verbose_debug_output, poppler_path)

        t_before = time.process_time()
        # Extract metadata.
        if excel_extraction.extract_metadata(metadata_directory + rel_file_path, full_file_name) == -1:
            print("Skip file, no metadata found.")
            skipped_files += 1
            continue

        try:
            # Initialize the calculations.
            excel_extraction.prepare_workbook()

            # Skip some steps if loading previously saved intermediate data.
            calculate_times[0] = time.process_time() - t_before
            t_before = time.process_time()
            if not load_extract_annotations or not excel_extraction.load_extract_annotations():
                # Loading not enabled or failed.
                t_before = time.process_time()
                if not load_calculate_indices or not excel_extraction.load_calculate_indices():
                    # Loading not enabled or failed.
                    t_before = time.process_time()
                    if not load_calculate_bboxes or not excel_extraction.load_calculate_bboxes():
                        # Loading not enabled or failed.
                        t_before = time.process_time()
                        if not load_calculate_annotations or not excel_extraction.load_calculate_annotations():
                            # Loading not enabled or failed.
                            t_before = time.process_time()
                            # Calculate new table array.
                            excel_extraction.calculate_new_table_array()

                            calculate_times[1] = time.process_time() - t_before

                        load_times[1] = t_before - time.process_time()
                        t_before = time.process_time()
                        # Calculate annotations.
                        excel_extraction.calculate_annotations()

                        calculate_times[2] = time.process_time() - t_before

                    load_times[2] = t_before - time.process_time()
                    t_before = time.process_time()
                    # Calculate bbox_dict.
                    excel_extraction.calculate_bboxes()

                    calculate_times[3] = time.process_time() - t_before

                load_times[3] = time.process_time() - t_before
                t_before = time.process_time()
                # Calculate cell_indices.
                excel_extraction.calculate_indices()

                calculate_times[4] = time.process_time() - t_before

            load_times[4] = t_before - time.process_time()
            t_before = time.process_time()
            # Extract annotations.
            excel_extraction.extract_annotations()

            calculate_times[5] = time.process_time() - t_before

        except com_error as e:
            print("ERROR: Workbook could not be prepared. The Workbook probably is password protected.")
            com_error_files += 1
            continue

        except Exception as e:
            print("Exception: General Exception raised.")
            traceback.print_exc()
            print(e)
            continue

        # Timer and file mapping
        time_passed = time.process_time() - start
        with open(timing_file, 'a+', encoding='utf8') as time_file:
            time_file.write("{\"id\":%d, \"time\":%d},\n" % (file_index, time_passed))
        timing.append(time_passed)

        file_name_mapping[file_path_name] = file_index
        with open(mapping_file, 'a+', encoding='utf8') as name_mapping_file:
            name_mapping_file.write("{\"id\": %d, \"name\": \"%s\", \"path\": \"%s\"},\n" % (file_index, file_name, file_path_name.replace('\\', '\\\\')))

        # Fine grained timing
        print(time_passed)
        print("extract_metadata: ", calculate_times[0])
        for index in range(1, 5):
            if calculate_times[index] > -1:
                print("calculate_{}".format(times_strings[index]), calculate_times[index])
            else:
                print("load_{}".format(times_strings[index]), load_times[index])
        print("extract_annotations: ", calculate_times[5])

        file_index += 1

    # Finish timing and mapping file to match json format.
    with open(timing_file, 'a+', encoding='utf8') as time_file:
        time_file.write("{\"id\":%d, \"totalTime\":%d, \"avgTime\":%f}]" % (file_index, sum(timing), sum(timing) / len(timing)))
    with open(mapping_file, 'a+', encoding='utf8') as name_mapping_file:
        name_mapping_file.write("]")

    # Finished processing
    print("\nFinished generation of annotations.")
    print("Processed: {} Documents ({} skipped, {} com-error, {} failed)".format(len(file_name_mapping), skipped_files, com_error_files, len(process_file_names) - len(file_name_mapping) - skipped_files - com_error_files))
    print("totalTime: {}, averageTime: {}".format(sum(timing), sum(timing) / len(timing)))


if __name__ == '__main__':
    run()
