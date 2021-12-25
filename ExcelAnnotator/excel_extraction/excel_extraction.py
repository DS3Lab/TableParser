import json
import os
import traceback
from datetime import date

import pdf2image
import win32com.client as win32
import pythoncom
import bisect
import pickle

from de_exceleration.extract_metadata import MetadataExtraction
from parse_helper.helper import next_fill_color, color2hex_bgr


class ExcelExtraction:
    """
    Class which handles the data extraction from excel files.
    """

    def __init__(self, abs_file_path, rel_file_path, file_name, file_extension, file_index, abs_output_file_path, output_file_ending, store_intermediate, abs_intermediate_path,
                 debug_output, verbose_debug_output, very_verbose_debug_output, poppler_path):
        """
        Initializes the instance, sets the environment for the data extraction
        :param abs_file_path: Absolute Path to the source file root directory
        :param rel_file_path: Relative Path from current directory to XLSX file
        :param file_name: Name of the XLSX file (without extension)
        :param file_extension: Extension of the XLSX file (.XLSX/.XLS)
        :param file_index: Index of file, replaces the filename in the output file
        :param abs_output_file_path: Absolute Path the teh output root directory
        :param output_file_ending: Ending of the output file (-data.json)
        :param store_intermediate: Boolean defining whether intermediate status should be stored. If True stores all the data which has been extracted before analyzing
        :param abs_intermediate_path: Absolute path to where intermediate data needs to be stored.
        :param debug_output: Boolean defining whether debug output is printed to the console
        :param verbose_debug_output: Boolean defining whether verbose debug output is printed to the console. If True, debug_output is shown as well
        :param very_verbose_debug_output: Boolean defining whether very verbose debug output is printed to the console. If True, verbose_debug_output is shown
        :param poppler_path: Path to the poppler utility program, needed for json.
        """

        self.workbook_prepared = False

        self.file_index = str(file_index)
        self.abs_file_path = abs_file_path
        self.output_file_path_prefix = abs_output_file_path + self.file_index
        self.output_file_path = []
        self.results_file_path_prefix = "{}results\\{}".format(abs_output_file_path, self.file_index)
        self.results_file_path = []

        self.rel_file_path = rel_file_path
        self.file_name = file_name
        self.file_extension = file_extension
        self.output_file_ending = output_file_ending
        self.xl_app = win32.Dispatch('Excel.Application')
        self.xl_app.EnableEvents = False
        self.xl_app.Visible = False
        self.xl_app.DisplayAlerts = False
        self.xl_app.Interactive = False
        self.valid_worksheets = []
        self.worksheet_names = []

        self.color_dict = {}
        self.num_rows = []
        self.num_cols = []
        self.row_empty = {}
        self.col_empty = {}
        self.table_data = {}
        self.new_table = {}
        self.meta_data_rows = {}

        self.poppler_path = poppler_path
        self.json_data = None
        self.object_id = -1
        self.current_document_id = -1
        self.current_table_id = -1
        self.current_tabular_id = -1
        self.initialize_json()

        self.illegal_colors = [(0, 0, 0), (0xFF, 0xFF, 0xFF)]  # By default black and white are considered illegal
        self.bbox_dict = {}
        self.cell_index_mapping = []

        self.store_intermediate = store_intermediate
        self.abs_intermediate_path = abs_intermediate_path
        self.intermediate_path_prefix = "{}{}\\".format(self.abs_intermediate_path, self.file_index)
        if self.store_intermediate and not os.path.exists(self.intermediate_path_prefix):
            os.makedirs(self.intermediate_path_prefix, exist_ok=True)

        self.very_verbose_debug_output = very_verbose_debug_output
        self.verbose_debug_output = verbose_debug_output or self.very_verbose_debug_output
        self.debug_output = debug_output or self.verbose_debug_output

        self.metadata_extraction = None
        self.meta_data = None
        self.meta_rows = []
        self.meta_cols = []

        self.max_cells_sheet = 60000

    def __del__(self):
        """
        Release com object on instance destruction
        :return: None
        """

        try:
            self.xl_app.quit()
        except Exception:
            pass

    def prepare_workbook(self):
        """
        Prepares the workbook. Sets row and col numbers and variables depending on worksheets.
        :return: None
        """
        workbook = None
        try:
            workbook = self.xl_app.Workbooks.Open(self.abs_file_path + self.rel_file_path + self.file_name + self.file_extension)
            num_worksheets = len(workbook.Worksheets)
            self.worksheet_names = [""] * num_worksheets
            self.output_file_path = [""] * num_worksheets
            self.results_file_path = [""] * num_worksheets
            self.num_rows = [-1] * num_worksheets
            self.num_cols = [-1] * num_worksheets

            for worksheet_index in range(num_worksheets):
                worksheet = workbook.Worksheets[worksheet_index]
                worksheet.Visible = -1

                self.worksheet_names[worksheet_index] = worksheet.Name
                self.set_used_range(worksheet, worksheet_index)

                if check_for_empty_or_large_sheet(self.num_rows[worksheet_index], self.num_cols[worksheet_index], self.max_cells_sheet, self.debug_output):
                    continue
                else:
                    self.valid_worksheets.append(worksheet_index)

                # Export PDF file - Re-Fill cell (max_row, max_col) such that the scaling is the same as for the annotated file.
                self.set_output_file_path(worksheet_index)
                cell = worksheet.Cells(self.num_rows[worksheet_index], self.num_cols[worksheet_index])
                interior_color = cell.Interior.Color
                cell.Interior.Color = interior_color

                pdf_file_name = "{}-{}".format(self.file_index, worksheet_index)
                worksheet.PageSetup.Zoom = False
                worksheet.PageSetup.FitToPagesTall = False
                worksheet.PageSetup.FitToPagesWide = 1
                try:
                    worksheet.ExportAsFixedFormat(0, "{}{}.pdf".format(self.results_file_path[worksheet_index], pdf_file_name))
                except pythoncom.com_error:
                    self.valid_worksheets.pop()
                    continue

            self.workbook_prepared = True

        except Exception:
            if workbook:
                workbook.Close(False)
            raise

        finally:
            if workbook:
                workbook.Close(False)

    def initialize_json(self):
        """
        Initializes the auxiliary json file for data output.
        self.json_data gets overwritten.
        :return: None
        """

        self.object_id = 1
        self.json_data = [{"id": self.object_id, "category": "unk", "parent": None}, {"id": self.object_id + 1, "category": "meta", "parent": None}]
        self.object_id += 2

        self.json_data.append({"id": self.object_id, "category": "document", "parent": None})
        self.current_document_id = self.object_id
        self.object_id += 1

    def load_calculate_annotations(self):
        """
        Loads all previously calculated variables from files needed to recalculate calculate_annotations and succeeding steps.
        :return: Whether loading succeeded.
        """

        # Load new table from file.
        pickle_new_table_path = "{}new_table.pickle".format(self.intermediate_path_prefix)
        if os.path.isfile(pickle_new_table_path):
            with open(pickle_new_table_path, 'rb') as pickle_file:
                self.new_table = pickle.load(pickle_file)
            return True
        return False

    def load_calculate_bboxes(self):
        """
        Loads all previously calculated variables from files needed to recalculate calculate_bboxes and succeeding steps.
        :return: Whether loading succeeded.
        """

        # Load color_dict, new_table, meta_data_rows, row_empty, col_empty
        pickle_calculated_annotations_path = "{}calc_annotations.pickle".format(self.intermediate_path_prefix)
        if os.path.isfile(pickle_calculated_annotations_path):
            with open(pickle_calculated_annotations_path, 'rb') as pickle_file:
                [self.color_dict, self.new_table, self.meta_data_rows, self.row_empty, self.col_empty] = pickle.load(pickle_file)
            return True
        return False

    def load_calculate_indices(self):
        """
        Loads all previously calculated variables from files needed to recalculate calculate_indices and succeeding steps.
        :return: Whether loading succeeded.
        """

        # Load new_table, meta_data_rows, table_data, row_empty, col_empty, bbox_dict
        pickle_calculated_bboxes = "{}calc_bboxes.pickle".format(self.intermediate_path_prefix)
        if os.path.isfile(pickle_calculated_bboxes):
            with open(pickle_calculated_bboxes, 'rb') as pickle_file:
                [self.new_table, self.meta_data_rows, self.row_empty, self.col_empty, self.table_data, self.bbox_dict] = pickle.load(pickle_file)
            return True
        return False

    def load_extract_annotations(self):
        """
        Loads all previously calculated variables from files needed to reextract all annotations into json files.s
        :return:
        """

        # Load new_table, meta_data_rows, table_data, bbox_dict, cell_index_mapping
        pickle_calculated_indices = "{}calc_indices.pickle".format(self.intermediate_path_prefix)
        if os.path.isfile(pickle_calculated_indices):
            with open(pickle_calculated_indices, 'rb') as pickle_file:
                [self.new_table, self.meta_data_rows, self.table_data, self.bbox_dict, self.cell_index_mapping] = pickle.load(pickle_file)
            return True
        return False

    def calculate_annotations(self):
        """
        Calculates the annotations from input file
        :return: None
        """

        if not self.workbook_prepared:
            self.prepare_workbook()

        workbook = None
        try:
            if self.debug_output:
                print('Calculate annotations')

            workbook = self.xl_app.Workbooks.Open(self.abs_file_path + self.rel_file_path + self.file_name + self.file_extension)

            # Store workbook for easier comparison.
            if len(self.output_file_path) > 0:
                workbook.SaveAs(self.output_file_path[0] + self.file_name + self.file_extension)

            for worksheet_index in self.valid_worksheets:
                if worksheet_index > 0:
                    workbook.Close(False)
                    workbook = self.xl_app.Workbooks.Open(self.abs_file_path + self.rel_file_path + self.file_name + self.file_extension)
                worksheet = workbook.Worksheets[worksheet_index]
                worksheet.Visible = -1

                if self.debug_output:
                    print(worksheet.Name)

                # Recognition for rows containing metadata
                self.meta_data_rows[worksheet_index] = []

                # Optimize removing empty rows & columns
                self.row_empty[worksheet_index] = []
                self.col_empty[worksheet_index] = []

                self.table_data[worksheet_index] = []
                self.color_dict[worksheet_index] = {}

                curr_row = 0
                table_index = -1
                num_cells_processed = 0
                annotated_index = 0
                while curr_row < self.num_rows[worksheet_index]:
                    if curr_row in self.new_table[worksheet_index]:
                        table_index += 1
                        if len(self.new_table[worksheet_index]) - 1 <= table_index:
                            rows_table_i = self.num_rows[worksheet_index] - self.new_table[worksheet_index][table_index]
                        else:
                            rows_table_i = self.new_table[worksheet_index][table_index + 1] - self.new_table[worksheet_index][table_index]
                        self.table_data[worksheet_index].append([{} for i in range(0, rows_table_i)])
                        self.row_empty[worksheet_index].append([True] * rows_table_i)
                        self.col_empty[worksheet_index].append([True] * self.num_cols[worksheet_index])

                    curr_col = 0
                    while curr_col < self.num_cols[worksheet_index]:
                        if num_cells_processed > 60000:
                            # Save intermediate annotated file, as only 64k different cell styles are allowed.
                            self.export_annotated_pdf(workbook, worksheet, worksheet_index, annotated_index)

                            # Reopen workbook - discard all changes.
                            workbook.Close(False)
                            workbook = self.xl_app.Workbooks.Open(self.abs_file_path + self.rel_file_path + self.file_name + self.file_extension)
                            worksheet = workbook.Worksheets[worksheet_index]
                            worksheet.Visible = -1
                            num_cells_processed = 0
                            annotated_index += 1

                        curr_cell = worksheet.Cells(curr_row + 1, curr_col + 1)
                        num_cells_processed += 1

                        if curr_cell.value:
                            # Update this Row/Col as not empty.
                            self.row_empty[worksheet_index][table_index][curr_row - self.new_table[worksheet_index][table_index]] = False
                            self.col_empty[worksheet_index][table_index][curr_col] = False

                            self.table_data[worksheet_index][table_index][curr_row - self.new_table[worksheet_index][table_index]][curr_col] = curr_cell.value

                            if curr_cell.value in self.meta_data:
                                # Update meta_data_rows array.
                                self.meta_data_rows[worksheet_index].append(curr_row)

                        # Set background color for cell, remove every other content and store into mapping.
                        fill_color = next_fill_color(self.illegal_colors)
                        hex_fill_color = color2hex_bgr(fill_color)

                        curr_cell.Borders.LineStyle = 0
                        if curr_cell.value in self.meta_data:
                            # Color font in same color to capture overflowing metadata.
                            curr_cell.Font.Color = hex_fill_color
                        else:
                            # Color font uniquely with black (0xFFFFFF)
                            curr_cell.Font.Color = 0xFFFFFF
                        curr_cell.Interior.Color = hex_fill_color

                        if self.very_verbose_debug_output:
                            print("Fill_color: ", fill_color)

                        # Store fill_color into dictionary
                        if hex_fill_color in self.color_dict[worksheet_index]:
                            print("ERROR: Color has been used twice or maximal color reached. Can be solved by introducing 'instances'")
                            exit(1)

                        self.color_dict[worksheet_index][hex_fill_color] = (curr_row, curr_col)

                        curr_col += 1
                    curr_row += 1

                if self.verbose_debug_output:
                    # Print table data for debugging purpose.
                    for table_index in range(len(self.table_data[worksheet_index])):
                        table_data = self.table_data[worksheet_index][table_index]
                        print("Worksheet ", table_index)
                        for table in table_data:
                            for row in table:
                                print(row)
                            print()

                if self.very_verbose_debug_output:
                    # Print color_dict.
                    print(json.dumps(self.color_dict, indent=2))
                    print()

                # Export annotated PDF file.
                self.export_annotated_pdf(workbook, worksheet, worksheet_index, annotated_index)

            if self.store_intermediate:
                # Store intermediate data.
                pickle_calculated_annotations_path = "{}calc_annotations.pickle".format(self.intermediate_path_prefix)
                with open(pickle_calculated_annotations_path, 'wb') as pickle_file:
                    pickle.dump([self.color_dict, self.new_table, self.meta_data_rows, self.row_empty, self.col_empty], pickle_file)

        except Exception as e:
            print('Exception:', end=' ')
            traceback.print_exc()
            print(e)

        finally:
            if workbook:
                workbook.Close(False)

    def export_annotated_pdf(self, workbook, worksheet, worksheet_index, annotated_index):
        """
        Export the annotated pdf file.
        :param workbook: Workbook to export
        :param worksheet: Annotated worksheet of workbook
        :param worksheet_index: Index to current worksheet
        :param annotated_index: Index to allow more than one annotated pdf for a worksheet
        :return: None
        """

        annotated_output_file_name = "{}-annotated-{}-{}".format(self.file_name, worksheet.Name, annotated_index)
        annotated_output_file_name = annotated_output_file_name.replace(' ', '_')
        worksheet.PageSetup.Zoom = False
        worksheet.PageSetup.FitToPagesTall = False
        worksheet.PageSetup.FitToPagesWide = 1
        worksheet.ExportAsFixedFormat(0, "{}{}.pdf".format(self.output_file_path[worksheet_index], annotated_output_file_name))

        if self.very_verbose_debug_output:
            # Store annotated excel file
            self.xl_app.DisplayAlerts = False
            workbook.SaveAs(self.output_file_path[worksheet_index] + annotated_output_file_name + self.file_extension)

    def calculate_bboxes(self):
        """
        Calculate bboxes of current workbook.
        :return: None
        """

        if not self.workbook_prepared:
            self.prepare_workbook()

        self.bbox_dict = {}
        if self.debug_output:
            print('Export annotation')

        for worksheet_index in self.valid_worksheets:
            self.bbox_dict[worksheet_index] = {}
            annotated_index = -1
            while True:
                annotated_index += 1
                pages = None
                try:
                    annotated_file_name_post = "-annotated-{}-{}.pdf".format(self.worksheet_names[worksheet_index], annotated_index).replace(' ', '_')
                    pages = pdf2image.convert_from_path(self.output_file_path[worksheet_index] + self.file_name.replace(' ', '_') + annotated_file_name_post,
                                                        poppler_path=self.poppler_path)
                except pdf2image.exceptions.PDFPageCountError as e:
                    # Ignore errors, is due to trying opening a file which does not exist.
                    break

                # Generate bbox_dict - Dict of cell index to bbox ([upper_left, lower_left, upper_right, upper_left], bbox_page)
                page_number = 0
                for page in pages:
                    if self.debug_output:
                        print('Page', page)

                    pixels = page.load()
                    x = 0
                    while x < page.width:
                        y = 0
                        while y < page.height:
                            color = color2hex_bgr(pixels[x, y])
                            if color in self.color_dict[worksheet_index]:
                                cell_index = self.color_dict[worksheet_index][color]
                                rgb_color = pixels[x, y]

                                # Get bounding box of this color.
                                bbox_index = -1
                                first_occurrence = False

                                if cell_index in self.bbox_dict[worksheet_index]:
                                    [[min_x, min_y, width, height], cell_page] = self.bbox_dict[worksheet_index][cell_index][-1]
                                    if cell_page == page_number:
                                        # Color already found on this page, adapt existing bbox.
                                        if x < min_x:
                                            width += min_x - x
                                            min_x = x
                                        if y < min_y:
                                            height += min_y - y
                                            min_y = y

                                        while y < page.height - 1 and pixels[x, y + 1] == rgb_color:
                                            y += 1

                                        if x > min_x + width:
                                            width = x - min_x
                                        if y > min_y + height:
                                            height = y - min_y

                                        # Store updated bbox.
                                        self.bbox_dict[worksheet_index][cell_index][-1][0] = [min_x, min_y, width, height]

                                    else:
                                        first_occurrence = True
                                else:
                                    first_occurrence = True

                                if first_occurrence:
                                    # New color found, create bbox.
                                    if bbox_index == -1:
                                        self.bbox_dict[worksheet_index][cell_index] = []

                                    min_y = y
                                    while y < page.height - 1 and pixels[x, y + 1] == rgb_color:
                                        y += 1

                                    self.bbox_dict[worksheet_index][cell_index].append([[x, min_y, 0, y - min_y], page_number])
                            y += 1
                        x += 1
                    page_number += 1

            self.remove_empty_row_col(worksheet_index)

            if self.very_verbose_debug_output:
                # Print bbox_dict of current worksheet.
                for j in sorted(self.bbox_dict[worksheet_index].keys()):
                    print(j)
                print()

        if self.store_intermediate:
            # Store intermediate data.
            pickle_calculated_bboxes = "{}calc_bboxes.pickle".format(self.intermediate_path_prefix)
            with open(pickle_calculated_bboxes, 'wb') as pickle_file:
                pickle.dump([self.new_table, self.meta_data_rows, self.row_empty, self.col_empty, self.table_data, self.bbox_dict], pickle_file)

    def calculate_indices(self):
        """
        Calculates the row and col ranges in all worksheets for current workbook.
        :return: None
        """

        if not self.workbook_prepared:
            self.prepare_workbook()

        self.cell_index_mapping = {}
        if self.debug_output:
            print('Export annotation')

        for worksheet_index in self.valid_worksheets:
            threshold = 4
            row_heights = []
            col_widths = []
            row_starts = []
            col_starts = []
            remaining_heights = [0] * self.num_cols[worksheet_index]
            remaining_widths = [0] * self.num_rows[worksheet_index]

            for table_index in range(len(self.new_table[worksheet_index])):
                # row_starts, row_heights separated for each table and page.
                row_starts.append({})
                row_heights.append({})
                # col_starts, col_widths separated for each table.
                col_starts.append([])
                col_widths.append([])

                table_min_row = self.new_table[worksheet_index][table_index]
                table_max_row = self.new_table[worksheet_index][table_index + 1] if table_index + 1 < len(self.new_table[worksheet_index]) else self.num_rows[worksheet_index]
                for row_index in range(table_min_row, table_max_row):
                    if row_index in self.meta_data_rows[worksheet_index]:
                        # We can skip metadata.
                        continue

                    row_bbox_entries = [self.bbox_dict[worksheet_index][(row_index, col_index)][0] for col_index in range(self.num_cols[worksheet_index])
                                        if (row_index, col_index) in self.bbox_dict[worksheet_index]]

                    row_bboxes = [row_bbox_entries[x][0] for x in range(len(row_bbox_entries))]
                    row_page_nums = [row_bbox_entries[x][1] for x in range(len(row_bbox_entries))]

                    # Check whether boxes start at same height.
                    if abs(min([x[1] for x in row_bboxes], default=-1000) - max([x[1] for x in row_bboxes], default=-1000)) > threshold:
                        print("ERROR: BBoxes of same row don't start at same height. (row:", row_index, ")", end='')
                        print(min([x[1] for x in row_bboxes], default=-1000) - max([x[1] for x in row_bboxes], default=-1000), " > ", threshold)

                    # Check whether boxes are all on same page.
                    if len(row_page_nums) > 0 and not all(element == row_page_nums[0] for element in row_page_nums):
                        print("ERROR: BBoxes of same row are not on same page. (row:", row_index, ")", end='')
                        print("row_page_nums: ", row_page_nums)

                    page = row_page_nums[0] if len(row_page_nums) > 0 else -1
                    if page == -1:
                        # Skip empty row.
                        continue

                    if page not in row_starts[table_index]:
                        curr_start = row_bboxes[0][1]
                        row_starts[table_index][page] = []
                        row_heights[table_index][page] = []
                    elif len(row_bboxes) > 0:
                        if abs(row_bboxes[0][1] - row_starts[table_index][page][-1] - row_heights[table_index][page][-1]) > threshold:
                            print("ERROR: New row does not start after last one (row:", row_index, ")", end='')
                            print(row_bboxes[0][1], " - ", row_starts[table_index][page][-1], " - ", row_heights[table_index][page][-1], " > ", threshold)

                        # If height does not match the difference of the starts, adjust remaining_heights.
                        row_diff = min([x[1] for x in row_bboxes], default=0) - row_starts[table_index][page][-1] - row_heights[table_index][page][-1]
                        if row_diff > 0:
                            remaining_heights = [x - row_diff if x - row_diff > threshold else 0 for x in remaining_heights]
                            row_heights[table_index][page][-1] += row_diff

                        curr_start = row_bboxes[0][1]
                    else:
                        curr_start = row_starts[table_index][page][-1] + row_heights[table_index][page][-1]

                    min_height = min([x[3] for x in row_bboxes] + [x for x in remaining_heights if x > 0], default=-1)
                    if min_height == -1:
                        # Skip empty row.
                        continue

                    row_starts[table_index][page].append(curr_start)
                    row_heights[table_index][page].append(min_height)

                    for col_index in range(self.num_cols[worksheet_index]):
                        if (row_index, col_index) in self.bbox_dict[worksheet_index]:
                            if remaining_heights[col_index] > threshold:
                                print("ERROR: New row overlaps with previous. ([row,col]=", [row_index, col_index], ")", end='')
                                print("remaining_heights: ", remaining_heights)

                            remaining_heights[col_index] = self.bbox_dict[worksheet_index][(row_index, col_index)][0][0][3] - min_height
                            if remaining_heights[col_index] < threshold:
                                remaining_heights[col_index] = 0
                        elif remaining_heights[col_index] > 0:
                            if remaining_heights[col_index] - min_height < -threshold:
                                print("ERROR: Cell has not enough remaining height. ([row,col]=", [row_index, col_index], ")", end='')
                                print("remaining_heights: ", remaining_heights, " // ", remaining_heights[col_index], " - ", min_height, " < ", -threshold)

                            remaining_heights[col_index] -= min_height
                            if remaining_heights[col_index] < threshold:
                                remaining_heights[col_index] = 0

                    while all(remaining_heights[x] > 0 or self.col_empty[worksheet_index][table_index][x] for x in range(self.num_cols[worksheet_index])):
                        min_height = min(x for x in remaining_heights if x > 0)
                        row_starts[table_index][page].append(row_starts[table_index][page][-1] + row_heights[table_index][page][-1])
                        row_heights[table_index][page].append(min_height)

                        remaining_heights = [x - min_height if x - min_height > threshold else 0 for x in remaining_heights]

                for col_index in range(self.num_cols[worksheet_index]):
                    col_bbox_entries = [self.bbox_dict[worksheet_index][(row_index, col_index)][0] for row_index in range(table_min_row, table_max_row)
                                        if (row_index, col_index) in self.bbox_dict[worksheet_index] and row_index not in self.meta_data_rows[worksheet_index]]

                    # Columns may range over more than a page, thus we can ignore the page of the bboxes.
                    col_bboxes = [col_bbox_entries[x][0] for x in range(len(col_bbox_entries))]

                    # Check whether boxes start at same width.
                    if abs(min([x[0] for x in col_bboxes], default=-1000) - max([x[0] for x in col_bboxes], default=-1000)) > threshold:
                        print("ERROR: BBoxes of same col don't start at same width. (col:", col_index, ")", end='')
                        print(min([x[0] for x in col_bboxes], default=-1000) - max([x[0] for x in col_bboxes], default=-1000), " > ", threshold)

                    # If width does not match the difference of the starts, adjust remaining_widths.
                    if len(col_starts[table_index]) > 0:
                        col_diff = min([x[0] for x in col_bboxes], default=-1) - col_starts[table_index][-1] - col_widths[table_index][-1]
                        if col_diff > 0 and len(col_bboxes) > 0:
                            remaining_widths = [x - col_diff if x - col_diff > threshold else 0 for x in remaining_widths]
                            col_widths[table_index][-1] += col_diff

                    min_width = min([x[2] for x in col_bboxes] + [remaining_widths[x] for x in range(table_min_row, table_max_row) if remaining_widths[x] > 0], default=-1)
                    if min_width == -1:
                        # Skip empty col.
                        continue

                    if len(col_starts[table_index]) == 0:
                        col_starts[table_index].append(col_bboxes[0][0])
                    elif len(col_bboxes) > 0:
                        if abs(col_bboxes[0][0] - col_starts[table_index][-1] - col_widths[table_index][-1]) > threshold:
                            print("ERROR: New col does not start after last one (col:", col_index, ")", end='')
                            print(col_bboxes[0][0], " - ", col_starts[table_index][-1], " - ", col_widths[table_index][-1], " > ", threshold)

                        col_starts[table_index].append(col_bboxes[0][0])
                    else:
                        col_starts[table_index].append(col_starts[table_index][-1] + col_widths[table_index][-1])

                    col_widths[table_index].append(min_width)

                    for row_index in range(table_min_row, table_max_row):
                        if (row_index, col_index) in self.bbox_dict[worksheet_index]:
                            if row_index in self.meta_data_rows[worksheet_index]:
                                # We can skip metadata.
                                continue

                            if remaining_widths[row_index] > threshold:
                                print("ERROR: New col overlaps with previous. ([row,col]=", [row_index, col_index], ")", end='')
                                print("remaining_widths: ", remaining_widths)

                            remaining_widths[row_index] = self.bbox_dict[worksheet_index][(row_index, col_index)][0][0][2] - min_width
                            if remaining_widths[row_index] < threshold:
                                remaining_widths[row_index] = 0
                        elif remaining_widths[row_index] > 0:
                            if remaining_widths[row_index] - min_width < -threshold:
                                print("ERROR: Cell has not enough remaining width. ([row,col]=", [row_index, col_index], ")", end='')
                                print("remaining_widths: ", remaining_widths, " // ", remaining_widths[row_index], " - ", min_width, " < ", -threshold)

                            remaining_widths[row_index] -= min_width
                            if remaining_widths[row_index] < threshold:
                                remaining_widths[row_index] = 0

                    while all(remaining_widths[x] > 0 or self.row_empty[worksheet_index][table_index][x - self.new_table[worksheet_index][table_index]]
                              for x in range(table_min_row, table_max_row)):
                        min_width = min(x for x in remaining_widths if x > 0)
                        col_starts[table_index].append(col_starts[table_index][-1] + col_widths[table_index][-1])
                        col_widths[table_index].append(min_width)

                        remaining_widths = [x - min_width if x - min_width > threshold else 0 for x in remaining_widths]

            # Calculate indices from col_widths and row_heights
            self.cell_index_mapping[worksheet_index] = {}
            for cell_index in self.bbox_dict[worksheet_index]:
                if cell_index[0] in self.meta_data_rows[worksheet_index]:
                    # Metadata does not have indices.
                    continue

                [[curr_x, curr_y, curr_width, curr_height], curr_page] = self.bbox_dict[worksheet_index][cell_index][0]
                table_index = bisect.bisect_left(self.new_table[worksheet_index], cell_index[0])
                if table_index >= len(self.new_table[worksheet_index]):
                    table_index = len(self.new_table[worksheet_index]) - 1
                if self.new_table[worksheet_index][table_index] > cell_index[0]:
                    # We want the last element which is < or =, thus decrease index.
                    table_index -= 1
                    if self.new_table[worksheet_index][table_index] > cell_index[0]:
                        print(self.new_table[worksheet_index][table_index], ">", cell_index[0])

                # Get row and col min, max.
                (row_min, row_max) = calculate_cell_range(row_starts[table_index][curr_page], row_heights[table_index][curr_page], curr_y, curr_height, threshold)
                (col_min, col_max) = calculate_cell_range(col_starts[table_index], col_widths[table_index], curr_x, curr_width, threshold)

                # Adjust row for offset from previous pages.
                for page in range(0, curr_page):
                    if page in row_starts[table_index]:
                        page_offset = len(row_starts[table_index][page])
                        row_min += page_offset
                        row_max += page_offset

                self.cell_index_mapping[worksheet_index][cell_index] = (row_min, row_max, col_min, col_max)

        if self.store_intermediate:
            # Store intermediate data.
            pickle_calculated_indices = "{}calc_indices.pickle".format(self.intermediate_path_prefix)
            with open(pickle_calculated_indices, 'wb') as pickle_file:
                pickle.dump([self.new_table, self.meta_data_rows, self.table_data, self.bbox_dict, self.cell_index_mapping], pickle_file)

    def extract_annotations(self):
        """
        Extracts the data and stores it into the auxiliary json file.
        :return: None
        """

        if not self.workbook_prepared:
            self.prepare_workbook()

        try:
            if self.debug_output:
                print('Export annotation')

            for worksheet_index in self.valid_worksheets:

                # Export PDF file pages as image.
                pdf_file_name = "{}-{}".format(self.file_index, worksheet_index)
                pages = pdf2image.convert_from_path("{}{}.pdf".format(self.results_file_path[worksheet_index], pdf_file_name), poppler_path=self.poppler_path)
                page_index = 0
                for page in pages:
                    page.save("{}{}-{}.png".format(self.results_file_path[worksheet_index], pdf_file_name, page_index))
                    page_index += 1

                # Create main json file.
                main_json_id = "{}-{}".format(self.file_index, worksheet_index)
                main_json_data = {"id": main_json_id,
                                  "subject": ["Automated"],
                                  "pages": len(pages),
                                  "title": main_json_id,
                                  "date": date.today().strftime("%d.%m.%Y")}

                main_json_file_path = "{}{}.json".format(self.results_file_path[worksheet_index], main_json_id)
                with open(main_json_file_path, 'w', encoding='utf8') as output_file:
                    json.dump(main_json_data, output_file, indent=2, ensure_ascii=False)

                # Generate annotations.
                table_index = -1
                table_row = 0
                for row_index in range(0, self.num_rows[worksheet_index]):
                    if row_index in self.new_table[worksheet_index]:
                        # New table.
                        self.json_data.append({"id": self.object_id, "category": "table", "parent": self.current_document_id})
                        self.current_table_id = self.object_id
                        self.current_tabular_id = -1
                        self.object_id += 1
                        table_index += 1

                    meta_data_row = row_index in self.meta_data_rows[worksheet_index]

                    table_col = 0
                    for col_index in range(0, self.num_cols[worksheet_index]):
                        if (row_index, col_index) in self.bbox_dict[worksheet_index]:
                            # Get all bbox_entries for this cell_index.
                            bbox_entries = self.bbox_dict[worksheet_index][(row_index, col_index)]
                        else:
                            # We don't have a bbox for this cell_index, thus we can skip it. (We do not make a check whether the cell should exist)
                            if self.debug_output:
                                print("BBox dict might not contain the bbox for ", (row_index, col_index))
                            continue

                        # Add cell content including bounding box to json_data.
                        cell_content = None
                        is_table_cell = True
                        cell_parent = self.current_tabular_id
                        parent_id = -1
                        if col_index in self.table_data[worksheet_index][table_index][row_index - self.new_table[worksheet_index][table_index]]:
                            # Append all data to json_data.
                            cell_content = self.table_data[worksheet_index][table_index][row_index - self.new_table[worksheet_index][table_index]][col_index]
                            if cell_content in self.meta_data:
                                meta_data_row = True
                                is_table_cell = False

                                self.json_data.append({"id": self.object_id, "category": "table_caption", "parent": self.current_table_id})
                                parent_id = self.object_id
                                self.object_id += 1

                        if is_table_cell:
                            if meta_data_row and (cell_content is None or cell_content == ""):
                                continue

                            # Add tabular object before adding any table_cells
                            if self.current_tabular_id == -1:
                                self.json_data.append({"id": self.object_id, "category": "tabular", "parent": self.current_table_id})
                                self.current_tabular_id = self.object_id
                                cell_parent = self.current_tabular_id
                                self.object_id += 1

                            [row_min, row_max, col_min, col_max] = self.cell_index_mapping[worksheet_index][(row_index, col_index)]
                            properties = "{}-{},{}-{}".format(row_min, row_max, col_min, col_max)
                            row_range = (row_min, row_max)
                            col_range = (col_min, col_max)

                            self.json_data.append({"id": self.object_id, "category": "table_cell", "properties": properties,
                                                   "row_range": row_range, "col_range": col_range, "parent": cell_parent})

                            parent_id = self.object_id
                            self.object_id += 1

                        for (bbox, page_number) in bbox_entries:
                            if cell_content:
                                self.json_data.append(
                                    {"id": self.object_id, "category": "box", "page": page_number, "bbox": bbox, "text": cell_content, "parent": parent_id})
                            else:
                                self.json_data.append({"id": self.object_id, "category": "box", "page": page_number, "bbox": bbox, "parent": parent_id})

                            self.object_id += 1
                        table_col += 1
                    table_row += 1

                # Store output file.
                json_file_path_name = "{}{}-{}{}".format(self.results_file_path[worksheet_index], self.file_index, worksheet_index, self.output_file_ending)
                with open(json_file_path_name, 'w', encoding='utf8') as output_file:
                    json.dump(self.json_data, output_file, indent=2, ensure_ascii=False)
                    self.initialize_json()

        except Exception as e:
            print('Exception:', end=' ')
            traceback.print_exc()
            print(e)

    def calculate_new_table_array(self):
        """
        Calculates the new table array depending on the meta_data extracted with deExcelerator.
        :return: None
        """

        if not self.workbook_prepared:
            self.prepare_workbook()

        workbook = None
        try:
            workbook = self.xl_app.Workbooks.Open(self.abs_file_path + self.rel_file_path + self.file_name + self.file_extension)

            if self.debug_output:
                print('Calculate new table array')

            for worksheet_index in self.valid_worksheets:
                worksheet = workbook.Worksheets[worksheet_index]
                worksheet.Visible = -1
                print(worksheet.Name)

                curr_row = 0
                stack = []
                cell_category_enum = ['empty', 'content', 'meta']
                self.new_table[worksheet_index] = []

                while curr_row < self.num_rows[worksheet_index]:
                    curr_col = 0
                    curr_category = cell_category_enum[0]
                    while curr_col < self.num_cols[worksheet_index]:
                        curr_cell = worksheet.Cells(curr_row + 1, curr_col + 1)

                        if curr_cell.value in self.meta_data:
                            curr_category = cell_category_enum[2]
                        elif curr_cell.value and curr_category == cell_category_enum[0]:
                            curr_category = cell_category_enum[1]

                        curr_col += 1

                    # Fill stack with correct category.
                    if len(stack) == 0:
                        # 'empty' is not needed to push onto empty stack.
                        if curr_category != cell_category_enum[0]:
                            # Push onto stack.
                            stack.append((curr_category, curr_row))
                    elif curr_category != stack[-1][0]:
                        if stack[-1][0] == cell_category_enum[0] and curr_category == stack[-2][0]:
                            stack.pop()
                        else:
                            stack.append((curr_category, curr_row))

                    curr_row += 1

                # Generate new_table list from stack.
                meta_start = -1
                empty_after = -1

                has_meta = False
                has_empty_before = False
                has_empty_after = False

                for stack_index in range(len(stack)):
                    if stack[stack_index][0] == cell_category_enum[1]:
                        # We have content, add new table.
                        if has_meta:
                            if has_empty_before and not has_empty_after:
                                # Metadata belongs to the table, table starts at beginning of metadata.
                                new_table_row = meta_start
                            else:
                                # Last row of metadata belongs to table.
                                if has_empty_after:
                                    new_table_row = empty_after - 1
                                else:
                                    new_table_row = stack[stack_index][1] - 1
                        else:
                            # Table starts at first line of table content.
                            new_table_row = stack[stack_index][1]

                        self.new_table[worksheet_index].append(new_table_row)

                        # Reset table.
                        has_meta = False
                        has_empty_before = False
                        has_empty_after = False

                    elif stack[stack_index][0] == cell_category_enum[2]:
                        meta_start = stack[stack_index][1]
                        has_meta = True
                    else:
                        if has_meta:
                            empty_after = stack[stack_index][1]
                            has_empty_after = True
                        else:
                            has_empty_before = True

                # Fix the start of first table to be set to 0.
                if self.new_table[worksheet_index][0] != 0:
                    self.new_table[worksheet_index][0] = 0

                if self.store_intermediate:
                    # Store intermediate data.
                    pickle_new_table_path = "{}new_table.pickle".format(self.intermediate_path_prefix)
                    with open(pickle_new_table_path, 'wb') as pickle_file:
                        pickle.dump(self.new_table, pickle_file)

        except Exception as e:
            print('Exception:', end=' ')
            traceback.print_exc()
            print(e)

        finally:
            if workbook:
                workbook.Close(False)

    def extract_metadata(self, metadata_directory, full_file_name):
        """
        Extract the metadata with help of metadata extraction object.
        :param metadata_directory: Directory where generated metadata files are located.
        :param full_file_name: File name of processed file.
        :return: 0 if successful, -1 if error occurred
        """

        self.metadata_extraction = MetadataExtraction(metadata_directory, full_file_name)
        if self.metadata_extraction.extract_metadata(self.verbose_debug_output) == -1:
            return -1

        self.meta_data = [j for sub in self.metadata_extraction.meta_data for j in sub]
        self.meta_rows = self.metadata_extraction.rows
        self.meta_cols = self.metadata_extraction.cols
        return 0

    def set_output_file_path(self, worksheet_index):
        """
        ¨Sets the output and results file path for this worksheet.
        :param worksheet_index: Index of current worksheet.
        :return: None
        """

        self.output_file_path[worksheet_index] = "{}-{}\\".format(self.output_file_path_prefix, worksheet_index)
        if not os.path.exists(self.output_file_path[worksheet_index]):
            os.makedirs(self.output_file_path[worksheet_index], exist_ok=True)
        self.results_file_path[worksheet_index] = "{}-{}\\".format(self.results_file_path_prefix, worksheet_index)
        if not os.path.exists(self.results_file_path[worksheet_index]):
            os.makedirs(self.results_file_path[worksheet_index], exist_ok=True)

    def remove_empty_row_col(self, worksheet_index):
        """
        Remove all the bboxes from bbox_dict for empty rows and cols.
        :param worksheet_index: Index to current worksheet
        :return: None
        """

        table_index = -1
        table_content_before = False
        row_candidates = []

        # Delete empty rows.
        for row_index in range(self.num_rows[worksheet_index]):
            if row_index in self.new_table[worksheet_index]:
                table_index += 1

            if row_index in self.meta_data_rows[worksheet_index]:
                # Delete row_candidates.
                for r in row_candidates:
                    for col_index in range(self.num_rows[worksheet_index]):
                        if (r, col_index) in self.bbox_dict[worksheet_index]:
                            del self.bbox_dict[worksheet_index][(r, col_index)]
                # Reset.
                row_candidates = []
                table_content_before = False
            elif self.row_empty[worksheet_index][table_index][row_index - self.new_table[worksheet_index][table_index]]:
                row_candidates.append(row_index)
            else:
                if not table_content_before:
                    # Delete row_candidates.
                    for r in row_candidates:
                        for col_index in range(self.num_rows[worksheet_index]):
                            if (r, col_index) in self.bbox_dict[worksheet_index]:
                                del self.bbox_dict[worksheet_index][(r, col_index)]
                # Reset.
                row_candidates = []
                table_content_before = True

        if len(row_candidates) > 0:
            # Delete row_candidates.
            for r in row_candidates:
                for col_index in range(self.num_rows[worksheet_index]):
                    if (r, col_index) in self.bbox_dict[worksheet_index]:
                        del self.bbox_dict[worksheet_index][(r, col_index)]

        # Delete empty columns.
        for table_index in range(len(self.new_table[worksheet_index])):
            for col_index in range(self.num_cols[worksheet_index]):
                if self.col_empty[worksheet_index][table_index][col_index]:
                    # Remove bbox(es) from dict.
                    table_min_row = self.new_table[worksheet_index][table_index]
                    table_max_row = self.new_table[worksheet_index][table_index + 1] if table_index + 1 < len(self.new_table[worksheet_index]) else self.num_rows[worksheet_index]
                    for row_index in range(table_min_row, table_max_row):
                        if (row_index, col_index) in self.bbox_dict[worksheet_index]:
                            del self.bbox_dict[worksheet_index][(row_index, col_index)]

    def set_used_range(self, worksheet, worksheet_index):
        """
        Set num_rows and num_cols.
        :param worksheet: Current worksheet
        :param worksheet_index: Index to current worksheet
        :return: None
        """

        if worksheet_index >= len(self.meta_rows):
            self.num_rows[worksheet_index] = worksheet.UsedRange.Rows.Count
            self.num_cols[worksheet_index] = worksheet.UsedRange.Columns.Count
        else:
            self.num_rows[worksheet_index] = min(worksheet.UsedRange.Rows.Count, self.meta_rows[worksheet_index] + 50)
            self.num_cols[worksheet_index] = min(worksheet.UsedRange.Columns.Count, self.meta_cols[worksheet_index] + 50)


def check_for_empty_or_large_sheet(num_rows, num_cols, max_cells_sheet, debug_output):
    """
    Check for empty or large sheet.
    :param num_rows: Number of rows in current sheet
    :param num_cols: Number of cols in current sheet
    :param max_cells_sheet: Maximal allowed cells in sheet still processing
    :param debug_output: Variable whether debug output should be printed.
    :return: Whether sheet is either empty or large
    """
    result = (num_rows <= 1 and num_cols <= 1) or num_rows * num_cols > max_cells_sheet

    if result and debug_output:
        print("Skip empty or large worksheet.")

    return result


def calculate_cell_range(cell_starts, cell_measurement, curr_coordinate, curr_size, threshold):
    """
    Calculate the cell range for current cell.
    :param cell_starts: List of cell start coordinates (either row_starts or col_starts)
    :param cell_measurement: List of cell measurement (either row_heights or col_widths)
    :param curr_coordinate: Coordinate of current cell (either curr_y or curr_x)
    :param curr_size: Size of current cell (either curr_height or curr_width)
    :param threshold: threshold of accuracy for mapping onto columns.
    :return: (min, max) range of current cell (either for row or column)
    """

    # Binary search for row/col.
    cell_min = bisect.bisect_left(cell_starts, curr_coordinate)
    if cell_min > 0 and abs(cell_starts[cell_min - 1] - curr_coordinate) <= threshold:
        cell_min -= 1

    cell_max = cell_min
    remaining_measurement = curr_size - cell_measurement[cell_max]
    while remaining_measurement > threshold and cell_max + 1 < len(cell_measurement):
        cell_max += 1
        remaining_measurement -= cell_measurement[cell_max]

    return cell_min, cell_max
