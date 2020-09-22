import os
from itertools import chain

class MarkdownReport:
    """ Class to generate a Markdown report from a study """

    def __init__(self, outname='results.md'):
        self.outname = outname
        self.toc = []
        self.has_introduction = False
        self.markdown = []

    def add_introduction(self, adict):
        """ Generate document header from dictionary """
        first = f"# CSEP Testing Results: {adict['simulation_name']}  \n" \
                f"**Forecast Name:** {adict['forecast_name']}  \n" \
                f"**Simulation Start Time:** {adict['origin_time']}  \n" \
                f"**Evaluation Time:** {adict['evaluation_time']}  \n" \
                f"**Catalog Source:** {adict['catalog_source']}  \n" \
                f"**Number Simulations:** {adict['num_simulations']}\n"

        # used to determine to place TOC at beginning of document or after introduction.
        self.has_introduction = True
        self.markdown.append(first)
        return first

    def add_text(self, text):
        """
        text should be a list of strings where each string will be on its own line.
        each add_text command represents a paragraph.

        Args:
            text (list): lines to write

        Returns:

        """
        self.markdown.append('  '.join(text) + '\n\n')

    def add_result_figure(self, title, level, relative_filepaths, ncols=3, add_ext=True, text='', caption=''):
        """
        this function expects a list of filepaths. if you want the output stacked, select a
        value of ncols. ncols should be divisible by filepaths. todo: modify formatted_paths to work when not divis.

        Args:
            title: name of the figure
            level (int): value 1-6 depending on the heading
            relative_filepaths (str or List[Tuple[str]]): list of paths in order to make table

        Returns:

        """

        # convert relative_filepaths into a list with ncols

        # verify filepaths have proper extension should always be png
        total_items = len(relative_filepaths)
        is_single = True if total_items == 1 else False
        correct_paths = []
        if add_ext:
            for fp in relative_filepaths:
                correct_paths.append(fp + '.png')
        else:
            correct_paths = relative_filepaths

        # generate new lists with size ncols
        formatted_paths = [correct_paths[i:i+ncols] for i in range(0, len(correct_paths), ncols)]

        # convert str into a proper list, where each potential row is an iter not str
        def build_header(row):
            top = "|"
            bottom = "|"
            for i, _ in enumerate(row):
                if i == ncols:
                    break
                top +=  " |"
                bottom += " --- |"
            return top + '\n' + bottom

        def add_to_row(row):
            if len(row) == 1:
                return f"![]({row[0]})"
            string = '| '
            for item in row:
                string = string + f' ![]({item}) |'
            return string

        level_string = f"{level*'#'}"
        result_cell = []
        locator = title.lower().replace(" ", "_")
        result_cell.append(f'{level_string} {title}  <a name="{locator}"></a>\n')
        result_cell.append(f'{text}\n')

        for i, row in enumerate(formatted_paths):
            if i == 0 and not is_single:
                result_cell.append(build_header(row))
            result_cell.append(add_to_row(row))
        result_cell.append('\n')


        result_cell.append(f'{caption}')

        self.markdown.append('\n'.join(result_cell) + '\n')

        # generate metadata for TOC
        self.toc.append((title, level, locator))

    def add_sub_heading(self, title, level, text):
        # multipying char simply repeats it
        if isinstance(text, str):
            text = [text]
        cell = []
        level_string = f"{level*'#'}"
        locator = title.lower().replace(" ", "_")
        sub_heading = f'{level_string} {title} <a name="{locator}"></a>\n'
        cell.append(sub_heading)
        try:
            for item in list(text):
                cell.append(item)
        except:
            raise RuntimeWarning("Unable to add results document subheading, text must be iterable.")
        self.markdown.append('\n'.join(cell) + '\n')

        # generate metadata for TOC
        self.toc.append((title, level, locator))

    def _generate_table_of_contents(self):
        """ generates table of contents based on contents of document. """
        toc = []
        toc.append("# Table of Contents")
        # allows for 6 levels of subheadings
        inc = [0] * 6
        for title, level, locator in self.toc:
            space = '   ' * (level-1)
            toc.append(f"{space}1. [{title}](#{locator})")

        insert_loc = 1 if self.has_introduction else 0
        self.markdown.insert(insert_loc, '\n'.join(toc) + '\n')


    def get_table(self, data, use_header=True):
        """
        Generates table from HTML and styles using bootstrap class
        Args:
           data List[Tuple[str]]: should be (nrows, ncols) in size. all rows should be the
                         same sizes

        Returns:
            table (str): this can be added to subheading or other cell if desired.

        """
        table = []
        table.append('<div class="table table-striped">')
        table.append(f'<table>')
        def make_header(row):
            header = []
            header.append('<tr>')
            for item in row:
                header.append(f'<th>{item}</th>')
            header.append('</tr>')
            return '\n'.join(header)

        def add_row(row):
            table_row = []
            table_row.append('<tr>')
            for item in row:
                table_row.append(f"<td>{item}</td>")
            table_row.append('</tr>')
            return '\n'.join(table_row)

        for i, row in enumerate(data):
            if i==0 and use_header:
                table.append(make_header(row))
            else:
                table.append(add_row(row))
        table.append('</table>')
        table.append('</div>')
        table = '\n'.join(table)
        self.markdown.append(table + '\n')

    def finalize(self, save_dir):
        self._generate_table_of_contents()
        output = list(chain.from_iterable(self.markdown))
        full_md_fname = os.path.join(save_dir, self.outname)
        with open(full_md_fname, 'w') as f:
            f.writelines(output)