import panel as pn
from climakitae.core.data_interface import DataInterface


class ExportFormat:
    """
    Called by 'export' at the end of the workflow. Displays panel
    from which to select the export file format. Modifies 'user_export_format'
    object, which is used by data_export() to export data to the user in their
    specified format.
    """

    def __init__(self):
        self.user_export_format = DataInterface().export_type

        # reserved for later: text boxes for dataset to export
        # as well as a file name
        # file_name = pn.widgets.TextInput(name='File name',
        #                                 placeholder='Type file name here')
        # file_input_col = pn.Column(user_export_format.param, data_to_export, file_name)
        return pn.Row(self.user_export_format.param)
