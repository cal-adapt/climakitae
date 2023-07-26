import panel as pn
from climakitae.core.data_interface import ExportParameters


class Export:
    """
    Called by 'export' at the end of the workflow. Displays panel
    from which to select the export file format. Modifies 'user_export_format'
    object, which is used by data_export() to export data to the user in their
    specified format.
    """

    def __init__(self):
        self.export_parameters = ExportParameters()

    def export_as(self):
        # Show export file type panel
        export_panel = _display_export(self.export_parameters)
        return export_panel

    def export_dataset(self):
        return


def _display_export(self):
    return pn.Row(
        pn.widgets.Select.from_param(
            self.output_file_format,
            name="Choose File Format for Exporting",
        )
    )
