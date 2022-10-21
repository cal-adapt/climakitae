# Decorator to append transform call
# details as metadata to ds/da

from functools import wraps

def transform_details(func):
    @wraps(func)
    def update_metadata(*args, **kwargs):

        """
        Wrapper that pulls function parameters on execution of any
        transform 'func'.

        Usage: place @transform_details above transform definition.

        Inputs:
        args: function arguments.
        NOTE: Assumes that the xarray.Dataset or xarray.DataArray is
        the first argument.
        kwargs: function keyword arguments.

        Output:
        The resultant transformed ds/da with updated metadata.
        """

        # Getting the name, args, and kwargs of the called function
        fname = func.__name__
        ds = args[0]
        func_args = args[1:]
        func_kwargs = kwargs
        
        # get existing ds/da attributes as a dictionary
        # which can be updated with new information.
        orig_attrs = ds.attrs

        # flag that climakitae transform has been applied
        transform_attrs = {
            'post_processed' : 'true',
            'post_processed_by' : 'Cal-Adapt Analytics Engine v 0.0.1'
        }

        # build the transform and transform details dict
        # this is to make the process as traceable as possible
        # first get transform name
        transform_command = 'climakitae.transform.' + str(fname)
        transform_details = {
            'transform_function' : transform_command,
            'transform_arguments' : func_args
        }

        # add details to existing attrs dict
        transform_attrs.update(transform_details)
        transform_attrs.update(func_kwargs)
        orig_attrs.update(transform_attrs)

        # execute the transform and update attributes
        ds_transformed = func(*args, **kwargs)
        ds_transformed.attrs = orig_attrs
        return(ds_transformed)
    return(update_metadata)
