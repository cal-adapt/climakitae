import xarray as xr
import inspect
import sys
from transform_helpers import progress_bar
from dask import delayed, compute

# set global xarray to keep attributes
# otherwise we lose crucial metadata during transforms
xr.set_options(keep_attrs=True)


def generate_metadata(transform_name,func_args,
                        func_kwargs,metadata_dict):
    
    """
    returns dictionary of xarray dataset/data array attributes.
    this is called in each transform itself. See example_transform
    for a usage example.
    
    these are pulled via the code dropped into the 
    transform function itself; no need to manually call.
    transform_name: the function applied to the data
    func_args: user-defined arguments fed into the function
    func_kwargs: user-defined keyword arguments fed into function
    metadata_dict: author-defined dictionary containing 
    relevant metadata. 
    
    note: the first three arguments are defined flexibly
    from code that transform authors put at the beginning
    of their functions. I am looking for a way to call this
    code separately to reduce this redundancy.
    """
    
    # flag that climakitae transform has been applied
    transform_attrs = {'post_processed' : 'true',
                 'post_processed_by' : 'Cal-Adapt Analytics'+
                 'Engine v 0.1'}
    # append author-supplied metadata dict
    transform_attrs.update(metadata_dict)

    # transform and transform details dict
    # this is to make the process as traceable as possible
    # first get transform name
    transform_command = 'climakitae.transform.'+str(transform_name)
    transform_details = {'transform_function' : transform_command,
                'transform_arguments' : func_args}  
    
    # add details to existing attrs dict
    transform_attrs.update(transform_details)
    transform_attrs.update(func_kwargs)
    return(transform_attrs)


@progress_bar
@delayed
def example_transform(ds,*args,**kwargs):
    
    """
    returns a full-record temporal mean of a dataset (ds).
    args are not doing anything right now but changing metadata,
    but one kwarg does something just to show that options
    can be updated flexibly as metadata.
    """
    
    # ================================================================
    # CODE THAT MUST BE ADDED TO TRANSFORM
    # (transform authors don't need to change anything) 
    # could this be a class or something
    # so we can take it out of each transform?
        
    # define function name
    transform_name = inspect.stack()[0][3] # string
    transform_cmd = eval(transform_name) # functional form

    # get function *args and **kwargs
    # this is so simple
    # why didn't this show up first on google
    # I wasted so much time
    # let me know if it does not work
    # there are other ways
    func_args = args
    func_kwargs = kwargs

    # BEGIN TRANSFORM AUTHOR INPUT:
    # define the transform-specific things we need to pass as metadata
    # up to you what is important here, but be as explicit as possible
    description = 'temporal mean' # string summary of transform
    author = 'Beth McClenny, Eagle Rock Analytics' # transform author
                                            # and affiliation - keep?
    # then whatever other things you want to add
    # just make sure you add them to the dictionary below    
    metadata_dict = {'transform_description' : description,
                     'transform_author' : author}
    
    # define metadata dict from ds
    ds_attrs = ds.attrs
    
    # ================================================================
    # YOUR TRANSFORM CODE HERE:
    # simple example: a temporal mean
    # let's throw in an option for time slicing:    
    if ('time_bounds'):
        time_list = kwargs.get("time_bounds")
        t0 = time_list[0]
        t1 = time_list[1]
        
    else: # if operating over the entire record
        t0 = ds.time.isel(time=0).values
        t1 = ds.time.isel(time=1).values
            
    ds_transformed = ds.sel(time=slice(t0,t1)).mean(dim='time')     
    # ================================================================
    
    # APPEND TRANSFORM METADATA AT THE END
    # GENERIC CODE AGAIN
    # CALL generate_metadata FUNCTION to generate new attrs:
    # and add to current metadata dict
    ds_attrs.update(generate_metadata(transform_name,
                            func_args,func_kwargs,metadata_dict))
    
    ds_transformed.attrs = ds_attrs
    
    return(ds_transformed)