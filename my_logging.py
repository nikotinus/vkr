import inspect
import traceback

def get_func_name():
    return traceback.extract_stack(None, 2)[0][2]

def get_func_params_and_values():
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    return ([(i, values[i]) for i in args])

def get_func_params():
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    return [item for item in args]

	

