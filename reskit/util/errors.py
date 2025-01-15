import inspect

class ResError(Exception):
    pass  # this just creates an error that we can use


class RESKitDeprecationError(Exception):
    """Custom exception for deprecation errors."""
    def __init__(self, commit_hash):
        """
        Raises an error stating that the method is deprecated and suggests
        a commit to check out when function is needed for backward 
        compatibility.

        commit_hash : str
            The has to the commit with the last working version
            of the deprecated method.
        """
        # extract function name from where the error wwas raised
        caller_frame = inspect.currentframe().f_back
        calling_function_name = caller_frame.f_code.co_name
        # raise individual error
        message = f"'{calling_function_name}' is deprecated and should not be used anymore. Check out the last working version to reproduce results under: {commit_hash}"
        super().__init__(message)