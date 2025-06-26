##................CUSTOM EXCEPTION................##

import traceback  ## to traceback the error
import sys 

class CustomException(Exception):
    """
    Custom exception class to handle exceptions in the application.
    It captures the exception message and the traceback.
    """
    def __init__(self, error_message, error_details : sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)


    @staticmethod
    def get_detailed_error_message(error_message, error_details : sys):
        """
        Returns a detailed error message including the traceback.
        """
        _ , _ , exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] : [{error_message}]"

    def __str__(self):
        """ Returns the string representation of the error message.
        """
        return self.error_message