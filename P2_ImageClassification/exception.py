import sys
import logging

def get_err_msg_detail(err, err_detail: sys):
    _, _, exc_tb = err_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    err_msg = "Error occurred in python script name [{0}], line number [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(err)
    )

    return err_msg



class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = get_err_msg_detail(err = error_message, err_detail = error_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    logging.info("Logging has started")

    try:
        a = 1/0

    except Exception as e:
        logging.info("Division by Zerooooooooo")
        raise CustomException(e, sys)