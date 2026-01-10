class USVisaException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_detail = error_detail