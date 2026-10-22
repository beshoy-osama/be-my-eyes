from .BaseController import BaseController
from fastapi import UploadFile
from models.enums import ResponseEnums
import re
import os

import filetype


class DataController(BaseController):
    def __init__ (self):
        super().__init__()
        self.size_scale = 1048576

    ############### To_get_file_size ##################
    def get_file_size(self, file: UploadFile) -> int:
        if file.size is not None:
            return file.size

        # Some upload backends do not provide `size`; calculate it from the stream.
        current_position = file.file.tell()
        file.file.seek(0, 2)
        calculated_size = file.file.tell()
        file.file.seek(current_position)

        return calculated_size
    ####################################################

    async def validate_upload_file(self, file: UploadFile):


        file_size = self.get_file_size(file)

        # 1. Read the first 2048 bytes of the file to determine its type
        header = await file.read(2048)
        await file.seek(0)         

        # 2. Determine the actual type
        kind = filetype.guess(header)

        if kind is None or f".{kind.extension}" not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseEnums.FILE_TYPE_NOT_SUPPORTED.value

        if file_size > (self.app_settings.FILE_MAX_SIZE * self.size_scale):
            return False, ResponseEnums.FILE_SIZE_EXCEEDED.value

        return True, ResponseEnums.FILE_UPLOAD_SUCCESS.value

    def generate_unique_file_path(self, file_name: str, project_id: str):

        random_key = self.generate_random_string()

        project_path = os.path.join(self.files_dir, str(project_id))
        os.makedirs(project_path, exist_ok=True)

        cleaned_file_name = self.get_clean_file_name(file_name=file_name)

        stem, ext = os.path.splitext(cleaned_file_name)

        new_file_path = os.path.join(
            project_path,
            random_key + '_' + stem + ext
        )

        while os.path.exists(new_file_path):

            random_key = self.generate_random_string()

            new_file_path = os.path.join(
                project_path,
                random_key + '_' + stem + ext
            )

        file_id = random_key + '_' + stem + ext

        return new_file_path, file_id

    def get_file_path(self, file_id: str, project_id: str) -> str | None:
        """Return the stored path of an uploaded file, or None if it does not exist."""
        safe_project_id = os.path.basename(str(project_id))
        safe_file_id = os.path.basename(file_id)
        if not safe_project_id or not safe_file_id or safe_file_id != file_id:
            return None

        file_path = os.path.join(self.files_dir, safe_project_id, safe_file_id)
        return file_path if os.path.isfile(file_path) else None


    def get_clean_file_name(self, file_name: str):

        # Replace whitespace with underscore first
        cleaned_file_name = re.sub(r'\s+', '_', file_name.strip())
        
        # Remove any character that is not alphanumeric, underscore, or period
        cleaned_file_name = re.sub(r'[^\w.]', '', cleaned_file_name)

        return cleaned_file_name
