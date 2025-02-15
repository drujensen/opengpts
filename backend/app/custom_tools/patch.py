from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)
import re

header_pattern = re.compile("^@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@$")


def apply_patch(source, patch, revert=False):
    source_lines = source.splitlines(True)
    patch_lines = patch.splitlines(True)
    target_text = ''
    index = start_line = 0
    (midx, sign) = (1, '+') if not revert else (3, '-')

    while index < len(patch_lines) and patch_lines[index].startswith(("---", "+++")):
        index += 1  # skip header lines

    while index < len(patch_lines):
        match = header_pattern.match(patch_lines[index])
        if not match:
            raise Exception("Cannot process diff")

        index += 1
        line_number = int(match.group(midx)) - 1 + (match.group(midx + 1) == '0')
        target_text += ''.join(source_lines[start_line:line_number])
        start_line = line_number

        while index < len(patch_lines) and patch_lines[index][0] != '@':
            if index + 1 < len(patch_lines) and patch_lines[index + 1][0] == '\\':
                line = patch_lines[index][:-1]
                index += 2
            else:
                line = patch_lines[index]
                index += 1

            if len(line) > 0:
                if line[0] == sign or line[0] == ' ':
                    target_text += line[1:]
                start_line += (line[0] != sign)

    target_text += ''.join(source_lines[start_line:])
    return target_text


class PatchFileInput(BaseModel):
    """Input for PatchFileTool."""

    file_path: str = Field(..., description="name of file")
    diff: str = Field(..., description="diff to apply to file")


class PatchFileTool(BaseFileToolMixin, BaseTool):
    """Tool that patches a file with a diff."""

    name: str = "patch_file"
    args_schema: Type[BaseModel] = PatchFileInput
    description: str = "Write file to disk"

    def _run(
        self,
        file_path: str,
        diff: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            write_path = self.get_relative_path(file_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name="file_path", value=file_path)
        try:
            with open(write_path, "r") as f:
                content = f.read()
            patched_content = apply_patch(content, diff)
            with open(file_path, "w") as f:
                f.write(patched_content)
        except Exception as e:
            return "Error: " + str(e)
