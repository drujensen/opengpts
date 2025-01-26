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

_hdr_pat = re.compile("^@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@$")


def apply_patch(s, patch, revert=False):
    s = s.splitlines(True)
    p = patch.splitlines(True)
    t = ''
    i = sl = 0
    (midx, sign) = (1, '+') if not revert else (3, '-')
    while i < len(p) and p[i].startswith(("---", "+++")):
        i += 1  # skip header lines
    while i < len(p):
        m = _hdr_pat.match(p[i])
        if not m:
            raise Exception("Cannot process diff")
        i += 1
        l = int(m.group(midx))-1 + (m.group(midx+1) == '0')
        t += ''.join(s[sl:l])
        sl = l
        while i < len(p) and p[i][0] != '@':
            if i+1 < len(p) and p[i+1][0] == '\\':
                line = p[i][:-1]
                i += 2
            else:
                line = p[i]
                i += 1
            if len(line) > 0:
                if line[0] == sign or line[0] == ' ':
                    t += line[1:]
                sl += (line[0] != sign)
    t += ''.join(s[sl:])
    return t


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
