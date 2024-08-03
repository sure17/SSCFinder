""" 检测一个包是否包含空的描述信息
"""
import logging
from typing import Optional

from guarddog.analyzer.metadata.empty_information import EmptyInfoDetector

MESSAGE = "This package has an empty description on PyPi"

log = logging.getLogger("guarddog")


class PypiEmptyInfoDetector(EmptyInfoDetector):
    def detect(self, package_info, path: Optional[str] = 'None', name: Optional[str] = None,
               version: Optional[str] = None) -> tuple[bool, str]:
        log.debug(f"Running PyPI empty description heuristic on package {name} version {version}")
        return len(package_info["info"]["description"].strip()), EmptyInfoDetector.MESSAGE_TEMPLATE % "PyPI"
