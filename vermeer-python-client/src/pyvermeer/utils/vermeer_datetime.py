# !/usr/bin/env python3
"""
file: vermeer_datetime.py
author: wenyuxuan@baidu.com
"""

import datetime

from dateutil import parser


def parse_vermeer_time(vm_dt: str) -> datetime:
    """Parse a vermeer time string into a Python datetime object."""
    if vm_dt is None or len(vm_dt) == 0:
        return None
    dt = parser.parse(vm_dt)
    return dt


if __name__ == '__main__':
    print(parse_vermeer_time('2025-02-17T15:45:05.396311145+08:00').strftime("%Y%m%d"))
