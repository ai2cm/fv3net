from datetime import timedelta

import vcm


def add_offset(
    time: str, seconds: Union[int, float]
) -> str:
    # TODO refactor to vcm.convenience
    """Add offset_seconds to a timestamp in YYYYMMDD.HHMMSS format"""
    offset = timedelta(seconds=seconds)
    offset_datetime = vcm.parse_datetime_from_str(time) + offset
    return offset_datetime.strftime("%Y%m%d.%H%M%S")