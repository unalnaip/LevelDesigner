from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def parse_time_to_seconds(time_str: str) -> int:
    """Convert time string (HH:MM:SS or MM:SS) to seconds.
    
    Args:
        time_str: Time string in format "HH:MM:SS" or "MM:SS"
        
    Returns:
        int: Total seconds
        
    Raises:
        ValueError: If time string format is invalid
    """
    try:
        time_parts = time_str.split(':')
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            minutes, seconds = map(int, time_parts)
            hours = 0
        else:
            raise ValueError(f"Invalid time format: {time_str}")
            
        return int(timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        ).total_seconds())
    except Exception as e:
        logger.error(f"Error parsing time: {time_str}")
        raise ValueError(f"Could not parse time string: {time_str}") from e 