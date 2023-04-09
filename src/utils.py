"""Utilities used in the package."""

import time
import torch


def select_device(gpu=True):
    """Return GPU device if available, else return CPU."""

    # Prioritize CUDA
    if gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else: # fall-back option
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device


def get_timestamp():
  """Timestamp utility."""

  # Get the current time in seconds since the epoch
  current_time = time.time()
  # Convert the current time to a struct_time object
  time_struct = time.localtime(current_time)
  # Extract the day, month, year, hours, and minutes from the time_struct object
  day = time_struct.tm_mday
  month = time_struct.tm_mon
  year = time_struct.tm_year
  hours = time_struct.tm_hour
  minutes = time_struct.tm_min
  # Construct the timestamp string in the format DAY:MONTH:YEAR:HOURS:MINUTES
  timestamp = f"{day:02d}:{month:02d}:{year}:{hours:02d}:{minutes:02d}"

  return(timestamp)