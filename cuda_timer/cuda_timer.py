from cuda_timer import _C


def start_timer():
    event = _C.cuda_timer_start()
    return event


def stop_timer(event):
    """
    Given the timer start event, compute the time (in ms) since.
    """
    time = _C.cuda_timer_end(event)
    return time
