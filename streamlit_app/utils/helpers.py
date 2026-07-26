def format_score(score):

    if score is None:
        return "-"

    return f"{score:.3f}"


def format_time(seconds):

    if seconds is None:
        return "-"

    return f"{seconds*1000:.1f} ms"