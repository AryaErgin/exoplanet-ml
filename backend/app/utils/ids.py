import time, random, string
def make_run_id(prefix="EV"):
    return f"{prefix}-{int(time.time()*1000):x}{random.choice(string.hexdigits.lower())}{random.choice(string.hexdigits.lower())}"
