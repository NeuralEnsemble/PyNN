

def is_listlike(obj):
    return hasattr(obj, "__len__") and not isinstance(obj, basestring)
