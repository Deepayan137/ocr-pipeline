
def codebook(filename):
    keys = open(filename).read().splitlines()
    labels = list(map(fmt_to_unicode, keys))

    # Add the blank character.
    labels.insert(0, '')
    indices = list(range(len(labels)))

    lmap = dict(zip(labels, indices))
    invlmap = dict(zip(indices, labels))
    return (lmap, invlmap)


def fmt_to_unicode(fmt):
    codepoint_repr = fmt[1:]
    codepoint_value = int(codepoint_repr, 16)
    return chr(codepoint_value)
