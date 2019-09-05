def line_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    olines = []
    for atom in atoms:
        i = int(atom["LineNo"])
        try:
            olines.append(lines[i-1])
        except Exception as e:
            print(e)
            pass
    return (olines, atoms)


def word_mapping_f(text, atoms):
    lines = list(filter(lambda x: x, text.split('\n')))
    words = list(map(lambda x: x.split(), lines))
    owords = []
    for atom in atoms:
        i, j = list(map(lambda x: int(atom[x]), ["LineNo", "SerialNo"]))
        owords.append(words[i-1][j-1])
    return (owords, atoms)
