import os
from lxml import etree
from .mapping import line_mapping_f, word_mapping_f
from .imgproc import extract_units
import pdb
from tqdm import *
# Parse line.xml file.


def parse_ocr_xml(xml_file):
    with open(xml_file, encoding='utf-8') as f:
        #print("Read", xml_file)
        root = etree.parse(f)
        rows = root.xpath("row")

        # Subfunction to extract from a row
        def extract(field): return (field.get("name"), field.text)

        def extract_field(x): return dict(map(extract, x.xpath("field")))
        export = list(map(extract_field, rows))
        return export


def group(text_data, units):
    udict = {}
    required_keys = ["ImageLoc", "Text"]
    udict["page"] = {}
    for entry in text_data:
        if not "BookCode" in udict:
            udict["BookCode"] = entry["BookCode"]
        pno = int(entry["PageNo"])
        if pno not in udict["page"]:
            udict["page"][pno] = dict((key, entry[key])
                                      for key in required_keys)
            udict["page"][pno]["units"] = []

    for unit in units:
        pno = int(unit["PageNo"])
        udict["page"][pno]["units"].append(unit)

    # Return ultimate dict
    return udict


def images_and_truths(udict, mapping_f):
    result = []
    prefix = udict["prefix"]
    for pno in (udict["page"]):
        def extract_required(x): return udict["page"][pno][x]
        required_keys = ['Text', 'ImageLoc', 'units']
        text, imgloc, units = list(map(extract_required, required_keys))

        # Order units by Key
        imagename = imgloc.split('/')[-1]
        # file = os.path.join(prefix, 'Predictions_CRNN', imagename+'.txt')
        # pred = open(file, 'r').read()
        unit_truths, units = mapping_f(text, units)
        unit_images = extract_units(prefix+imgloc, units)
        result.append((unit_images, unit_truths))
        #print(len(unit_images),  len(unit_truths))
        # result.append([imagename, (unit_images, pred)])
    return result


def read_book(**kwargs):
    book_dir_path = kwargs['book_path']
    opt_unit = kwargs['unit']

    def obtainxml(f): return book_dir_path + f + '.xml'
    filenames = map(obtainxml, ['line', 'word', 'text'])
    lines, words, text = list(map(parse_ocr_xml, filenames))
    units = words
    mapping_f = word_mapping_f
    if opt_unit == 'line':
        units = lines
        mapping_f = line_mapping_f
    ud = group(text, units)
    ud["prefix"] = book_dir_path
    pagewise = images_and_truths(ud, mapping_f)
    return pagewise
