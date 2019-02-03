import os
import csv


def relative_path(file, path):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(file)))
    return os.path.join(__location__, path)


def get_csv_lines(filename, delimeter=','):
    with open(filename, encoding='UTF-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimeter)
        for row in reader:
            yield row
