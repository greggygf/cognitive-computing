import csv
with open(rb'iris.data') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print(','.join(row))
