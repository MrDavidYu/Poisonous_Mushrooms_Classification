import csv

with open('mushrooms.csv', 'rb') as f:
    reader = csv.reader(f)
    header = next(header)
    rows = [header] + [ int(row[1])] for row in reader]
    rows = [row for row in reader if row[1] > 's']

for row in rows:
    print row
