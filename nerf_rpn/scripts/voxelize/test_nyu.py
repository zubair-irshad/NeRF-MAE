import csv

# Load data from CSV
data = {}
with open('BlenderProc/blenderproc/resources/front_3D/3D_front_nyu_mapping.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id_number = int(row['id'])
        name = row['name']
        if id_number in data:
            data[id_number].append(name)
        else:
            data[id_number] = [name]

sorted_data = dict(sorted(data.items()))

# Print the grouped items with line separators
for key, values in sorted_data.items():
    print(f"Number {key}:")
    print('\n'.join(values))
    print()
