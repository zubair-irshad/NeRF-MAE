# category_mapping = {
#     'void': 'void',
#     'smartcustomizedceiling': 'ceiling',
#     'cabinet/lightband': 'cabinet',
#     'tea table': 'coffee_table',
#     'cornice': 'ceiling',
#     'sewerpipe': 'others',
#     'children cabinet': 'cabinet',
#     'hole': 'others',
#     'ceiling lamp': 'ceiling lamp',
#     'chaise longue sofa': 'sofa',
#     'lazy sofa': 'sofa',
#     'appliance': 'appliances',
#     'round end table': 'table',
#     'build element': 'others',
#     'dining chair': 'chair',
#     'others': 'others',
#     'armchair': 'armchair',
#     'bed': 'bed',
#     'two-seat sofa': 'sofa',
#     'lighting': 'lamp',
#     'kids bed': 'kids_bed',
#     'pocket': 'others',
#     'storage unit': 'cabinet',
#     'media unit': 'media_unit',
#     'slabside': 'slab',
#     'footstool / sofastool / bed end stool / stool': 'stool',
#     '300 - on top of others': 'others',
#     'customizedplatform': 'others',
#     'sideboard / side cabinet / console': 'cabinet',
#     'plants': 'plants',
#     'ceiling': 'ceiling',
#     'slabtop': 'slab',
#     'pendant lamp': 'pendant lamp',
#     'lightband': 'lamp',
#     'electric': 'appliances',
#     'pier/stool': 'stool',
#     'table': 'table',
#     'extrusioncustomizedceilingmodel': 'ceiling',
#     'baseboard': 'wall',
#     'front': 'others',
#     'wallinner': 'wall',
#     'basin': 'basin',  # Remains as "basin"
#     'bath': 'bath',  # Remains as "bath"
#     'customizedpersonalizedmodel': 'others',
#     'baywindow': 'window',
#     'customizedfurniture': 'furniture',
#     'sofa': 'sofa',
#     'kitchen cabinet': 'cabinet',
#     'cabinet': 'cabinet',
#     'walltop': 'wall',
#     'chair': 'chair',
#     'floor': 'floor',  # Remains as "floor"
#     'customizedceiling': 'ceiling',
#     '500 - attach to ceiling': 'ceiling',
#     'customizedbackgroundmodel': 'others',
#     'drawer chest / corner cabinet': 'cabinet',
#     'tv stand': 'tv_stand',  # Renamed to "tv_stand"
#     '400 - attach to wall': 'wall',
#     'window': 'window',
#     'art': 'art',  # Remains as "art"
#     'back': 'others',
#     'accessory': 'others',
#     '200 - on the floor': 'floor',
#     'beam': 'column',
#     'stair': 'stairs',  # Renamed to "stairs"
#     'wine cooler': 'appliances',
#     'outdoor furniture': 'furniture',
#     'double bed': 'double_bed',
#     'dining table': 'table',
#     'cabinet/shelf/desk': 'cabinet',
#     'single bed': 'single_bed',
#     'classic chinese chair': 'chair',
#     'corner/side table': 'table',
#     'flue': 'others',
#     'shelf': 'shelf',  # Remains as "shelf"
#     'customizedfeaturewall': 'wall',
#     'nightstand': 'nightstand',
#     'recreation': 'others',
#     'lounge chair / book-chair / computer chair': 'chair',
#     'slabbottom': 'floor',
#     'dressing table': 'dressing table',
#     'desk': 'desk',  # Remains as "desk"
#     'column': 'column',
#     'dressing chair': 'dressing_chair',
#     'wardrobe': 'wardrobe',  # Remains as "wardrobe"
#     'extrusioncustomizedbackgroundwall': 'wall',
#     'electronics': 'electronics',  # Remains as "electronics"
#     'bunk bed': 'single_bed',  # Renamed to "single_bed"
#     'bed frame': 'bed',  # Renamed to "single_bed"
#     'three-seat / multi-person sofa': 'sofa',
#     'customizedfixedfurniture': 'fixed_furniture',
#     'bookcase / jewelry armoire': 'bookshelf',  # Renamed to "bookshelf"
#     'mirror': 'others',
#     'wallbottom': 'wall',
#     'barstool': 'stool',
#     'wallouter': 'wall',
#     'l-shaped sofa': 'sofa',
#     'customized_wainscot': 'wall',  # Renamed to "wall"
#     'door': 'door'  # Remains as "door"
# }

category_mapping = {
    "void": "Void",
    "smartcustomizedceiling": "Ceiling",
    "ceiling": "Ceiling",
    "extrusioncustomizedceilingmodel": "Ceiling",
    "customizedceiling": "Ceiling",
    "cabinet/lightband": "Cabinet",
    "children cabinet": "Cabinet",
    "storage unit": "Cabinet",
    "media unit": "Cabinet",
    "sideboard / side cabinet / console": "Cabinet",
    "cabinet": "Cabinet",
    "drawer chest / corner cabinet": "Cabinet",
    "tv stand": "Cabinet",
    "wine cooler": "Cabinet",
    "cabinet/shelf/desk": "Cabinet",
    "wardrobe": "Cabinet",
    "tea table": "Table",
    "round end table": "Table",
    "table": "Table",
    "dining table": "Table",
    "corner/side table": "Table",
    "dressing table": "Table",
    "cornice": "Building Element",
    "sewerpipe": "Building Element",
    "hole": "Building Element",
    "build element": "Building Element",
    "others": "Building Element",
    "slabside": "Building Element",
    "electric": "Building Element",
    "front": "Building Element",
    "500 - attach to ceiling": "Building Element",
    "customizedbackgroundmodel": "Building Element",
    "back": "Building Element",
    "beam": "Building Element",
    "stair": "Building Element",
    "slabbottom": "Building Element",
    "column": "Building Element",
    "extrusioncustomizedbackgroundwall": "Building Element",
    "electronics": "Building Element",
    "customizedfixedfurniture": "Building Element",
    "customized_wainscot": "Building Element",
    "ceiling lamp": "Lighting",
    "lighting": "Lighting",
    "pendant lamp": "Lighting",
    "lightband": "Lighting",
    "chaise longue sofa": "Sofa",
    "lazy sofa": "Sofa",
    "two-seat sofa": "Sofa",
    "sofa": "Sofa",
    "three-seat / multi-person sofa": "Sofa",
    "l-shaped sofa": "Sofa",
    "appliance": "Others",
    "plants": "Others",
    "slabtop": "Others",
    "baseboard": "Others",
    "customizedpersonalizedmodel": "Others",
    "customizedfurniture": "Others",
    "400 - attach to wall": "Others",
    "art": "Others",
    "200 - on the floor": "Others",
    "outdoor furniture": "Others",
    "recreation": "Others",
    "dining chair": "Chair",
    "armchair": "Chair",
    "footstool / sofastool / bed end stool / stool": "Chair",
    "pier/stool": "Chair",
    "chair": "Chair",
    "classic chinese chair": "Chair",
    "lounge chair / book-chair / computer chair": "Chair",
    "dressing chair": "Chair",
    "barstool": "Chair",
    "bed": "Bed",
    "kids bed": "Bed",
    "double bed": "Bed",
    "single bed": "Bed",
    "bunk bed": "Bed",
    "bed frame": "Bed",
    "pocket": "Pocket",
    "300 - on top of others": "Accessory",
    "accessory": "Accessory",
    "flue": "Accessory",
    "customizedplatform": "Floor",
    "floor": "Floor",
    "wallinner": "Wall",
    "walltop": "Wall",
    "customizedfeaturewall": "Wall",
    "wallbottom": "Wall",
    "wallouter": "Wall",
    "basin": "Basin",
    "bath": "Bath",
    "baywindow": "Window",
    "window": "Window",
    "kitchen cabinet": "Kitchen Cabinet",
    "shelf": "Shelf",
    "nightstand": "Nightstand",
    "desk": "Desk",
    "bookcase / jewelry armoire": "Bookcase",
    "mirror": "Mirror",
    "door": "Door"
}

# unique_categories = sorted(set(category_mapping.values()))
# unique_categories.remove('void')  # Remove 'void' from the list of unique categories
# unique_categories = ['void'] + unique_categories  # Add 'void' back to the beginning of the list

# category_to_number = {category: index for index, category in enumerate(unique_categories)}

# num_to_category = {v: k for k, v in category_to_number.items()}
# print(category_to_number)

# unique_categories = sorted(set(category_mapping.values()))
# category_to_number = {category: index for index, category in enumerate(unique_categories)}

# print(category_to_number)

category_to_number = {
    "Void": 0,
    "Wall": 1,
    "Floor": 2,
    "Cabinet": 3,
    "Bed": 4,
    "Chair": 5,
    "Sofa": 6,
    "Table": 7,
    "Door": 8,
    "Window": 9,
    "Bookcase": 10,
    "Kitchen Cabinet": 12,
    "Desk": 14,
    "Shelf": 15,
    "Mirror": 19,
    "Ceiling": 22,
    "Nightstand": 32,
    "Basin": 34,
    "Lighting": 35,
    "Bath": 36,
    "Pocket": 37,
    "Building Element": 38,
    "Others": 39,
    "Accessory": 40
}

num_to_category = {
    0: "Void",
    1: "Wall",
    2: "Floor",
    3: "Cabinet",
    4: "Bed",
    5: "Chair",
    6: "Sofa",
    7: "Table",
    8: "Door",
    9: "Window",
    10: "Bookcase",
    12: "Kitchen Cabinet",
    14: "Desk",
    15: "Shelf",
    19: "Mirror",
    22: "Ceiling",
    32: "Nightstand",
    34: "Basin",
    35: "Lighting",
    36: "Bath",
    37: "Pocket",
    38: "Building Element",
    39: "Others",
    40: "Accessory"
}
