import numpy as np

response = np.load('./R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response.npy')

good_rows = []
bad_rows = []

where_bad_row = (response == -1).sum(1) == response.shape[1]
bad_rows = np.where(where_bad_row)[0]
good_rows = np.where(~where_bad_row)[0]
response = response[good_rows]

good_columns = []
bad_columns = []

all_columns = np.arange(response.shape[1])
bad_columns = np.where(response.sum(0) == -response.shape[0])[0]
good_columns = np.array(list(set(all_columns) - set(bad_columns)))

response_good = response[:, good_columns]

good_columns2 = []
bad_columns2 = []

for i in range(response_good.shape[1]):
    response_i = response_good[:, i]
    response_i = response_i[response_i != -1]
    response_i = np.unique(response_i)

    if len(response_i) == 1:
        bad_columns2.append(i)
    else:
        good_columns2.append(i)

bad_columns2 = np.array(bad_columns2).astype(np.int)
good_columns2 = np.array(good_columns2).astype(np.int)

response_good2 = response_good[:, good_columns2]

np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response_edited.npy',
    response_good2,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_rows.npy',
    good_rows,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_rows.npy',
    bad_rows,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_columns1.npy',
    good_columns,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_columns1.npy',
    bad_columns,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_columns2.npy',
    good_columns2,
)
np.save(
    './R_data/pisa2015_science_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_columns2.npy',
    bad_columns2,
)
