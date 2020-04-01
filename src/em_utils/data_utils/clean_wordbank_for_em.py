import numpy as np

good_columns = []
bad_columns = []
response = np.load('./R_data/wordbank_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response.npy')

for i in range(response.shape[1]):
    response_i = response[:, i]
    response_i = response_i[response_i != -1]
    response_i = np.unique(response_i)

    if len(response_i) == 1:
        bad_columns.append(i)
    else:
        good_columns.append(i)

bad_columns = np.array(bad_columns).astype(np.int)
good_columns = np.array(good_columns).astype(np.int)

response_good = response[:, good_columns]
np.save(
    './R_data/wordbank_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response_edited.npy',
    response_good,
)
np.save(
    './R_data/wordbank_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_columns.npy',
    good_columns,
)
np.save(
    './R_data/wordbank_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_columns.npy',
    bad_columns,
)