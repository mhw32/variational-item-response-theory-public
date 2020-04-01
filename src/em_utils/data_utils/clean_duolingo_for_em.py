import numpy as np

good_columns = []
bad_columns = []
response = np.load('./R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response.npy')

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
    './R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/response_edited.npy',
    response_good2,
)
np.save(
    './R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_columns1.npy',
    good_columns,
)
np.save(
    './R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_columns1.npy',
    bad_columns,
)
np.save(
    './R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/good_columns2.npy',
    good_columns2,
)
np.save(
    './R_data/duolingo_language_1000person_100item_Nonemaxperson_Nonemaxitem_1ability/bad_columns2.npy',
    bad_columns2,
)
