import os
import ast
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter, OrderedDict
import torch.utils.data

from src.config import (
    CHILDREN_LANG_DIR, 
    DUOLINGO_LANG_DIR,
    WORDBANKR_LANG_DIR, 
    PISA2015_DIR,
    DATA_DIR,
)


def load_dataset(dataset_name, train=True, **kwargs):
    if dataset_name == '1pl_simulation':
        return IRTSimulation(train=train, irt_model='1pl', **kwargs)
    elif dataset_name == '2pl_simulation':
        return IRTSimulation(train=train, irt_model='2pl', **kwargs)
    elif dataset_name == '3pl_simulation':
        return IRTSimulation(train=train, irt_model='3pl', **kwargs)
    elif dataset_name == '1pl_nonlinear':
        return IRTSimulation(train=train, irt_model='1pl', nonlinear=True, **kwargs)
    elif dataset_name == '2pl_nonlinear':
        return IRTSimulation(train=train, irt_model='2pl', nonlinear=True, **kwargs)
    elif dataset_name == '3pl_nonlinear':
        return IRTSimulation(train=train, irt_model='3pl', nonlinear=True, **kwargs)
    elif dataset_name == 'critlangacq':
        return Children_LanguageAcquisition(train=train, **kwargs)
    elif dataset_name == 'duolingo':
        return DuoLingo_LanguageAcquisition(train=train, binarize=True, **kwargs)
    elif dataset_name == 'wordbank':
        return WordBank_Language(train=train, **kwargs)
    elif dataset_name == 'pisa2015_science':
        return PISAScience2015(train=train, **kwargs)
    else:
        raise Exception(f'Dataset {dataset_name} is not supported.')


def artificially_mask_dataset(old_dataset, perc):
    dataset = copy.deepcopy(old_dataset)
    assert perc >= 0 and perc <= 1
    response = dataset.response
    mask = dataset.mask

    if np.ndim(mask) == 2:
        row, col = np.where(mask != 0)
    elif np.ndim(mask) == 3:
        row, col = np.where(mask[:, :, 0] != 0)
    pool = np.array(list(zip(row, col)))
    num_all = pool.shape[0]
    num = int(perc * num_all)
    
    rs = np.random.RandomState(42)
    indices = np.sort(
        rs.choice(np.arange(num_all), size=num, replace=False),
    )
    label_indices = pool[indices]
    labels = []
    for idx in label_indices:
        label = copy.deepcopy(response[idx[0], idx[1]])
        labels.append(label)
        mask[idx[0], idx[1]] = 0
        response[idx[0], idx[1]] = -1
    labels = np.array(labels)

    dataset.response = response
    dataset.mask = mask
    dataset.missing_labels = labels
    dataset.missing_indices = label_indices

    return dataset


def load_1pl_simulation(
        num_person = 1000, 
        num_item = 100, 
        ability_dim = 1, 
        nonlinear = False,
    ):
    data_dir = os.path.join(
        DATA_DIR, 
        f'1pl_simulation_{num_person}person_{num_item}item_{ability_dim}ability',
    )
    if nonlinear: data_dir = data_dir + '_nonlinear'
    dataset = torch.load(os.path.join(data_dir, 'simulation.pth'))
    return dataset


def load_2pl_simulation(
        num_person = 1000, 
        num_item = 100, 
        ability_dim = 1, 
        nonlinear = False,
    ):
    data_dir = os.path.join(
        DATA_DIR, 
        f'2pl_simulation_{num_person}person_{num_item}item_{ability_dim}ability',
    )
    if nonlinear: data_dir = data_dir + '_nonlinear'
    dataset = torch.load(os.path.join(data_dir, 'simulation.pth'))
    return dataset


def load_3pl_simulation(
        num_person = 1000, 
        num_item = 100, 
        ability_dim = 1, 
        nonlinear = False,
    ):
    data_dir = os.path.join(
        DATA_DIR, 
        f'3pl_simulation_{num_person}person_{num_item}item_{ability_dim}ability',
    )
    if nonlinear: data_dir = data_dir + '_nonlinear'
    dataset = torch.load(os.path.join(data_dir, 'simulation.pth'))
    return dataset


def load_duolingo(filename):
    """
    This method loads and returns the data in filename. If the data is 
    labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional):  if you specified training data, a dict of 
                            instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')
    instance_properties = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if 'prompt' in line:
                    instance_properties['prompt'] = line.split(':')[1]
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'countries':
                            value = value.split('|')
                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']
        self.prompt = instance_properties.get('prompt', None)

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

    def to_features(self):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        to_return['bias'] = 1.0
        to_return['user:' + self.user] = 1.0
        to_return['format:' + self.format] = 1.0
        to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0

        return to_return


class Children_LanguageAcquisition(torch.utils.data.Dataset):
    """
    A critical period for second language acquisition: Evidence from 
    2/3 million English speakers.

    Abstract from the paper: 

    Children learn language more easily than adults, though when 
    and why this ability declines have been obscure for both empirical 
    reasons (underpowered studies) and conceptual reasons (measuring 
    the ultimate attainment of learners who started at different ages 
    cannot by itself reveal changes in underlying learning ability). 
    We address both limitations with a dataset of unprecedented size 
    (669,498 native and non-native English speakers) and a computational 
    model that estimates the trajectory of underlying learning ability 
    by disentangling current age, age at first exposure, and years of 
    experience. This allows us to provide the first direct estimate of 
    how grammar-learning ability changes with age, finding that it is 
    preserved almost to the crux of adulthood (17.4 years old) and then 
    declines steadily. This finding held not only for "difficult" 
    syntactic phenomena but also for "easy" syntactic phenomena that 
    are normally mastered early in acquisition. The results support the 
    existence of a sharply-defined critical period for language acquisition,
    but the age of offset is much later than previously speculated. The 
    size of the dataset also provides novel insight into several other 
    outstanding questions in language acquisition.

    Columns in the dataset: 

    Metadata
    --------
    'id': integer (unique identifier of the individual)
    'date',
    'time',

    Person Attributes
    -----------------
    'gender',
    'age',
    'natlangs',
    'primelangs',
    'dyslexia',
    'psychiatric',
    'education',
    'tests',
    'Eng_start',
    'Eng_country_yrs',
    'house_Eng',
    'dictionary',
    'already_participated',
    'countries',
    'currcountry',
    'US_region',
    'UK_region',
    'Can_region',
    'Ebonics',
    'Ir_region',
    'UK_constituency',
    'nat_Eng',
    'prime_Eng',
    'speaker_cat',
    'type',
    'Lived_Eng_per',
    'Eng_little',
    
    Questions
    ---------
    'q1': {0,1} correctness or not correct
    'q2': {0,1} correctness or not correct 
    ...

    Answers
    -------
    'correct': average correctness:
    'elogit':  this is a real number before squishing into [0,1]

    There is no missing data. There are 535598 persons and 95 items.
    """
    def __init__(self, train=True, max_num_person=None, max_num_item=None, **kwargs):
        super().__init__()
        data_path = os.path.join(CHILDREN_LANG_DIR, 'data.csv')
        data_df = pd.read_csv(data_path)
        metadata = self.get_metadata(data_df)
        item_keys = [
            'q1', 'q2', 'q3', 'q5', 'q6', 'q7', 'q9_1', 'q9_4', 'q10_2', 
            'q10_4', 'q11_3', 'q11_4', 'q12_1', 'q12_2', 'q12_4', 'q13_3', 
            'q13_4', 'q14_3', 'q14_4', 'q15_1', 'q15_2', 'q15_3', 'q16_3', 
            'q16_4', 'q17_1', 'q17_3', 'q17_4', 'q18_2', 'q18_3', 'q18_4',
            'q19_1', 'q19_2', 'q19_3', 'q19_4', 'q20_1', 'q20_2', 'q20_3', 
            'q20_4', 'q21_1', 'q21_2', 'q21_3', 'q21_4', 'q22_1', 'q22_2', 
            'q22_3', 'q22_4', 'q23_3', 'q23_4', 'q24_1', 'q24_2', 'q24_3', 
            'q24_4', 'q25_1', 'q25_2', 'q25_3', 'q25_4', 'q26_1', 'q26_2', 
            'q26_3', 'q26_4', 'q27_1', 'q27_2', 'q27_3', 'q27_4', 'q28_1', 
            'q28_2', 'q29_1', 'q29_2', 'q29_3', 'q29_4', 'q30_1', 'q30_2', 
            'q30_3', 'q30_4', 'q31_1', 'q31_4', 'q32_5', 'q32_6', 'q32_8', 
            'q33_4', 'q33_5', 'q33_6', 'q33_7', 'q34_1', 'q34_2', 'q34_3', 
            'q34_4', 'q34_6', 'q34_8', 'q35_1', 'q35_2', 'q35_4', 'q35_5', 
            'q35_7', 'q35_8',
        ]
        item_id = np.arange(len(item_keys))
        # NOTE: no missing data here? 
        response = np.asarray(data_df[item_keys])
    
        rs = np.random.RandomState(42)
        swapper = np.arange(response.shape[0])
        rs.shuffle(swapper)
        response = response[swapper]

        num_person = response.shape[0]
        num_train  = int(0.8 * num_person)

        if train:
            response = response[:num_train]
        else:
            response = response[num_train:]

        if max_num_person is not None:
            response = response[:max_num_person]
        
        if max_num_item is not None:
            response = response[:, :max_num_item]
            item_id = item_id[:max_num_item]

        response_mask = np.ones_like(response)
        response_mask[response == -1] = 0

        self.response = response
        self.metadata = metadata
        self.item_id = item_id
        self.mask = response_mask
        self.length = response.shape[0]
        self.num_person = response.shape[0]
        self.num_item = response.shape[1]

    def get_metadata(self, df):
        age = np.asarray(df['age'])
        education = np.asarray(df['education'])
        metadata = {
            'age': age,
            'education': education,
        }
        return metadata

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        response = self.response[index]
        item_id = self.item_id.copy()
        # -1 in questions represents missing data
        item_id[response == -1] = -1
        mask = self.mask[index]

        response = torch.from_numpy(response).float().unsqueeze(1)
        item_id = torch.from_numpy(item_id).long().unsqueeze(1)
        mask = torch.from_numpy(mask).bool().unsqueeze(1)

        return index, response, item_id, mask


class DuoLingo_LanguageAcquisition(torch.utils.data.Dataset):
    """
    2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM).

    -   There will be three (3) tracks for learners of English, Spanish, 
        and French. Teams are encouraged to explore features which generalize 
        across all three languages.
    -   Anonymized user IDs and time data will be provided. This allows teams 
        to explore various personalized, adaptive SLA modeling approaches.
    -   The sequential nature of the data also allows teams to model language 
        learning (and forgetting) over time.

    Data contains a bunch of exercise groups containing metadata:

    -   user: a B64 encoded, 8-digit, anonymized, unique identifier for each 
        student (may include / or + characters)
    -   countries: a pipe (|) delimited list of 2-character country codes from 
        which this user has done exercises
    -   days: the number of days since the student started learning this 
        language on Duolingo
    -   client: the student's device platform (one of: android, ios, or web)
    -   session: the session type (one of: lesson, practice, or test; explanation below)
    -   format: the exercise format (one of: reverse_translate, reverse_tap, 
        or listen; see figures above)
    -   time: the amount of time (in seconds) it took for the student to 
        construct and submit their whole answer (note: for some exercises, this 
        can be null due to data logging issues)
    
    There are three different sessions:

    -   lesson: 77% of dataset
    -   practice: 22% of dataset (only previously seen words)
    -   test: 1% quizzes that allow students to skip units

    We are ONLY going to use the `lesson` series for now.

    The rest of data is organized as follows:

    -   Unique 12-digit ID for each token instance: the first 8 digits are a 
        B64-encoded ID representing the session, the next 2 digits denote 
        the index of this exercise within the session, and the last 2 digits 
        denote the index of the token (word) in this exercise
    -   The token (word)
    -   Part of speech in Universal Dependencies (UD) format
    -   Morphological features in UD format
    -   Dependency edge label in UD format
    -   Dependency edge head in UD format (this corresponds to the last 1-2 
        digits of the ID in the first column)
    -   The label to be predicted (0 or 1)
    """
    def __init__(
            self, 
            train = True, 
            sub_problem = 'en_es', 
            max_num_person = None, 
            max_num_item = None,
            binarize = True,
            **kwargs
        ):
        super().__init__()
        assert sub_problem in ['fr_en', 'en_es', 'es_en']

        cache_score_matrix_file = os.path.join(DUOLINGO_LANG_DIR, f'score_matrix.npy')
        cache_token_id_file = os.path.join(DUOLINGO_LANG_DIR, f'token_id.npy')
        if (os.path.isfile(cache_score_matrix_file) and \
            os.path.isfile(cache_token_id_file)):

            response = np.load(cache_score_matrix_file)
            item_id = np.load(cache_token_id_file)
        else:
            response, item_id = self.make_score_matrix(sub_problem)
            np.save(cache_score_matrix_file, response)
            np.save(cache_token_id_file, item_id)

        if binarize:
            response = np.round(response)

        rs = np.random.RandomState(42)
        swapper = np.arange(response.shape[0])
        rs.shuffle(swapper)
        response = response[swapper] 

        num_person = response.shape[0]
        num_train = int(0.8 * num_person)

        if train:
            response = response[:num_train]
        else:
            response = response[num_train:]

        if max_num_person is not None:
            response = response[:max_num_person]
        
        if max_num_item is not None:
            response = response[:, :max_num_item]
            item_id = item_id[:max_num_item]

        rows_to_remove = np.sum(response, 1) == (-1 * response.shape[1])
        response = response[~rows_to_remove]

        response_mask = np.ones_like(response)
        response_mask[response == -1] = 0

        self.binarize = binarize 
        self.response = response
        self.item_id = item_id
        self.mask = response_mask
        self.length = response.shape[0]
        self.num_person = response.shape[0]
        self.num_item = response.shape[1]

    def make_score_matrix(self, sub_problem):
        filename = os.path.join(
            DUOLINGO_LANG_DIR, f'{sub_problem}.slam.20190204.train')
        instances, labels = load_duolingo(filename)

        words = []
        for i in tqdm(range(len(instances))):
            instance = instances[i]
            if instance.session != 'lesson':
                continue
            word = instance.token
            words.append(word)
        words = list(set(words))

        person_ids, tokens, responses = [], [], []

        for i in tqdm(range(len(instances))):
            instance = instances[i]

            if instance.session != 'lesson':
                continue
            
            user = instance.user
            response = labels[instance.instance_id]
            word = instance.token
            token = words.index(word)

            person_ids.append(user)
            tokens.append(token)
            responses.append(response)

        num_tokens = len(list(set(tokens)))
        unique_ids = list(set(person_ids))
        unique_ids = dict(zip(unique_ids, range(len(unique_ids))))

        person_ids = np.array(person_ids)
        tokens = np.array(tokens)
        responses = np.array(responses)

        num_kept = len(person_ids)

        person_ids_int = []
        for i in tqdm(range(num_kept)):
            person_ids_int.append(unique_ids[person_ids[i]])
        person_ids = np.array(person_ids_int)
        unique_person_ids = np.unique(person_ids)

        num_persons = len(unique_person_ids)
        # -1 => missing data (we might have every student answer every q)
        score_matrix = np.zeros((num_persons, num_tokens))
        count_matrix = np.zeros((num_persons, num_tokens))

        for i in tqdm(range(num_kept)):
            score_matrix[person_ids[i], tokens[i]] += responses[i]
            count_matrix[person_ids[i], tokens[i]] += 1.

        score_matrix[count_matrix == 0] = -1
        count_matrix[count_matrix == 0] = 1  # do not divide by 0
        score_matrix = score_matrix / count_matrix
        token_ids = np.arange(num_tokens)

        return score_matrix, token_ids

    def get_unique_person_ids(self, instances):
        return [instance.user for instance in instances] 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        response = self.response[index]
        item_id = self.item_id.copy()
        # -1 in questions represents missing data
        item_id[response == -1] = -1
        mask = self.mask[index]

        response = torch.from_numpy(response).float().unsqueeze(1)
        item_id = torch.from_numpy(item_id).long().unsqueeze(1)
        mask = torch.from_numpy(mask).bool().unsqueeze(1)

        return index, response, item_id, mask


class WordBank_Language(torch.utils.data.Dataset):
    """
    Downloaded from the following: http://langcog.github.io/wordbankr/
    I used `get_administration_data()`. Looks like the following:

    ## # A tibble: 82,055 x 15
    ##    data_id   age comprehension production language form  birth_order
    ##      <dbl> <int>         <int>      <int> <chr>    <chr> <fct>      
    ##  1   29821    13           293         88 Croatian WG    <NA>       
    ##  2   29822    16           122         12 Croatian WG    <NA>       
    ##  3   29823     9             3          0 Croatian WG    <NA>       
    ##  4   29824    12             0          0 Croatian WG    <NA>       
    ##  5   29825    12            44          0 Croatian WG    <NA>       
    ##  6   29826     8            14          5 Croatian WG    <NA>       
    ##  7   29827     9             2          1 Croatian WG    <NA>       
    ##  8   29828    10            44          1 Croatian WG    <NA>       
    ##  9   29829    13           172         51 Croatian WG    <NA>       
    ## 10   29830    16           241         68 Croatian WG    <NA>       
    ## # … with 82,045 more rows, and 8 more variables: ethnicity <fct>,
    ## #   sex <fct>, zygosity <chr>, norming <lgl>, mom_ed <fct>,
    ## #   longitudinal <lgl>, source_name <chr>, license <chr>

    For now, we will combine all languages together.

    There are no missing data with 5520 persons and 797 items.
    """
    def __init__(self, train=True, max_num_person=None, max_num_item=None, **kwargs):
        super().__init__()

        raw_data = pd.read_csv(os.path.join(WORDBANKR_LANG_DIR, 'wordbankr_english.csv'))

        unique_item = np.unique(np.asarray(raw_data['num_item_id']))
        unique_person = np.unique(np.asarray(raw_data['data_id']))

        response = self.make_score_matrix(raw_data, unique_person, unique_item)

        num_person = response.shape[0]
        num_train  = int(0.8 * num_person)

        if train:
            response = response[:num_train]
        else:
            response = response[num_train:]

        if max_num_person is not None:
            response = response[:max_num_person]
        
        if max_num_item is not None:
            response = response[:, :max_num_item]
            item_id = item_id[:max_num_item]

        response_mask = np.ones_like(response)
        response_mask[response == -1] = 0

        self.response = response
        self.length = response.shape[0]
        self.mask = response_mask
        self.num_person = response.shape[0]
        self.num_item = response.shape[1]

    def make_score_matrix(self, raw_data, unique_person, unique_item):
        cache_file = os.path.join(WORDBANKR_LANG_DIR, 'score_matrix.npy')
        if os.path.isfile(cache_file):
            return np.load(cache_file)
        else:
            num_person = unique_person.shape[0]
            num_item   = unique_item.shape[0]

            # there will be missing data (-1)
            score_matrix = np.ones((num_person, num_item)) * -1

            item = np.asarray(raw_data['num_item_id'])
            response = np.asarray(raw_data['value'])
            response = (response == 'produces').astype(np.int)
            person = np.asarray(raw_data['data_id'])

            pbar = tqdm(total=len(item))
            for item_i, response_i, person_i in zip(item, response, person):
                person_idx = np.where(unique_person == person_i)[0][0]
                item_idx   = np.where(unique_item == item_i)[0][0]
                score_matrix[person_idx, item_idx] = response_i
                pbar.update()
            pbar.close()

            np.save(cache_file, score_matrix)
            return score_matrix

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        response = self.response[index]
        item_id = np.arange(self.response.shape[1])
        # -1 in questions represents missing data
        item_id[response == -1] = -1
        mask = self.mask[index]

        response = torch.from_numpy(response).float().unsqueeze(1)
        item_id = torch.from_numpy(item_id).long().unsqueeze(1)
        mask = torch.from_numpy(mask).bool().unsqueeze(1)

        return index, response, item_id, mask


class PISAScience2015(torch.utils.data.Dataset):
    """
    Processed version of the 2015 PISA OECD Dataset
    https://www.oecd.org/pisa/data/2015database/

    Filtered to only science questions.
    """
    def __init__(self, train=True, max_num_person=None, max_num_item=None, **kwargs):
        super().__init__()

        cache_file = os.path.join(PISA2015_DIR, 'score_matrix.npy')

        if os.path.isfile(cache_file):
            response = np.load(cache_file)
        else:
            # each row is a student 
            data_df = pd.read_csv(os.path.join(PISA2015_DIR, 'cog_science.csv'))
            item_keys = [
                'DS269Q01C', 'DS269Q03C', 'CS269Q04S', 'CS408Q01S', 'DS408Q03C',
                'CS408Q04S', 'CS408Q05S', 'CS521Q02S', 'CS521Q06S', 'DS519Q01C',
                'CS519Q02S', 'DS519Q03C', 'CS527Q01S', 'CS527Q03S', 'CS527Q04S',
                'CS466Q01S', 'CS466Q07S', 'CS466Q05S', 'DS326Q01C', 'DS326Q02C',
                'CS326Q03S', 'CS326Q04S', 'CS256Q01S', 'CS478Q01S', 'CS478Q02S',
                'CS478Q03S', 'CS413Q06S', 'CS413Q04S', 'CS413Q05S', 'CS498Q02S',
                'CS498Q03S', 'DS498Q04C', 'DS425Q03C', 'CS425Q05S', 'CS425Q02S',
                'DS425Q04C', 'DS465Q01C', 'CS465Q02S', 'CS465Q04S', 'DS131Q02C',
                'DS131Q04C', 'CS428Q01S', 'CS428Q03S', 'DS428Q05C', 'DS514Q02C',
                'DS514Q03C', 'DS514Q04C', 'CS438Q01S', 'CS438Q02S', 'DS438Q03C',
                'CS415Q07S', 'CS415Q02S', 'CS415Q08S', 'CS476Q01S', 'CS476Q02S',
                'CS476Q03S', 'CS495Q04S', 'CS495Q01S', 'CS495Q02S', 'DS495Q03C',
                'CS268Q01S', 'DS268Q02C', 'CS268Q06S', 'CS524Q06S', 'DS524Q07C',
                'CS510Q01S', 'DS510Q04C', 'CS437Q01S', 'CS437Q03S', 'CS437Q04S',
                'DS437Q06C', 'DS304Q01C', 'CS304Q02S', 'DS416Q01C', 'DS458Q01C',
                'CS458Q02S', 'CS421Q01S', 'CS421Q02S', 'CS421Q03S', 'CS252Q01S',
                'CS252Q02S', 'CS252Q03S', 'CS327Q01S', 'DS327Q02C', 'CS627Q01S',
                'CS627Q03S', 'CS627Q04S', 'CS635Q01S', 'CS635Q02S', 'DS635Q03C',
                'CS635Q04S', 'DS635Q05C', 'CS603Q01S', 'DS603Q02C', 'CS603Q03S',
                'CS603Q04S', 'CS603Q05S', 'CS602Q01S', 'CS602Q02S', 'DS602Q03C',
                'CS602Q04S', 'CS607Q01S', 'CS607Q02S', 'DS607Q03C', 'CS646Q01S',
                'CS646Q02S', 'CS646Q03S', 'DS646Q04C', 'DS646Q05C', 'CS608Q01S',
                'CS608Q02S', 'CS608Q03S', 'DS608Q04C', 'CS605Q01S', 'CS605Q02S',
                'CS605Q03S', 'DS605Q04C', 'CS649Q01S', 'DS649Q02C', 'CS649Q03S',
                'CS649Q04S', 'CS634Q01S', 'CS634Q02S', 'DS634Q03C', 'DS634Q05C',
                'CS634Q04S', 'CS620Q01S', 'CS620Q02S', 'DS620Q04C', 'CS638Q01S',
                'CS638Q02S', 'CS638Q04S', 'DS638Q05C', 'DS625Q01C', 'CS625Q02S',
                'CS625Q03S', 'CS615Q07S', 'CS615Q01S', 'CS615Q02S', 'CS615Q05S',
                'CS604Q02S', 'DS604Q04C', 'CS645Q01S', 'CS645Q03S', 'DS645Q04C',
                'DS645Q05C', 'CS657Q01S', 'CS657Q02S', 'CS657Q03S', 'DS657Q04C',
                'CS656Q01S', 'DS656Q02C', 'CS656Q04S', 'DS643Q03C', 'CS643Q01S',
                'CS643Q02S', 'CS643Q04S', 'DS643Q05C', 'DS629Q01C', 'CS629Q02S',
                'DS629Q03C', 'CS629Q04S', 'DS648Q01C', 'CS648Q02S', 'CS648Q03S',
                'DS648Q05C', 'CS641Q01S', 'CS641Q02S', 'CS641Q03S', 'CS641Q04S',
                'DS637Q01C', 'CS637Q02S', 'DS637Q05C', 'CS601Q01S', 'CS601Q02S',
                'CS601Q04S', 'DS610Q01C', 'CS610Q02S', 'CS610Q04S', 'CS626Q01S',
                'CS626Q02S', 'CS626Q03S', 'DS626Q04C'
            ]
            data_df = data_df[item_keys]
            data_df = np.asarray(data_df).astype(str)
            response = np.ones(data_df.shape) * -1

            response[data_df == '0 - No credit'] = 0
            response[data_df == '01 - No credit'] = 0
            response[data_df == '02 - No credit'] = 0
            response[data_df == '03 - No credit'] = 0
            response[data_df == '04 - No credit'] = 0
            response[data_df == 'No credit'] = 0
            response[data_df == '1 - Full credit'] = 1
            response[data_df == '1 - Partial credit'] = 1
            response[data_df == '11 - Full credit'] = 1
            response[data_df == '11 - Partial credit'] = 1
            response[data_df == '12 - Full credit'] = 1
            response[data_df == '12 - Partial credit'] = 1
            response[data_df == '2 - Full credit'] = 1
            response[data_df == '21 - Full credit'] = 1
            response[data_df == 'Full credit'] = 1

            np.save(cache_file, response)
      
        rs = np.random.RandomState(42)
        swapper = np.arange(response.shape[0])
        rs.shuffle(swapper)
        response = response[swapper]

        num_person = response.shape[0]
        num_train  = int(0.8 * num_person)

        if train:
            response = response[:num_train]
        else:
            response = response[num_train:]

        if max_num_person is not None:
            response = response[:max_num_person]
        
        if max_num_item is not None:
            response = response[:, :max_num_item]

        rows_to_remove = np.sum(response, 1) == (-1 * response.shape[1])
        response = response[~rows_to_remove]

        response_mask = np.ones_like(response)
        response_mask[response == -1] = 0
        
        self.response = response
        self.mask = response_mask
        self.num_person = response.shape[0]
        self.num_item = response.shape[1]

    def __len__(self):
        return self.num_person

    def __getitem__(self, index):
        response = self.response[index]
        item_id = np.arange(self.num_item)
        # -1 in questions represents missing data
        item_id[response == -1] = -1
        mask = self.mask[index]

        response = torch.from_numpy(response).float().unsqueeze(1)
        item_id = torch.from_numpy(item_id).long().unsqueeze(1)
        mask = torch.from_numpy(mask).bool().unsqueeze(1)

        return index, response, item_id, mask


class IRTSimulation(torch.utils.data.Dataset):
    def __init__(
            self, 
            train = True, 
            irt_model = '3pl', 
            num_person = 1000, 
            num_item = 100, 
            ability_dim = 1, 
            nonlinear = False,
            **kwargs
        ):
        super().__init__()
        if irt_model == '1pl':
            dataset = load_1pl_simulation(num_person, num_item, ability_dim, nonlinear)
            response = dataset['response']
            true_ability = dataset['ability']
            true_item_feat = dataset['item_feat']
        elif irt_model == '2pl':
            dataset = load_2pl_simulation(num_person, num_item, ability_dim, nonlinear)
            response = dataset['response']
            true_ability = dataset['ability']
            true_item_feat = dataset['item_feat']
        elif irt_model == '3pl':
            dataset = load_3pl_simulation(num_person, num_item, ability_dim, nonlinear)
            response = dataset['response']
            true_ability = dataset['ability']
            true_item_feat = dataset['item_feat']
        else:
            raise Exception('irt_model {} not supported'.format(irt_model))
        response = response.numpy()
        true_ability = true_ability.numpy()
        true_item_feat = true_item_feat.numpy()

        num_person = response.shape[0]
        num_question = response.shape[1]
        num_train = int(0.8 * num_person)
        item_id = np.arange(num_question)

        if train:
            response = response[:num_train]
            true_ability = true_ability[:num_train]
            true_item_feat = true_item_feat[:num_train]
        else:
            response = response[num_train:]
            true_ability = true_ability[num_train:]
            true_item_feat = true_item_feat[num_train:]

        response_mask = np.ones_like(response)
        response_mask[response == -1] = 0

        self.response = response
        self.true_ability = true_ability
        self.true_item_feat = true_item_feat
        self.item_id = item_id
        self.mask = response_mask
        self.length = response.shape[0]
        self.num_person = response.shape[0]
        self.num_item = response.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        response = self.response[index]

        item_id = self.item_id.copy()
        # -1 in items represents missing data
        item_id[response.flatten() == -1] = -1
        mask = self.mask[index]
       
        response = torch.from_numpy(response).float()
        item_id = torch.from_numpy(item_id).long()
        mask = torch.from_numpy(mask).bool()

        return index, response, item_id, mask


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def wide_to_long_form_torch(response_wideform, mask_longform):
    response_longform = []

    n_person = response_wideform.size(0)
    n_item = response_wideform.size(1)
    device = response_wideform.device

    for person_id in range(n_person):
        response = response_wideform[person_id].view(-1)
        mask = mask_longform[person_id].view(-1)
        item = torch.arange(n_item, device=device).float()
        item = item[mask]
        response = response[mask]
        num = response.size(0)
        person = torch.ones(num, device=device).float() * person_id
        longform = torch.stack([response, person, item]).T
        response_longform.append(longform)
    
    response_longform = torch.cat(response_longform, dim=0)
    return response_longform


def wide_to_long_form(response_wideform):
    """
    wide form: # people by # items matrix.
               response_wideform will contain -1's for missing data.
    long form: # all rows by 3 (response, person id, item id)
    """
    long_form, mask_form = [], []
    n_item = response_wideform.shape[1]
    n_person = response_wideform.shape[0]
    
    for person_id in range(n_person):
        row = response_wideform[person_id]
        mask = np.ones(n_item)
        item_ids = np.arange(n_item)
        item_ids = item_ids[row != -1]
        mask[row == -1] = 0
        num = len(item_ids)
        row = row[row != -1]
        
        person_data = np.vstack([row, np.ones(num) * person_id, item_ids]).T
        long_form.append(person_data)
        mask_form.append(mask)

    long_form = np.concatenate(long_form, axis=0)
    mask_form = np.concatenate(mask_form, axis=0)
    
    return long_form, mask_form


def long_to_wide_form(response_longform):
    """Collapse long form into a dense matrix with missing data"""
    n_person = np.unique(response_longform[:, 1])
    n_item = np.unique(response_longform[:, 2])

    response_wideform = np.ones((n_person, n_item)) * -1

    for i in range(response_longform.shape[0]):
        response_i = response_longform[i, 0]
        person_i = response_longform[i, 1]
        item_i = response_longform[i, 2]
        response_wideform[person_i, item_i] = response_i

    return response_wideform


if __name__ == "__main__":
    # dataset_name = 'critlangacq'
    dataset_name = 'duolingo'
    # dataset_name = 'wordbank'
    # dataset_name = 'pisa2015_science'
    dset = load_dataset(dataset_name, train=True)
    print(dset.num_person, dset.num_item)
