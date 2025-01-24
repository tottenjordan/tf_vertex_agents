#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Prepare TF.Examples for on-device recommendation model.

Following functions are included: 1) downloading raw data 2) processing to user
activity sequence and splitting to train/test data 3) convert to TF.Examples
and write in output location.

More information about the movielens dataset can be found here:
https://grouplens.org/datasets/movielens/
"""

import collections
import numpy as np
import pickle as pkl
import random
import base64
import json
import glob
import sys
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import app
from absl import flags
from absl import logging
import pandas as pd

import tensorflow as tf

from google.cloud import storage

import warnings
warnings.filterwarnings('ignore')

FLAGS = flags.FLAGS

# genmoe tag dataset
TAG_GENOME_URL = "http://files.grouplens.org/datasets/tag-genome/tag-genome.zip"
TAG_GENOME_ZIP_FILENAME = "tag-genome.zip"
TAG_GENOME_FILENAME = "tag-genome/tag_relevance.dat"

# Permalinks to download movielens data.
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ZIP_FILENAME = "ml-1m.zip"
MOVIELENS_ZIP_HASH = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MOVIELENS_EXTRACTED_DIR = "ml-1m"
RATINGS_FILE_NAME = "ratings.dat"
MOVIES_FILE_NAME = "movies.dat"
USERS_FILE_NAME = "users.dat"
RATINGS_DATA_COLUMNS = ['user_id', 'movie_id', 'rating', 'timestamp']
MOVIES_DATA_COLUMNS = ["movie_id", "title", "genres"]
USERS_DATA_COLUMNS = ['user_id', 'user_gender', 'user_age', 'occupation', 'zip_code']

OUTPUT_MOVIE_VOCAB_FILENAME = "movie_vocab.json"
OUTPUT_MOVIE_YEAR_VOCAB_FILENAME = "movie_year_vocab.txt"
OUTPUT_MOVIE_GENRE_VOCAB_FILENAME = "movie_genre_vocab.txt"
PAD_MOVIE_ID = 0 # "UNK" | 0
PAD_USER_ID = 0  # "UNK" | 0
PAD_RATING = 0.0
PAD_MOVIE_YEAR = 0
UNKNOWN_STR = "UNK"
VOCAB_MOVIE_ID_INDEX = 0
VOCAB_COUNT_INDEX = 3

OCCUPATION_MAP_DICT = {
    0: 'other',
    1: 'academic/educator',
    2: 'artist',
    3: 'clerical/admin',
    4: 'college/grad student',
    5: 'customer service',
    6: 'doctor/health care',
    7: 'executive/managerial',
    8: 'farmer',
    9: 'homemaker',
    10: 'K-12 student',
    11: 'lawyer',
    12: 'programmer',
    13: 'retired',
    14: 'sales/marketing',
    15: 'scientist',
    16: 'self-employed',
    17: 'technician/engineer',
    18: 'tradesman/craftsman',
    19: 'unemployed',
    20: 'writer'
}

# ====================================================
# tag genmoe utils
# ====================================================
def get_genome_dataset(url):
    """
    Add in tag relevance to movies dataframe for a single tag.
    """
    data = urllib.request.urlopen(url).read()
    downloaded_zip = zipfile.ZipFile(io.BytesIO(data))
    logging.info('Downloaded zip file containing: %s', downloaded_zip.namelist())
    
    tags_df = pd.read_csv(
        downloaded_zip.open('tag-genome/tags.dat', 'r'),
          sep='\t',
          names=['tag_id', 'tag', 'tag_popularity'],
          encoding='iso-8859-1'
    )
    
    tag_relevance_df = pd.read_csv(
        downloaded_zip.open('tag-genome/tag_relevance.dat', 'r'),
        sep='\t',
        names=['movie_id', 'tag_id', 'relevance'],
        encoding='iso-8859-1'
    )
    
    print(f"tags_df shape          : {tags_df.shape}")
    print(f"tag_relevance_df shape : {tag_relevance_df.shape}")
    
    return tags_df, tag_relevance_df

def merge_with_genome_data(url, target_tag, dataframes):
    """
    Add in tag relevance to movies dataframe for a single tag.
    """
    data = urllib.request.urlopen(url).read()
    downloaded_zip = zipfile.ZipFile(io.BytesIO(data))
    logging.info('Downloaded zip file containing: %s', downloaded_zip.namelist())
    tags_df = pd.read_csv(
        downloaded_zip.open('tag-genome/tags.dat', 'r'),
        sep='\t',
        names=['tag_id', 'tag', 'tag_popularity'],
        encoding='iso-8859-1'
    )

    target_tag_id = tags_df[tags_df.tag == target_tag].tag_id.values[0]
    logging.info('%s corresponds to tag %d', target_tag, target_tag_id)

    tag_relevance_df = pd.read_csv(
      downloaded_zip.open('tag-genome/tag_relevance.dat', 'r'),
      sep='\t',
      names=['movie_id', 'tag_id', 'relevance'],
      encoding='iso-8859-1')

    # Filter for rows that contain the target tag id.
    tag_relevance_df = tag_relevance_df[tag_relevance_df.tag_id == target_tag_id]

    # Merge tag relevance values on to the movies dataframe.
    movies_df, users_df, ratings_df = dataframes
    movies_df = movies_df.merge(
      tag_relevance_df, on='movie_id', how='left').fillna(0)
    movies_df.rename(
      columns={'relevance': '%s_tag_relevance' % target_tag}, inplace=True)

    logging.info('Movies df has keys %s', list(movies_df.keys()))
    logging.info('Movies df now looks like this %s', movies_df.head().to_string())
    return movies_df, users_df, ratings_df

# ====================================================
# movielens utils
# ====================================================
# def define_flags():
#     """
#     Define flags.
#     """
flags.DEFINE_string("project_id", 
                    "hybrid-vertex",
                    "your Google Cloud project ID")

flags.DEFINE_string("gcs_bucket_name", 
                    "rec-bandits-v2-hybrid-vertex-bucket",
                    "GCS bucket name only; no gs:// prefix")

flags.DEFINE_string("gcs_data_path_prefix", 
                    "data/movielens/movielens-1m-gen",
                    "subfolders of bucket")

flags.DEFINE_string("tfrecord_prefix", "ml-1m-gen",
                    "string to prefix all tfrecords")

flags.DEFINE_integer("num_train_tfrecords", 8,
                    "num records to save tf-examples to")

flags.DEFINE_integer("num_test_tfrecords", 2,
                    "num records to save tf-examples to")

flags.DEFINE_string("local_data_dir", "/tmp",
                    "Path to download and store movielens data.")

flags.DEFINE_string("local_output_dir", None,
                    "Path to the directory of output files.")

flags.DEFINE_integer("min_timeline_length", 3,
                     "The minimum timeline length to construct examples.")

flags.DEFINE_integer("max_context_length", 10,
                     "The maximum length of user context history.")

flags.DEFINE_integer("max_context_movie_genre_length", 10,
                     "The maximum length of user genre context history.")

flags.DEFINE_integer("min_rating", None,
                     "Min rating of movie to be used in training data")

flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")

flags.DEFINE_bool("build_vocabs", True,
                  "If yes, generate movie feature vocabs.")

class MovieInfo(
    collections.namedtuple(
        "MovieInfo", ["movie_id", "timestamp", "rating", "title", "genres", "year"]
    )
):
    """Data holder of basic information of a movie."""
    __slots__ = ()

    def __new__(
        cls,
        movie_id=PAD_MOVIE_ID,
        timestamp=0,
        rating=PAD_RATING,
        year=0,
        title="",
        genres=""
    ):
        return super(MovieInfo, cls).__new__(
            cls, movie_id, timestamp, rating, title, genres, year
        )

class UserInfo(
    collections.namedtuple(
        "UserInfo", ["user_id", "user_age", "user_occupation_text", "user_zip_code", "user_gender"]
    )
):
    """Data holder of basic information of a user."""
    __slots__ = ()

    def __new__(
        cls,
        user_id=PAD_USER_ID,
        user_age=0,
        user_occupation_text="",
        user_zip_code="",
        user_gender="",
    ):
        return super(UserInfo, cls).__new__(
            cls, user_id, user_age, user_occupation_text, user_zip_code, user_gender
    )

# ====================================================
# helper functions
# ====================================================
def _no_gaps(sequence):
    """
    Returns True if a sequence has all values between 0..N with no gaps.
    """
    return set(sequence) == set(range(len(sequence)))


def reindex(dataframes):
    """
    Returns dataframes that have been reindexed to remove gaps.
    """
    print(f"Reindexing dataframes...")
    movies, users, ratings = dataframes
    index_dict = pd.Series(
        np.arange(movies.shape[0]), 
        index=movies['movie_id'].values
    ).to_dict()
    movies['movie_id'] = np.arange(movies.shape[0])
    ratings['movie_id'] = [index_dict[iden] for iden in ratings['movie_id']]
    ratings['user_id'] -= 1
    users['user_id'] -= 1
    assert _no_gaps(movies['movie_id'])
    assert _no_gaps(users['user_id'])

    return movies, users, ratings

def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(
                local_file, 
                bucket, 
                gcs_path + "/" + os.path.basename(local_file)
            )
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

def write_csv_output(dataframes, local_dir=None, gcs_path=""):
    """
    Write csv file outputs.
    """
    directory = None
    movies, users, ratings = dataframes
    
    if local_dir is not None:
        if tf.io.gfile.exists(local_dir):
            print(f"{local_dir} exists")
            return local_dir
        if not tf.io.gfile.exists(local_dir):
            print(f"{local_dir} does not exist")
            tf.io.gfile.makedirs(local_dir)
            print("Directory '%s' created" % local_dir)
        directory = local_dir
        print(f"writing CSVs to local dir: {directory}")
    if gcs_path:
        directory = f"{gcs_path}/csv_files"
        print(f"writing CSVs to Cloud Storage: {directory}")
        
    # write dataframes to CSV; save CSVs to directory
    users.to_csv(
        f'{directory}/users.csv',
        index=False,
        # columns=['user_id']
    )
    movies.to_csv(
        f'{directory}/movies.csv',
        index=False
    )
    ratings.to_csv(
        f'{directory}/ratings.csv',
        index=False
    )
    print(f"writing CSVs complete!")


def download_and_extract_data(
    data_directory,
    url=MOVIELENS_1M_URL,
    fname=MOVIELENS_ZIP_FILENAME,
    file_hash=MOVIELENS_ZIP_HASH,
    extracted_dir_name=MOVIELENS_EXTRACTED_DIR
):
    """
    Download and extract zip containing MovieLens data to a given directory.

    Args:
      data_directory: Local path to extract dataset to.
      url: Direct path to MovieLens dataset .zip file. See constants above for
          examples.
      fname: str, zip file name to download.
      file_hash: str, SHA-256 file hash.
      extracted_dir_name: str, extracted dir name under data_directory.

    Returns:
      Downloaded and extracted data file directory.
    """
    if not tf.io.gfile.exists(data_directory):
        tf.io.gfile.makedirs(data_directory)
    path_to_zip = tf.keras.utils.get_file(
        fname=fname,
        origin=url,
        file_hash=file_hash,
        hash_algorithm="sha256",
        extract=True,
        cache_dir=data_directory
    )
    extracted_file_dir = os.path.join(
        os.path.dirname(path_to_zip), extracted_dir_name
    )
    return extracted_file_dir


def read_data(data_directory, min_rating=None, local_output_dir=None):
    """
    Read movielens ratings.dat and movies.dat file into dataframe.
    """
    ratings_df = pd.read_csv(
        os.path.join(data_directory, RATINGS_FILE_NAME),
        sep="::",
        names=RATINGS_DATA_COLUMNS,
        encoding="unicode_escape"
    )  # May contain unicode. Need to escape.
    ratings_df["timestamp"] = ratings_df["timestamp"].apply(int)
    # ratings_df["user_id"] = ratings_df["user_id"].apply(str)
    # ratings_df["movie_id"] = ratings_df["movie_id"].apply(str)
    
    if min_rating is not None:
        ratings_df = ratings_df[ratings_df["rating"] >= min_rating]
    
    movies_df = pd.read_csv(
        os.path.join(data_directory, MOVIES_FILE_NAME),
        sep="::",
        names=MOVIES_DATA_COLUMNS,
        encoding="unicode_escape"
    )  # May contain unicode. Need to escape.
    # movies_df["movie_id"] = movies_df["movie_id"].apply(str)
    movies_df['year'] = movies_df['title'].apply(
        lambda movie_name: re.search('\((\d*)\)', movie_name).groups(1)[0]
    )
    movies_df["year"] = movies_df["year"].apply(int)
    users_df = pd.read_csv(
        os.path.join(data_directory, USERS_FILE_NAME),
        sep='::',
        names=USERS_DATA_COLUMNS,
        encoding='unicode_escape'
    )
    users_df['occupation'] = users_df['occupation'].replace(OCCUPATION_MAP_DICT)
    # users_df["user_id"] = users_df["user_id"].apply(str)
    
    # write raw dfs to csv; store in GCS
    dataframes = movies_df, users_df, ratings_df
    write_csv_output(dataframes, local_dir=local_output_dir, gcs_path="")
    movies_df, users_df, ratings_df = reindex(dataframes)
    
    return ratings_df, movies_df, users_df


def convert_to_timelines(ratings_df):
    """
    Convert ratings data to user.
    """

    # create timelines index
    timelines = collections.defaultdict(list)
    movie_counts = collections.Counter()
    
    # get ratings
    for user_id, movie_id, rating, timestamp in ratings_df.values:
        timelines[user_id].append(
            MovieInfo(
                movie_id=movie_id,
                timestamp=int(timestamp),
                rating=rating
            ),
        )
        movie_counts[movie_id] += 1
    
    # Sort per-user timeline by timestamp
    for (user_id, context) in timelines.items():
        context.sort(key=lambda x: x.timestamp)
        timelines[user_id] = context
    return timelines, movie_counts


def generate_movies_dict(movies_df):
    """Generates movies dictionary from movies dataframe."""
    movies_dict = {
        movie_id: MovieInfo(
            movie_id=movie_id, 
            title=title, 
            genres=genres,
            year=year
        )
        for movie_id, title, genres, year in movies_df.values
    }     
    movies_dict[0] = MovieInfo()
    return movies_dict


def generate_users_dict(users_df):
    """Generates users dictionary from users dataframe."""
    users_dict = {
        user_id: UserInfo(
            user_id=user_id, 
            user_gender=user_gender, 
            user_age=user_age,
            user_occupation_text=user_occupation_text,
            user_zip_code=user_zip_code
        )
        for user_id, user_gender, user_age, user_occupation_text, user_zip_code in users_df.values
    }
    users_dict[0] = UserInfo()
    return users_dict


def extract_year_from_title(title):
    year = re.search(r"\((\d{4})\)", title)
    if year:
        return int(year.group(1))
    return 0


def generate_feature_of_movie_years(movies_dict, movies):
    """Extracts year feature for movies from movie title."""
    return [
        extract_year_from_title(movies_dict[movie.movie_id].title)
        for movie in movies
    ]


def generate_movie_genres(movies_dict, movies):
    """Create a feature of the genre of each movie.

    Save genre as a feature for the movies.

    Args:
      movies_dict: Dict of movies, keyed by movie_id with value of (title, genre)
      movies: list of movies to extract genres.

    Returns:
      movie_genres: list of genres of all input movies.
    """
    movie_genres = []
    for movie in movies:
        if not movies_dict[movie.movie_id].genres:
            continue
        genres = [
            tf.compat.as_bytes(genre)
            for genre in movies_dict[movie.movie_id].genres.split("|")
        ]
        movie_genres.extend(genres)

    return movie_genres

def get_single_movies_genres(genre_str):
    list_o_genres = [
        tf.compat.as_bytes(genre)
        for genre in genre_str.split("|")
    ]
    return list_o_genres

def generate_movie_titles(movies_dict, movies):
    """
    Create a feature for the title of each movie.
    Save title as a feature for the movies.

    Args:
      movies_dict: Dict of movies, keyed by movie_id with value of (title, genre)
      movies: list of movies to extract titles.

    Returns:
      movie_titles: list of titles of all input movies.
    """
    movie_titles = []
    for movie in movies:
        if not movies_dict[movie.movie_id].title:
            continue
        movie_titles.append(
            tf.compat.as_bytes(
                movies_dict[movie.movie_id].title
            )
        )
    return movie_titles


def _pad_or_truncate_movie_feature(feature, max_len, pad_value):
    feature.extend(
        [
            pad_value for _ in range(max_len - len(feature))
        ]
    )
    return feature[:max_len]


def generate_examples_from_single_timeline(
    timeline,
    movies_dict,
    users_dict,
    max_context_len=100,
    max_context_movie_genre_len=320
):
    """
    Generate TF examples from a single user timeline.

    Generate TF examples from a single user timeline. Timeline with length less
    than minimum timeline length will be skipped. And if context user history
    length is shorter than max_context_len, features will be padded with default
    values.

    Args:
      timeline: The timeline to generate TF examples from.
      movies_dict: Dictionary of all MovieInfos.
      users_dict: TODO
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.
      max_context_movie_genre_len: The length of movie genre feature.

    Returns:
      examples: Generated examples from this single timeline.
    """

    # gather movie features for context sequences
    examples = []
    for label_idx in range(1, len(timeline)):
        start_idx = max(0, label_idx - max_context_len)
        context = timeline[start_idx:label_idx]

        # Pad context with out-of-vocab movie id 0.
        while len(context) < max_context_len:
            context.append(MovieInfo())

        # get target/label item features (non-sequence)
        label_movie_id = timeline[label_idx].movie_id
        label_movie_rating = timeline[label_idx].rating
        label_movie_timestamp = timeline[label_idx].timestamp
        
        # use this to get features in movies_dict
        target_movie_dict = movies_dict[label_movie_id]

        label_movie_title = tf.compat.as_bytes(target_movie_dict.title)
        # label_movie_year = int(target_movie_dict[5])
        label_movie_year = int(target_movie_dict.year)

        label_movie_genres = get_single_movies_genres(target_movie_dict.genres)
        label_movie_genres = _pad_or_truncate_movie_feature(
            label_movie_genres, 
            max_context_movie_genre_len,
            tf.compat.as_bytes(UNKNOWN_STR)
        )

        # get context sequence item features
        context_movie_id = [tf.compat.as_bytes(str(movie.movie_id)) for movie in context]
        context_movie_rating = [movie.rating for movie in context]
        context_timestamp = [movie.timestamp for movie in context]
        context_movie_title = generate_movie_titles(movies_dict, context)
        context_movie_title = _pad_or_truncate_movie_feature(
            context_movie_title, 
            max_context_len,
            tf.compat.as_bytes(UNKNOWN_STR)
        )
        context_movie_year = generate_feature_of_movie_years(movies_dict, context)
        context_movie_genres = generate_movie_genres(movies_dict, context)
        context_movie_genres = _pad_or_truncate_movie_feature(
            context_movie_genres, 
            max_context_movie_genre_len,
            tf.compat.as_bytes(UNKNOWN_STR)
        )

        # create tf.example features
        feature = {
            # context sequence item features
            "context_movie_id":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=context_movie_id)),
            "context_movie_rating":
                tf.train.Feature(
                    float_list=tf.train.FloatList(value=context_movie_rating)),
            "context_rating_timestamp":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=context_timestamp)),
            "context_movie_genre":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=context_movie_genres)),
            "context_movie_year":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=context_movie_year)),
            "context_movie_title":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=context_movie_title)),
            
            # target/label item features
            "target_movie_id":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(label_movie_id))])),
            "target_movie_rating":
                tf.train.Feature(
                    float_list=tf.train.FloatList(value=[label_movie_rating])),
            "target_rating_timestamp":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label_movie_timestamp])),
            "target_movie_genres":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=label_movie_genres)),
            "target_movie_year":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label_movie_year])),
            "target_movie_title":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[label_movie_title])),
            
            # global context user features
            "user_id":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(users_dict.user_id))])),
            "user_gender":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(users_dict.user_gender)])),
            "user_age":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(users_dict.user_age)])),
            "user_occupation_text":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(users_dict.user_occupation_text)])),
            "user_zip_code":
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(users_dict.user_zip_code)])),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)

    return examples


def generate_examples_from_timelines(
    timelines,
    movies_df,
    users_df,
    min_timeline_len=3,
    max_context_len=100,
    max_context_movie_genre_len=320,
    train_data_fraction=0.9,
    random_seed=None,
    shuffle=True
):
    """
    Convert user timelines to tf examples.

    Convert user timelines to tf examples by adding all possible context-label
    pairs in the examples pool.

    Args:
      timelines: The user timelines to process.
      movies_df: The dataframe of all movies.
      users_df: TODO
      min_timeline_len: The minimum length of timeline. If the timeline length is
        less than min_timeline_len, empty examples list will be returned.
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.
      max_context_movie_genre_len: The length of movie genre feature.
      train_data_fraction: Fraction of training data.
      random_seed: Seed for randomization.
      shuffle: Whether to shuffle the examples before splitting train and test
        data.

    Returns:
      train_examples: TF example list for training.
      test_examples: TF example list for testing.
    """
    examples = []
    users_dict = generate_users_dict(users_df)
    movies_dict = generate_movies_dict(movies_df)
    progress_bar = tf.keras.utils.Progbar(len(timelines))
    # for timeline in timelines.values():
    for (user_id, timeline) in timelines.items():
        user = users_dict[user_id]
        if len(timeline) < min_timeline_len:
            progress_bar.add(1)
            continue
        single_timeline_examples = generate_examples_from_single_timeline(
            timeline=timeline,
            movies_dict=movies_dict,
            users_dict=user,
            max_context_len=max_context_len,
            max_context_movie_genre_len=max_context_movie_genre_len
        )
        examples.extend(single_timeline_examples)
        progress_bar.add(1)
    # Split the examples into train, test sets.
    if shuffle:
        random.seed(random_seed)
        random.shuffle(examples)
    last_train_index = round(len(examples) * train_data_fraction)

    train_examples = examples[:last_train_index]
    test_examples = examples[last_train_index:]
    return train_examples, test_examples


def generate_movie_feature_vocabs(movies_df, movie_counts):
    """
    Generate vocabularies for movie features.

    Generate vocabularies for movie features (movie_id, genre, year), sorted by
    usage count. Vocab id 0 will be reserved for default padding value.

    Args:
      movies_df: Dataframe for movies.
      movie_counts: Counts that each movie is rated.

    Returns:
      movie_id_vocab: List of all movie ids paired with movie usage count, and
        sorted by counts.
      movie_genre_vocab: List of all movie genres, sorted by genre usage counts.
      movie_year_vocab: List of all movie years, sorted by year usage counts.
    """
    movie_vocab = []
    movie_id_counter = collections.Counter()
    movie_genre_counter = collections.Counter()
    movie_year_counter = collections.Counter()
    movie_title_counter = collections.Counter()
    for movie_id, title, genres, year in movies_df.values:
        count = movie_counts.get(movie_id) or 0
        movie_vocab.append([movie_id, title, genres, year, count])
        movie_id_counter[movie_id] += 1
        movie_year_counter[year] += 1
        movie_title_counter[title] += 1
        for genre in genres.split("|"):
            movie_genre_counter[genre] += 1

    movie_vocab.sort(key=lambda x: x[VOCAB_COUNT_INDEX], reverse=True)  # by count
    movie_id_vocab = [UNKNOWN_STR] + [
        x for x, _ in movie_id_counter.most_common()
    ]
    movie_year_vocab = [0] + [
        x for x, _ in movie_year_counter.most_common()
    ]
    movie_genre_vocab = [UNKNOWN_STR] + [
        x for x, _ in movie_genre_counter.most_common()
    ]
    movie_title_vocab = [UNKNOWN_STR] + [
        x for x, _ in movie_title_counter.most_common()
    ]
    return (
        movie_id_vocab, 
        movie_year_vocab, 
        movie_genre_vocab, 
        movie_title_vocab
    )

def generate_user_feature_vocabs(user_df):
    """
    Generate vocabs for user_df
    """
    user_vocab = []
    user_id_counter = collections.Counter()
    user_gender_counter = collections.Counter()
    user_age_counter = collections.Counter()
    user_occ_counter = collections.Counter()
    user_zipcode_counter = collections.Counter()
    for user_id, user_sex, user_age, user_occ, user_zip in user_df.values:
        user_vocab.append([user_id, user_sex, user_age, user_occ, user_zip])
        user_id_counter[user_id] += 1
        user_gender_counter[user_sex] += 1
        user_age_counter[user_age] += 1
        user_occ_counter[user_occ] += 1
        user_zipcode_counter[user_zip] += 1

    user_vocab.sort(key=lambda x: x[VOCAB_COUNT_INDEX], reverse=True)  # by count
    user_id_vocab = [UNKNOWN_STR] + [
        x for x, _ in user_id_counter.most_common()
    ]
    user_gender_vocab = [UNKNOWN_STR] + [
        x for x, _ in user_gender_counter.most_common()
    ]
    user_age_vocab = [0] + [
        int(x) for x, _ in user_age_counter.most_common()
    ]
    user_occ_vocab = [UNKNOWN_STR] + [
        x for x, _ in user_occ_counter.most_common()
    ]
    user_zip_vocab = [UNKNOWN_STR] + [
        x for x, _ in user_zipcode_counter.most_common()
    ]
    return (
        user_id_vocab, 
        user_gender_vocab, 
        user_age_vocab, 
        user_occ_vocab, 
        user_zip_vocab
    )

def generate_rating_feature_vocabs(ratings_df):
    """
    Generate vocabs for ratings_df
    """
    rating_vocab = []
    rating_timestamp_counter = collections.Counter()
    for user_id, movie_id, rating, timestamp in ratings_df.values:
        rating_vocab.append([user_id, movie_id, rating, timestamp])
        # year = extract_year_from_title(title)
        rating_timestamp_counter[timestamp] += 1

    rating_vocab.sort(key=lambda x: x[VOCAB_COUNT_INDEX], reverse=True)  # by count
    timestamp_vocab = [0] + [
        x for x, _ in rating_timestamp_counter.most_common()
    ]
    min_ts = min(timestamp_vocab)
    max_ts = max(timestamp_vocab)
    timestamp_buckets = np.linspace(
        min_ts, max_ts, num=1000
    )
    return (
        min_ts, 
        max_ts, 
        timestamp_buckets,
    )


def write_tfrecords(tf_examples, filename):
    """Writes tf examples to tfrecord file, and returns the count."""
    with tf.io.TFRecordWriter(filename) as file_writer:
        length = len(tf_examples)
        progress_bar = tf.keras.utils.Progbar(length)
        for example in tf_examples:
            file_writer.write(example.SerializeToString())
            progress_bar.add(1)
        return length

def create_records(tf_examples, output_dir, num_of_records=5, prefix="movielens-1m"):
    """
    Takes list of tf_examples + number of records, and creates TFRecords.
    Saves records in output_dir
    """
    tf_examples_len = len(tf_examples)
    files_per_record = int(tf_examples_len // num_of_records)
    print(f"prefix           : {prefix}")
    print(f"tf_examples_len  : {tf_examples_len}")
    print(f"num_of_records   : {num_of_records}")
    print(f"files_per_record : {files_per_record}\n")
    
    file_list = []
    total_record_size = 0
    chunk_number = 1
    for i in range(0, tf_examples_len, files_per_record):
        print(f"Writing chunk: {str(chunk_number)}")
        print(f"chunk range  : [{i},{i+files_per_record}]")
        this_chunk = tf_examples[i:i+files_per_record]

        if num_of_records == 1:
            record_file = f"{prefix}.tfrecord"
        else:
            record_file = f"{prefix}-{str(chunk_number).zfill(3)}-of-{str(num_of_records).zfill(3)}.tfrecord"
        
        record_path = os.path.join(output_dir, record_file)
        LOCAL_TF_RECORD_FILE = f"./{record_path}"
        print(f"writing examples to: {LOCAL_TF_RECORD_FILE}")

        record_size = write_tfrecords(tf_examples=this_chunk, filename=LOCAL_TF_RECORD_FILE) # record_file
        total_record_size += record_size
        file_list.append(record_file)
        chunk_number += 1
        
    print(f"Complete: wrote {total_record_size} tf_examples to {len(file_list)} tfrecords")
    return total_record_size, file_list


def write_vocab_json(vocab, filename):
    """Write generated movie vocabulary to specified file."""
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(vocab, jsonfile, indent=2)


def write_vocab_txt(vocab, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in vocab:
            f.write(str(item) + "\n")


def _get_temp_dir(dirpath, subdir):
    temp_dir = os.path.join(dirpath, subdir)
    if not tf.io.gfile.exists(temp_dir):
        tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def generate_datasets(
    project_id,
    num_train_tfrecords,
    num_test_tfrecords,
    gcs_bucket_name,
    gcs_data_path_prefix,
    # local dirs
    extracted_data_dir,
    output_dir,
    # data configs
    min_timeline_length,
    max_context_length,
    max_context_movie_genre_length,
    min_rating=1,
    build_vocabs=True,
    train_data_fraction=0.8,
    # output files
    tfrecord_filename_prefix="movielens-1m",
    vocab_filename=OUTPUT_MOVIE_VOCAB_FILENAME,
    vocab_year_filename=OUTPUT_MOVIE_YEAR_VOCAB_FILENAME,
    vocab_genre_filename=OUTPUT_MOVIE_GENRE_VOCAB_FILENAME,
):
    """
    Generates train and test datasets as TFRecord, and returns stats.
    """

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    LOCAL_CSV_PATH = _get_temp_dir(output_dir, "csvs")
    LOCAL_TRAIN_PATH = _get_temp_dir(output_dir, "train")
    LOCAL_VAL_PATH = _get_temp_dir(output_dir, "val")
    LOCAL_VOCAB_PATH = _get_temp_dir(output_dir, "vocabs")

    logging.info("Reading data to dataframes..\n")
    ratings_df, movies_df, users_df = read_data(
        extracted_data_dir, 
        min_rating=min_rating, 
        local_output_dir=LOCAL_CSV_PATH
    )

    logging.info("Generating movie rating user timelines..\n")
    timelines, movie_counts = convert_to_timelines(ratings_df)

    logging.info("Generating train and test examples..\n")
    train_examples, test_examples = generate_examples_from_timelines(
        timelines=timelines,
        movies_df=movies_df,
        users_df=users_df,
        min_timeline_len=min_timeline_length,
        max_context_len=max_context_length,
        max_context_movie_genre_len=max_context_movie_genre_length,
        train_data_fraction=train_data_fraction
    )

    logging.info("Writing generated training examples..\n")
    train_size, train_files = create_records(
        tf_examples=train_examples, 
        output_dir=LOCAL_TRAIN_PATH, 
        num_of_records=num_train_tfrecords, 
        prefix=tfrecord_filename_prefix
    )
    logging.info("Writing generated testing examples..\n")
    test_size, test_files = create_records(
        tf_examples=test_examples, 
        output_dir=LOCAL_VAL_PATH,
        num_of_records=num_test_tfrecords, 
        prefix=tfrecord_filename_prefix
    )
    stats = {
        "train_size": train_size,
        "test_size": test_size,
        "train_files": train_files,
        "test_files": test_files,
    }

    # save files to google cloud storage
    print(f"Saving TF Records to Cloud Storage..\n")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)

    # train tfrecords
    upload_local_directory_to_gcs(
        local_path=LOCAL_TRAIN_PATH, 
        bucket=bucket, 
        gcs_path=f"{gcs_data_path_prefix}/train"
    )
    print(f"saved train tfrecords to gs://{gcs_bucket_name}/{gcs_data_path_prefix}/train")

    # val tfrecords
    upload_local_directory_to_gcs(
        local_path=LOCAL_VAL_PATH, 
        bucket=bucket, 
        gcs_path=f"{gcs_data_path_prefix}/val"
    )
    print(f"saved val tfrecords to gs://{gcs_bucket_name}/{gcs_data_path_prefix}/val")

    # create vocabs & save to Google Cloud Storage
    if build_vocabs:
        print("building vocabs..")
        (movie_vocab, movie_year_vocab, movie_genre_vocab, movie_title_vocab) = (
            generate_movie_feature_vocabs(
                movies_df=movies_df, 
                movie_counts=movie_counts
            )
        )
        (user_vocab, user_gender_vocab, user_age_vocab, user_occ_vocab, user_zip_vocab) = (
            generate_user_feature_vocabs(
                user_df=users_df,
            )
        )
        (min_ts, max_ts, timestamp_buckets) = (
            generate_rating_feature_vocabs(
                ratings_df=ratings_df,
            )
        )
        vocab_file = os.path.join(LOCAL_VOCAB_PATH, vocab_filename)
        # write_vocab_json(movie_vocab, filename=vocab_file)
        # stats.update({
        #     "vocab_size": len(movie_vocab),
        #     "vocab_file": vocab_file,
        #     "vocab_max_id": max([arr[VOCAB_MOVIE_ID_INDEX] for arr in movie_vocab])
        # })
        # for vocab, filename, key in zip(
        #     [movie_year_vocab, movie_genre_vocab],
        #     [vocab_year_filename, vocab_genre_filename],
        #     ["year_vocab", "genre_vocab"]
        # ):
        #     vocab_file = os.path.join(LOCAL_VOCAB_PATH, filename)
        #     write_vocab_txt(vocab, filename=vocab_file)
        #     stats.update({
        #         key + "_size": len(vocab),
        #         key + "_file": vocab_file,
        #     })
        
        VOCAB_PKL_NAME = 'vocab_dict.pkl'
        LOCAL_VOCAB_PKL = os.path.join(LOCAL_VOCAB_PATH, VOCAB_PKL_NAME)
        vocab_dict = {
            'movie_id': [str(x).encode('utf-8') for x in movie_vocab],
            'movie_year': movie_year_vocab,
            'movie_genre': [str(x).encode('utf-8') for x in movie_genre_vocab],
            'movie_title': [str(x).encode('utf-8') for x in movie_title_vocab],
            'user_id': [str(x).encode('utf-8') for x in user_vocab],
            'user_gender_vocab': [str(x).encode('utf-8') for x in user_gender_vocab],
            'user_age_vocab': user_age_vocab,
            'user_occ_vocab': [str(x).encode('utf-8') for x in user_occ_vocab],
            'user_zip_vocab': [str(x).encode('utf-8') for x in user_zip_vocab],
            'min_timestamp': min_ts,
            "max_timestamp": max_ts,
            "timestamp_buckets": timestamp_buckets,
        }
        filehandler = open(LOCAL_VOCAB_PKL, 'wb')
        pkl.dump(vocab_dict, filehandler)
        filehandler.close()

        # save vocabs to gcs
        print(f"Saving Vocab files to Cloud Storage..")
        upload_local_directory_to_gcs(
            local_path=LOCAL_VOCAB_PATH, 
            bucket=bucket, 
            gcs_path=f"{gcs_data_path_prefix}/vocabs"
        )
        print(f"saved vocab files to gs://{gcs_bucket_name}/{gcs_data_path_prefix}/vocabs\n")

    return stats


# def main(_):
def main(argv):
    
    print(f"\n{sys.argv[1:]}\n")

    logging.info("Downloading and extracting data.") # TODO: if FLAGS.skip_download:
    extracted_data_dir = download_and_extract_data(data_directory=FLAGS.local_data_dir)

    stats = generate_datasets(
        extracted_data_dir=extracted_data_dir,
        project_id=FLAGS.project_id,
        gcs_bucket_name=FLAGS.gcs_bucket_name,
        gcs_data_path_prefix=FLAGS.gcs_data_path_prefix,
        num_train_tfrecords=FLAGS.num_train_tfrecords,
        num_test_tfrecords=FLAGS.num_test_tfrecords,
        output_dir=FLAGS.local_output_dir,
        min_timeline_length=FLAGS.min_timeline_length,
        max_context_length=FLAGS.max_context_length,
        max_context_movie_genre_length=FLAGS.max_context_movie_genre_length,
        min_rating=FLAGS.min_rating,
        build_vocabs=FLAGS.build_vocabs,
        train_data_fraction=FLAGS.train_data_fraction,
        tfrecord_filename_prefix=FLAGS.tfrecord_prefix
    )
    logging.info("Generated dataset!")

    logging.info("train_size: %s", stats['train_size'])
    logging.info("test_size: %s", stats['test_size'])

    # if FLAGS.build_vocabs:
    #     logging.info("vocab_size: %s", stats['vocab_size'])
    #     logging.info("vocab_max_id: %s", stats['vocab_max_id'])
    #     logging.info("year_vocab_size: %s", stats['year_vocab_size'])
    #     logging.info("genre_vocab_size: %s", stats['genre_vocab_size'])

if __name__ == "__main__":
    # define_flags()
    app.run(main)