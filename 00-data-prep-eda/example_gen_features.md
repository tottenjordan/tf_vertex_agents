### features

**Context features**

If user preferences are relatively stable across contexts and time, context features may not provide much benefit. If, however, users preferences are highly contextual, adding context will improve the model significantly. 
> For example, day of the week may be an important feature when deciding whether to recommend a short clip or a movie: users may only have time to watch short content during the week, but can relax and enjoy a full-length movie during the weekend. 
> Similarly, query timestamps may play an important role in modelling popularity dynamics: one movie may be highly popular around the time of its release, but decay quickly afterwards. Conversely, other movies may be evergreens that are happily watched time and time again.

**Data sparsity** 

Using non-id features may be critical if data is sparse. With few observations available for a given user or item, the model may struggle with estimating a good per-user or per-item representation. To build an accurate model, other features such as item categories, descriptions, and images have to be used to help the model generalize beyond the training data. This is especially relevant in cold-start situations, where relatively little data is available on some items or users.

### our tf examples

> a single tf-example in our generated dataset should look similar to the following:

```
features {
  feature {
    key: "context_movie_genre"
    value {
      bytes_list {
        value: "Comedy"
        value: "Drama"
        value: "Drama"
        value: "War"
        value: "Drama"
        value: "Drama"
        value: "Thriller"
        value: "Drama"
        value: "Romance"
        value: "Children\'s"
      }
    }
  }
  feature {
    key: "context_movie_id"
    value {
      bytes_list {
        value: "2858"
        value: "527"
        value: "515"
        value: "593"
        value: "265"
        value: "34"
        value: "1704"
        value: "3418"
        value: "1179"
        value: "150"
      }
    }
  }
  feature {
    key: "context_movie_rating"
    value {
      float_list {
        value: 4.0
        value: 5.0
        value: 4.0
        value: 3.0
        value: 5.0
        value: 5.0
        value: 4.0
        value: 4.0
        value: 4.0
        value: 4.0
      }
    }
  }
  feature {
    key: "context_movie_title"
    value {
      bytes_list {
        value: "American Beauty (1999)"
        value: "Schindler\'s List (1993)"
        value: "Remains of the Day, The (1993)"
        value: "Silence of the Lambs, The (1991)"
        value: "Like Water for Chocolate (Como agua para chocolate) (1992)"
        value: "Babe (1995)"
        value: "Good Will Hunting (1997)"
        value: "Thelma & Louise (1991)"
        value: "Grifters, The (1990)"
        value: "Apollo 13 (1995)"
      }
    }
  }
  feature {
    key: "context_movie_year"
    value {
      int64_list {
        value: 1999
        value: 1993
        value: 1993
        value: 1991
        value: 1992
        value: 1995
        value: 1997
        value: 1991
        value: 1990
        value: 1995
      }
    }
  }
  feature {
    key: "context_rating_timestamp"
    value {
      int64_list {
        value: 962765672
        value: 962765704
        value: 962765731
        value: 962765760
        value: 962765816
        value: 962765816
        value: 962765845
        value: 962765918
        value: 962765918
        value: 962765918
      }
    }
  }
  feature {
    key: "target_movie_genres"
    value {
      bytes_list {
        value: "Drama"
        value: "Romance"
        value: "War"
      }
    }
  }
  feature {
    key: "target_movie_id"
    value {
      bytes_list {
        value: "1094"
      }
    }
  }
  feature {
    key: "target_movie_rating"
    value {
      float_list {
        value: 4.0
      }
    }
  }
  feature {
    key: "target_movie_title"
    value {
      bytes_list {
        value: "Crying Game, The (1992)"
      }
    }
  }
  feature {
    key: "target_movie_year"
    value {
      int64_list {
        value: 1992
      }
    }
  }
  feature {
    key: "target_rating_timestamp"
    value {
      int64_list {
        value: 962765918
      }
    }
  }
  feature {
    key: "user_age"
    value {
      int64_list {
        value: 35
      }
    }
  }
  feature {
    key: "user_gender"
    value {
      bytes_list {
        value: "F"
      }
    }
  }
  feature {
    key: "user_id"
    value {
      bytes_list {
        value: "4876"
      }
    }
  }
  feature {
    key: "user_occupation_text"
    value {
      bytes_list {
        value: "technician/engineer"
      }
    }
  }
  feature {
    key: "user_zip_code"
    value {
      bytes_list {
        value: "98201"
      }
    }
  }
}
```