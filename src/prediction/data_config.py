PROJECT_ID          = "hybrid-vertex"
REGION              = "us-central1"
PREFIX              = "rec-bandits-v2"
BUCKET_NAME         = "rec-bandits-v2-hybrid-vertex-bucket"
EXISTING_VOCAB_FILE = "gs://rec-bandits-v2-hybrid-vertex-bucket/vocabs/vocab_dict.pkl"
eval_batch_size     = 1
PER_ARM_DIM         = 64
GLOBAL_DIM          = 64