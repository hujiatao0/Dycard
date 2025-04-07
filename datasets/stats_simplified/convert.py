import pandas as pd
from datetime import datetime

def convert_to_timestamp(date_str):
    """Convert datetime string to Unix timestamp"""
    return int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())

files_config = {
    'badges.csv': {
        'time_cols': ['Date'],
        'int_cols': ['Id', 'UserId']
    },
    'comments.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'PostId', 'Score', 'UserId']
    },
    'users.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'Reputation', 'Views', 'UpVotes', 'DownVotes']
    },
    'posts.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'PostTypeId', 'Score', 'ViewCount', 'OwnerUserId', 
                    'AnswerCount', 'CommentCount', 'FavoriteCount', 'LastEditorUserId']
    },
    'votes.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'PostId', 'VoteTypeId', 'UserId', 'BountyAmount']
    },
    'postHistory.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'PostHistoryTypeId', 'PostId', 'UserId']
    },
    'postLinks.csv': {
        'time_cols': ['CreationDate'],
        'int_cols': ['Id', 'PostId', 'RelatedPostId', 'LinkTypeId']
    }
}

for file_name, config in files_config.items():
    file_path = f'datasets/stats_simplified/{file_name}'
    print(f"Processing {file_path}...")
    
    dtype_dict = {col: 'Int64' for col in config['int_cols']}
    df = pd.read_csv(file_path, dtype=dtype_dict)

    for col in config['time_cols']:
        df[col] = df[col].apply(lambda x: convert_to_timestamp(x) if pd.notnull(x) else None)
    for col in config['int_cols']:
        df[col] = df[col].fillna(0).astype(int)  
    
    df.to_csv(file_path, index=False, float_format='%.0f')

print("All files converted!")