# This .py used to merge all the data file from kaggle to a big csv file.
# 


# pip install chardet

import pandas as pd
import os
import json
from chardet import detect

# list all the countries code
countries = ['CA', 'DE', 'FR', 'GB', 'IN', 'KR', 'MX', 'JP', 'US', 'RU']

# initialize a df to store data
merged_data = pd.DataFrame()

def detect_encoding(file_path):
    '''
    Detect encoding type
    '''
    with open(file_path, 'rb') as f:
        rawdata = f.read(50000)
    return detect(rawdata)['encoding']

# deal each country
for country in countries:
    try:
        csv_path = f'{country}videos.csv'
        
        # detect file encoding type
        try:
            encoding = detect_encoding(csv_path)
            print(f"Detected encoding for {csv_path}: {encoding}")
            
            # try to read file with encoding type that detected
            df = pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            print(f"Failed with detected encoding, trying fallback encodings for {csv_path}: {e}")
            # try to read file with other encoding type
            for enc in ['utf-8', 'ISO-8859-1', 'Windows-1252', 'latin1']:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    print(f"Successfully read {csv_path} with {enc} encoding")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not read {csv_path} with any tried encoding")
        
        # add country column
        df['country'] = country
        
        # merge
        merged_data = pd.concat([merged_data, df], ignore_index=True)
        
    except FileNotFoundError:
        print(f"Warning: {country}videos.csv not found, skipping...")
    except Exception as e:
        print(f"Error processing {country}videos.csv: {e}")

# store
merged_data.to_csv('all_countries_videos.csv', index=False, encoding='utf-8')
print("All data merged successfully into all_countries_videos.csv")


