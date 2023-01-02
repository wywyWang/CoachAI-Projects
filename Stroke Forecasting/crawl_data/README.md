# Manual for crawling badminton labeled data from the database

## Usage

### check available matches: Print the number of available matches and the detailed match ids.
```
python get_data.py --model check_match
```

### Download match information: Download a `match.csv` file with match information
```
python get_data.py --model get_match
```

### Download corresponding homography matrics: Download a `homography.csv` file with all available homography matrics
```
python get_data.py --model get_homography
```

### Download stroke-level records: Download all detailed matches with stroke-level information
```
python get_data.py --model get_set
```