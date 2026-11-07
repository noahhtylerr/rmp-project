import os
import time
import pandas as pd
from rmp_client import RMPClient

# ===== settings =====
# schools to scrape, as (RMP school number, short name for files, review cap or None for no cap)
schools = [
    ('1331', 'rochester', 12000),
    ('807', 'rit', 12000),
    ('963', 'geneseo', 12000),
    ('1549', 'brockport', 12000),
    ('661', 'nazareth', 12000),
    ('1628', 'fisher', 12000),
]

resume = False # False = start each school from scratch, True = continue a stopped run

# data folders
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
schools_folder = os.path.join(data_folder, 'schools')
raw_folder = os.path.join(data_folder, 'raw')
os.makedirs(schools_folder, exist_ok=True)
os.makedirs(raw_folder, exist_ok=True)


# turns whatever object the package returns into a plain dictionary
def to_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, 'model_dump'):
        return item.model_dump()
    return dict(vars(item))


total_start = time.time()

with RMPClient() as client:
    for school_id, school_name, max_reviews in schools:
        print("\n========", school_name, "========")
        professors_file = os.path.join(schools_folder, f'{school_name}_professors.csv')
        ratings_file = os.path.join(schools_folder, f'{school_name}_ratings.csv')

        # starting from scratch, so remove any old files for this school
        if not resume:
            for old_file in [professors_file, ratings_file]:
                if os.path.exists(old_file):
                    os.remove(old_file)

        # ----- step 1, list every professor at the school -----
        if resume and os.path.exists(professors_file):
            professors_df = pd.read_csv(professors_file)
        else:
            professors = [to_dict(prof) for prof in client.iter_professors_for_school(school_id)]
            professors_df = pd.DataFrame(professors)
            print("Professors on RMP:", len(professors_df),
                  " Total reviews on RMP:", professors_df['num_ratings'].sum())

            # only professors with reviews
            professors_df = professors_df[professors_df['num_ratings'] >= 0]

            # if the school is over its cap, randomly sample whole professors until the cap is reached
            if max_reviews is not None and professors_df['num_ratings'].sum() > max_reviews:
                shuffled = professors_df.sample(frac=1, random_state=42)
                professors_df = shuffled[shuffled['num_ratings'].cumsum() <= max_reviews]
                print("Sampled professors to stay under", max_reviews, "reviews")

            professors_df['school'] = school_name
            professors_df.to_csv(professors_file, index=False)

        print("Professors to scrape:", len(professors_df))
        print("Expected reviews:", professors_df['num_ratings'].sum())

        # ----- step 2, collect every review for each professor -----
        done_ids = set()
        if resume and os.path.exists(ratings_file):
            done_ids = set(pd.read_csv(ratings_file)['professor_id'].astype(str))
            print("Already done:", len(done_ids))

        start_time = time.time()

        for i, prof_id in enumerate(professors_df['id'].astype(str)):
            if prof_id in done_ids:
                continue

            # try up to 3 times, waiting longer after each failure
            rows = None
            for attempt in range(3):
                try:
                    rows = [to_dict(rating) for rating in client.iter_professor_ratings(prof_id)]
                    break
                except Exception as error:
                    print("Error on", prof_id, "attempt", attempt + 1, error)
                    time.sleep(5 * (attempt + 1))

            if rows is None:
                print("Skipped", prof_id, "after 3 failures")
                continue

            # add the professor and school to each review
            for row in rows:
                row['professor_id'] = prof_id
                row['school'] = school_name

            # save after every professor so nothing is lost if the run stops
            if rows:
                pd.DataFrame(rows).to_csv(ratings_file, mode='a', index=False, header=not os.path.exists(ratings_file))
            done_ids.add(prof_id)

            if i % 50 == 0:
                minutes = (time.time() - start_time) / 60
                print(i, "of", len(professors_df), "professors done,", round(minutes, 1), "minutes")

            time.sleep(0.5)   # pause so we do not overload the site

        print(school_name, "finished in", round((time.time() - start_time) / 60, 1), "minutes")


# ----- combine every school into one file -----
all_ratings = []
all_professors = []
for school_id, school_name, max_reviews in schools:
    ratings_path = os.path.join(schools_folder, f'{school_name}_ratings.csv')
    if os.path.exists(ratings_path):
        all_ratings.append(pd.read_csv(ratings_path))
        all_professors.append(pd.read_csv(os.path.join(schools_folder, f'{school_name}_professors.csv')))

combined_ratings = pd.concat(all_ratings, ignore_index=True)
combined_professors = pd.concat(all_professors, ignore_index=True)
combined_ratings.to_csv(os.path.join(raw_folder, 'all_ratings.csv'), index=False)
combined_professors.to_csv(os.path.join(raw_folder, 'all_professors.csv'), index=False)

print("\nTotal reviews:", len(combined_ratings))
print(combined_ratings['school'].value_counts())
print("Total time:", round((time.time() - total_start) / 60, 1), "minutes")