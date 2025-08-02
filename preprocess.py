import pandas as pd 
import re 
"""
The purpose of this script is to combine the course evaluation datasets with their respective course descriptions and preprocess the datasets.
"""
def preprocessing(new_data): 
    """Remove any links, punctuation, commas, periods, and make everything lowercase"""
    # Make everything lowercase 
    new_data['Division'] = new_data["Division"].str.lower() 
    new_data['Dept'] = new_data["Dept"].str.lower() 
    new_data['course'] = new_data["course"].str.lower() 
    new_data['Term'] = new_data["Term"].str.lower() 
    new_data["Last Name"] = new_data["Last Name"].str.lower() 
    new_data['description'] = new_data["description"].str.lower() 
    
    # Remove links
    # Referenced: https://stackoverflow.com/questions/56358888/how-to-remove-https-links-from-a-string-column-in-pandas for regex pattern
    new_data["description"] = new_data["description"].str.replace('https?:\/\/[^\s<>"]+|www\.[^\s<>"]+', "", regex=True)
    
    # Remove punctuation
    # Referenced: https://stackoverflow.com/questions/39782418/remove-punctuations-in-pandas for regex pattern
    new_data["description"] = new_data['description'].str.replace('[^\w\s]+',' ', regex=True)
    
    # Remove commas
    new_data["description"] = new_data['description'].str.replace(',','')
    
    # Remove periods
    new_data["description"] = new_data['description'].str.replace('.','')
    
    # Remove extra spacing
    new_data["description"] = new_data['description'].str.replace(' +', ' ', regex=True)

    return new_data 

def return_matching_substring(value):
    """Returns the matching substring of value"""
    # https://stackoverflow.com/questions/68759305/which-pattern-was-matched-among-those-that-i-passed-through-a-regular-expression
    # https://www.geeksforgeeks.org/python/pattern-matching-python-regex/
    return value.group(1)

if __name__ == '__main__':
    dataset_1 = pd.read_csv("data/raw_data/artsci.csv")
    dataset_2 = pd.read_csv("data/raw_data/course_evals.csv")
    # Referenced: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
    course_descriptions = pd.read_csv("data/raw_data/course_description.csv")
    # print(course_descriptions.head())
    
    # Filter only for CS courses at UofT's Arts & Science 
    dataset_1 = dataset_1[dataset_1["Dept"] == "CSC"]
    dataset_2 = dataset_2[dataset_2["Dept"] == "CSC"]
    
    # Grab dataset_1's last name column and course and merge with dataset_2
    new_data = pd.merge(dataset_2, dataset_1[["Course", "Last Name"]], on=['Course'], how='inner')
    new_data = new_data.rename(columns={"Course": "course"}) # Rename
    
    # If a column value is in the other dataset's course string, then we found the matching course 
    # Convert course description dataframes into a string and combine with | (OR)
    # https://github.com/learnbyexample/py_regular_expressions/blob/master/py_regex.md
    # https://regex101.com/
    combined_as_string = '|'.join(re.escape(a_course_code) for a_course_code in course_descriptions["course"].astype(str))
    the_regex_pattern = f'.*?({combined_as_string})*.?'
    # Find the Course that contains a substring that is in combined_as_string and replace the string with the substring (course code)
    # https://flexiple.com/python/python-regex-replace
    # https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values 
    # https://stackoverflow.com/questions/68759305/which-pattern-was-matched-among-those-that-i-passed-through-a-regular-expression
    # https://www.geeksforgeeks.org/python/pattern-matching-python-regex/
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html
    new_data['course'] = new_data['course'].str.replace(the_regex_pattern, return_matching_substring, regex=True)
    
    # Merge with course description 
    new_data = pd.merge(new_data, course_descriptions, on=['course'], how='inner')
    
    # Preprocess the new dataset
    new_data = preprocessing(new_data)
    
    # Make column names lowercase
    new_data.columns = map(str.lower, new_data.columns)
    
    # Make column for the target variable, which is 1 (if it's recommended, which is when the "i would recommend this course" is >= 4) and 0 (if it's "not recommended/somewhat recommend", which is when "i would recommend this course" is <4)
    # https://www.geeksforgeeks.org/python/ways-to-apply-an-if-condition-in-pandas-dataframe/
    new_data["recommended"] = new_data["i would recommend this course"].apply(lambda x: 1 if x >=4 else 0)
    
    # Output new CSV for cleaned dataset
    new_data.to_csv("data/clean_data/new_data.csv", index=False)

