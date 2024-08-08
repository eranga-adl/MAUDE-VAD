#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud


# In[5]:


# Define the folder paths
folder_path_1 = 'foiText Files'
folder_path_2 = 'device Files'

# List all the text files in each folder
text_files_1 = ['foitext2020.txt','foitext2021.txt','foitext2022.txt', 'foitext2023.txt']
text_files_2 = ['DEVICE2020.txt', 'DEVICE2021.txt','DEVICE2022.txt', 'DEVICE2023.txt']


# In[6]:


# Function to read and concatenate text files from a given folder
def read_and_concatenate(folder_path, text_files):
    dataframes = []
    for file in text_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, delimiter='|', encoding='ISO-8859-1', on_bad_lines='skip')
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


# In[7]:


foitext_files = read_and_concatenate(folder_path_1, text_files_1)


# In[12]:


device_files = read_and_concatenate(folder_path_2, text_files_2)


# ## Preprocess

# In[15]:


device_files = device_files[device_files['MDR_REPORT_KEY'].apply(type) == int]
device_files = device_files.set_index('MDR_REPORT_KEY')


# In[19]:


device_files.head()


# ## Joining foiText and Device files
# 
# 

# In[20]:


foitext_files = foitext_files.join(device_files, on = 'MDR_REPORT_KEY', how = 'left', rsuffix = '_device')


# In[21]:


foitext_files.head(2)


# In[22]:


foitext_files.columns


# In[82]:


foitext_files[foitext_files.GENERIC_NAME == 'VENTRICULAR (ASSIST) BYPASS']


# In[23]:


filter_values = [
    'VENTRICULAR (ASSIST) BYPASS',
    'VENTRICULAR (ASSISST) BYPASS',
    'VENTRICULAR ASSIST DEVICE',
    'LEFT VENTRICULAR ASSIST DEVICE'
]


# In[24]:


filtered_foitext_files = foitext_files[foitext_files['GENERIC_NAME'].isin(filter_values)]


# In[25]:


filtered_foitext_files


# ### Saving the Filtered File(DO NOT RUN AGAIN)

# In[26]:


filtered_foitext_files.to_csv('filtered_foitext_files.csv', index=False)


# ## Start from Here

# In[8]:


filtered_foitext_files = pd.read_csv('filtered_foitext_files.csv')


# In[27]:


filtered_foitext_files.head(2)


# ## Pre-processing Data

# In[9]:


filtered_foitext_files.columns


# ## Identify unique values

# In[10]:


def unique_values_in_columns(df):
    unique_values = {}
    for column in df.columns:
        unique_vals = df[column].unique()
        if len(unique_vals) > 40:
            unique_values[column] = 'More than 40 unique values'
        else:
            unique_values[column] = unique_vals.tolist()
    return unique_values


# In[11]:


# Example usage
unique_values = unique_values_in_columns(filtered_foitext_files)
for column, values in unique_values.items():
    print(f"Column: {column}")
    print(f"Unique Values: {values}\n")


# In[12]:


filtered_foitext_files["FOI_TEXT"].str.contains("PAIN|INFECTION|SICK|BLEEDING|EROSION|SEVERE|DEMAGE|TIGHT|HEALTH PROBLEM|ABNORMAL|ANXIETY|NEGATIVE|DIFFICULT|ACHES|PAINFUL|DIARRHEA|BOWEL OBSTRUCTION|INCONTINENCE|DIED|BOWEL PROBLEMS|ANAL|DYSPAREUNIA|PAINfUL SEXUAL INTERCOURSE|REMOVAL|COME OUT|WEAKNESS|NUMBNESS").value_counts()


# ## Reading Main Data File

# In[27]:


mdrfoiMaster = pd.read_csv(
    'mdrfoiThru2023.txt', 
    sep='|', 
    quoting=3, 
    encoding = "ISO-8859-1", 
    on_bad_lines='skip', 
    low_memory=False, 
    usecols=["MDR_REPORT_KEY", "REPORT_SOURCE_CODE",
             "DATE_RECEIVED", "ADVERSE_EVENT_FLAG",
             "PRODUCT_PROBLEM_FLAG", 
             "SUMMARY_REPORT", "EVENT_TYPE"])


# In[29]:


mdrfoiMaster.to_csv('mdrfoiMaster_exported.csv', index=False)


# In[ ]:





# ## READ MDRFOIMASTER

# In[31]:


mdrfoiMaster = pd.read_csv('mdrfoiMaster_exported.csv')


# In[35]:


mdrfoiMaster


# In[33]:


filtered_foitext_files.head(2)


# ## Filter out non-numeric and apply INT - mdrfoiMaster

# In[34]:


# Filter out non-numeric values from MDR_REPORT_KEY
mdrfoiMaster = mdrfoiMaster[pd.to_numeric(mdrfoiMaster['MDR_REPORT_KEY'], errors='coerce').notna()]

# Convert MDR_REPORT_KEY to int
mdrfoiMaster['MDR_REPORT_KEY'] = mdrfoiMaster['MDR_REPORT_KEY'].astype(int)


# In[36]:


merged_df = filtered_foitext_files.merge(mdrfoiMaster,
                                         on='MDR_REPORT_KEY',
                                         how='left',
                                         suffixes=('', '_master'))


# In[37]:


merged_df.head(2)


# In[38]:


merged_df.columns


# ------------------------------------------------------------------------------------------

# In[39]:


merged_df.to_csv('merged_df_exported_v2.csv', index=False)


# In[40]:


merged_df


# ## Start From HERE !!!

# In[13]:


foiText_merged = pd.read_csv('merged_df_exported_v2.csv')


# In[14]:


foiText_merged.head(2)


# In[15]:


foiText_merged.head(2)


# ## Date adjustment

# In[16]:


foiText_merged["DATE_RECEIVED"] = pd.to_datetime(foiText_merged["DATE_RECEIVED"])


# ## Identify Adverse Events (Flag-Y) and TEXT_TYPE_CODE (D)

# In[17]:


adverseEvents_full = foiText_merged[(foiText_merged["ADVERSE_EVENT_FLAG"] == "Y") & (foiText_merged["TEXT_TYPE_CODE"] == "D")]


# In[18]:


adverseEvents_full.columns


# In[19]:


# Remove duplicates based on 'FOI_TEXT'
adverseEvents = adverseEvents_full.drop_duplicates(subset='FOI_TEXT')


# In[20]:


adverseEvents


# In[21]:


adverseEvents["Year"]  = (adverseEvents["DATE_RECEIVED"]).dt.year


# In[22]:


yearly_counts = adverseEvents.groupby("Year")["ADVERSE_EVENT_FLAG"].count().reset_index(name='Counts')


# In[23]:


# Plotting
sns.barplot(x='Year', y='Counts', data=yearly_counts)
plt.title('Count of Adverse Events by Year')
plt.xlabel('Year')
plt.ylabel('Count of Adverse Events')
plt.show()


# ## Further Visualisations

# In[24]:


# Define subcategories of keywords
subcategories = {
    "Pain and Discomfort": ["pain", "tight", "weakness", "numbness"],
    "Bleeding and Circulation": ["bleeding", "bleed", "bled"],
    "Fatigue and Weakness": ["fatigue", "dizziness", "weakness"],
    "Infections": ["infection", "septicemia", "sepsis", "viral infection", "bacterial"],
    "Cardiac Conditions": ["cardiac tamponade", "myocardial infarction", "cardiogenic shock", "heart failure", "pump failure"],
    "Circulatory and Blood Conditions": ["low cardiac output", "thrombosis", "stroke", "pericardial effusion", "ventricular arrhythmias"],
    "Mechanical Failures": ["malfunction", "device dislocation", "device wear and tear", "motor failure", "sensor malfunction", "fluid leakage", "low flow alarms","battery failure", "electrical failure", "improper implantation", "device adjustment", "calibration error", "programming error"],
    "Psychological Issues": ["distress", "psychological", "cognitive impairment"],
    "Surgical Complications": ["surgical site infection", "graft failure", "reoperation", "postoperative complications"],
    "Systemic Outcomes": ["hemolysis", "anticoagulation dysfunction", "readmission", "device-related mortality"],
    "Fatal Outcomes": ["death", "died", "expired", "passed away"],
    "Digestive System Issues": ["diarrhea", "bowel obstruction", "incontinence", "bowel problems", "abdominal pain", "constipation"],
    "Respiratory Health": ["shortness of breath", "respiratory arrest", "respiratory failure"]
}


# ## Don't run again ( Only if want to save the subcategories again )

# In[25]:


# Create a DataFrame
data = {"Category": [], "Keywords": []}
for category, keywords in subcategories.items():
    data["Category"].append(category)
    # Join keywords into a string with each keyword in single quotes
    data["Keywords"].append(", ".join(f"'{keyword}'" for keyword in keywords))

df = pd.DataFrame(data)

# Export the DataFrame to a CSV file
df.to_csv('Subcategories_and_Keywords.csv', index=False)


# ## Visualise Subcategories

# In[26]:


# Ensure FOI_TEXT is a string and handle any NaN values
adverseEvents['FOI_TEXT'].fillna('', inplace=True)

# Initialize a DataFrame to store counts per subcategory per document
subcategory_counts = pd.DataFrame()

# Iterate over subcategories and keywords
for subcategory_name, keywords in subcategories.items():
    subcategory_pattern = r'\b(' + '|'.join([re.escape(keyword) for keyword in keywords]) + r')\b'
    subcategory_counts[subcategory_name] = adverseEvents['FOI_TEXT'].str.contains(subcategory_pattern, case=False, na=False)

# Aggregate these boolean counts to get the number of documents mentioning each subcategory
subcategory_counts_sum = subcategory_counts.sum().reset_index()
subcategory_counts_sum.columns = ['Subcategory', 'Count']
subcategory_counts_sum = subcategory_counts_sum.sort_values(by='Count', ascending=False)

# Now visualize the top subcategories
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Subcategory', data=subcategory_counts_sum, palette='viridis')
plt.title('Top Adverse Event Subcategories in Texts (Per Document)')
plt.xlabel('Number of Documents Mentioning Subcategory')
plt.ylabel('Subcategories')
plt.show()


# In[27]:


# Ensure FOI_TEXT is a string and handle any NaN values
adverseEvents['FOI_TEXT'].fillna('', inplace=True)

# Initialize a DataFrame to store counts of each keyword
keyword_counts = pd.DataFrame(index=adverseEvents.index)

# Count each keyword's occurrences in each document and sum them by subcategory
for subcategory_name, keywords in subcategories.items():
    keyword_pattern = r'\b(' + '|'.join([re.escape(keyword) for keyword in keywords]) + r')\b'
    keyword_counts[subcategory_name] = adverseEvents['FOI_TEXT'].str.count(keyword_pattern, flags=re.IGNORECASE)

# Aggregate these counts to get the total mentions of each subcategory
subcategory_totals = keyword_counts.sum()

# Count the number of documents that mention each subcategory
document_mentions = (keyword_counts > 0).sum()

# Prepare the final DataFrame
final_counts = pd.DataFrame({
    'Adverse Events': subcategory_totals.index,
    'No. of Reports': document_mentions.values,
    'Freq': subcategory_totals.values,
    'Report (%)': (document_mentions / len(adverseEvents) * 100).values
}).sort_values(by='Report (%)', ascending=False)

# Display the DataFrame
print(final_counts)


# ## By Year Visualisation

# In[50]:


# Ensure adverseEvents is a standalone DataFrame
adverseEvents = adverseEvents.copy()

# Initialize the DataFrame to store subcategory trends
subcategory_trends = pd.DataFrame()

# Process each subcategory
for subcategory, keywords in subcategories.items():
    # Create a pattern that matches any of the keywords in the subcategory
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    adverseEvents['subcategory_match'] = adverseEvents['FOI_TEXT'].str.contains(pattern, case=False, na=False)
    # Group by 'Year' and sum the 'subcategory_match' column to count occurrences
    subcategory_counts = adverseEvents.groupby('Year')['subcategory_match'].sum()
    subcategory_trends[subcategory] = subcategory_counts

# Transpose to align years for plotting
subcategory_trends = subcategory_trends.transpose()

# Calculate total mentions for each subcategory and get the top ones
subcategory_sums = subcategory_trends.sum(axis=1)
top_subcategories = subcategory_sums.sort_values(ascending=False)

# Split into top 5 and the rest
top_5_subcategories = top_subcategories.head(5).index
bottom_subcategories = top_subcategories.tail(len(top_subcategories) - 5).index

# Data for plotting
top_5_data = subcategory_trends.loc[top_5_subcategories]
bottom_data = subcategory_trends.loc[bottom_subcategories]

# Plot data function with subplots
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))  # Set up 2 rows, 1 column

    # Plot the top 5 subcategories
    for subcategory in top_5_data.index:
        ax1.plot(top_5_data.columns, top_5_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax1.set_title('Top 5 Adverse Event Subcategories Across Years')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Mentions')
    ax1.legend(title="Subcategories", loc='upper right')
    ax1.set_xticks(top_5_data.columns)
    ax1.set_ylim(0, 2500)  # Set y-axis limit

    # Plot the remaining subcategories
    for subcategory in bottom_data.index:
        ax2.plot(bottom_data.columns, bottom_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax2.set_title('Remaining Adverse Event Subcategories Across Years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Mentions')
    ax2.legend(title="Subcategories", loc='upper right')
    ax2.set_xticks(bottom_data.columns)
    ax2.set_ylim(0, 2500)  # Set y-axis limit

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_data()

# Clean up the temporary column
adverseEvents.drop('subcategory_match', axis=1, inplace=True)


# In[48]:


# Ensure adverseEvents is a standalone DataFrame
adverseEvents = adverseEvents.copy()

# Ensure FOI_TEXT is a string and handle any NaN values
adverseEvents['FOI_TEXT'].fillna('', inplace=True)

# Initialize the DataFrame to store subcategory trends
subcategory_trends = pd.DataFrame()

# Process each subcategory
for subcategory, keywords in subcategories.items():
    # Create a pattern that matches any of the keywords in the subcategory
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    adverseEvents['subcategory_match'] = adverseEvents['FOI_TEXT'].str.contains(pattern, case=False, na=False)
    # Group by 'Year' and sum the 'subcategory_match' column to count occurrences
    adverseEvents['Year'] = adverseEvents['DATE_RECEIVED'].dt.year
    subcategory_counts = adverseEvents.groupby('Year')['subcategory_match'].sum()
    subcategory_trends[subcategory] = subcategory_counts

# Calculate the total number of reports per year
total_reports_per_year = adverseEvents.groupby('Year').size()

# Normalize the counts by the total number of reports per year
subcategory_trends_normalized = subcategory_trends.div(total_reports_per_year, axis=0)

# Replace NaN values with 0
subcategory_trends_normalized = subcategory_trends_normalized.fillna(0)

# Transpose to align years for plotting
subcategory_trends_normalized = subcategory_trends_normalized.transpose()

# Calculate total mentions for each subcategory and get the top ones
subcategory_sums = subcategory_trends_normalized.sum(axis=1)
top_subcategories = subcategory_sums.sort_values(ascending=False)

# Split into top 5 and the rest
top_5_subcategories = top_subcategories.head(5).index
bottom_subcategories = top_subcategories.tail(len(top_subcategories) - 5).index

# Data for plotting
top_5_data = subcategory_trends_normalized.loc[top_5_subcategories]
bottom_data = subcategory_trends_normalized.loc[bottom_subcategories]

# Plot data function with subplots
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))  # Set up 2 rows, 1 column

    # Plot the top 5 subcategories
    for subcategory in top_5_data.index:
        ax1.plot(top_5_data.columns, top_5_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax1.set_title('Top 5 Adverse Event Subcategories Proportions Across Years')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Proportion of Mentions')
    ax1.legend(title="Subcategories", loc='upper left')
    ax1.set_xticks(top_5_data.columns)
    ax1.set_ylim(0, 0.4)

    # Plot the remaining subcategories
    for subcategory in bottom_data.index:
        ax2.plot(bottom_data.columns, bottom_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax2.set_title('Remaining Adverse Event Subcategories Proportions Across Years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Proportion of Mentions')
    ax2.legend(title="Subcategories", loc='upper left')
    ax2.set_xticks(bottom_data.columns)
    ax2.set_ylim(0, 0.4)

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_data()

# Clean up the temporary column
adverseEvents.drop('subcategory_match', axis=1, inplace=True)


# ## By Year Visualisation (Quarterly Breakdown)

# In[52]:


# Ensure adverseEvents is a standalone DataFrame
adverseEvents = adverseEvents.copy()

# Ensure the DATE_RECEIVED column is in datetime format
adverseEvents['DATE_RECEIVED'] = pd.to_datetime(adverseEvents['DATE_RECEIVED'])

# Create a Quarter column
adverseEvents['Quarter'] = adverseEvents['DATE_RECEIVED'].dt.to_period('Q')

# Initialize the DataFrame to store subcategory trends
subcategory_trends = pd.DataFrame()

# Process each subcategory
for subcategory, keywords in subcategories.items():
    # Create a pattern that matches any of the keywords in the subcategory
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    adverseEvents['subcategory_match'] = adverseEvents['FOI_TEXT'].str.contains(pattern, case=False, na=False)
    # Group by 'Quarter' and sum the 'subcategory_match' column to count occurrences
    subcategory_counts = adverseEvents.groupby('Quarter')['subcategory_match'].sum()
    subcategory_trends[subcategory] = subcategory_counts

# Transpose to align quarters for plotting
subcategory_trends = subcategory_trends.transpose()

# Calculate total mentions for each subcategory and get the top ones
subcategory_sums = subcategory_trends.sum(axis=1)
top_subcategories = subcategory_sums.sort_values(ascending=False)

# Split into top 5 and the rest
top_5_subcategories = top_subcategories.head(5).index
bottom_subcategories = top_subcategories.tail(len(top_subcategories) - 5).index

# Data for plotting
top_5_data = subcategory_trends.loc[top_5_subcategories]
bottom_data = subcategory_trends.loc[bottom_subcategories]

# Plot data function with subplots
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))  # Set up 2 rows, 1 column

    # Plot the top 5 subcategories
    for subcategory in top_5_data.index:
        ax1.plot(top_5_data.columns.astype(str), top_5_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax1.set_title('Top 5 Adverse Event Subcategories Across Quarters')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Number of Mentions')
    ax1.legend(title="Subcategories", loc='upper right')
    ax1.set_ylim(0, 2500)  # Set y-axis limit

    # Plot the remaining subcategories
    for subcategory in bottom_data.index:
        ax2.plot(bottom_data.columns.astype(str), bottom_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax2.set_title('Remaining Adverse Event Subcategories Across Quarters')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Number of Mentions')
    ax2.legend(title="Subcategories", loc='upper right')
    ax2.set_ylim(0, 2500)  # Set y-axis limit

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_data()

# Clean up the temporary column
adverseEvents.drop('subcategory_match', axis=1, inplace=True)


# In[49]:


# Ensure adverseEvents is a standalone DataFrame
adverseEvents = adverseEvents.copy()

# Ensure the DATE_RECEIVED column is in datetime format
adverseEvents['DATE_RECEIVED'] = pd.to_datetime(adverseEvents['DATE_RECEIVED'])

# Create a Quarter column
adverseEvents['Quarter'] = adverseEvents['DATE_RECEIVED'].dt.to_period('Q').astype(str)

# Initialize the DataFrame to store subcategory trends
subcategory_trends = pd.DataFrame()

# Process each subcategory
for subcategory, keywords in subcategories.items():
    # Create a pattern that matches any of the keywords in the subcategory
    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    adverseEvents['subcategory_match'] = adverseEvents['FOI_TEXT'].str.contains(pattern, case=False, na=False)
    # Group by 'Quarter' and sum the 'subcategory_match' column to count occurrences
    subcategory_counts = adverseEvents.groupby('Quarter')['subcategory_match'].sum()
    subcategory_trends[subcategory] = subcategory_counts

# Calculate the total number of reports per quarter
total_reports_per_quarter = adverseEvents.groupby('Quarter').size()

# Normalize the counts by the total number of reports per quarter
subcategory_trends_normalized = subcategory_trends.div(total_reports_per_quarter, axis=0)

# Replace NaN values with 0
subcategory_trends_normalized = subcategory_trends_normalized.fillna(0)

# Transpose to align quarters for plotting
subcategory_trends_normalized = subcategory_trends_normalized.transpose()

# Calculate total mentions for each subcategory and get the top ones
subcategory_sums = subcategory_trends_normalized.sum(axis=1)
top_subcategories = subcategory_sums.sort_values(ascending=False)

# Split into top 5 and the rest
top_5_subcategories = top_subcategories.head(5).index
bottom_subcategories = top_subcategories.tail(len(top_subcategories) - 5).index

# Data for plotting
top_5_data = subcategory_trends_normalized.loc[top_5_subcategories]
bottom_data = subcategory_trends_normalized.loc[bottom_subcategories]

# Plot data function with subplots
def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))  # Set up 2 rows, 1 column

    # Plot the top 5 subcategories
    for subcategory in top_5_data.index:
        ax1.plot(top_5_data.columns, top_5_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax1.set_title('Top 5 Adverse Event Subcategories Proportions Across Quarters')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Proportion of Mentions')
    ax1.legend(title="Subcategories", loc='upper left')
    ax1.set_xticks(range(len(top_5_data.columns)))
    ax1.set_xticklabels(top_5_data.columns, rotation=45)
    ax1.set_ylim(0, 0.5)

    # Plot the remaining subcategories
    for subcategory in bottom_data.index:
        ax2.plot(bottom_data.columns, bottom_data.loc[subcategory], label=subcategory, marker='o', linewidth=2)
    ax2.set_title('Remaining Adverse Event Subcategories Proportions Across Quarters')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Proportion of Mentions')
    ax2.legend(title="Subcategories", loc='upper left')
    ax2.set_xticks(range(len(bottom_data.columns)))
    ax2.set_xticklabels(bottom_data.columns, rotation=45)
    ax2.set_ylim(0, 0.5) 

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_data()

# Clean up the temporary column
adverseEvents.drop('subcategory_match', axis=1, inplace=True)

