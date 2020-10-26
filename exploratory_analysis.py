"""
EXPLORATORY DATA ANALYSIS

In this notebook we will use a recent survey in the USA on the job market for software developers.
Our goal is to do an initial investigation of the data in order to detect problems with the data, need for more
variables, flaws in the organization and transformation needs.

Salary Survey conducted at https://www.freecodecamp.com/ with software developers in the USA who attended
Bootcamp training.
"""


import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import colorsys
import warnings
from platform import python_version
plt.style.use('seaborn-talk')
warnings.filterwarnings('ignore')


print('\n'*3 + f'Python version used in this analysis: {python_version()}')
print(f'Numpy version used in this analysis: {np.__version__}')
print(f'Pandas version used in this analysis: {pd.__version__}')
print(f'Matplotlib version used in this analysis: {mat.__version__}' + '\n'*3)


# Loading the dataset
df = pd.read_csv('Notebooks/Dados-Pesquisa.csv', sep=',', low_memory=False)
print(df)
print(df.describe())
print(list(df))


# What is the age distribution of the survey participants?
# Most professionals who work as programmers for
# software are in the age range between 20 and 30 years old, being 25 years old
# the most frequent age.


# Age distribution
df.Age.hist(bins=60)
plt.xlabel('Age')
plt.ylabel('Number of Professionals')
plt.title('Age Distribution')
plt.show()


# What is the gender distribution of survey participants?
# The vast majority of programmers are male


# Gender distribution
labels = df.Gender.value_counts().index
num = len(df.EmploymentField.value_counts().index)

HSVlist = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
RGBlist = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSVlist))

slices, text = plt.pie(df.Gender.value_counts(), colors=RGBlist, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(slices, labels, bbox_to_anchor=(1.05, 1))
plt.title('Gender')
plt.show()


# What are the main interests of the research participants?
# The main professional interest of programmers is web development (Full-Stack, Front-End and Back-End),
# followed by the Data Science area.


# Interest distribution
num = len(df.JobRoleInterest.value_counts().index)
HSVlist = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
RGBlist = list(map(lambda x: colorsys.hsv_to_rgb(*x), RGBlist))
labels = df.JobRoleInterest.value_counts().index
colors = ['OliveDrab', 'Orange', 'OrangeRed', 'DarkCyan', 'Salmon', 'Sienna', 'Maroon', 'LightSlateGrey', 'DimGray']

slices, text = plt.pie(df.JobRoleInterest.value_counts(), colors=RGBlist, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(slices, labels, bbox_to_anchor=(1.1, 1))
plt.title("Professional Interest")
plt.show()


# What are the business areas in which research participants work?
# Most programmers work in the field of
# software and IT, but other areas such as finance and health are also
# significant.


# Employability distribution
num = len(df.EmploymentField.value_counts().index)

# Creating the color list
HSVlist = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
RGBlist = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSVlist))
labels = df.EmploymentField.value_counts().index

# Pie chart
slices, text = plt.pie(df.EmploymentField.value_counts(), colors=RGBlist, startangle=90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(slices, labels, bbox_to_anchor=(1.05, 1))
plt.title("Current Work")
plt.show()


# What are the work preferences by age?
# Realize that as age increases, interest in work
# freelance also increases, being the model preferred by professionals
# over 60 years old. Younger professionals prefer to work in
# Startups or in your own business. Professionals between 20 and 50 years old
# prefer to work in medium-sized companies.


# Work preference by age
df_agearanges = df.copy()
bins = [0, 20, 30, 40, 50, 60, 100]

df_agearanges['AgeRanges'] = pd.cut(df_agearanges['Age'], bins, labels=["< 20", "20-30", "30-40", "40-50", "50-60", ">60"])

df2 = pd.crosstab(df_agearanges.AgeRanges, df_agearanges.JobPref).apply(lambda r: r/r.sum(), axis=1)


# Creating the color list
HSVlist = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
RGBlist = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSVlist))

# Bar chart
ax1 = df2.plot(kind="bar", stacked=True, color=RGBlist, title="Work Preference by Age")
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, bbox_to_anchor=(1.1, 1))
plt.show()


# What is the purpose of relocation?
# The desire to seek a new job decreases with age.
# Almost 80% of people under 30 are prepared for this.


# Job reallocation by age
df3 = pd.crosstab(df_agearanges.AgeRanges, df_agearanges.JobRelocateYesNo).apply(lambda r: r/r.sum(), axis=1)

# Setting the quantity
num = len(df_agearanges.AgeRanges.value_counts().index)

# Creating the color list
HSVlist = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
RGBlist = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSVlist))

# Bar chart (Stacked)
ax1 = df3.plot(kind="bar", stacked=True, color=RGBlist, title="Job Reallocation by Age")
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, ["No", "Yes"], loc='best')
plt.show()


# What is the relationship between age and learning hours?
# The age of the professionals does not affect the amount of time spent on training and education.


# Age x Learning Hours
df9 = df.copy()
df9 = df9.dropna(subset=["HoursLearning"])
df9 = df9[df['Age'].isin(range(0, 70))]

# Setting the values of x and y
x = df9.Age
y = df9.HoursLearning

# Computing the values and generating the graph
m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Age")
plt.ylabel("Training hours")
plt.title("Age per training hours")
plt.show()


# What is the relationship between investment in training and salary expectations?
# Professionals who invest time and money in training and
# training, in general, get higher salaries, although some
# professionals expect high salaries, investing 0 in training.


# Investment in Training x Salary Expectation
df5 = df.copy()
df5 = df5.dropna(subset=["ExpectedEarning"])
df5 = df5[df['MoneyForLearning'].isin(range(0, 60000))]

# Setting the values of x and y
x = df5.MoneyForLearning
y = df5.ExpectedEarning

# Computing the values and generating the graph
m, b = np.polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-', color="red")
plt.xlabel("Investment in Training")
plt.ylabel("Salary Expectation")
plt.title("Investment in Training vs Salary Expectation")
plt.show()
