import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

matplotlib.style.use('ggplot')
matplotlib.get_backend()
pd.options.display.max_rows = 100

'''
 Exploratory data analysis
   - Data extraction: load the dataset
   - Cleaning - fill in missing values
   - Plotting - create some interesting charts that will spot correlations
       and hidden insights out of the data
   - Assumptions - formulate hypotheses from the charts
'''

'''
    Two datasets are available: a training set and a test set
    Use training set  - to build our predictive model
    Use testing set - to score it (predictions)
'''


data = pd.read_csv('../input/train.csv')
data.head()
#   Statistically describe data - find missing values
data.describe()
#   Missing data - Age column - fill with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()

survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.show(block=True)

figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show(block=True)


figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show(block=True)

# combine the age, the fare and the survival on a single chart
plt.figure(figsize=(13,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
plt.show(block = True)


# the ticket fare correlates with the class
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(13,8), ax = ax)
plt.show(block=True)


# how the embarkation site affects the survival.
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.show(block=True)


# FEATURE ENGINEERING

def status(feature):

    print('Processing',feature,': ok')


def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')

    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined
combined = get_combined_data()
combined.shape
combined.head()

#   Extracting the passenger titles

def get_titles():
    global combined

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()
combined.head()

#   now we have an additional column called Title in out combined dataset

'''
    Simply replacing Age with median value might not be the
    best solution since the age may differ by groups and
    categories if passengers
    Consider : median age based on SEX, PCLASS and TITLE
'''


def process_age():
    global combined

    # a function that fills the missing values of the Age variable

    def fillAges(row):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

    combined.Age = combined.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

status('age')

process_age()
combined.info()

#   We noticed a missing value in 'Fare', two missing values in Embarked
#   and a lot of missing values in Cabin (295)


'''
    Drop the name column since we won't be using it anymore because
    we created a Title column then encode the Title column with
    dummy values
'''
def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)

    status('names')


process_names()
combined.head()

#   Replace missing Fare with mean
def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)

    status('fare')

process_fares()


#   Replace missing values of Embarked with the most frequent Embarked value
def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S', inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')

process_embarked()


#   Replace NaN values of Cabin with U(unknown) and then map each
#   Cabin value to the first letter, then encode the cabin values
#   using dummy values

def process_cabin():
    global combined

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')

process_cabin()
combined.info()

#   OK. NOW we are done with missing values

#   1- male , 0- female
def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    status('sex')

process_sex()


def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')

process_pclass()


def process_ticket():
    global combined
    combined.drop('Ticket', axis=1, inplace=True)
    status('ticket')

process_ticket()


def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')

process_family()

combined.shape
combined.head()


# The feature range in different intervals => NORMALIZATION
def scale_all_features():
    global combined

    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x / x.max(), axis=0)

    print('Features scaled successfully !')

scale_all_features()

# and MODEL
from sklearn.linear_model import LogisticRegression

'''
    1. Break the combined dataset in train set and test set.
    2. Use the train set to build a predictive model.
    3. Evaluate the model using the train set.
    4. Test the model using the test set and generate and output file for the submission.
'''

'''
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)
'''
print(combined.head())

def recover_train_test_target():
    global combined

    train0 = pd.read_csv('../input/train.csv')

    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]

    return train, test, targets

print('Recover all sets')
train, test, targets = recover_train_test_target()
print(train.head())




# Based on : http://hamelg.blogspot.ro/2015/11/python-for-data-analysis-part-28.html

# The columns we'll use to predict the target
train_features = pd.DataFrame([ train["Sex"], train["Age"], train["FamilySize"], train["Pclass_3"], train["Pclass_2"],
                               train["Parch"], train["Fare"], train["SibSp"], train["Cabin_U"],
                               train["Title_Master"], train["Title_Mr"], train["Title_Miss"],
                               train["Title_Mrs"]]).T

# Initialize our algorithm
log_model = LogisticRegression(random_state=1)
# Train the model
log_model.fit(X=train_features, y=targets)

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)

#   let's make class predictions using this model and then compare the predictons to the actual values
# Make predictions
preds = log_model.predict(X= train_features)

print('Predictions:')
print(preds)
# Generate table of predictions vs actual
print(pd.crosstab(preds, targets))

predictionAccuracy = (490 + 254)/889
print(predictionAccuracy)



# TEST
# Make predictions using the test set.
test_predictors = pd.DataFrame([ test["Sex"], test["Age"], test["FamilySize"], test["Pclass_3"], test["Pclass_2"],
                               test["Parch"], test["Fare"], test["SibSp"], test["Cabin_U"],
                               test["Title_Master"], test["Title_Mr"], test["Title_Miss"],
                               test["Title_Mrs"]]).T
test_predictions = log_model.predict(test_predictors)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
test_result = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_predictions
    })

print('Test result')
print(test_result)