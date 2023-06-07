# Imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# CSV file reading

try:

    df = pd.read_csv("https://pycourse.s3.amazonaws.com/temperature.csv")

# Setting the data frame index

    df = df.set_index('date')

    # Assigning column values to variables

    x, y = df[['temperatura']].values, df[['classification']].values

# Converting categorical data to numeric

    le = LabelEncoder()
    y = le.fit_transform(y.ravel())

# Logistical regression object and predicting data value labels

    clf = LogisticRegression()
    clf.fit(x, y)

    x_test = np.linspace(start=0., stop=45., num=100).reshape(-1, 1)
    y_pred = clf.predict(x_test)

    y_pred = le.inverse_transform(y_pred)

# Defining data frame outputs

    output = {'new_temp': x_test.ravel(), 'new_class': y_pred.ravel()}
    output = pd.DataFrame(output)
    output.head(), output.tail(), output.info(), output.describe()

    output['new_class'].value_counts().plot.bar(figsize=(10, 5), rot=0, title='# of new values generated')
    output.boxplot(by='new_class', figsize=(10, 5))

# Funcitons


    def classify_temp():
        ask = True
        while ask:
            temp = input("Insert the temperature (ºC): ")
            temp = np.array(float(temp)).reshape(-1, 1)
            class_temp = clf.predict(temp)
            class_temp = le.inverse_transform(class_temp)
            print(f"The classification of temperature {temp.ravel()[0]}ºC is:", class_temp[0])
            ask = input("New classification (y/n): ") == 'y'
        else:
            print("You have chosen not to continue. Finished program!")


# Calling the function

    classify_temp()

# Error handling

except ValueError:
    print('Incorrect data format! Please verify.')

# Standard exception

except Exception as exception:
    print(f'An error occurred while running the program. Error: {exception}')
