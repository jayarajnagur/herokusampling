import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

specimen_data=pd.read_excel("D:/PROJECT 69/MODEL DEPLOYMENT/sampla_data_08_05_2022.xlsx(1)")

specimen_data=specimen_data.rename(columns={"Cut-off Schedule":"Cut_off_Schedule","Cut-off time_HH_MM":"Cut_off_time_HH_MM"})

specimen_data.head()
specimen_data.columns

specimen_data=specimen_data.drop(["Patient_ID","Test_Booking_Date","Sample_Collection_Date","Mode_Of_Transport","Agent_ID"],axis=1)


#EDA
# Measures of Central Tendency / First moment business decision
specimen_data.mean()
specimen_data.median()
specimen_data.mode()

# Measures of Dispersion / Second moment business decision
specimen_data.std()
specimen_data.var()

# Third moment business decision
specimen_data.skew()

# Fourth moment business decision
specimen_data.kurt()


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()

specimen_data["Patient_Gender"]=le.fit_transform(specimen_data["Patient_Gender"])
specimen_data["Sample"]=le.fit_transform(specimen_data["Sample"])
specimen_data["Way_Of_Storage_Of_Sample"]=le.fit_transform(specimen_data["Way_Of_Storage_Of_Sample"])
specimen_data["Cut_off_Schedule"]=le.fit_transform(specimen_data["Cut_off_Schedule"])
specimen_data["Traffic_Conditions"]=le.fit_transform(specimen_data["Traffic_Conditions"])
specimen_data["Test_Name"]=le.fit_transform(specimen_data["Test_Name"])


specimen_data["Reached_On_Time"].unique()


specimen_data["Reached_On_Time"].value_counts()


# Input and Output Split
predictors = specimen_data.loc[:, specimen_data.columns!="Reached_On_Time"]

target = specimen_data["Reached_On_Time"]


# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

#Predictions on test data
confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))


# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(x_train))
accuracy_score(y_train, rf_clf.predict(x_train))



input_data = (32,1,6,0,1,16.1,10.15,0,13.15,0,12,72,3,9,54)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rf_clf.predict(input_data_reshaped)
print(prediction)


#checking for the results
list_value=pd.DataFrame(specimen_data.iloc[0:1,:15])
list_value

print(rf_clf.predict(list_value))


#saving the model
pickle.dump(rf_clf,open('model.pkl','wb'))

#load the model from disk
model=pickle.load(open('model.pkl','rb'))





























