<H3>ENTER YOUR NAME:swetha.M
<H3>ENTER YOUR REGISTER NO.212223040223
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
IMPORT LIBRARIES :

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

READ THE DATA:
df=pd.read_csv("Churn_Modelling.csv")

CHECK DATA:
df.head()
df.tail()
df.columns

CHECK THE MISSING DATA:
df.isnull().sum()
ASSIGNING X:
X = df.iloc[:,:-1].values
X
ASSIGNING Y:
Y = df.iloc[:,-1].values
Y
CHECK FOR OUTLIERS:
df.describe()

DROPPING STRING VALUES DATA FROM DATASET:
data = df.drop(['Surname', 'Geography','Gender'], axis=1)

CHECKING DATASETS AFTER DROPPING STRING VALUES DATA FROM DATASET:
data.head()

NORMALIE THE DATASET USING (MinMax Scaler):
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

SPLIT THE DATASET:
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print(X)
print(Y)

TRAINING AND TESTING MODEL:
X_train ,X_test ,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))

## OUTPUT:
DATA CHECKING:

![image](https://github.com/user-attachments/assets/5b5d8cde-7762-4325-9134-f8ba54e7be70)

MISSING DATA:


![image](https://github.com/user-attachments/assets/b80c0040-7e3f-401c-868f-c8583040f0c9)

DUPLICATES IDENTIFICATION:


![image](https://github.com/user-attachments/assets/e0c656eb-cf31-4bdc-ad7a-c52769bc58eb)

VALUE OF Y:


![image](https://github.com/user-attachments/assets/0d86b56f-89fc-4f4c-aade-d81b53c323fa)

OUTLIERS:


![image](https://github.com/user-attachments/assets/6310af08-274e-4cca-9721-cb19fa9a8d44)

CHECKING DATASET AFTER DROPPING STRING VALUES DATA FROM DATASET:

![image](https://github.com/user-attachments/assets/d9d5dded-e026-4440-8ff8-5425b0adc37b)

NORMALIZE THE DATASET:


![image](https://github.com/user-attachments/assets/98d3eacf-d5a2-484d-b916-e178d40dd36b)

SPLIT THE DATASET:

![image](https://github.com/user-attachments/assets/c05e11fb-4f76-4518-8b82-dd1d359728ac)

### TRAINING AND TESTING MODEL:
![image](https://github.com/user-attachments/assets/a7c0cb51-6751-407f-aff1-bf409c7815f8)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.






