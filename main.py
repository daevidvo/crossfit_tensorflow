# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# makes numpy nums easier to read
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

# reads the athletes.csv and takes out the labels
crossfit_data = pd.read_csv('athletes.csv', names=['athlete_id', 'name', 'region', 'team', 'affiliate', 'gender', 'age', 'height', 'weight',	'fran',	'helen', 'grace', 'filthy50', 'fgonebad', 'run400', 'run5k', 'candj', 'snatch', 'deadlift', 'backsq', 'pullups', 'eat', 'train', 'background', 'experience', 'schedule', 'howlong'], low_memory=False)
crossfit_data.head()

# prints the first five rows of athletes.csv; used to verify linkage
print(crossfit_data.head())

#makes a new list that has the backsq data in it
crossfit_features = crossfit_data.copy()
crossfit_labels = crossfit_features.pop('backsq')

#verify the new list; should print 305.0
print(crossfit_labels[1])






