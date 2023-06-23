import numpy as np
import pickle

load_model=pickle.load(open('F:\Coding\Project\House price predict\trained_model.sav','rb'))

input_data = (8.3252	,41.0	,6.984127,	1.023810,	322.0,	2.555556,	37.88,	-122.23)
id_asnumpy = np.asarray(input_data)
input_reshaped = id_asnumpy.reshape(1,-1)
predict = gb1.predict(input_reshaped)
print("The predicted value is:",predict)