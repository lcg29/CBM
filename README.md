# CBM-methane content prediction

## Revising depth parameters to improve the accuracy of machine learning to predict gas content.
The code and software of this project are open source.
<br/>The project is suggested to be run and completed in Jupyter Notebook.
<br/> The algorithms for the training model are based on an open source project Pycaret (https://github.com/pycaret/pycaret).

## Here is the list of libraries you need to install to execute the code.
python = 3.8.5
<br/> numpy
<br/> six 
<br/> pandas
<br/> catboost (https://catboost.ai/)
<br/> pycaret = 2.3  in this program
<br/> pycaret = 2.2  in published article
<br/> Note that in different versions, the number of decimal places calculated is different, which can lead to different results in the model.

## Input and Output 
The input data is in 'data.xlsx'
<br/>Jupyter is an open source interactive application, and the output of programming is also in *.ipynb.

## Example: CatBoost verifies the effect of improving the input depth parameter
### Before revising parameters
![MD as deep parameter](https://github.com/lcg29/CBM/blob/main/CBMgas-md.ipynb)  
<br/> 
![image](https://github.com/lcg29/CBM/blob/main/md%20input.png)
<br/> Picture based on CatBoost
### After revising parameters
![Z as deep parameter](https://github.com/lcg29/CBM/blob/main/CBMgas-z.ipynb)  
<br/> 
![image](https://github.com/lcg29/CBM/blob/main/Z%20input.png)
<br/> Picture based on CatBoost
<br/>The effects of other algorithms are not shown here, but are included in the code.
