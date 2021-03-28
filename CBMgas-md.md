# MD as deep parameter


```python
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
from pycaret.regression import *
```


```python
dataset=pd.read_excel("train set", sheet_name='pymd')
```


```python
print(dataset.columns)
dataset.shape
```

    Index(['MD', 'GR', 'RD', 'DEN', 'AC', 'CNL', 'C'], dtype='object')
    




    (40, 7)




```python
# Setting up Environment in PyCaret
```


```python
exp_regt = setup(data = dataset, target = 'C', train_size = 0.75 ,normalize = True, session_id=123,normalize_method='maxabs' ) 
```


<style  type="text/css" >
#T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row27_col1{
            background-color:  lightgreen;
        }</style><table id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row0_col1" class="data row0 col1" >123</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row1_col1" class="data row1 col1" >C</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row2_col0" class="data row2 col0" >Original Data</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row2_col1" class="data row2 col1" >(40, 7)</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row3_col0" class="data row3 col0" >Missing Values</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row3_col1" class="data row3 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row4_col0" class="data row4 col0" >Numeric Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row4_col1" class="data row4 col1" >6</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row5_col0" class="data row5 col0" >Categorical Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row5_col1" class="data row5 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row6_col0" class="data row6 col0" >Ordinal Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row6_col1" class="data row6 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row7_col0" class="data row7 col0" >High Cardinality Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row8_col0" class="data row8 col0" >High Cardinality Method</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row8_col1" class="data row8 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row9_col0" class="data row9 col0" >Transformed Train Set</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row9_col1" class="data row9 col1" >(30, 6)</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row10_col0" class="data row10 col0" >Transformed Test Set</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row10_col1" class="data row10 col1" >(10, 6)</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row11_col0" class="data row11 col0" >Shuffle Train-Test</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row11_col1" class="data row11 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row12_col0" class="data row12 col0" >Stratify Train-Test</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row12_col1" class="data row12 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row13_col0" class="data row13 col0" >Fold Generator</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row13_col1" class="data row13 col1" >KFold</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row14_col0" class="data row14 col0" >Fold Number</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row14_col1" class="data row14 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row15_col0" class="data row15 col0" >CPU Jobs</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row15_col1" class="data row15 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row16_col0" class="data row16 col0" >Use GPU</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row16_col1" class="data row16 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row17_col0" class="data row17 col0" >Log Experiment</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row18_col0" class="data row18 col0" >Experiment Name</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row18_col1" class="data row18 col1" >reg-default-name</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row19_col0" class="data row19 col0" >USI</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row19_col1" class="data row19 col1" >e930</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row20_col0" class="data row20 col0" >Imputation Type</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row20_col1" class="data row20 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row21_col0" class="data row21 col0" >Iterative Imputation Iteration</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row22_col0" class="data row22 col0" >Numeric Imputer</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row22_col1" class="data row22 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row23_col0" class="data row23 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row24_col0" class="data row24 col0" >Categorical Imputer</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row24_col1" class="data row24 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row25_col0" class="data row25 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row26_col0" class="data row26 col0" >Unknown Categoricals Handling</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row26_col1" class="data row26 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row27_col0" class="data row27 col0" >Normalize</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row27_col1" class="data row27 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row28_col0" class="data row28 col0" >Normalize Method</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row28_col1" class="data row28 col1" >maxabs</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row29_col0" class="data row29 col0" >Transformation</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row30_col0" class="data row30 col0" >Transformation Method</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row31_col0" class="data row31 col0" >PCA</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row32_col0" class="data row32 col0" >PCA Method</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row33_col0" class="data row33 col0" >PCA Components</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row34_col0" class="data row34 col0" >Ignore Low Variance</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row35_col0" class="data row35 col0" >Combine Rare Levels</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row35_col1" class="data row35 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row36_col0" class="data row36 col0" >Rare Level Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row36_col1" class="data row36 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row37_col0" class="data row37 col0" >Numeric Binning</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row38_col0" class="data row38 col0" >Remove Outliers</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row38_col1" class="data row38 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row39_col0" class="data row39 col0" >Outliers Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row39_col1" class="data row39 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row40_col0" class="data row40 col0" >Remove Multicollinearity</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row41_col0" class="data row41 col0" >Multicollinearity Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row42_col0" class="data row42 col0" >Clustering</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row43_col0" class="data row43 col0" >Clustering Iteration</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row44_col0" class="data row44 col0" >Polynomial Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row44_col1" class="data row44 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row45_col0" class="data row45 col0" >Polynomial Degree</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row45_col1" class="data row45 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row46_col0" class="data row46 col0" >Trignometry Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row46_col1" class="data row46 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row47_col0" class="data row47 col0" >Polynomial Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row47_col1" class="data row47 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row48_col0" class="data row48 col0" >Group Features</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row48_col1" class="data row48 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row49_col0" class="data row49 col0" >Feature Selection</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row49_col1" class="data row49 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row50_col0" class="data row50 col0" >Features Selection Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row50_col1" class="data row50 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row51_col0" class="data row51 col0" >Feature Interaction</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row52_col0" class="data row52 col0" >Feature Ratio</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row52_col1" class="data row52 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row53_col0" class="data row53 col0" >Interaction Threshold</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row53_col1" class="data row53 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row54_col0" class="data row54 col0" >Transform Target</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row54_col1" class="data row54 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row55_col0" class="data row55 col0" >Transform Target Method</td>
                        <td id="T_e6dc43cf_5ef6_11eb_b0c0_54e1ada8ef40row55_col1" class="data row55 col1" >box-cox</td>
            </tr>
    </tbody></table>


# SVM


```python
svm=create_model('svm',fold=5);
print(svm);
```


<style  type="text/css" >
#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col0,#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col1,#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col2,#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col3,#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col4,#T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col0" class="data row0 col0" >0.7120</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col1" class="data row0 col1" >0.6741</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col2" class="data row0 col2" >0.8211</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col3" class="data row0 col3" >0.4333</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col4" class="data row0 col4" >0.0452</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row0_col5" class="data row0 col5" >0.0414</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col0" class="data row1 col0" >0.6884</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col1" class="data row1 col1" >0.8073</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col2" class="data row1 col2" >0.8985</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col3" class="data row1 col3" >0.5210</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col4" class="data row1 col4" >0.0540</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row1_col5" class="data row1 col5" >0.0448</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col0" class="data row2 col0" >0.5554</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col1" class="data row2 col1" >0.3914</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col2" class="data row2 col2" >0.6256</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col3" class="data row2 col3" >0.7644</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col4" class="data row2 col4" >0.0370</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row2_col5" class="data row2 col5" >0.0340</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col0" class="data row3 col0" >0.4358</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col1" class="data row3 col1" >0.2675</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col2" class="data row3 col2" >0.5172</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col3" class="data row3 col3" >0.5991</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col4" class="data row3 col4" >0.0282</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row3_col5" class="data row3 col5" >0.0246</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col0" class="data row4 col0" >0.5878</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col1" class="data row4 col1" >0.4109</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col2" class="data row4 col2" >0.6410</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col3" class="data row4 col3" >0.6957</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col4" class="data row4 col4" >0.0361</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row4_col5" class="data row4 col5" >0.0353</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col0" class="data row5 col0" >0.5959</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col1" class="data row5 col1" >0.5102</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col2" class="data row5 col2" >0.7007</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col3" class="data row5 col3" >0.6027</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col4" class="data row5 col4" >0.0401</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row5_col5" class="data row5 col5" >0.0360</td>
            </tr>
            <tr>
                        <th id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col0" class="data row6 col0" >0.0994</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col1" class="data row6 col1" >0.1990</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col2" class="data row6 col2" >0.1389</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col3" class="data row6 col3" >0.1185</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col4" class="data row6 col4" >0.0088</td>
                        <td id="T_eb673e71_5ef6_11eb_a054_54e1ada8ef40row6_col5" class="data row6 col5" >0.0070</td>
            </tr>
    </tbody></table>


    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    


```python
#plot_model(svm)
```


```python
#predict_model(svm)
```


```python
#evaluate_model(svm)
```


```python
tuned_svm = tune_model(svm,n_iter =100, choose_better=True, fold=5)
print(tuned_svm)
```


<style  type="text/css" >
#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col0,#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col1,#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col2,#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col3,#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col4,#T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col0" class="data row0 col0" >0.8863</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col1" class="data row0 col1" >0.9281</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col2" class="data row0 col2" >0.9634</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col3" class="data row0 col3" >0.2198</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col4" class="data row0 col4" >0.0526</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row0_col5" class="data row0 col5" >0.0509</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col0" class="data row1 col0" >0.6610</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col1" class="data row1 col1" >0.7245</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col2" class="data row1 col2" >0.8512</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col3" class="data row1 col3" >0.5701</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col4" class="data row1 col4" >0.0509</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row1_col5" class="data row1 col5" >0.0424</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col0" class="data row2 col0" >0.6435</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col1" class="data row2 col1" >0.5327</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col2" class="data row2 col2" >0.7299</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col3" class="data row2 col3" >0.6793</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col4" class="data row2 col4" >0.0424</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row2_col5" class="data row2 col5" >0.0389</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col0" class="data row3 col0" >0.5800</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col1" class="data row3 col1" >0.3891</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col2" class="data row3 col2" >0.6238</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col3" class="data row3 col3" >0.4167</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col4" class="data row3 col4" >0.0338</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row3_col5" class="data row3 col5" >0.0328</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col0" class="data row4 col0" >0.5257</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col1" class="data row4 col1" >0.4552</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col2" class="data row4 col2" >0.6747</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col3" class="data row4 col3" >0.6629</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col4" class="data row4 col4" >0.0392</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row4_col5" class="data row4 col5" >0.0319</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col0" class="data row5 col0" >0.6593</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col1" class="data row5 col1" >0.6059</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col2" class="data row5 col2" >0.7686</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col3" class="data row5 col3" >0.5098</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col4" class="data row5 col4" >0.0438</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row5_col5" class="data row5 col5" >0.0394</td>
            </tr>
            <tr>
                        <th id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col0" class="data row6 col0" >0.1232</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col1" class="data row6 col1" >0.1964</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col2" class="data row6 col2" >0.1233</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col3" class="data row6 col3" >0.1724</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col4" class="data row6 col4" >0.0071</td>
                        <td id="T_eebd5827_5ef6_11eb_9b01_54e1ada8ef40row6_col5" class="data row6 col5" >0.0069</td>
            </tr>
    </tbody></table>


    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    


```python
evaluate_model(tuned_svm)
```


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
predict_model(tuned_svm)
```


<style  type="text/css" >
</style><table id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col0" class="data row0 col0" >Support Vector Regression</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col1" class="data row0 col1" >0.6390</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col2" class="data row0 col2" >0.5775</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col3" class="data row0 col3" >0.7600</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col4" class="data row0 col4" >0.5733</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col5" class="data row0 col5" >0.0447</td>
                        <td id="T_ef706c08_5ef6_11eb_a36c_54e1ada8ef40row0_col6" class="data row0 col6" >0.0404</td>
            </tr>
    </tbody></table>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MD</th>
      <th>GR</th>
      <th>RD</th>
      <th>DEN</th>
      <th>AC</th>
      <th>CNL</th>
      <th>C</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.641712</td>
      <td>1.003763</td>
      <td>0.418927</td>
      <td>0.901557</td>
      <td>0.842278</td>
      <td>0.768814</td>
      <td>14.910000</td>
      <td>15.631070</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.857525</td>
      <td>0.729330</td>
      <td>0.963617</td>
      <td>0.942158</td>
      <td>0.827880</td>
      <td>0.587210</td>
      <td>16.170000</td>
      <td>16.198327</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.575194</td>
      <td>0.930752</td>
      <td>0.039959</td>
      <td>0.928810</td>
      <td>0.881682</td>
      <td>0.427478</td>
      <td>16.639999</td>
      <td>17.236607</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.778647</td>
      <td>0.680355</td>
      <td>0.233072</td>
      <td>0.825918</td>
      <td>0.811381</td>
      <td>0.568453</td>
      <td>18.450001</td>
      <td>18.057534</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.646029</td>
      <td>0.758685</td>
      <td>0.462177</td>
      <td>0.789210</td>
      <td>1.021404</td>
      <td>0.752855</td>
      <td>14.860000</td>
      <td>16.172209</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.660101</td>
      <td>0.545505</td>
      <td>0.478695</td>
      <td>0.933259</td>
      <td>0.864255</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>16.862626</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.648719</td>
      <td>0.521168</td>
      <td>0.305116</td>
      <td>0.800334</td>
      <td>0.837680</td>
      <td>0.812915</td>
      <td>17.260000</td>
      <td>17.795333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.646676</td>
      <td>0.831446</td>
      <td>0.489261</td>
      <td>0.833704</td>
      <td>0.925579</td>
      <td>0.910972</td>
      <td>14.570000</td>
      <td>15.790109</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.644691</td>
      <td>0.439274</td>
      <td>0.921610</td>
      <td>0.779199</td>
      <td>0.982503</td>
      <td>0.921437</td>
      <td>16.650000</td>
      <td>16.773139</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.992806</td>
      <td>0.659747</td>
      <td>0.110600</td>
      <td>0.814238</td>
      <td>0.814287</td>
      <td>0.450711</td>
      <td>17.150000</td>
      <td>18.168087</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_svm = finalize_model(tuned_svm)
print(final_svm)
save_model(tuned_svm,'Final-svm-md')
```

    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    Transformation Pipeline and Model Succesfully Saved
    




    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[], ml_usecase='regression',
                                           numerical_features=[], target='C',
                                           time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_strategy='mean...
                     ('dummy', Dummify(target='C')),
                     ('fix_perfect', Remove_100(target='C')),
                     ('clean_names', Clean_Colum_Names()),
                     ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                     ('dfs', 'passthrough'), ('pca', 'passthrough'),
                     ['trained_model',
                      SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
                          gamma='scale', kernel='rbf', max_iter=-1, shrinking=True,
                          tol=0.001, verbose=False)]],
              verbose=False),
     'Final-svm-md.pkl')




```python
#svm.get_params().keys()
```


```python
help(plot_model)
```

    Help on function plot_model in module pycaret.regression:
    
    plot_model(estimator, plot: str = 'residuals', scale: float = 1, save: bool = False, fold: Union[int, Any, NoneType] = None, fit_kwargs: Union[dict, NoneType] = None, groups: Union[str, Any, NoneType] = None, use_train_data: bool = False, verbose: bool = True) -> str
        This function analyzes the performance of a trained model on holdout set. 
        It may require re-training the model in certain cases.
        
        
        Example
        --------
        >>> from pycaret.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> plot_model(lr, plot = 'residual')
        
        
        estimator: scikit-learn compatible object
            Trained model object
        
        
        plot: str, default = 'residual'
            List of available plots (ID - Name):
        
            * 'residuals' - Residuals Plot
            * 'error' - Prediction Error Plot
            * 'cooks' - Cooks Distance Plot
            * 'rfe' - Recursive Feat. Selection
            * 'learning' - Learning Curve
            * 'vc' - Validation Curve
            * 'manifold' - Manifold Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'tree' - Decision Tree
        
        
        scale: float, default = 1
            The resolution scale of the figure.
        
        
        save: bool, default = False
            When set to True, plot is saved in the current working directory.
        
        
        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy`` 
            parameter of the ``setup`` function is used. When an integer is passed, 
            it is interpreted as the 'n_splits' parameter of the CV generator in the 
            ``setup`` function.
        
        
        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.
        
        
        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as 
            the column name in the dataset containing group labels.
        
        
        use_train_data: bool, default = False
            When set to true, train data will be used for plots, instead
            of test data.
        
        
        verbose: bool, default = True
            When set to False, progress bar is not displayed.
        
        
        Returns:
            None
    
    

# GBDT


```python
dt=create_model('dt',fold=5);
tuned_dt= tune_model(dt,n_iter =100, choose_better=True, fold=5);
print(tuned_dt);
bagged_dt = ensemble_model(tuned_dt)
boosted_dt = ensemble_model(tuned_dt, method = 'Boosting',fold=5) 
```


<style  type="text/css" >
#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col0,#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col1,#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col2,#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col3,#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col4,#T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col0" class="data row0 col0" >0.8592</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col1" class="data row0 col1" >0.9175</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col2" class="data row0 col2" >0.9579</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col3" class="data row0 col3" >0.2287</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col4" class="data row0 col4" >0.0533</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row0_col5" class="data row0 col5" >0.0510</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col0" class="data row1 col0" >0.8483</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col1" class="data row1 col1" >1.0738</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col2" class="data row1 col2" >1.0363</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col3" class="data row1 col3" >0.3628</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col4" class="data row1 col4" >0.0614</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row1_col5" class="data row1 col5" >0.0543</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col0" class="data row2 col0" >0.6258</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col1" class="data row2 col1" >0.5224</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col2" class="data row2 col2" >0.7228</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col3" class="data row2 col3" >0.6855</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col4" class="data row2 col4" >0.0393</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row2_col5" class="data row2 col5" >0.0367</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col0" class="data row3 col0" >0.4667</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col1" class="data row3 col1" >0.4225</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col2" class="data row3 col2" >0.6500</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col3" class="data row3 col3" >0.3667</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col4" class="data row3 col4" >0.0358</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row3_col5" class="data row3 col5" >0.0271</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col0" class="data row4 col0" >0.4167</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col1" class="data row4 col1" >0.2738</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col2" class="data row4 col2" >0.5232</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col3" class="data row4 col3" >0.7973</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col4" class="data row4 col4" >0.0284</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row4_col5" class="data row4 col5" >0.0246</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col0" class="data row5 col0" >0.6433</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col1" class="data row5 col1" >0.6420</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col2" class="data row5 col2" >0.7780</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col3" class="data row5 col3" >0.4882</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col4" class="data row5 col4" >0.0436</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row5_col5" class="data row5 col5" >0.0387</td>
            </tr>
            <tr>
                        <th id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col0" class="data row6 col0" >0.1852</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col1" class="data row6 col1" >0.3035</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col2" class="data row6 col2" >0.1915</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col3" class="data row6 col3" >0.2155</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col4" class="data row6 col4" >0.0120</td>
                        <td id="T_f1a8f0b7_5ef6_11eb_88c7_54e1ada8ef40row6_col5" class="data row6 col5" >0.0121</td>
            </tr>
    </tbody></table>



```python
evaluate_model(boosted_dt) 
```


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
predict_model(boosted_dt) 
```


<style  type="text/css" >
</style><table id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col0" class="data row0 col0" >Decision Tree Regressor</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col1" class="data row0 col1" >0.7895</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col2" class="data row0 col2" >1.0655</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col3" class="data row0 col3" >1.0322</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col4" class="data row0 col4" >0.2127</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col5" class="data row0 col5" >0.0612</td>
                        <td id="T_f1d5b323_5ef6_11eb_9925_54e1ada8ef40row0_col6" class="data row0 col6" >0.0498</td>
            </tr>
    </tbody></table>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MD</th>
      <th>GR</th>
      <th>RD</th>
      <th>DEN</th>
      <th>AC</th>
      <th>CNL</th>
      <th>C</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.641712</td>
      <td>1.003763</td>
      <td>0.418927</td>
      <td>0.901557</td>
      <td>0.842278</td>
      <td>0.768814</td>
      <td>14.910000</td>
      <td>16.850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.857525</td>
      <td>0.729330</td>
      <td>0.963617</td>
      <td>0.942158</td>
      <td>0.827880</td>
      <td>0.587210</td>
      <td>16.170000</td>
      <td>16.850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.575194</td>
      <td>0.930752</td>
      <td>0.039959</td>
      <td>0.928810</td>
      <td>0.881682</td>
      <td>0.427478</td>
      <td>16.639999</td>
      <td>15.850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.778647</td>
      <td>0.680355</td>
      <td>0.233072</td>
      <td>0.825918</td>
      <td>0.811381</td>
      <td>0.568453</td>
      <td>18.450001</td>
      <td>18.380</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.646029</td>
      <td>0.758685</td>
      <td>0.462177</td>
      <td>0.789210</td>
      <td>1.021404</td>
      <td>0.752855</td>
      <td>14.860000</td>
      <td>14.810</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.660101</td>
      <td>0.545505</td>
      <td>0.478695</td>
      <td>0.933259</td>
      <td>0.864255</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>16.850</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.648719</td>
      <td>0.521168</td>
      <td>0.305116</td>
      <td>0.800334</td>
      <td>0.837680</td>
      <td>0.812915</td>
      <td>17.260000</td>
      <td>17.235</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.646676</td>
      <td>0.831446</td>
      <td>0.489261</td>
      <td>0.833704</td>
      <td>0.925579</td>
      <td>0.910972</td>
      <td>14.570000</td>
      <td>15.810</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.644691</td>
      <td>0.439274</td>
      <td>0.921610</td>
      <td>0.779199</td>
      <td>0.982503</td>
      <td>0.921437</td>
      <td>16.650000</td>
      <td>14.810</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.992806</td>
      <td>0.659747</td>
      <td>0.110600</td>
      <td>0.814238</td>
      <td>0.814287</td>
      <td>0.450711</td>
      <td>17.150000</td>
      <td>17.980</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_gbdt = finalize_model(boosted_dt)
print(final_gbdt)
save_model(boosted_dt,'Final-gbdt-md')
```

    AdaBoostRegressor(base_estimator=DecisionTreeRegressor(ccp_alpha=0.0,
                                                           criterion='mae',
                                                           max_depth=11,
                                                           max_features='log2',
                                                           max_leaf_nodes=None,
                                                           min_impurity_decrease=0.0002,
                                                           min_impurity_split=None,
                                                           min_samples_leaf=3,
                                                           min_samples_split=7,
                                                           min_weight_fraction_leaf=0.0,
                                                           presort='deprecated',
                                                           random_state=123,
                                                           splitter='best'),
                      learning_rate=1.0, loss='linear', n_estimators=10,
                      random_state=123)
    Transformation Pipeline and Model Succesfully Saved
    




    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[], ml_usecase='regression',
                                           numerical_features=[], target='C',
                                           time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_strategy='mean...
                      AdaBoostRegressor(base_estimator=DecisionTreeRegressor(ccp_alpha=0.0,
                                                                             criterion='mae',
                                                                             max_depth=11,
                                                                             max_features='log2',
                                                                             max_leaf_nodes=None,
                                                                             min_impurity_decrease=0.0002,
                                                                             min_impurity_split=None,
                                                                             min_samples_leaf=3,
                                                                             min_samples_split=7,
                                                                             min_weight_fraction_leaf=0.0,
                                                                             presort='deprecated',
                                                                             random_state=123,
                                                                             splitter='best'),
                                        learning_rate=1.0, loss='linear',
                                        n_estimators=10, random_state=123)]],
              verbose=False),
     'Final-gbdt-md.pkl')



# CatBoost


```python
catboost=create_model('catboost',fold=5);
print(catboost)
```


<style  type="text/css" >
#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col0,#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col1,#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col2,#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col3,#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col4,#T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col0" class="data row0 col0" >0.5620</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col1" class="data row0 col1" >0.4302</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col2" class="data row0 col2" >0.6559</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col3" class="data row0 col3" >0.6383</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col4" class="data row0 col4" >0.0360</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row0_col5" class="data row0 col5" >0.0327</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col0" class="data row1 col0" >0.5248</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col1" class="data row1 col1" >0.5750</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col2" class="data row1 col2" >0.7583</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col3" class="data row1 col3" >0.6588</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col4" class="data row1 col4" >0.0453</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row1_col5" class="data row1 col5" >0.0338</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col0" class="data row2 col0" >0.4849</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col1" class="data row2 col1" >0.2695</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col2" class="data row2 col2" >0.5192</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col3" class="data row2 col3" >0.8377</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col4" class="data row2 col4" >0.0291</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row2_col5" class="data row2 col5" >0.0288</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col0" class="data row3 col0" >0.5334</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col1" class="data row3 col1" >0.3864</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col2" class="data row3 col2" >0.6216</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col3" class="data row3 col3" >0.4208</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col4" class="data row3 col4" >0.0340</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row3_col5" class="data row3 col5" >0.0305</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col0" class="data row4 col0" >0.2345</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col1" class="data row4 col1" >0.0958</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col2" class="data row4 col2" >0.3096</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col3" class="data row4 col3" >0.9290</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col4" class="data row4 col4" >0.0168</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row4_col5" class="data row4 col5" >0.0137</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col0" class="data row5 col0" >0.4679</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col1" class="data row5 col1" >0.3514</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col2" class="data row5 col2" >0.5729</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col3" class="data row5 col3" >0.6969</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col4" class="data row5 col4" >0.0322</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row5_col5" class="data row5 col5" >0.0279</td>
            </tr>
            <tr>
                        <th id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col0" class="data row6 col0" >0.1193</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col1" class="data row6 col1" >0.1609</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col2" class="data row6 col2" >0.1522</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col3" class="data row6 col3" >0.1760</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col4" class="data row6 col4" >0.0093</td>
                        <td id="T_0ee3bf5e_5ef7_11eb_a115_54e1ada8ef40row6_col5" class="data row6 col5" >0.0073</td>
            </tr>
    </tbody></table>


    <catboost.core.CatBoostRegressor object at 0x000001CD4BA45E20>
    


```python
tuned_catboost= tune_model(catboost,n_iter =100, choose_better=True,fold=5);
```


<style  type="text/css" >
#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col0,#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col1,#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col2,#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col3,#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col4,#T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col0" class="data row0 col0" >0.6284</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col1" class="data row0 col1" >0.5418</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col2" class="data row0 col2" >0.7360</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col3" class="data row0 col3" >0.5446</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col4" class="data row0 col4" >0.0404</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row0_col5" class="data row0 col5" >0.0366</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col0" class="data row1 col0" >0.4955</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col1" class="data row1 col1" >0.4893</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col2" class="data row1 col2" >0.6995</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col3" class="data row1 col3" >0.7097</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col4" class="data row1 col4" >0.0419</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row1_col5" class="data row1 col5" >0.0319</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col0" class="data row2 col0" >0.4828</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col1" class="data row2 col1" >0.2987</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col2" class="data row2 col2" >0.5465</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col3" class="data row2 col3" >0.8202</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col4" class="data row2 col4" >0.0320</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row2_col5" class="data row2 col5" >0.0294</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col0" class="data row3 col0" >0.5274</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col1" class="data row3 col1" >0.3515</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col2" class="data row3 col2" >0.5929</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col3" class="data row3 col3" >0.4731</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col4" class="data row3 col4" >0.0328</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row3_col5" class="data row3 col5" >0.0302</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col0" class="data row4 col0" >0.1927</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col1" class="data row4 col1" >0.0768</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col2" class="data row4 col2" >0.2771</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col3" class="data row4 col3" >0.9431</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col4" class="data row4 col4" >0.0153</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row4_col5" class="data row4 col5" >0.0114</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col0" class="data row5 col0" >0.4654</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col1" class="data row5 col1" >0.3516</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col2" class="data row5 col2" >0.5704</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col3" class="data row5 col3" >0.6981</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col4" class="data row5 col4" >0.0325</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row5_col5" class="data row5 col5" >0.0279</td>
            </tr>
            <tr>
                        <th id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col0" class="data row6 col0" >0.1456</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col1" class="data row6 col1" >0.1634</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col2" class="data row6 col2" >0.1620</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col3" class="data row6 col3" >0.1728</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col4" class="data row6 col4" >0.0094</td>
                        <td id="T_453e4a2a_5ef7_11eb_b849_54e1ada8ef40row6_col5" class="data row6 col5" >0.0086</td>
            </tr>
    </tbody></table>



```python
#interpret_model(tuned_catboost,plot = 'reason')
```


```python
final_catboost = finalize_model(tuned_catboost)
print(final_catboost)
#save_model(final_catboost,'Final-catboost-md') 
```

    <catboost.core.CatBoostRegressor object at 0x000001CD4BA140D0>
    


```python
blender_specific2 = blend_models(estimator_list = [tuned_et,tuned_catboost],fold=5) 
```


<style  type="text/css" >
#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col0,#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col1,#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col2,#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col3,#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col4,#T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col0" class="data row0 col0" >0.6443</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col1" class="data row0 col1" >0.5263</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col2" class="data row0 col2" >0.7255</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col3" class="data row0 col3" >0.5576</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col4" class="data row0 col4" >0.0398</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row0_col5" class="data row0 col5" >0.0374</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col0" class="data row1 col0" >0.5625</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col1" class="data row1 col1" >0.5806</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col2" class="data row1 col2" >0.7620</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col3" class="data row1 col3" >0.6555</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col4" class="data row1 col4" >0.0458</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row1_col5" class="data row1 col5" >0.0364</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col0" class="data row2 col0" >0.4632</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col1" class="data row2 col1" >0.3015</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col2" class="data row2 col2" >0.5491</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col3" class="data row2 col3" >0.8185</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col4" class="data row2 col4" >0.0326</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row2_col5" class="data row2 col5" >0.0284</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col0" class="data row3 col0" >0.5825</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col1" class="data row3 col1" >0.4181</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col2" class="data row3 col2" >0.6466</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col3" class="data row3 col3" >0.3733</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col4" class="data row3 col4" >0.0359</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row3_col5" class="data row3 col5" >0.0334</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col0" class="data row4 col0" >0.3128</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col1" class="data row4 col1" >0.1605</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col2" class="data row4 col2" >0.4006</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col3" class="data row4 col3" >0.8811</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col4" class="data row4 col4" >0.0230</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row4_col5" class="data row4 col5" >0.0189</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col0" class="data row5 col0" >0.5131</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col1" class="data row5 col1" >0.3974</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col2" class="data row5 col2" >0.6168</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col3" class="data row5 col3" >0.6572</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col4" class="data row5 col4" >0.0354</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row5_col5" class="data row5 col5" >0.0309</td>
            </tr>
            <tr>
                        <th id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col0" class="data row6 col0" >0.1158</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col1" class="data row6 col1" >0.1522</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col2" class="data row6 col2" >0.1304</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col3" class="data row6 col3" >0.1826</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col4" class="data row6 col4" >0.0076</td>
                        <td id="T_49ae1d1d_5ef7_11eb_a8da_54e1ada8ef40row6_col5" class="data row6 col5" >0.0068</td>
            </tr>
    </tbody></table>



```python
evaluate_model(blender_specific2) 
```


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
predict_model(blender_specific2)
```


<style  type="text/css" >
</style><table id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col0" class="data row0 col0" >Voting Regressor</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col1" class="data row0 col1" >0.6132</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col2" class="data row0 col2" >0.7431</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col3" class="data row0 col3" >0.8620</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col4" class="data row0 col4" >0.4509</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col5" class="data row0 col5" >0.0517</td>
                        <td id="T_49e56130_5ef7_11eb_a232_54e1ada8ef40row0_col6" class="data row0 col6" >0.0400</td>
            </tr>
    </tbody></table>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MD</th>
      <th>GR</th>
      <th>RD</th>
      <th>DEN</th>
      <th>AC</th>
      <th>CNL</th>
      <th>C</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.641712</td>
      <td>1.003763</td>
      <td>0.418927</td>
      <td>0.901557</td>
      <td>0.842278</td>
      <td>0.768814</td>
      <td>14.910000</td>
      <td>16.324407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.857525</td>
      <td>0.729330</td>
      <td>0.963617</td>
      <td>0.942158</td>
      <td>0.827880</td>
      <td>0.587210</td>
      <td>16.170000</td>
      <td>16.494188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.575194</td>
      <td>0.930752</td>
      <td>0.039959</td>
      <td>0.928810</td>
      <td>0.881682</td>
      <td>0.427478</td>
      <td>16.639999</td>
      <td>16.693497</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.778647</td>
      <td>0.680355</td>
      <td>0.233072</td>
      <td>0.825918</td>
      <td>0.811381</td>
      <td>0.568453</td>
      <td>18.450001</td>
      <td>18.163569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.646029</td>
      <td>0.758685</td>
      <td>0.462177</td>
      <td>0.789210</td>
      <td>1.021404</td>
      <td>0.752855</td>
      <td>14.860000</td>
      <td>16.383713</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.660101</td>
      <td>0.545505</td>
      <td>0.478695</td>
      <td>0.933259</td>
      <td>0.864255</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>16.645711</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.648719</td>
      <td>0.521168</td>
      <td>0.305116</td>
      <td>0.800334</td>
      <td>0.837680</td>
      <td>0.812915</td>
      <td>17.260000</td>
      <td>17.292471</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.646676</td>
      <td>0.831446</td>
      <td>0.489261</td>
      <td>0.833704</td>
      <td>0.925579</td>
      <td>0.910972</td>
      <td>14.570000</td>
      <td>16.118544</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.644691</td>
      <td>0.439274</td>
      <td>0.921610</td>
      <td>0.779199</td>
      <td>0.982503</td>
      <td>0.921437</td>
      <td>16.650000</td>
      <td>16.688973</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.992806</td>
      <td>0.659747</td>
      <td>0.110600</td>
      <td>0.814238</td>
      <td>0.814287</td>
      <td>0.450711</td>
      <td>17.150000</td>
      <td>17.833757</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_gbrcb = finalize_model(blender_specific2)
print(final_gbrcb)
save_model(blender_specific2,'Final-catboost-md') 
```

    TunableVotingRegressor(estimators=[('et',
                                        ExtraTreesRegressor(bootstrap=False,
                                                            ccp_alpha=0.0,
                                                            criterion='mse',
                                                            max_depth=8,
                                                            max_features='sqrt',
                                                            max_leaf_nodes=None,
                                                            max_samples=None,
                                                            min_impurity_decrease=0.0001,
                                                            min_impurity_split=None,
                                                            min_samples_leaf=2,
                                                            min_samples_split=5,
                                                            min_weight_fraction_leaf=0.0,
                                                            n_estimators=240,
                                                            n_jobs=-1,
                                                            oob_score=False,
                                                            random_state=123,
                                                            verbose=0,
                                                            warm_start=False)),
                                       ('catboost',
                                        <catboost.core.CatBoostRegressor object at 0x000001CD4BB536D0>)],
                           n_jobs=-1, verbose=False, weight_0=1, weight_1=1,
                           weights=[1, 1])
    Transformation Pipeline and Model Succesfully Saved
    




    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[], ml_usecase='regression',
                                           numerical_features=[], target='C',
                                           time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_strategy='mean...
                                                                              min_impurity_decrease=0.0001,
                                                                              min_impurity_split=None,
                                                                              min_samples_leaf=2,
                                                                              min_samples_split=5,
                                                                              min_weight_fraction_leaf=0.0,
                                                                              n_estimators=240,
                                                                              n_jobs=-1,
                                                                              oob_score=False,
                                                                              random_state=123,
                                                                              verbose=0,
                                                                              warm_start=False)),
                                                         ('catboost',
                                                          <catboost.core.CatBoostRegressor object at 0x000001CD4CB62970>)],
                                             n_jobs=-1, verbose=False, weight_0=1,
                                             weight_1=1, weights=[1, 1])]],
              verbose=False),
     'Final-catboost-md.pkl')



# Predict


```python
data_unseen=pd.read_excel("your prediction test set", sheet_name='pymd')
```


```python
saved_final_lightgbm = load_model('chose a saved model')
new_prediction = predict_model(saved_final_lightgbm, data=data_unseen)
new_prediction
```

    Transformation Pipeline and Model Successfully Loaded
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MD</th>
      <th>GR</th>
      <th>RD</th>
      <th>DEN</th>
      <th>AC</th>
      <th>CNL</th>
      <th>C</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>448.75</td>
      <td>59.257</td>
      <td>126.334</td>
      <td>1.7980</td>
      <td>468.939</td>
      <td>69.3440</td>
      <td>14.41</td>
      <td>14.827212</td>
    </tr>
    <tr>
      <th>1</th>
      <td>449.44</td>
      <td>49.708</td>
      <td>140.488</td>
      <td>1.4990</td>
      <td>434.040</td>
      <td>71.6370</td>
      <td>14.57</td>
      <td>16.118544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>445.29</td>
      <td>53.567</td>
      <td>186.942</td>
      <td>1.6890</td>
      <td>437.061</td>
      <td>65.3320</td>
      <td>14.57</td>
      <td>14.915502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>445.69</td>
      <td>49.385</td>
      <td>141.334</td>
      <td>1.7620</td>
      <td>422.793</td>
      <td>65.1980</td>
      <td>14.66</td>
      <td>14.999235</td>
    </tr>
    <tr>
      <th>4</th>
      <td>447.99</td>
      <td>50.230</td>
      <td>134.283</td>
      <td>1.5490</td>
      <td>417.491</td>
      <td>61.9360</td>
      <td>14.81</td>
      <td>15.360285</td>
    </tr>
    <tr>
      <th>5</th>
      <td>448.99</td>
      <td>45.358</td>
      <td>132.711</td>
      <td>1.4190</td>
      <td>478.976</td>
      <td>59.2030</td>
      <td>14.86</td>
      <td>16.383713</td>
    </tr>
    <tr>
      <th>6</th>
      <td>445.99</td>
      <td>60.010</td>
      <td>120.292</td>
      <td>1.6210</td>
      <td>394.977</td>
      <td>60.4580</td>
      <td>14.91</td>
      <td>16.324407</td>
    </tr>
    <tr>
      <th>7</th>
      <td>597.66</td>
      <td>42.015</td>
      <td>181.055</td>
      <td>1.6620</td>
      <td>422.045</td>
      <td>47.1230</td>
      <td>15.81</td>
      <td>16.008908</td>
    </tr>
    <tr>
      <th>8</th>
      <td>591.81</td>
      <td>55.963</td>
      <td>61.885</td>
      <td>1.7510</td>
      <td>435.674</td>
      <td>42.3400</td>
      <td>15.85</td>
      <td>16.015930</td>
    </tr>
    <tr>
      <th>9</th>
      <td>598.10</td>
      <td>42.296</td>
      <td>195.644</td>
      <td>1.5900</td>
      <td>400.829</td>
      <td>44.7920</td>
      <td>16.17</td>
      <td>16.353359</td>
    </tr>
    <tr>
      <th>10</th>
      <td>595.98</td>
      <td>43.603</td>
      <td>276.696</td>
      <td>1.6940</td>
      <td>388.225</td>
      <td>46.1770</td>
      <td>16.17</td>
      <td>16.494188</td>
    </tr>
    <tr>
      <th>11</th>
      <td>458.77</td>
      <td>32.613</td>
      <td>137.454</td>
      <td>1.6780</td>
      <td>405.283</td>
      <td>70.7390</td>
      <td>16.42</td>
      <td>16.645711</td>
    </tr>
    <tr>
      <th>12</th>
      <td>399.76</td>
      <td>55.645</td>
      <td>11.474</td>
      <td>1.6700</td>
      <td>413.455</td>
      <td>33.6160</td>
      <td>16.64</td>
      <td>16.693497</td>
    </tr>
    <tr>
      <th>13</th>
      <td>448.06</td>
      <td>26.262</td>
      <td>264.634</td>
      <td>1.4010</td>
      <td>460.734</td>
      <td>72.4600</td>
      <td>16.65</td>
      <td>16.688973</td>
    </tr>
    <tr>
      <th>14</th>
      <td>598.46</td>
      <td>32.835</td>
      <td>287.143</td>
      <td>1.4700</td>
      <td>406.958</td>
      <td>49.0680</td>
      <td>16.85</td>
      <td>16.945806</td>
    </tr>
    <tr>
      <th>15</th>
      <td>451.97</td>
      <td>34.896</td>
      <td>141.935</td>
      <td>1.7250</td>
      <td>415.225</td>
      <td>71.0750</td>
      <td>16.85</td>
      <td>16.455688</td>
    </tr>
    <tr>
      <th>16</th>
      <td>563.53</td>
      <td>59.785</td>
      <td>10.833</td>
      <td>1.6140</td>
      <td>374.273</td>
      <td>61.4610</td>
      <td>16.93</td>
      <td>17.006454</td>
    </tr>
    <tr>
      <th>17</th>
      <td>395.16</td>
      <td>47.975</td>
      <td>27.665</td>
      <td>1.5940</td>
      <td>395.635</td>
      <td>35.3850</td>
      <td>16.95</td>
      <td>16.990730</td>
    </tr>
    <tr>
      <th>18</th>
      <td>681.20</td>
      <td>43.319</td>
      <td>85.641</td>
      <td>1.5310</td>
      <td>392.306</td>
      <td>63.9810</td>
      <td>17.06</td>
      <td>17.096106</td>
    </tr>
    <tr>
      <th>19</th>
      <td>682.46</td>
      <td>38.032</td>
      <td>59.641</td>
      <td>1.4560</td>
      <td>409.366</td>
      <td>62.6980</td>
      <td>17.13</td>
      <td>17.202805</td>
    </tr>
    <tr>
      <th>20</th>
      <td>690.00</td>
      <td>39.443</td>
      <td>31.758</td>
      <td>1.4640</td>
      <td>381.851</td>
      <td>35.4430</td>
      <td>17.15</td>
      <td>17.833757</td>
    </tr>
    <tr>
      <th>21</th>
      <td>681.86</td>
      <td>40.175</td>
      <td>72.481</td>
      <td>1.4350</td>
      <td>400.396</td>
      <td>63.3050</td>
      <td>17.15</td>
      <td>17.197687</td>
    </tr>
    <tr>
      <th>22</th>
      <td>450.06</td>
      <td>49.554</td>
      <td>77.954</td>
      <td>1.4710</td>
      <td>402.261</td>
      <td>78.6380</td>
      <td>17.21</td>
      <td>17.043108</td>
    </tr>
    <tr>
      <th>23</th>
      <td>450.86</td>
      <td>31.158</td>
      <td>87.612</td>
      <td>1.4390</td>
      <td>392.821</td>
      <td>63.9260</td>
      <td>17.26</td>
      <td>17.292471</td>
    </tr>
    <tr>
      <th>24</th>
      <td>449.94</td>
      <td>33.152</td>
      <td>134.821</td>
      <td>1.4640</td>
      <td>394.501</td>
      <td>67.7990</td>
      <td>17.26</td>
      <td>17.128083</td>
    </tr>
    <tr>
      <th>25</th>
      <td>693.73</td>
      <td>43.390</td>
      <td>122.136</td>
      <td>1.4990</td>
      <td>352.824</td>
      <td>40.0530</td>
      <td>17.46</td>
      <td>17.495861</td>
    </tr>
    <tr>
      <th>26</th>
      <td>692.60</td>
      <td>40.296</td>
      <td>53.823</td>
      <td>1.4880</td>
      <td>377.837</td>
      <td>37.3310</td>
      <td>17.55</td>
      <td>17.688729</td>
    </tr>
    <tr>
      <th>27</th>
      <td>562.67</td>
      <td>55.567</td>
      <td>22.648</td>
      <td>1.5670</td>
      <td>374.069</td>
      <td>53.6670</td>
      <td>17.61</td>
      <td>17.514944</td>
    </tr>
    <tr>
      <th>28</th>
      <td>695.00</td>
      <td>43.682</td>
      <td>20.852</td>
      <td>1.4770</td>
      <td>352.033</td>
      <td>40.1130</td>
      <td>17.65</td>
      <td>17.694327</td>
    </tr>
    <tr>
      <th>29</th>
      <td>560.80</td>
      <td>55.151</td>
      <td>44.874</td>
      <td>1.4525</td>
      <td>369.438</td>
      <td>56.5270</td>
      <td>17.87</td>
      <td>17.769101</td>
    </tr>
    <tr>
      <th>30</th>
      <td>691.89</td>
      <td>38.683</td>
      <td>51.671</td>
      <td>1.4280</td>
      <td>362.710</td>
      <td>35.8370</td>
      <td>17.98</td>
      <td>17.940408</td>
    </tr>
    <tr>
      <th>31</th>
      <td>562.20</td>
      <td>51.731</td>
      <td>11.697</td>
      <td>1.5270</td>
      <td>378.496</td>
      <td>43.7620</td>
      <td>18.03</td>
      <td>17.847262</td>
    </tr>
    <tr>
      <th>32</th>
      <td>561.30</td>
      <td>52.781</td>
      <td>21.387</td>
      <td>1.4630</td>
      <td>355.365</td>
      <td>51.1615</td>
      <td>18.13</td>
      <td>17.948228</td>
    </tr>
    <tr>
      <th>33</th>
      <td>693.18</td>
      <td>41.060</td>
      <td>45.010</td>
      <td>1.4760</td>
      <td>419.446</td>
      <td>37.8290</td>
      <td>18.19</td>
      <td>17.922963</td>
    </tr>
    <tr>
      <th>34</th>
      <td>692.80</td>
      <td>37.858</td>
      <td>41.294</td>
      <td>1.5120</td>
      <td>409.452</td>
      <td>37.2540</td>
      <td>18.31</td>
      <td>18.050467</td>
    </tr>
    <tr>
      <th>35</th>
      <td>690.88</td>
      <td>36.340</td>
      <td>61.893</td>
      <td>1.5020</td>
      <td>351.382</td>
      <td>33.2920</td>
      <td>18.37</td>
      <td>18.135735</td>
    </tr>
    <tr>
      <th>36</th>
      <td>542.86</td>
      <td>42.029</td>
      <td>51.798</td>
      <td>1.4860</td>
      <td>374.708</td>
      <td>42.6910</td>
      <td>18.39</td>
      <td>18.188518</td>
    </tr>
    <tr>
      <th>37</th>
      <td>542.50</td>
      <td>39.456</td>
      <td>64.961</td>
      <td>1.4390</td>
      <td>369.073</td>
      <td>43.3610</td>
      <td>18.41</td>
      <td>18.238239</td>
    </tr>
    <tr>
      <th>38</th>
      <td>541.16</td>
      <td>40.675</td>
      <td>66.925</td>
      <td>1.4850</td>
      <td>380.488</td>
      <td>44.7020</td>
      <td>18.45</td>
      <td>18.163569</td>
    </tr>
    <tr>
      <th>39</th>
      <td>542.15</td>
      <td>39.324</td>
      <td>82.187</td>
      <td>1.4750</td>
      <td>386.268</td>
      <td>46.7130</td>
      <td>18.58</td>
      <td>18.224483</td>
    </tr>
  </tbody>
</table>
</div>


