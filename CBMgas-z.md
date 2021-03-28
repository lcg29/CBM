#Z as deep parameter


```python
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
from pycaret.regression import *
```


```python
dataset=pd.read_excel(train set, sheet_name='pyt')
```


```python
print(dataset.columns)
dataset.shape
```

    Index(['T', 'GR', 'RD', 'DEN', 'AC', 'CNL', 'C'], dtype='object')
    




    (40, 7)




```python
exp_regt = setup(data = dataset, target = 'C', train_size = 0.75,session_id=111 ,normalize = True, normalize_method='maxabs' ) 
```


<style  type="text/css" >
#T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row27_col1{
            background-color:  lightgreen;
        }</style><table id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row0_col1" class="data row0 col1" >111</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row1_col1" class="data row1 col1" >C</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row2_col0" class="data row2 col0" >Original Data</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row2_col1" class="data row2 col1" >(40, 7)</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row3_col0" class="data row3 col0" >Missing Values</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row3_col1" class="data row3 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row4_col0" class="data row4 col0" >Numeric Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row4_col1" class="data row4 col1" >6</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row5_col0" class="data row5 col0" >Categorical Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row5_col1" class="data row5 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row6_col0" class="data row6 col0" >Ordinal Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row6_col1" class="data row6 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row7_col0" class="data row7 col0" >High Cardinality Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row8_col0" class="data row8 col0" >High Cardinality Method</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row8_col1" class="data row8 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row9_col0" class="data row9 col0" >Transformed Train Set</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row9_col1" class="data row9 col1" >(30, 6)</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row10_col0" class="data row10 col0" >Transformed Test Set</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row10_col1" class="data row10 col1" >(10, 6)</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row11_col0" class="data row11 col0" >Shuffle Train-Test</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row11_col1" class="data row11 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row12_col0" class="data row12 col0" >Stratify Train-Test</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row12_col1" class="data row12 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row13_col0" class="data row13 col0" >Fold Generator</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row13_col1" class="data row13 col1" >KFold</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row14_col0" class="data row14 col0" >Fold Number</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row14_col1" class="data row14 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row15_col0" class="data row15 col0" >CPU Jobs</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row15_col1" class="data row15 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row16_col0" class="data row16 col0" >Use GPU</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row16_col1" class="data row16 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row17_col0" class="data row17 col0" >Log Experiment</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row18_col0" class="data row18 col0" >Experiment Name</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row18_col1" class="data row18 col1" >reg-default-name</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row19_col0" class="data row19 col0" >USI</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row19_col1" class="data row19 col1" >1874</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row20_col0" class="data row20 col0" >Imputation Type</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row20_col1" class="data row20 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row21_col0" class="data row21 col0" >Iterative Imputation Iteration</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row22_col0" class="data row22 col0" >Numeric Imputer</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row22_col1" class="data row22 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row23_col0" class="data row23 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row24_col0" class="data row24 col0" >Categorical Imputer</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row24_col1" class="data row24 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row25_col0" class="data row25 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row26_col0" class="data row26 col0" >Unknown Categoricals Handling</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row26_col1" class="data row26 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row27_col0" class="data row27 col0" >Normalize</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row27_col1" class="data row27 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row28_col0" class="data row28 col0" >Normalize Method</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row28_col1" class="data row28 col1" >maxabs</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row29_col0" class="data row29 col0" >Transformation</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row30_col0" class="data row30 col0" >Transformation Method</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row31_col0" class="data row31 col0" >PCA</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row32_col0" class="data row32 col0" >PCA Method</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row33_col0" class="data row33 col0" >PCA Components</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row34_col0" class="data row34 col0" >Ignore Low Variance</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row35_col0" class="data row35 col0" >Combine Rare Levels</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row35_col1" class="data row35 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row36_col0" class="data row36 col0" >Rare Level Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row36_col1" class="data row36 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row37_col0" class="data row37 col0" >Numeric Binning</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row38_col0" class="data row38 col0" >Remove Outliers</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row38_col1" class="data row38 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row39_col0" class="data row39 col0" >Outliers Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row39_col1" class="data row39 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row40_col0" class="data row40 col0" >Remove Multicollinearity</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row41_col0" class="data row41 col0" >Multicollinearity Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row42_col0" class="data row42 col0" >Clustering</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row43_col0" class="data row43 col0" >Clustering Iteration</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row44_col0" class="data row44 col0" >Polynomial Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row44_col1" class="data row44 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row45_col0" class="data row45 col0" >Polynomial Degree</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row45_col1" class="data row45 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row46_col0" class="data row46 col0" >Trignometry Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row46_col1" class="data row46 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row47_col0" class="data row47 col0" >Polynomial Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row47_col1" class="data row47 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row48_col0" class="data row48 col0" >Group Features</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row48_col1" class="data row48 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row49_col0" class="data row49 col0" >Feature Selection</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row49_col1" class="data row49 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row50_col0" class="data row50 col0" >Features Selection Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row50_col1" class="data row50 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row51_col0" class="data row51 col0" >Feature Interaction</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row52_col0" class="data row52 col0" >Feature Ratio</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row52_col1" class="data row52 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row53_col0" class="data row53 col0" >Interaction Threshold</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row53_col1" class="data row53 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row54_col0" class="data row54 col0" >Transform Target</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row54_col1" class="data row54 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row55_col0" class="data row55 col0" >Transform Target Method</td>
                        <td id="T_78807cbf_5ef5_11eb_b853_54e1ada8ef40row55_col1" class="data row55 col1" >box-cox</td>
            </tr>
    </tbody></table>


# SVM


```python
svm=create_model('svm',fold=5);
```


<style  type="text/css" >
#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col0,#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col1,#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col2,#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col3,#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col4,#T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col0" class="data row0 col0" >0.3693</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col1" class="data row0 col1" >0.1573</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col2" class="data row0 col2" >0.3966</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col3" class="data row0 col3" >0.5054</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col4" class="data row0 col4" >0.0214</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row0_col5" class="data row0 col5" >0.0208</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col0" class="data row1 col0" >0.5274</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col1" class="data row1 col1" >0.3900</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col2" class="data row1 col2" >0.6245</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col3" class="data row1 col3" >0.7872</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col4" class="data row1 col4" >0.0375</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row1_col5" class="data row1 col5" >0.0337</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col0" class="data row2 col0" >0.5811</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col1" class="data row2 col1" >0.3919</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col2" class="data row2 col2" >0.6260</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col3" class="data row2 col3" >0.6904</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col4" class="data row2 col4" >0.0359</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row2_col5" class="data row2 col5" >0.0354</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col0" class="data row3 col0" >0.3790</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col1" class="data row3 col1" >0.2158</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col2" class="data row3 col2" >0.4645</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col3" class="data row3 col3" >0.6441</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col4" class="data row3 col4" >0.0253</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row3_col5" class="data row3 col5" >0.0223</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col0" class="data row4 col0" >0.6166</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col1" class="data row4 col1" >0.4588</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col2" class="data row4 col2" >0.6773</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col3" class="data row4 col3" >0.7350</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col4" class="data row4 col4" >0.0379</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row4_col5" class="data row4 col5" >0.0368</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col0" class="data row5 col0" >0.4947</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col1" class="data row5 col1" >0.3228</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col2" class="data row5 col2" >0.5578</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col3" class="data row5 col3" >0.6724</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col4" class="data row5 col4" >0.0316</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row5_col5" class="data row5 col5" >0.0298</td>
            </tr>
            <tr>
                        <th id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col0" class="data row6 col0" >0.1025</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col1" class="data row6 col1" >0.1154</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col2" class="data row6 col2" >0.1078</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col3" class="data row6 col3" >0.0960</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col4" class="data row6 col4" >0.0069</td>
                        <td id="T_7cf847a7_5ef5_11eb_9e17_54e1ada8ef40row6_col5" class="data row6 col5" >0.0068</td>
            </tr>
    </tbody></table>



```python
#predict_model(svm)
```


```python
#evaluate_model(svm)
```


```python
#plot_model(svm) 
```


```python
tuned_svm = tune_model(svm,n_iter =100, choose_better=True, fold=5)
print(tuned_svm)
```


<style  type="text/css" >
#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col0,#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col1,#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col2,#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col3,#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col4,#T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col0" class="data row0 col0" >0.7775</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col1" class="data row0 col1" >0.6293</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col2" class="data row0 col2" >0.7933</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col3" class="data row0 col3" >-0.9786</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col4" class="data row0 col4" >0.0427</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row0_col5" class="data row0 col5" >0.0433</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col0" class="data row1 col0" >0.5965</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col1" class="data row1 col1" >0.5095</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col2" class="data row1 col2" >0.7138</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col3" class="data row1 col3" >0.7220</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col4" class="data row1 col4" >0.0408</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row1_col5" class="data row1 col5" >0.0364</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col0" class="data row2 col0" >0.6764</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col1" class="data row2 col1" >0.6047</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col2" class="data row2 col2" >0.7776</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col3" class="data row2 col3" >0.5223</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col4" class="data row2 col4" >0.0449</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row2_col5" class="data row2 col5" >0.0410</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col0" class="data row3 col0" >0.2904</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col1" class="data row3 col1" >0.1381</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col2" class="data row3 col2" >0.3716</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col3" class="data row3 col3" >0.7723</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col4" class="data row3 col4" >0.0202</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row3_col5" class="data row3 col5" >0.0168</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col0" class="data row4 col0" >0.7233</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col1" class="data row4 col1" >0.7861</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col2" class="data row4 col2" >0.8867</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col3" class="data row4 col3" >0.5460</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col4" class="data row4 col4" >0.0495</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row4_col5" class="data row4 col5" >0.0424</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col0" class="data row5 col0" >0.6128</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col1" class="data row5 col1" >0.5336</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col2" class="data row5 col2" >0.7086</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col3" class="data row5 col3" >0.3168</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col4" class="data row5 col4" >0.0396</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row5_col5" class="data row5 col5" >0.0360</td>
            </tr>
            <tr>
                        <th id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col0" class="data row6 col0" >0.1718</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col1" class="data row6 col1" >0.2168</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col2" class="data row6 col2" >0.1773</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col3" class="data row6 col3" >0.6549</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col4" class="data row6 col4" >0.0101</td>
                        <td id="T_8093058f_5ef5_11eb_a726_54e1ada8ef40row6_col5" class="data row6 col5" >0.0099</td>
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
</style><table id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col0" class="data row0 col0" >Support Vector Regression</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col1" class="data row0 col1" >0.3688</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col2" class="data row0 col2" >0.1656</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col3" class="data row0 col3" >0.4069</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col4" class="data row0 col4" >0.9004</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col5" class="data row0 col5" >0.0236</td>
                        <td id="T_8130254f_5ef5_11eb_be96_54e1ada8ef40row0_col6" class="data row0 col6" >0.0229</td>
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
      <th>T</th>
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
      <td>0.975670</td>
      <td>0.987452</td>
      <td>0.439969</td>
      <td>1.026842</td>
      <td>0.979045</td>
      <td>0.881813</td>
      <td>14.410000</td>
      <td>14.973567</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.892342</td>
      <td>0.704816</td>
      <td>0.681347</td>
      <td>0.908053</td>
      <td>0.836846</td>
      <td>0.569597</td>
      <td>16.170000</td>
      <td>15.810190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.076577</td>
      <td>0.644609</td>
      <td>0.179949</td>
      <td>0.815534</td>
      <td>0.757261</td>
      <td>0.455721</td>
      <td>17.980000</td>
      <td>18.153417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.098446</td>
      <td>0.727912</td>
      <td>0.072619</td>
      <td>0.843518</td>
      <td>0.734970</td>
      <td>0.510097</td>
      <td>17.650000</td>
      <td>18.087123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.081570</td>
      <td>0.671488</td>
      <td>0.187443</td>
      <td>0.849800</td>
      <td>0.788843</td>
      <td>0.474720</td>
      <td>17.549999</td>
      <td>18.116870</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.173195</td>
      <td>0.543459</td>
      <td>0.478695</td>
      <td>0.958309</td>
      <td>0.846145</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>16.977741</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.542648</td>
      <td>0.925962</td>
      <td>0.078874</td>
      <td>0.894917</td>
      <td>0.780976</td>
      <td>0.682456</td>
      <td>17.610001</td>
      <td>17.905601</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.221011</td>
      <td>0.581503</td>
      <td>0.494301</td>
      <td>0.985151</td>
      <td>0.866901</td>
      <td>0.903825</td>
      <td>16.850000</td>
      <td>16.837587</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.997187</td>
      <td>0.822946</td>
      <td>0.492208</td>
      <td>1.006282</td>
      <td>0.882702</td>
      <td>0.829090</td>
      <td>14.660000</td>
      <td>15.079806</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.981014</td>
      <td>0.837027</td>
      <td>0.467652</td>
      <td>0.884637</td>
      <td>0.871632</td>
      <td>0.787609</td>
      <td>14.810000</td>
      <td>15.111844</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_svm = finalize_model(tuned_svm)
print(final_svm)
save_model(tuned_svm,'Final-svm-t')
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
     'Final-svm-t.pkl')




```python
#svm.get_params().keys()
```


```python

```

# GBDT


```python
dt=create_model('dt',fold=5);
tuned_dt= tune_model(dt,n_iter =100, choose_better=True, fold=5 );
```


<style  type="text/css" >
#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col0,#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col1,#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col2,#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col3,#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col4,#T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col0" class="data row0 col0" >0.1321</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col1" class="data row0 col1" >0.0268</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col2" class="data row0 col2" >0.1638</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col3" class="data row0 col3" >0.9156</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col4" class="data row0 col4" >0.0088</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row0_col5" class="data row0 col5" >0.0075</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col0" class="data row1 col0" >0.7633</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col1" class="data row1 col1" >0.8105</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col2" class="data row1 col2" >0.9003</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col3" class="data row1 col3" >0.5577</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col4" class="data row1 col4" >0.0494</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row1_col5" class="data row1 col5" >0.0457</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col0" class="data row2 col0" >0.2474</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col1" class="data row2 col1" >0.1519</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col2" class="data row2 col2" >0.3897</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col3" class="data row2 col3" >0.8800</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col4" class="data row2 col4" >0.0225</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row2_col5" class="data row2 col5" >0.0149</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col0" class="data row3 col0" >0.7550</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col1" class="data row3 col1" >0.7242</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col2" class="data row3 col2" >0.8510</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col3" class="data row3 col3" >-0.1944</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col4" class="data row3 col4" >0.0472</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row3_col5" class="data row3 col5" >0.0450</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col0" class="data row4 col0" >0.4761</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col1" class="data row4 col1" >0.5242</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col2" class="data row4 col2" >0.7240</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col3" class="data row4 col3" >0.6973</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col4" class="data row4 col4" >0.0423</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row4_col5" class="data row4 col5" >0.0299</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col0" class="data row5 col0" >0.4748</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col1" class="data row5 col1" >0.4475</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col2" class="data row5 col2" >0.6058</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col3" class="data row5 col3" >0.5712</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col4" class="data row5 col4" >0.0340</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row5_col5" class="data row5 col5" >0.0286</td>
            </tr>
            <tr>
                        <th id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col0" class="data row6 col0" >0.2573</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col1" class="data row6 col1" >0.3094</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col2" class="data row6 col2" >0.2839</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col3" class="data row6 col3" >0.4040</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col4" class="data row6 col4" >0.0158</td>
                        <td id="T_82c1996c_5ef5_11eb_b81a_54e1ada8ef40row6_col5" class="data row6 col5" >0.0155</td>
            </tr>
    </tbody></table>



```python
bagged_dt = ensemble_model(tuned_dt)
boosted_dt = ensemble_model(tuned_dt, method = 'Boosting',fold=5 )
```


<style  type="text/css" >
#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col0,#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col1,#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col2,#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col3,#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col4,#T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col0" class="data row0 col0" >0.1233</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col1" class="data row0 col1" >0.0230</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col2" class="data row0 col2" >0.1515</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col3" class="data row0 col3" >0.9278</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col4" class="data row0 col4" >0.0084</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row0_col5" class="data row0 col5" >0.0070</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col0" class="data row1 col0" >0.6917</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col1" class="data row1 col1" >0.7018</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col2" class="data row1 col2" >0.8377</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col3" class="data row1 col3" >0.6171</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col4" class="data row1 col4" >0.0492</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row1_col5" class="data row1 col5" >0.0425</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col0" class="data row2 col0" >0.3767</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col1" class="data row2 col1" >0.3005</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col2" class="data row2 col2" >0.5481</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col3" class="data row2 col3" >0.7626</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col4" class="data row2 col4" >0.0304</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row2_col5" class="data row2 col5" >0.0227</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col0" class="data row3 col0" >0.5483</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col1" class="data row3 col1" >0.4088</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col2" class="data row3 col2" >0.6394</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col3" class="data row3 col3" >0.3258</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col4" class="data row3 col4" >0.0358</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row3_col5" class="data row3 col5" >0.0330</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col0" class="data row4 col0" >0.7233</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col1" class="data row4 col1" >0.8011</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col2" class="data row4 col2" >0.8951</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col3" class="data row4 col3" >0.5373</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col4" class="data row4 col4" >0.0502</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row4_col5" class="data row4 col5" >0.0431</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col0" class="data row5 col0" >0.4927</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col1" class="data row5 col1" >0.4470</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col2" class="data row5 col2" >0.6144</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col3" class="data row5 col3" >0.6341</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col4" class="data row5 col4" >0.0348</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row5_col5" class="data row5 col5" >0.0297</td>
            </tr>
            <tr>
                        <th id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col0" class="data row6 col0" >0.2217</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col1" class="data row6 col1" >0.2804</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col2" class="data row6 col2" >0.2638</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col3" class="data row6 col3" >0.2037</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col4" class="data row6 col4" >0.0152</td>
                        <td id="T_8369f48b_5ef5_11eb_b9c8_54e1ada8ef40row6_col5" class="data row6 col5" >0.0135</td>
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
</style><table id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col0" class="data row0 col0" >Decision Tree Regressor</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col1" class="data row0 col1" >0.3910</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col2" class="data row0 col2" >0.2237</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col3" class="data row0 col3" >0.4730</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col4" class="data row0 col4" >0.8654</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col5" class="data row0 col5" >0.0264</td>
                        <td id="T_838e0609_5ef5_11eb_88b2_54e1ada8ef40row0_col6" class="data row0 col6" >0.0235</td>
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
      <th>T</th>
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
      <td>0.975670</td>
      <td>0.987452</td>
      <td>0.439969</td>
      <td>1.026842</td>
      <td>0.979045</td>
      <td>0.881813</td>
      <td>14.410000</td>
      <td>14.910000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.892342</td>
      <td>0.704816</td>
      <td>0.681347</td>
      <td>0.908053</td>
      <td>0.836846</td>
      <td>0.569597</td>
      <td>16.170000</td>
      <td>16.850000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.076577</td>
      <td>0.644609</td>
      <td>0.179949</td>
      <td>0.815534</td>
      <td>0.757261</td>
      <td>0.455721</td>
      <td>17.980000</td>
      <td>18.370001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.098446</td>
      <td>0.727912</td>
      <td>0.072619</td>
      <td>0.843518</td>
      <td>0.734970</td>
      <td>0.510097</td>
      <td>17.650000</td>
      <td>17.459999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.081570</td>
      <td>0.671488</td>
      <td>0.187443</td>
      <td>0.849800</td>
      <td>0.788843</td>
      <td>0.474720</td>
      <td>17.549999</td>
      <td>18.370001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.173195</td>
      <td>0.543459</td>
      <td>0.478695</td>
      <td>0.958309</td>
      <td>0.846145</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>16.850000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.542648</td>
      <td>0.925962</td>
      <td>0.078874</td>
      <td>0.894917</td>
      <td>0.780976</td>
      <td>0.682456</td>
      <td>17.610001</td>
      <td>16.930000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.221011</td>
      <td>0.581503</td>
      <td>0.494301</td>
      <td>0.985151</td>
      <td>0.866901</td>
      <td>0.903825</td>
      <td>16.850000</td>
      <td>16.930000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.997187</td>
      <td>0.822946</td>
      <td>0.492208</td>
      <td>1.006282</td>
      <td>0.882702</td>
      <td>0.829090</td>
      <td>14.660000</td>
      <td>14.570000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.981014</td>
      <td>0.837027</td>
      <td>0.467652</td>
      <td>0.884637</td>
      <td>0.871632</td>
      <td>0.787609</td>
      <td>14.810000</td>
      <td>14.860000</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_gbdt = finalize_model(boosted_dt)
print(final_gbdt)
save_model(boosted_dt,'Final-gbdt-t')
```

    AdaBoostRegressor(base_estimator=DecisionTreeRegressor(ccp_alpha=0.0,
                                                           criterion='mse',
                                                           max_depth=None,
                                                           max_features=None,
                                                           max_leaf_nodes=None,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None,
                                                           min_samples_leaf=1,
                                                           min_samples_split=2,
                                                           min_weight_fraction_leaf=0.0,
                                                           presort='deprecated',
                                                           random_state=111,
                                                           splitter='best'),
                      learning_rate=1.0, loss='linear', n_estimators=10,
                      random_state=111)
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
                                                                             criterion='mse',
                                                                             max_depth=None,
                                                                             max_features=None,
                                                                             max_leaf_nodes=None,
                                                                             min_impurity_decrease=0.0,
                                                                             min_impurity_split=None,
                                                                             min_samples_leaf=1,
                                                                             min_samples_split=2,
                                                                             min_weight_fraction_leaf=0.0,
                                                                             presort='deprecated',
                                                                             random_state=111,
                                                                             splitter='best'),
                                        learning_rate=1.0, loss='linear',
                                        n_estimators=10, random_state=111)]],
              verbose=False),
     'Final-gbdt-t.pkl')



# CatBoost


```python
catboost=create_model('catboost',fold=5);
#print(catboost)
```


<style  type="text/css" >
#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col0,#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col1,#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col2,#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col3,#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col4,#T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col0" class="data row0 col0" >0.2760</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col1" class="data row0 col1" >0.1044</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col2" class="data row0 col2" >0.3231</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col3" class="data row0 col3" >0.6717</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col4" class="data row0 col4" >0.0170</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row0_col5" class="data row0 col5" >0.0152</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col0" class="data row1 col0" >0.7123</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col1" class="data row1 col1" >0.6860</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col2" class="data row1 col2" >0.8283</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col3" class="data row1 col3" >0.6257</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col4" class="data row1 col4" >0.0489</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row1_col5" class="data row1 col5" >0.0451</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col0" class="data row2 col0" >0.3809</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col1" class="data row2 col1" >0.1645</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col2" class="data row2 col2" >0.4056</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col3" class="data row2 col3" >0.8700</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col4" class="data row2 col4" >0.0232</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row2_col5" class="data row2 col5" >0.0230</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col0" class="data row3 col0" >0.3860</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col1" class="data row3 col1" >0.1694</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col2" class="data row3 col2" >0.4116</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col3" class="data row3 col3" >0.7206</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col4" class="data row3 col4" >0.0232</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row3_col5" class="data row3 col5" >0.0231</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col0" class="data row4 col0" >0.5461</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col1" class="data row4 col1" >0.3982</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col2" class="data row4 col2" >0.6310</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col3" class="data row4 col3" >0.7701</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col4" class="data row4 col4" >0.0348</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row4_col5" class="data row4 col5" >0.0322</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col0" class="data row5 col0" >0.4603</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col1" class="data row5 col1" >0.3045</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col2" class="data row5 col2" >0.5199</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col3" class="data row5 col3" >0.7316</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col4" class="data row5 col4" >0.0294</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row5_col5" class="data row5 col5" >0.0277</td>
            </tr>
            <tr>
                        <th id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col0" class="data row6 col0" >0.1528</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col1" class="data row6 col1" >0.2155</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col2" class="data row6 col2" >0.1849</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col3" class="data row6 col3" >0.0843</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col4" class="data row6 col4" >0.0113</td>
                        <td id="T_9b290e4b_5ef5_11eb_964a_54e1ada8ef40row6_col5" class="data row6 col5" >0.0102</td>
            </tr>
    </tbody></table>



```python
tuned_catboost= tune_model(catboost,n_iter =100, choose_better=True, fold=5);
print(tuned_catboost);
```


<style  type="text/css" >
#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col0,#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col1,#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col2,#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col3,#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col4,#T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col0" class="data row0 col0" >0.2605</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col1" class="data row0 col1" >0.0964</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col2" class="data row0 col2" >0.3105</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col3" class="data row0 col3" >0.6969</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col4" class="data row0 col4" >0.0163</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row0_col5" class="data row0 col5" >0.0144</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col0" class="data row1 col0" >0.7431</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col1" class="data row1 col1" >0.7837</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col2" class="data row1 col2" >0.8853</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col3" class="data row1 col3" >0.5724</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col4" class="data row1 col4" >0.0522</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row1_col5" class="data row1 col5" >0.0471</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col0" class="data row2 col0" >0.3714</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col1" class="data row2 col1" >0.1711</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col2" class="data row2 col2" >0.4137</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col3" class="data row2 col3" >0.8648</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col4" class="data row2 col4" >0.0240</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row2_col5" class="data row2 col5" >0.0226</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col0" class="data row3 col0" >0.4056</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col1" class="data row3 col1" >0.1818</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col2" class="data row3 col2" >0.4263</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col3" class="data row3 col3" >0.7002</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col4" class="data row3 col4" >0.0239</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row3_col5" class="data row3 col5" >0.0243</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col0" class="data row4 col0" >0.5547</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col1" class="data row4 col1" >0.4316</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col2" class="data row4 col2" >0.6570</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col3" class="data row4 col3" >0.7507</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col4" class="data row4 col4" >0.0367</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row4_col5" class="data row4 col5" >0.0331</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col0" class="data row5 col0" >0.4670</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col1" class="data row5 col1" >0.3329</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col2" class="data row5 col2" >0.5386</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col3" class="data row5 col3" >0.7170</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col4" class="data row5 col4" >0.0306</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row5_col5" class="data row5 col5" >0.0283</td>
            </tr>
            <tr>
                        <th id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col0" class="data row6 col0" >0.1670</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col1" class="data row6 col1" >0.2522</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col2" class="data row6 col2" >0.2071</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col3" class="data row6 col3" >0.0944</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col4" class="data row6 col4" >0.0126</td>
                        <td id="T_c997dd97_5ef5_11eb_9323_54e1ada8ef40row6_col5" class="data row6 col5" >0.0111</td>
            </tr>
    </tbody></table>


    <catboost.core.CatBoostRegressor object at 0x0000023A27BD1F40>
    


```python
#predict_model(tuned_catboost)
```


```python
final_catboost = finalize_model(tuned_catboost)
print(final_catboost)
#save_model(final_catboost,'Final-catboost-t')
```

    <catboost.core.CatBoostRegressor object at 0x0000023A28D64FD0>
    


```python
blender_specific2 = blend_models(estimator_list = [tuned_et,tuned_catboost],fold=5 ) 
```


<style  type="text/css" >
#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col0,#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col1,#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col2,#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col3,#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col4,#T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col5{
            background:  yellow;
        }</style><table id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col0" class="data row0 col0" >0.2385</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col1" class="data row0 col1" >0.0808</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col2" class="data row0 col2" >0.2843</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col3" class="data row0 col3" >0.7459</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col4" class="data row0 col4" >0.0150</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row0_col5" class="data row0 col5" >0.0132</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col0" class="data row1 col0" >0.4560</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col1" class="data row1 col1" >0.3442</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col2" class="data row1 col2" >0.5867</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col3" class="data row1 col3" >0.8122</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col4" class="data row1 col4" >0.0343</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row1_col5" class="data row1 col5" >0.0285</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col0" class="data row2 col0" >0.3877</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col1" class="data row2 col1" >0.1914</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col2" class="data row2 col2" >0.4375</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col3" class="data row2 col3" >0.8488</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col4" class="data row2 col4" >0.0248</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row2_col5" class="data row2 col5" >0.0233</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col0" class="data row3 col0" >0.3856</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col1" class="data row3 col1" >0.1751</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col2" class="data row3 col2" >0.4184</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col3" class="data row3 col3" >0.7112</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col4" class="data row3 col4" >0.0234</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row3_col5" class="data row3 col5" >0.0230</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col0" class="data row4 col0" >0.4706</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col1" class="data row4 col1" >0.3238</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col2" class="data row4 col2" >0.5690</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col3" class="data row4 col3" >0.8130</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col4" class="data row4 col4" >0.0310</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row4_col5" class="data row4 col5" >0.0277</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col0" class="data row5 col0" >0.3877</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col1" class="data row5 col1" >0.2231</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col2" class="data row5 col2" >0.4592</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col3" class="data row5 col3" >0.7862</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col4" class="data row5 col4" >0.0257</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row5_col5" class="data row5 col5" >0.0231</td>
            </tr>
            <tr>
                        <th id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col0" class="data row6 col0" >0.0822</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col1" class="data row6 col1" >0.0983</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col2" class="data row6 col2" >0.1105</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col3" class="data row6 col3" >0.0501</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col4" class="data row6 col4" >0.0067</td>
                        <td id="T_d1b3819b_5ef5_11eb_a8f4_54e1ada8ef40row6_col5" class="data row6 col5" >0.0054</td>
            </tr>
    </tbody></table>



```python
evaluate_model(blender_specific2) #
```


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
predict_model(blender_specific2)
```


<style  type="text/css" >
</style><table id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col0" class="data row0 col0" >Voting Regressor</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col1" class="data row0 col1" >0.2252</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col2" class="data row0 col2" >0.0843</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col3" class="data row0 col3" >0.2903</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col4" class="data row0 col4" >0.9493</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col5" class="data row0 col5" >0.0165</td>
                        <td id="T_d1e800ce_5ef5_11eb_928a_54e1ada8ef40row0_col6" class="data row0 col6" >0.0138</td>
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
      <th>T</th>
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
      <td>0.975670</td>
      <td>0.987452</td>
      <td>0.439969</td>
      <td>1.026842</td>
      <td>0.979045</td>
      <td>0.881813</td>
      <td>14.410000</td>
      <td>14.779909</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.892342</td>
      <td>0.704816</td>
      <td>0.681347</td>
      <td>0.908053</td>
      <td>0.836846</td>
      <td>0.569597</td>
      <td>16.170000</td>
      <td>16.396258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.076577</td>
      <td>0.644609</td>
      <td>0.179949</td>
      <td>0.815534</td>
      <td>0.757261</td>
      <td>0.455721</td>
      <td>17.980000</td>
      <td>17.844084</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.098446</td>
      <td>0.727912</td>
      <td>0.072619</td>
      <td>0.843518</td>
      <td>0.734970</td>
      <td>0.510097</td>
      <td>17.650000</td>
      <td>17.723027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.081570</td>
      <td>0.671488</td>
      <td>0.187443</td>
      <td>0.849800</td>
      <td>0.788843</td>
      <td>0.474720</td>
      <td>17.549999</td>
      <td>18.044377</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.173195</td>
      <td>0.543459</td>
      <td>0.478695</td>
      <td>0.958309</td>
      <td>0.846145</td>
      <td>0.899552</td>
      <td>16.420000</td>
      <td>17.011934</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.542648</td>
      <td>0.925962</td>
      <td>0.078874</td>
      <td>0.894917</td>
      <td>0.780976</td>
      <td>0.682456</td>
      <td>17.610001</td>
      <td>17.718203</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.221011</td>
      <td>0.581503</td>
      <td>0.494301</td>
      <td>0.985151</td>
      <td>0.866901</td>
      <td>0.903825</td>
      <td>16.850000</td>
      <td>16.981524</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.997187</td>
      <td>0.822946</td>
      <td>0.492208</td>
      <td>1.006282</td>
      <td>0.882702</td>
      <td>0.829090</td>
      <td>14.660000</td>
      <td>14.716478</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.981014</td>
      <td>0.837027</td>
      <td>0.467652</td>
      <td>0.884637</td>
      <td>0.871632</td>
      <td>0.787609</td>
      <td>14.810000</td>
      <td>14.874264</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_gbrcb = finalize_model(blender_specific2)
print(final_gbrcb)
save_model(blender_specific2,'Final-catboost-t')
```

    TunableVotingRegressor(estimators=[('et',
                                        ExtraTreesRegressor(bootstrap=False,
                                                            ccp_alpha=0.0,
                                                            criterion='mse',
                                                            max_depth=None,
                                                            max_features='auto',
                                                            max_leaf_nodes=None,
                                                            max_samples=None,
                                                            min_impurity_decrease=0.0,
                                                            min_impurity_split=None,
                                                            min_samples_leaf=1,
                                                            min_samples_split=2,
                                                            min_weight_fraction_leaf=0.0,
                                                            n_estimators=100,
                                                            n_jobs=-1,
                                                            oob_score=False,
                                                            random_state=111,
                                                            verbose=0,
                                                            warm_start=False)),
                                       ('catboost',
                                        <catboost.core.CatBoostRegressor object at 0x0000023A28D7BF70>)],
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
                                                                              min_impurity_decrease=0.0,
                                                                              min_impurity_split=None,
                                                                              min_samples_leaf=1,
                                                                              min_samples_split=2,
                                                                              min_weight_fraction_leaf=0.0,
                                                                              n_estimators=100,
                                                                              n_jobs=-1,
                                                                              oob_score=False,
                                                                              random_state=111,
                                                                              verbose=0,
                                                                              warm_start=False)),
                                                         ('catboost',
                                                          <catboost.core.CatBoostRegressor object at 0x0000023A27D8D790>)],
                                             n_jobs=-1, verbose=False, weight_0=1,
                                             weight_1=1, weights=[1, 1])]],
              verbose=False),
     'Final-catboost-t.pkl')




```python
#help(plot_model) 
```

# Predict


```python
data_unseen=pd.read_excel("your prediction test set", sheet_name='pyt')
```


```python
saved_f= load_model('chose a saved model')
new_prediction = predict_model(saved_f, data=data_unseen)
```

    Transformation Pipeline and Model Successfully Loaded
    


```python
new_prediction
```




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
      <th>T</th>
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
      <td>138.75</td>
      <td>59.257</td>
      <td>126.334</td>
      <td>1.7980</td>
      <td>468.939</td>
      <td>69.3440</td>
      <td>14.41</td>
      <td>14.779909</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138.06</td>
      <td>49.708</td>
      <td>140.488</td>
      <td>1.4990</td>
      <td>434.040</td>
      <td>71.6370</td>
      <td>14.57</td>
      <td>14.569961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>142.21</td>
      <td>53.567</td>
      <td>186.942</td>
      <td>1.6890</td>
      <td>437.061</td>
      <td>65.3320</td>
      <td>14.57</td>
      <td>14.570649</td>
    </tr>
    <tr>
      <th>3</th>
      <td>141.81</td>
      <td>49.385</td>
      <td>141.334</td>
      <td>1.7620</td>
      <td>422.793</td>
      <td>65.1980</td>
      <td>14.66</td>
      <td>14.716478</td>
    </tr>
    <tr>
      <th>4</th>
      <td>139.51</td>
      <td>50.230</td>
      <td>134.283</td>
      <td>1.5490</td>
      <td>417.491</td>
      <td>61.9360</td>
      <td>14.81</td>
      <td>14.874264</td>
    </tr>
    <tr>
      <th>5</th>
      <td>138.51</td>
      <td>45.358</td>
      <td>132.711</td>
      <td>1.4190</td>
      <td>478.976</td>
      <td>59.2030</td>
      <td>14.86</td>
      <td>14.861534</td>
    </tr>
    <tr>
      <th>6</th>
      <td>141.51</td>
      <td>60.010</td>
      <td>120.292</td>
      <td>1.6210</td>
      <td>394.977</td>
      <td>60.4580</td>
      <td>14.91</td>
      <td>14.910495</td>
    </tr>
    <tr>
      <th>7</th>
      <td>127.34</td>
      <td>42.015</td>
      <td>181.055</td>
      <td>1.6620</td>
      <td>422.045</td>
      <td>47.1230</td>
      <td>15.81</td>
      <td>15.810805</td>
    </tr>
    <tr>
      <th>8</th>
      <td>133.19</td>
      <td>55.963</td>
      <td>61.885</td>
      <td>1.7510</td>
      <td>435.674</td>
      <td>42.3400</td>
      <td>15.85</td>
      <td>15.850616</td>
    </tr>
    <tr>
      <th>9</th>
      <td>126.90</td>
      <td>42.296</td>
      <td>195.644</td>
      <td>1.5900</td>
      <td>400.829</td>
      <td>44.7920</td>
      <td>16.17</td>
      <td>16.396258</td>
    </tr>
    <tr>
      <th>10</th>
      <td>129.02</td>
      <td>43.603</td>
      <td>276.696</td>
      <td>1.6940</td>
      <td>388.225</td>
      <td>46.1770</td>
      <td>16.17</td>
      <td>16.169158</td>
    </tr>
    <tr>
      <th>11</th>
      <td>24.63</td>
      <td>32.613</td>
      <td>137.454</td>
      <td>1.6780</td>
      <td>405.283</td>
      <td>70.7390</td>
      <td>16.42</td>
      <td>17.011934</td>
    </tr>
    <tr>
      <th>12</th>
      <td>100.84</td>
      <td>55.645</td>
      <td>11.474</td>
      <td>1.6700</td>
      <td>413.455</td>
      <td>33.6160</td>
      <td>16.64</td>
      <td>16.640390</td>
    </tr>
    <tr>
      <th>13</th>
      <td>35.34</td>
      <td>26.262</td>
      <td>264.634</td>
      <td>1.4010</td>
      <td>460.734</td>
      <td>72.4600</td>
      <td>16.65</td>
      <td>16.650350</td>
    </tr>
    <tr>
      <th>14</th>
      <td>126.54</td>
      <td>32.835</td>
      <td>287.143</td>
      <td>1.4700</td>
      <td>406.958</td>
      <td>49.0680</td>
      <td>16.85</td>
      <td>16.850722</td>
    </tr>
    <tr>
      <th>15</th>
      <td>31.43</td>
      <td>34.896</td>
      <td>141.935</td>
      <td>1.7250</td>
      <td>415.225</td>
      <td>71.0750</td>
      <td>16.85</td>
      <td>16.981524</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-78.03</td>
      <td>59.785</td>
      <td>10.833</td>
      <td>1.6140</td>
      <td>374.273</td>
      <td>61.4610</td>
      <td>16.93</td>
      <td>16.931308</td>
    </tr>
    <tr>
      <th>17</th>
      <td>105.44</td>
      <td>47.975</td>
      <td>27.665</td>
      <td>1.5940</td>
      <td>395.635</td>
      <td>35.3850</td>
      <td>16.95</td>
      <td>16.949667</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-17.55</td>
      <td>43.319</td>
      <td>85.641</td>
      <td>1.5310</td>
      <td>392.306</td>
      <td>63.9810</td>
      <td>17.06</td>
      <td>17.060500</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-18.81</td>
      <td>38.032</td>
      <td>59.641</td>
      <td>1.4560</td>
      <td>409.366</td>
      <td>62.6980</td>
      <td>17.13</td>
      <td>17.131071</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-9.00</td>
      <td>39.443</td>
      <td>31.758</td>
      <td>1.4640</td>
      <td>381.851</td>
      <td>35.4430</td>
      <td>17.15</td>
      <td>17.152387</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-18.21</td>
      <td>40.175</td>
      <td>72.481</td>
      <td>1.4350</td>
      <td>400.396</td>
      <td>63.3050</td>
      <td>17.15</td>
      <td>17.150489</td>
    </tr>
    <tr>
      <th>22</th>
      <td>33.34</td>
      <td>49.554</td>
      <td>77.954</td>
      <td>1.4710</td>
      <td>402.261</td>
      <td>78.6380</td>
      <td>17.21</td>
      <td>17.209488</td>
    </tr>
    <tr>
      <th>23</th>
      <td>32.54</td>
      <td>31.158</td>
      <td>87.612</td>
      <td>1.4390</td>
      <td>392.821</td>
      <td>63.9260</td>
      <td>17.26</td>
      <td>17.260274</td>
    </tr>
    <tr>
      <th>24</th>
      <td>33.46</td>
      <td>33.152</td>
      <td>134.821</td>
      <td>1.4640</td>
      <td>394.501</td>
      <td>67.7990</td>
      <td>17.26</td>
      <td>17.258568</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-12.73</td>
      <td>43.390</td>
      <td>122.136</td>
      <td>1.4990</td>
      <td>352.824</td>
      <td>40.0530</td>
      <td>17.46</td>
      <td>17.461483</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-11.60</td>
      <td>40.296</td>
      <td>53.823</td>
      <td>1.4880</td>
      <td>377.837</td>
      <td>37.3310</td>
      <td>17.55</td>
      <td>18.044377</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-77.17</td>
      <td>55.567</td>
      <td>22.648</td>
      <td>1.5670</td>
      <td>374.069</td>
      <td>53.6670</td>
      <td>17.61</td>
      <td>17.718203</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-14.00</td>
      <td>43.682</td>
      <td>20.852</td>
      <td>1.4770</td>
      <td>352.033</td>
      <td>40.1130</td>
      <td>17.65</td>
      <td>17.723027</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-75.30</td>
      <td>55.151</td>
      <td>44.874</td>
      <td>1.4525</td>
      <td>369.438</td>
      <td>56.5270</td>
      <td>17.87</td>
      <td>17.870470</td>
    </tr>
    <tr>
      <th>30</th>
      <td>-10.89</td>
      <td>38.683</td>
      <td>51.671</td>
      <td>1.4280</td>
      <td>362.710</td>
      <td>35.8370</td>
      <td>17.98</td>
      <td>17.844084</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-76.70</td>
      <td>51.731</td>
      <td>11.697</td>
      <td>1.5270</td>
      <td>378.496</td>
      <td>43.7620</td>
      <td>18.03</td>
      <td>18.029514</td>
    </tr>
    <tr>
      <th>32</th>
      <td>-75.80</td>
      <td>52.781</td>
      <td>21.387</td>
      <td>1.4630</td>
      <td>355.365</td>
      <td>51.1615</td>
      <td>18.13</td>
      <td>18.128694</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-12.18</td>
      <td>41.060</td>
      <td>45.010</td>
      <td>1.4760</td>
      <td>419.446</td>
      <td>37.8290</td>
      <td>18.19</td>
      <td>18.188576</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-11.80</td>
      <td>37.858</td>
      <td>41.294</td>
      <td>1.5120</td>
      <td>409.452</td>
      <td>37.2540</td>
      <td>18.31</td>
      <td>18.309466</td>
    </tr>
    <tr>
      <th>35</th>
      <td>-9.88</td>
      <td>36.340</td>
      <td>61.893</td>
      <td>1.5020</td>
      <td>351.382</td>
      <td>33.2920</td>
      <td>18.37</td>
      <td>18.369147</td>
    </tr>
    <tr>
      <th>36</th>
      <td>-65.56</td>
      <td>42.029</td>
      <td>51.798</td>
      <td>1.4860</td>
      <td>374.708</td>
      <td>42.6910</td>
      <td>18.39</td>
      <td>18.391007</td>
    </tr>
    <tr>
      <th>37</th>
      <td>-65.20</td>
      <td>39.456</td>
      <td>64.961</td>
      <td>1.4390</td>
      <td>369.073</td>
      <td>43.3610</td>
      <td>18.41</td>
      <td>18.409882</td>
    </tr>
    <tr>
      <th>38</th>
      <td>-63.86</td>
      <td>40.675</td>
      <td>66.925</td>
      <td>1.4850</td>
      <td>380.488</td>
      <td>44.7020</td>
      <td>18.45</td>
      <td>18.449539</td>
    </tr>
    <tr>
      <th>39</th>
      <td>-64.85</td>
      <td>39.324</td>
      <td>82.187</td>
      <td>1.4750</td>
      <td>386.268</td>
      <td>46.7130</td>
      <td>18.58</td>
      <td>18.578198</td>
    </tr>
  </tbody>
</table>
</div>


