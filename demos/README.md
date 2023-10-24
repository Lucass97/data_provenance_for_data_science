# Demos

In this section, you will find some examples that are useful for understanding how the provenance capture of the [DPDS](../README.md) tool works. There are some toy examples that implement [simple pipelines](simple-pipelines) as well as [real-world pipelines](real-world-pipelines) that involve the use of real datasets.

> It is recommended to use the [simple client](../client/README.md) to query the generated provenance and the [query tester](../test/README.md) to perform tests.

## Simple Pipelines

These are simple pipelines designed for different types of small synthetic datasets. There are two pipelines available:

- **Demo Shell Categorical**: This pipeline is designed to work with synthetic categorical datasets.
- **Demo Shell Numerical**: This pipeline is designed to work with synthetic numerical datasets.

---

Follow these steps to use them:

1. Open a terminal and navigate to the project directory.

2. Navigate to the `demos` folder using the following command:
   ```sh
   cd demos
   ```

3. Ensure that you have set up the required dependencies and the virtual environment, as described in the [installation](../README.md#installation) instructions.

4. Run the specific pipeline using one of the following commands:

    - [`Demo Shell Categorical`](demo_shell_categorical.py)

        ```sh
        python3 demo_shell_categorical.py
        ```
    - [`Demo Shell Numerical`](demo_shell_numerical.py)

        ```sh
        python3 demo_shell_numerical.py
        ```

## Real World Pipelines

In this section, you will find examples of pipelines that utilize real datasets and capture their provenance using the [DPDS](../README.md) tool.

---

Follow these steps to use them:

You can execute any of the following pipelines with different commands to perform specific operations. Each pipeline has its own script, and they all work in a similar way. Below are examples of how to run the programs for each pipeline:

1. Open a terminal and navigate to the project directory.

2. Navigate to the `real_world_pipelines` folder using the following command:
   ```sh
   cd demos/real_world_pipelines
   ```

3. Ensure that you have set up the required dependencies and the virtual environment, as described in the [installation](../README.md#installation) instructions.

4. Run the specific pipeline using one of the following commands:

    - [Census Pipeline](#census-pipeline)

        ```sh
        python3 census_pipeline.py 
                --dataset <DATASET_PATH>
                --frac <FRACTION>
        ```
    - [Compas Pipeline](#compas-pipeline)

        ```sh
        python3 compas_pipeline.py 
                --dataset <DATASET_PATH>
                --frac <FRACTION>
        ```
    - [German Pipeline](#german-pipeline)

        ```sh
        python3 german_pipeline.py 
                --dataset <DATASET_PATH>
                --frac <FRACTION>
        ```

### Command Line Arguments

The following command line arguments are available for each of the pipelines:

- `<DATASET_PATH>` is the relative path to the dataset file.
- `<FRACTION>` is the sampling fraction, which should be in the range of 0.0 to 1.0. It determines the portion of data to be sampled. The default is 0.0, meaning no sampling by default. This is useful for debugging purposes.

Here is an example of how to use the command line arguments for each pipeline:

```sh
# For the Census Pipeline
python3 census_pipeline.py --dataset ./datasets/census_dataset.csv --frac 0.2

# For the Compas Pipeline
python3 compas_pipeline.py --dataset ./datasets/compas_dataset.csv --frac 0.2

# For the German Pipeline
python3 german_pipeline.py --dataset ./datasets/german_dataset.csv --frac 0.2
```

These examples run the respective scripts using the specified dataset files (contained in the [datasets](real_world_pipelines/datasets) folder) and sampling fractions.

---

### Census Pipeline

The ***Census Pipeline*** processes census data, including operations like whitespace removal, handling missing values, one-hot encoding of categorical features, and more. Below is an explanation of the operations included in the ***Census Pipeline***:

1. **Remove Whitespace from 9 Columns**: This operation removes leading and trailing whitespaces from specific columns in the dataset.

2. **Replace '?' Character with 'NaN' Value**: This operation replaces '?' characters with 'NaN' values in the dataset.

3. **One-Hot Encode Categorical Features**: This operation one-hot encodes 7 categorical features in the dataset.

4. **Assign Binary Values to `Sex` and `Label`**: This operation assigns binary values (0 and 1) to the `Sex` and `Label` columns, respectively.

5. **Drop `fnlwg` Column**: This operation removes the `fnlwg` column from the dataset.

### Compas Pipeline

The ***Compas Pipeline*** processes data related to the COMPAS dataset, performing various data preprocessing operations. Below is an explanation of the operations included in the ***Compas Pipeline***:

1. **Selection of Relevant Columns**: This operation selects a subset of relevant columns from the dataset. The selected columns are:
   - `age`
   - `c_charge_degree`
   - `race`
   - `sex`
   - `priors_count`
   - `days_b_screening_arrest`
   - `two_year_recid`
   - `c_jail_in`
   - `c_jail_out`

2. **Remove Missing Values**: This operation removes rows with missing values ('NaN') from the dataset.

3. **Make Race Feature Binary**: In this operation, the `race` feature is transformed into a binary feature. If `race` is not 'Caucasian', it is assigned a value of 0; otherwise, it is assigned a value of 1.

4. **Rename and Transform the Label Column**: The `two_year_recid` column is renamed to `label`. Additionally, the values in the `label` column are transformed such that 1 means no recidivism ('good') and 0 means recidivism ('bad').

5. **Create and Convert Jailtime Column**: This operation calculates the 'jailtime' for each record by subtracting the `c_jail_in` datetime from the `c_jail_out` datetime and converting the result to days.

6. **Drop `c_jail_in` and `c_jail_out` Features**: The `c_jail_in` and `c_jail_out` columns are dropped from the dataset as they are no longer needed.

7. **Value Transformation of `c_charge_degree` Column**: In this operation, the `c_charge_degree` column is transformed into binary values. 'M' (*misconduct*) is assigned a value of 0, and 'F' (*felony*) is assigned a value of 1.

### German Credit Pipeline

The ***German Credit Pipeline*** processes data related to the German credit dataset, performing various data preprocessing operations. Below is an explanation of the operations included in the ***German Credit Pipeline***:

1. **Value Transformation of 13 Distinct Columns**: This operation involves the transformation of values in 13 distinct columns in the dataset. Each column is transformed as follows:
   - `checking`: Values 'A11', 'A12', 'A13', and 'A14' are transformed to 'check_low', 'check_mid', 'check_high', and 'check_none' respectively.
   - `credit_history`: Values 'A30', 'A31', 'A32', 'A33', and 'A34' are transformed to 'debt_none', 'debt_noneBank', 'debt_onSchedule', 'debt_delay', and 'debt_critical' respectively.
   - `purpose`: Values 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', and 'A410' are transformed to 'pur_newCar', 'pur_usedCar', 'pur_furniture', 'pur_tv', 'pur_appliance', 'pur_repairs', 'pur_education', 'pur_vacation', 'pur_retraining', 'pur_business', and 'pur_other' respectively.
   - `savings`: Values 'A61', 'A62', 'A63', 'A64', and 'A65' are transformed to 'sav_small', 'sav_medium', 'sav_large', 'sav_xlarge', and 'sav_none' respectively.
   - `employment`: Values 'A71', 'A72', 'A73', 'A74', and 'A75' are transformed to 'emp_unemployed', 'emp_lessOne', 'emp_lessFour', 'emp_lessSeven', and 'emp_moreSeven' respectively.
   - `other_debtors`: Values 'A101', 'A102', and 'A103' are transformed to 'debtor_none', 'debtor_coApp', and 'debtor_guarantor' respectively.
   - `property`: Values 'A121', 'A122', 'A123', and 'A124' are transformed to 'prop_realEstate', 'prop_agreement', 'prop_car', and 'prop_none' respectively.
   - `other_inst`: Values 'A141', 'A142', and 'A143' are transformed to 'oi_bank', 'oi_stores', and 'oi_none' respectively.
   - `housing`: Values 'A151', 'A152', and 'A153' are transformed to 'hous_rent', 'hous_own', and 'hous_free' respectively.
   - `job`: Values 'A171', 'A172', 'A173', and 'A174' are transformed to 'job_unskilledNR', 'job_unskilledR', 'job_skilled', and 'job_highSkill' respectively.
   - `phone`: Values 'A191' and 'A192' are transformed to binary values 0 and 1 respectively.
   - `foreigner`: Values 'A201' and 'A202' are transformed to binary values 1 and 0 respectively.
   - `label`: Value 2 is transformed to 0.

2. **Generation of Two New Columns from the `personal_status` Column**: This operation involves creating two new columns, `status` and `gender`, based on the values in the `personal_status` column. The `status` column is derived from the `personal_status` values, translating them into 'divorced', 'single', or 'married'. The `gender` column is determined based on `personal_status`, where 'A92' and 'A95' values are assigned 0 (indicating female), while others are assigned 1 (indicating male).

3. **Drop 'personal_status' Column**: The `personal_status` column is dropped from the dataset as it is no longer needed.

4. **One-Hot Encode of 11 Categorical Columns**: In this operation, one-hot encoding is performed on 11 categorical columns in the dataset. Each categorical column is one-hot encoded, and the original column is dropped. The one-hot encoding results in binary columns for each category within the original columns.