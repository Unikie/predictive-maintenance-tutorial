# Predictive Maintenance Tutorial

## Instruction to run the code
1. Clone this repository to your laptop
2. Run all the scripts project folder `predictive-maintenance-tutorial`
    * The command for the example solution would be `python solution/run.py`
3. The code has tested with Python `3.8.3` and the following libraries:
    * numpy `1.19.0`
    * pandas `1.0.5`
    * scikit-learn `0.23.1`
    * matplotlib `3.3.2`

## The assignment
Gadget company Ltd is creating a predictive algorithm to detect their gadget breakdowns.

When a breakdown happes the component easy to replace without significant downtime. The problem is, the spare part it is extremely expensive.

It is known that the past vibration, pressure and temperature measurements have something to do with the breakdowns.

**Your task is to develop a suitable machine learning model.**. Take the sample data `datasets/measurements.csv` and `datasets/failures.csv` to explore the data. Read more about the dataset structure below.

The summary of the example solution can be found [here](./solution/solution.md) and the example Python script [here](./solution/run.py).

## Datasets

### datasets/measurements.csv
Sensor data from three different gadgets. There is an hourly measurement from these sensors:

sensor | description | unit
---  | --- | ---
vibration_x | How much the gadget vibrates horizontally | cm
vibration_y | How much the gadget vibrates vertically | cm |
pressure | Pressure in hose | bar
temperature | Internal temperature of the gadget | Celcius degrees

There is maintenance data from one week resulting 1008 rows of observations:

```
[6 gadgets] X [168 hours per week] = 1008 rows
```

### datasets/failures.csv

The maintenance data contains information about when the failures happened to each gadget. 105 rows in total.

## Splitting the training testing data
- Gadget ids from `1` to `4` for training the model
- Gadget ids `5` and `6` to test the model