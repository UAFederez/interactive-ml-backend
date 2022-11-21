# Interactive Machine Learning Backend

This serves as the web service exposing a RESTful API to interact with various machine learning models implemented from the ground up.

## Usage

The RESTful API, implemented with `flask_restful` can be initiated as,

```
python api.py
```

## Univariate Linear Regression API

<strong>`POST`</strong> `/api/linear-regression-uni`

### Request JSON Body Payload Parameters

| Parameter       | Default | Type      | Allowable values                | Description                                                                                                                                      |
| --------------- | ------- | --------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `method`        | `none`  | `string`  | `"direct"`/`"gradient_descent"` | Method to calculate optimal parameters `direct` for analytical solution (system of equations), `gradient_descent` for iterative gradient descent |
| `train_x`       | `none`  | `float[]` | non-empty float array           | X values of the dataset                                                                                                                          |
| `train_y`       | `none`  | `float[]` | non-empty float array           | Y values of the dataset                                                                                                                          |
| `epochs`        | `none`  | `integer` | Positive integer                | Number of epochs in the training phase                                                                                                           |
| `learning_rate` | `none`  | `float`   | Positive float                  | Learning rate in the training phase                                                                                                              |
| `include_hist`  | `true`  | `boolean` | `true`/`false`                  | Include the history of parameter updates and the loss values only when `method`=`"direct"`                                                       |

### Response JSON Body

-   If `status` has a value of `invalid_method`, then the remaining fields are not included in the response

| Key      | Type     | Description                                                 |
| -------- | -------- | ----------------------------------------------------------- |
| `status` | `string` | `success` or `invalid_method` if the parameters are invalid |
| `weight` | `float`  | Approximation of the true value of the coefficient, $w$     |
| `bias`   | `float`  | Approximation of the true value of the intercept, $b$       |

## Multivariate Linear Regression

<strong>`POST`</strong> `/api/linear-regression-mul`

### Request JSON Body Payload Parameters

| Parameter       | Default | Type        | Allowable values                    | Description                                                                                                                                      |
| --------------- | ------- | ----------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `method`        | `none`  | `string`    | `"normal_eq"`, `"gradient_descent"` | Method to calculate optimal parameters `normal_eq` for analytical solution (normal equations), `gradient_descent` for iterative gradient descent |
| `train_x`       | `none`  | `float[][]` | non-empty 2D float array            | An $m\times n$ array where $m$ is the number of features and $n$ is the number of training examples.                                             |
| `train_y`       | `none`  | `float[]`   | non-empty float array               | Y values of the dataset                                                                                                                          |
| `epochs`        | `none`  | `integer`   | Any positive integer                | Number of epochs in the training phase                                                                                                           |
| `learning_rate` | `none`  | `float`     | Any positive float                  | Learning rate in the training phase                                                                                                              |

### Response JSON Body

-   If `status` has a value of `invalid_method`, then the remaining fields are not included in the response

| Key       | Type     | Description                                                                    |
| --------- | -------- | ------------------------------------------------------------------------------ |
| `status`  | `string` | `sucess` or `invalid_method` if the parameters are invalid                     |
| `weights` | `float`  | Approximation of the true value of all coefficients (bias is the last element) |
