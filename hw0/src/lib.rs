use numpy::PyArrayDyn;
use pyo3::prelude::*;

fn softmax(x: &Vec<f32>, rows: u32, cols: u32) -> Vec<f32> {
    let mut exp_x = x.clone();
    for (i, el) in x.iter().enumerate() {
        exp_x[i] = el.exp()
    }
    for i in 0..rows {
        // sum rows
        let mut temp = 0f32;
        for j in 0..cols {
            temp += exp_x[(rows * i + j) as usize]
        }
        // divide by sum
        for j in 0..cols {
            exp_x[(rows * i + j) as usize] /= temp;
        }
    }
    return exp_x;
}

fn matmul(
    a: &Vec<f32>,
    b: &Vec<f32>,
    a_rows: u32,
    a_cols: u32,
    b_rows: u32,
    b_cols: u32,
) -> Vec<f32> {
    assert_eq!(
        a_cols, b_rows,
        "Matrix dimensions do not match for multiplication."
    );

    let mut result: Vec<f32> = Vec::with_capacity((a_rows * b_cols) as usize);

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[(i * a_rows + j) as usize] +=
                    a[(i * a_rows + k) as usize] * b[(k * b_rows + j) as usize];
            }
        }
    }

    result
}

#[pyfunction]
fn softmax_regression_epoch_rust(
    x: &PyArrayDyn<f32>,
    // x: Vec<f32>,
    y: &PyArrayDyn<u8>,
    // Vec<u8>,
    theta: &PyArrayDyn<f32>,
    // mut theta: Vec<f32>,
    m: u32, // n_examples
    n: u32, // input dim
    k: u32, // n_classes
    lr: f32,
    batch: u32,
) -> PyResult<()> {
    let x = x.to_vec().unwrap();
    let y = y.to_vec().unwrap();
    let mut theta = theta.to_vec().unwrap();

    let n_batches = m / batch;

    for _ in 0..n_batches {
        let x_batch = &x[(n * batch) as usize..((n + 1) * batch) as usize].to_vec();
        let y_batch = &y[(n * batch) as usize..((n + 1) * batch) as usize].to_vec();

        let x_theta = matmul(x_batch, &theta, batch, n, n, k);
        let mut z = softmax(&x_theta, batch, k);

        let mut i_y = vec![0f32; (batch * k) as usize];
        let mut row_index = 0;
        for &i in y_batch {
            i_y[(row_index * n + (i as u32)) as usize] = 1.0;
            row_index += 1;
        }

        for j in 0..z.len() {
            z[j] -= i_y[j];
        }
        let mut grad = matmul(x_batch, &z, n, batch, batch, k);
        for j in 0..grad.len() {
            grad[j] /= batch as f32;
        }

        for j in 0..theta.len() {
            theta[j] -= lr * grad[j];
        }
    }

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn hw0(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(softmax_regression_epoch_rust, m)?)?;
    Ok(())
}
