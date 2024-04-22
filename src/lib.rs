use kiddo::{ImmutableKdTree, NearestNeighbour, SquaredEuclidean};
use pyo3::{prelude::*, types::PyTuple};

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pyclass]
struct Py2KDTree {
    tree: ImmutableKdTree<f32, 2>,
}

fn two_d_query<'py>(query_: PyReadonlyArray1<'py, f32>) -> [f32; 2] {
    let query = query_.as_array();
    let query_view = query.as_slice().unwrap();
    if query_view.len() != 2 {
        panic!("Query should be a 2D point")
    }
    [query_view[0], query_view[1]]
}

fn nearest_neighbours_to_object<'py>(
    py: Python<'py>,
    neighbours: Vec<NearestNeighbour<f32, u64>>,
) -> PyObject {
    let mut distances_vec: Vec<f32> = Vec::new();
    let mut indices_vec: Vec<u64> = Vec::new();

    for n in neighbours {
        distances_vec.push(n.distance);
        indices_vec.push(n.item);
    }

    let distance = PyArray1::from_vec_bound(py, distances_vec);
    let indices = PyArray1::from_vec_bound(py, indices_vec);
    PyTuple::new_bound(py, &[indices.to_object(py), distance.to_object(py)]).to_object(py)
}

#[pymethods]
impl Py2KDTree {
    #[new]
    fn new<'py>(x_: PyReadonlyArray2<'py, f32>) -> Self {
        let x = x_.as_array();
        let shape = x.shape();
        let dim = 2;
        if shape[1] != dim {
            panic!("Shape should be [N, 2]. Only 2D points are supported.")
        }
        let n_points = shape[0];
        let mut x_arr: Vec<[f32; 2]> = Vec::with_capacity(n_points);
        for row in x.rows() {
            let row_slice = row.as_slice().unwrap();
            x_arr.push([row_slice[0], row_slice[1]]);
        }
        let tree = ImmutableKdTree::new_from_slice(x_arr.as_slice());
        return Self { tree };
    }

    fn nearest_n_within<'py>(
        &self,
        py: Python<'py>,
        query_: PyReadonlyArray1<'py, f32>,
        dist: f32,
        max_neighbours: usize,
        sorted: bool,
    ) -> PyObject {
        let query_view_2 = two_d_query(query_);
        let neighbours = self.tree.nearest_n_within::<SquaredEuclidean>(
            &query_view_2,
            dist,
            max_neighbours,
            sorted,
        );
        let neighbours_vec = Vec::from_iter(neighbours);
        nearest_neighbours_to_object(py, neighbours_vec)
    }

    fn count_within<'py>(&self, query_: PyReadonlyArray1<'py, f32>, dist: f32) -> usize {
        let query_view_2 = two_d_query(query_);
        let neighbours = self
            .tree
            .within_unsorted::<SquaredEuclidean>(&query_view_2, dist);
        neighbours.len()
    }

    fn within<'py>(
        &self,
        py: Python<'py>,
        query_: PyReadonlyArray1<'py, f32>,
        dist: f32,
    ) -> PyObject {
        let query_view_2 = two_d_query(query_);
        let neighbours = self
            .tree
            .within_unsorted::<SquaredEuclidean>(&query_view_2, dist);
        let neighbours_vec = Vec::from_iter(neighbours);
        nearest_neighbours_to_object(py, neighbours_vec)
    }
}

#[pymodule]
fn kiddo_python_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Py2KDTree>()?;
    Ok(())
}
