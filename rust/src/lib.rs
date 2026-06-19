use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct Bm25Index {
    doc_offsets: Vec<usize>,   // len = n_docs + 1
    doc_term_ids: Vec<u32>,    // flattened, parallel to doc_term_freqs
    doc_term_freqs: Vec<u32>,
    doc_lens: Vec<f64>,        // len = n_docs
    avgdl: f64,
    k1: f64,
    b: f64,
}

#[pymethods]
impl Bm25Index {
    #[new]
    fn new(
        doc_offsets: Vec<usize>,
        doc_term_ids: Vec<u32>,
        doc_term_freqs: Vec<u32>,
        doc_lens: Vec<f64>,
        avgdl: f64,
        k1: f64,
        b: f64,
    ) -> Self {
        Bm25Index { doc_offsets, doc_term_ids, doc_term_freqs, doc_lens, avgdl, k1, b }
    }


    fn score(&self, query_term_ids: Vec<u32>, query_idfs: Vec<f64>) -> Vec<f64> {

        let qmap: HashMap<u32, f64> =
            query_term_ids.into_iter().zip(query_idfs).collect();

        let n_docs = self.doc_lens.len();

        (0..n_docs)
            .into_par_iter()
            .map(|i| {
                let start = self.doc_offsets[i];
                let end = self.doc_offsets[i + 1];
                let dl = self.doc_lens[i];

                let mut score = 0.0;
                for j in start..end {
                    if let Some(&idf) = qmap.get(&self.doc_term_ids[j]) {
                        let f = self.doc_term_freqs[j] as f64;
                        let denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                        score += idf * (f * (self.k1 + 1.0)) / denom;
                    }
                }
                score
            })
            .collect()
    }
}

#[pymodule]
fn scholar_rag_bm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bm25Index>()?;
    Ok(())
}