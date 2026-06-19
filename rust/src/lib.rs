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


    fn top_k(
        &self,
        query_term_ids: Vec<u32>,
        query_idfs: Vec<f64>,
        k: usize,
    ) -> Vec<(f64, usize)> {
        let qmap: HashMap<u32, f64> =
            query_term_ids.into_iter().zip(query_idfs).collect();

        let n_docs = self.doc_lens.len();

        let mut scored: Vec<(f64, usize)> = (0..n_docs)
            .into_par_iter()
            .filter_map(|i| {
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
                if score > 0.0 { Some((score, i)) } else { None }
            })
            .collect();

        let cmp = |a: &(f64, usize), b: &(f64, usize)| {
            b.0.total_cmp(&a.0).then(b.1.cmp(&a.1))
        };

        let kk = k.min(scored.len());
        if kk == 0 {
            return Vec::new();
        }
        if kk < scored.len() {
            scored.select_nth_unstable_by(kk - 1, cmp);
            scored.truncate(kk);
        }
        scored.sort_unstable_by(cmp); // only kk elements now
        scored
    }
}

/// Inverted index: instead of "doc -> its terms", we store "term -> the docs
/// containing it" (each term's list is a *postings list*, flattened CSR-style by
/// term id). A query only walks the postings of its own terms, so documents that
/// share no words with the query are never visited.
#[pyclass]
struct Bm25Inverted {
    term_offsets: Vec<usize>,  // len = vocab_size + 1; term t's postings are [t..t+1)
    posting_docs: Vec<u32>,    // flattened doc ids, parallel to posting_freqs
    posting_freqs: Vec<u32>,
    doc_lens: Vec<f64>,        // len = n_docs
    avgdl: f64,
    k1: f64,
    b: f64,
}

#[pymethods]
impl Bm25Inverted {
    #[new]
    fn new(
        term_offsets: Vec<usize>,
        posting_docs: Vec<u32>,
        posting_freqs: Vec<u32>,
        doc_lens: Vec<f64>,
        avgdl: f64,
        k1: f64,
        b: f64,
    ) -> Self {
        Bm25Inverted { term_offsets, posting_docs, posting_freqs, doc_lens, avgdl, k1, b }
    }

    fn top_k(
        &self,
        query_term_ids: Vec<u32>,
        query_idfs: Vec<f64>,
        k: usize,
    ) -> Vec<(f64, usize)> {
        let n_docs = self.doc_lens.len();
        let mut scores = vec![0.0f64; n_docs];
        let mut touched: Vec<usize> = Vec::new(); // docs we actually scored, to skip an O(n) scan

        for (qi, &tid) in query_term_ids.iter().enumerate() {
            let idf = query_idfs[qi];
            let start = self.term_offsets[tid as usize];
            let end = self.term_offsets[tid as usize + 1];
            for p in start..end {
                let doc = self.posting_docs[p] as usize;
                let f = self.posting_freqs[p] as f64;
                let dl = self.doc_lens[doc];
                let denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                if scores[doc] == 0.0 {
                    touched.push(doc);
                }
                scores[doc] += idf * (f * (self.k1 + 1.0)) / denom;
            }
        }

        let cmp = |a: &(f64, usize), b: &(f64, usize)| {
            b.0.total_cmp(&a.0).then(b.1.cmp(&a.1))
        };
        let mut scored: Vec<(f64, usize)> =
            touched.into_iter().map(|d| (scores[d], d)).collect();

        let kk = k.min(scored.len());
        if kk == 0 {
            return Vec::new();
        }
        if kk < scored.len() {
            scored.select_nth_unstable_by(kk - 1, cmp);
            scored.truncate(kk);
        }
        scored.sort_unstable_by(cmp);
        scored
    }

    /// Parallel variant: each task accumulates query terms into its OWN score
    /// buffer (no shared-write race), then the buffers are summed (reduce).
    /// Mostly here to demonstrate that at small per-query work the overhead
    /// (per-thread n_docs buffers + the reduce + load imbalance across uneven
    /// postings) outweighs the gain
    fn top_k_parallel(
        &self,
        query_term_ids: Vec<u32>,
        query_idfs: Vec<f64>,
        k: usize,
    ) -> Vec<(f64, usize)> {
        let n_docs = self.doc_lens.len();

        let scores = query_term_ids
            .par_iter()
            .zip(query_idfs.par_iter())
            .fold(
                || vec![0.0f64; n_docs],
                |mut acc, (&tid, &idf)| {
                    let start = self.term_offsets[tid as usize];
                    let end = self.term_offsets[tid as usize + 1];
                    for p in start..end {
                        let doc = self.posting_docs[p] as usize;
                        let f = self.posting_freqs[p] as f64;
                        let dl = self.doc_lens[doc];
                        let denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                        acc[doc] += idf * (f * (self.k1 + 1.0)) / denom;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n_docs],
                |mut a, b| {
                    for i in 0..n_docs {
                        a[i] += b[i];
                    }
                    a
                },
            );

        let cmp = |a: &(f64, usize), b: &(f64, usize)| {
            b.0.total_cmp(&a.0).then(b.1.cmp(&a.1))
        };
        let mut scored: Vec<(f64, usize)> = scores
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s > 0.0)
            .map(|(i, s)| (s, i))
            .collect();

        let kk = k.min(scored.len());
        if kk == 0 {
            return Vec::new();
        }
        if kk < scored.len() {
            scored.select_nth_unstable_by(kk - 1, cmp);
            scored.truncate(kk);
        }
        scored.sort_unstable_by(cmp);
        scored
    }
}

#[pymodule]
fn scholar_rag_bm25(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bm25Index>()?;
    m.add_class::<Bm25Inverted>()?;
    Ok(())
}