use std::alloc::{alloc, dealloc, Layout};
use std::cmp::Ordering;

#[repr(C)]
pub struct BrowserIndex {
    count: usize,
    dim: usize,
    words: usize,
    vectors: Vec<f32>,
    binary: Vec<u64>,
    query_bits: Vec<u64>,
    shortlist: Vec<(usize, u32)>,
    scored: Vec<(usize, f32)>,
}

#[no_mangle]
pub extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    alloc_typed::<f32>(len)
}

#[no_mangle]
/// # Safety
///
/// `ptr` must have been returned by [`alloc_f32`] for exactly `len` elements
/// and must not have been freed already.
pub unsafe extern "C" fn free_f32(ptr: *mut f32, len: usize) {
    free_typed(ptr, len);
}

#[no_mangle]
pub extern "C" fn alloc_u32(len: usize) -> *mut u32 {
    alloc_typed::<u32>(len)
}

#[no_mangle]
/// # Safety
///
/// `ptr` must have been returned by [`alloc_u32`] for exactly `len` elements
/// and must not have been freed already.
pub unsafe extern "C" fn free_u32(ptr: *mut u32, len: usize) {
    free_typed(ptr, len);
}

#[no_mangle]
/// # Safety
///
/// `vectors_ptr` must point to at least `count * dim` initialized `f32`
/// values for the duration of this call.
pub unsafe extern "C" fn index_new(
    vectors_ptr: *const f32,
    count: usize,
    dim: usize,
) -> *mut BrowserIndex {
    if vectors_ptr.is_null() || count == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    let Some(len) = count.checked_mul(dim) else {
        return std::ptr::null_mut();
    };
    let words = dim.div_ceil(64);
    let Some(binary_len) = count.checked_mul(words) else {
        return std::ptr::null_mut();
    };

    let vectors = unsafe { std::slice::from_raw_parts(vectors_ptr, len) }.to_vec();
    let mut binary = vec![0u64; binary_len];
    for row in 0..count {
        pack_sign_words(
            &vectors[row * dim..(row + 1) * dim],
            &mut binary[row * words..(row + 1) * words],
        );
    }

    Box::into_raw(Box::new(BrowserIndex {
        count,
        dim,
        words,
        vectors,
        binary,
        query_bits: vec![0u64; words],
        shortlist: Vec::with_capacity(count),
        scored: Vec::new(),
    }))
}

#[no_mangle]
/// # Safety
///
/// `index` must be null or a live pointer returned by [`index_new`], and it
/// must not be freed more than once.
pub unsafe extern "C" fn index_free(index: *mut BrowserIndex) {
    if !index.is_null() {
        unsafe {
            drop(Box::from_raw(index));
        }
    }
}

#[no_mangle]
/// # Safety
///
/// `index` must be null or a live pointer returned by [`index_new`].
pub unsafe extern "C" fn index_count(index: *const BrowserIndex) -> usize {
    let Some(index) = (unsafe { index.as_ref() }) else {
        return 0;
    };
    index.count
}

#[no_mangle]
/// # Safety
///
/// `index` must be null or a live pointer returned by [`index_new`].
pub unsafe extern "C" fn index_dim(index: *const BrowserIndex) -> usize {
    let Some(index) = (unsafe { index.as_ref() }) else {
        return 0;
    };
    index.dim
}

#[no_mangle]
/// # Safety
///
/// `index` must be a live pointer returned by [`index_new`]. `query_ptr`
/// must reference `index_dim(index)` initialized `f32` values. `out_ids` and
/// `out_scores` must each reference writable storage for `top_k` values.
pub unsafe extern "C" fn index_search(
    index: *mut BrowserIndex,
    query_ptr: *const f32,
    dense_n: usize,
    top_k: usize,
    out_ids: *mut u32,
    out_scores: *mut f32,
) -> usize {
    let Some(index) = (unsafe { index.as_mut() }) else {
        return 0;
    };
    if query_ptr.is_null()
        || out_ids.is_null()
        || out_scores.is_null()
        || dense_n == 0
        || top_k == 0
    {
        return 0;
    }

    let query = unsafe { std::slice::from_raw_parts(query_ptr, index.dim) };
    pack_sign_words(query, &mut index.query_bits);

    let shortlist_n = dense_n.max(top_k).min(index.count);
    index.shortlist.clear();
    for doc_id in 0..index.count {
        let start = doc_id * index.words;
        let distance = hamming_words(&index.query_bits, &index.binary[start..start + index.words]);
        index.shortlist.push((doc_id, distance));
    }
    if shortlist_n < index.shortlist.len() {
        index
            .shortlist
            .select_nth_unstable_by(shortlist_n, |left, right| left.1.cmp(&right.1));
        index.shortlist.truncate(shortlist_n);
    }

    index.scored.clear();
    index.scored.reserve(shortlist_n);
    for (doc_id, _) in &index.shortlist {
        let start = *doc_id * index.dim;
        let score = dot(query, &index.vectors[start..start + index.dim]);
        index.scored.push((*doc_id, score));
    }
    index
        .scored
        .sort_unstable_by(|left, right| cmp_f32_desc(left.1, right.1));

    let result_len = top_k.min(index.scored.len());
    let out_ids = unsafe { std::slice::from_raw_parts_mut(out_ids, top_k) };
    let out_scores = unsafe { std::slice::from_raw_parts_mut(out_scores, top_k) };
    for (position, (doc_id, score)) in index.scored.iter().take(result_len).enumerate() {
        out_ids[position] = *doc_id as u32;
        out_scores[position] = *score;
    }
    result_len
}

fn alloc_typed<T>(len: usize) -> *mut T {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::array::<T>(len) else {
        return std::ptr::null_mut();
    };
    unsafe { alloc(layout) as *mut T }
}

fn free_typed<T>(ptr: *mut T, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::array::<T>(len) {
        unsafe {
            dealloc(ptr.cast::<u8>(), layout);
        }
    }
}

fn pack_sign_words(vector: &[f32], output: &mut [u64]) {
    output.fill(0);
    for (index, value) in vector.iter().enumerate() {
        if *value > 0.0 {
            output[index / 64] |= 1u64 << (index % 64);
        }
    }
}

fn hamming_words(left: &[u64], right: &[u64]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cmp_f32_desc(left: f32, right: f32) -> Ordering {
    right.total_cmp(&left)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_vector_is_ranked_first() {
        let vectors = [
            1.0, -1.0, -1.0, -1.0, //
            -1.0, 1.0, -1.0, -1.0, //
            -1.0, -1.0, 1.0, -1.0, //
            -1.0, -1.0, -1.0, 1.0, //
        ];
        unsafe {
            let index = index_new(vectors.as_ptr(), 4, 4);
            assert!(!index.is_null());

            let query = [0.9, -1.0, -1.0, -1.0];
            let mut ids = [u32::MAX; 2];
            let mut scores = [0.0f32; 2];
            let found = index_search(
                index,
                query.as_ptr(),
                4,
                2,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
            );
            assert_eq!(found, 2);
            assert_eq!(ids[0], 0);
            index_free(index);
        }
    }
}
