// Byte-level BPE tokenizer, ported from tokenizer.mojo (same algorithm,
// same GPT-2 byte<->codepoint remap, same ASCII-focused pretokenizer).

use crate::gguf::Gguf;
use std::collections::HashMap;

pub fn byte_to_cp(b: u32) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
        b
    } else if b <= 32 {
        256 + b
    } else if (127..=160).contains(&b) {
        289 + (b - 127)
    } else {
        323
    }
}

pub fn cp_to_byte(cp: u32) -> u32 {
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        cp
    } else if (256..=288).contains(&cp) {
        cp - 256
    } else if (289..=322).contains(&cp) {
        cp - 289 + 127
    } else if cp == 323 {
        173
    } else {
        panic!("byte-level codepoint out of range: {}", cp)
    }
}

fn byte_encode(piece: &str) -> String {
    piece
        .bytes()
        .map(|b| char::from_u32(byte_to_cp(b as u32)).unwrap())
        .collect()
}

pub struct Tokenizer {
    pub id_to_tok: Vec<String>,
    pub tok_to_id: HashMap<String, i64>,
    pub merge_rank: HashMap<String, i64>,
    pub bos: i64,
    pub eos: i64,
}

pub fn load_tokenizer(g: &Gguf) -> Tokenizer {
    assert!(!g.tokens.is_empty(), "GGUF has no tokenizer.ggml.tokens");
    let id_to_tok = g.tokens.clone();
    let mut tok_to_id = HashMap::with_capacity(id_to_tok.len());
    for (i, t) in id_to_tok.iter().enumerate() {
        tok_to_id.insert(t.clone(), i as i64);
    }
    let mut merge_rank = HashMap::with_capacity(g.merges.len());
    for (i, m) in g.merges.iter().enumerate() {
        merge_rank.insert(m.clone(), i as i64);
    }
    Tokenizer {
        id_to_tok,
        tok_to_id,
        merge_rank,
        bos: 128000,
        eos: 128001,
    }
}

fn is_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
}
fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}
fn is_space(c: char) -> bool {
    c == ' ' || (('\u{9}'..='\u{d}').contains(&c))
}

fn pretokenize(text: &str) -> Vec<String> {
    let cps: Vec<char> = text.chars().collect();
    let n = cps.len();
    let mut pieces = Vec::new();
    let mut i = 0usize;
    while i < n {
        let c = cps[i];
        let lead_space = c == ' ' && i + 1 < n && is_letter(cps[i + 1]);
        if is_letter(c) || lead_space {
            let start = i;
            if lead_space {
                i += 1;
            }
            while i < n && is_letter(cps[i]) {
                i += 1;
            }
            pieces.push(cps[start..i].iter().collect());
            continue;
        }
        if is_space(c) {
            let j0 = i;
            let mut j = i;
            while j < n && is_space(cps[j]) {
                j += 1;
            }
            pieces.push(cps[j0..j].iter().collect());
            i = j;
            continue;
        }
        if is_digit(c) {
            let start = i;
            while i < n && is_digit(cps[i]) {
                i += 1;
            }
            let mut d = start;
            while i - d > 0 {
                let take = if i - d >= 3 { 3 } else { i - d };
                pieces.push(cps[d..d + take].iter().collect());
                d += take;
            }
            continue;
        }
        let start = i;
        while i < n && !is_space(cps[i]) && !is_letter(cps[i]) && !is_digit(cps[i]) {
            i += 1;
        }
        pieces.push(cps[start..i].iter().collect());
    }
    pieces
}

fn bpe(t: &Tokenizer, word: &str) -> Vec<String> {
    let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    if syms.len() <= 1 {
        return syms;
    }
    loop {
        let mut best_rank = i64::MAX;
        let mut best_i: isize = -1;
        for k in 0..syms.len() - 1 {
            let key = format!("{} {}", syms[k], syms[k + 1]);
            if let Some(&r) = t.merge_rank.get(&key) {
                if r < best_rank {
                    best_rank = r;
                    best_i = k as isize;
                }
            }
        }
        if best_i < 0 {
            break;
        }
        let best_i = best_i as usize;
        let merged = format!("{}{}", syms[best_i], syms[best_i + 1]);
        let mut ns = Vec::with_capacity(syms.len() - 1);
        for (k, s) in syms.iter().enumerate() {
            if k == best_i {
                ns.push(merged.clone());
            } else if k == best_i + 1 {
                continue;
            } else {
                ns.push(s.clone());
            }
        }
        syms = ns;
    }
    syms
}

pub fn encode(t: &Tokenizer, text: &str, add_bos: bool) -> Vec<i64> {
    let mut out = Vec::new();
    if add_bos {
        out.push(t.bos);
    }
    for piece in pretokenize(text) {
        let benc = byte_encode(&piece);
        for tok in bpe(t, &benc) {
            match t.tok_to_id.get(&tok) {
                Some(&id) => out.push(id),
                None => panic!("BPE piece not in vocab: '{}'", tok),
            }
        }
    }
    out
}

pub fn decode(t: &Tokenizer, ids: &[i64], skip_special: bool) -> String {
    let mut bytes = Vec::new();
    for &tid in ids {
        if skip_special && tid >= 128000 {
            continue;
        }
        let piece = &t.id_to_tok[tid as usize];
        for c in piece.chars() {
            bytes.push(cp_to_byte(c as u32) as u8);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
