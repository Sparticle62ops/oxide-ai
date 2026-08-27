use std::collections::{HashMap, HashSet};

/// Zero-copy sequence chunking iterator for TBPTT training batches (Inputs, Targets)
pub struct TokenChunkIterator<'a> {
    tokens: &'a [usize],
    chunk_len: usize,
    cursor: usize,
}

impl<'a> TokenChunkIterator<'a> {
    pub fn new(tokens: &'a [usize], chunk_len: usize) -> Self {
        Self {
            tokens,
            chunk_len: chunk_len.max(1),
            cursor: 0,
        }
    }
}

impl<'a> Iterator for TokenChunkIterator<'a> {
    type Item = (&'a [usize], &'a [usize]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor + 1 >= self.tokens.len() {
            return None;
        }

        let remaining = (self.tokens.len() - 1) - self.cursor;
        let len = self.chunk_len.min(remaining);
        if len == 0 {
            return None;
        }

        let inputs = &self.tokens[self.cursor..self.cursor + len];
        let targets = &self.tokens[self.cursor + 1..self.cursor + 1 + len];
        self.cursor += len;

        Some((inputs, targets))
    }
}

pub struct Tokenizer {
    pub vocab_size: usize,
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub synsets: Vec<HashSet<String>>,
    pub token_counts: Vec<usize>,
    pub unigram_table: Vec<usize>,
}

impl Tokenizer {
    fn clean_and_tokenize(text: &str, lower: bool) -> Vec<String> {
        let mut tokens = Vec::new();

        for raw in text.split_whitespace() {
            let processed = if lower {
                raw.to_lowercase()
            } else {
                raw.to_string()
            };

            let mut word_buf = String::new();
            for c in processed.chars() {
                if c.is_alphanumeric() || c == '-' || c == '\'' {
                    word_buf.push(c);
                } else if c == '.' || c == ',' || c == '?' || c == '!' {
                    if !word_buf.is_empty() {
                        tokens.push(word_buf.clone());
                        word_buf.clear();
                    }
                    tokens.push(c.to_string());
                }
            }

            if !word_buf.is_empty() {
                tokens.push(word_buf);
            }
        }

        tokens
    }

    pub fn from_corpus(corpus: &str, lower: bool) -> Self {
        let raw_tokens = Self::clean_and_tokenize(corpus, lower);

        let mut frequency_map: HashMap<String, usize> = HashMap::new();
        for tok in &raw_tokens {
            *frequency_map.entry(tok.clone()).or_insert(0) += 1;
        }

        let mut sorted_tokens: Vec<(String, usize)> = frequency_map.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut token_counts = Vec::new();

        token_to_id.insert("<unk>".to_string(), 0);
        id_to_token.insert(0, "<unk>".to_string());
        token_counts.push(0);

        let mut idx = 1;
        let max_vocab_cap = 10_000;
        let mut unk_count = 0;
        let min_freq = if raw_tokens.len() < 1000 { 1 } else { 2 };

        for (word, count) in sorted_tokens {
            if idx >= max_vocab_cap {
                unk_count += count;
                continue;
            }
            if count >= min_freq || word == "." || word == "," {
                token_to_id.insert(word.clone(), idx);
                id_to_token.insert(idx, word);
                token_counts.push(count);
                idx += 1;
            } else {
                unk_count += count;
            }
        }

        token_counts[0] = unk_count.max(1);

        let table_size = 100_000;
        let mut unigram_table = Vec::with_capacity(table_size);

        let mut sum_pow = 0.0f64;
        for &count in &token_counts {
            sum_pow += (count as f64).powf(0.75);
        }

        if sum_pow > 0.0 && !token_counts.is_empty() {
            let mut curr_token = 0;
            let mut cumulative_prob = (token_counts[0] as f64).powf(0.75) / sum_pow;

            for i in 0..table_size {
                let p = (i as f64) / (table_size as f64);
                while p > cumulative_prob && curr_token + 1 < idx {
                    curr_token += 1;
                    cumulative_prob += (token_counts[curr_token] as f64).powf(0.75) / sum_pow;
                }
                unigram_table.push(curr_token);
            }
        } else {
            for i in 0..table_size {
                unigram_table.push(i % idx.max(1));
            }
        }

        let mut synsets = Vec::new();
        let mut syn1 = HashSet::new();
        syn1.insert("fast".to_string());
        syn1.insert("rapid".to_string());
        synsets.push(syn1);

        Self {
            vocab_size: idx,
            token_to_id,
            id_to_token,
            synsets,
            token_counts,
            unigram_table,
        }
    }

    pub fn encode(&self, text: &str, lower: bool) -> Vec<usize> {
        let raw_tokens = Self::clean_and_tokenize(text, lower);
        raw_tokens
            .into_iter()
            .map(|w| self.token_to_id.get(&w).copied().unwrap_or(0))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn are_synonyms(&self, id_a: usize, id_b: usize) -> bool {
        if let (Some(w_a), Some(w_b)) = (self.id_to_token.get(&id_a), self.id_to_token.get(&id_b)) {
            for set in &self.synsets {
                if set.contains(w_a) && set.contains(w_b) {
                    return true;
                }
            }
        }
        false
    }
}

pub struct DatasetManager;

impl DatasetManager {
    pub const SCIENCE_REFERENCE_CORPUS: &'static str = r#"
the solar system consists of the sun and the planetary objects orbiting it .
the four inner terrestrial planets are mercury , venus , earth , and mars , composed primarily of rock and metal .
quantum mechanics is the branch of physics studying the behavior of matter and light at atomic scale .
computer science is the study of computation , information , and the theoretical foundations of computation .
artificial intelligence focuses on building computational models and software that learn .
photosynthesis is the biological process used by plants to convert light into energy .
dna contains the genetic instructions necessary for the development and reproduction of living organisms .
social science is the study of human societies and interconnected relationships .
data science involves analyzing large volumes of information to extract patterns .
"#;

    pub fn load_dataset(sources_arg: Option<&str>) -> String {
        let raw_arg = match sources_arg {
            Some(s) if !s.trim().is_empty() => s,
            _ => return Self::SCIENCE_REFERENCE_CORPUS.to_string(),
        };

        let mut combined_corpus = String::new();
        let sources: Vec<&str> = raw_arg
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        for source in sources {
            if source == "science" {
                combined_corpus.push_str(Self::SCIENCE_REFERENCE_CORPUS);
                combined_corpus.push('\n');
            } else if source.starts_with("hf:") {
                let repo = &source[3..];
                if let Ok(content) = Self::download_huggingface_dataset(repo) {
                    combined_corpus.push_str(&content);
                    combined_corpus.push('\n');
                }
            } else if source.starts_with("http://") || source.starts_with("https://") {
                if let Ok(content) = Self::download_url_dataset(source) {
                    combined_corpus.push_str(&content);
                    combined_corpus.push('\n');
                }
            } else {
                let path = std::path::Path::new(source);
                if path.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(path) {
                        for entry in entries.flatten() {
                            let file_path = entry.path();
                            if let Ok(content) = std::fs::read_to_string(&file_path) {
                                let cleaned = Self::extract_clean_text(&content);
                                combined_corpus.push_str(&cleaned);
                                combined_corpus.push('\n');
                            }
                        }
                    }
                } else if path.exists() {
                    if let Ok(content) = std::fs::read_to_string(path) {
                        let cleaned = Self::extract_clean_text(&content);
                        combined_corpus.push_str(&cleaned);
                        combined_corpus.push('\n');
                    }
                } else if let Ok(content) = Self::download_huggingface_dataset(source) {
                    combined_corpus.push_str(&content);
                    combined_corpus.push('\n');
                }
            }
        }

        if combined_corpus.trim().is_empty() {
            Self::SCIENCE_REFERENCE_CORPUS.to_string()
        } else {
            combined_corpus
        }
    }

    pub fn download_url_dataset(url: &str) -> Result<String, String> {
        println!("===> Connecting to: {}", url);
        let resp = ureq::get(url)
            .set("User-Agent", "oxide-ai/0.4.0")
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let body = resp
            .into_string()
            .map_err(|e| format!("Failed to read response body: {}", e))?;
        let extracted = Self::extract_clean_text(&body);

        if extracted.is_empty() {
            Ok(body)
        } else {
            Ok(extracted)
        }
    }

    pub fn download_huggingface_dataset(dataset_repo: &str) -> Result<String, String> {
        println!("===> Querying Hugging Face repository: '{}'", dataset_repo);

        let server_endpoints = [
            format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&split=train&offset=0&limit=1000",
                dataset_repo
            ),
            format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config=default&split=train&offset=0&limit=1000",
                dataset_repo
            ),
            format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config=20231101.en&split=train&offset=0&limit=1000",
                dataset_repo
            ),
            format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config=plain_text&split=train&offset=0&limit=1000",
                dataset_repo
            ),
        ];

        for endpoint in &server_endpoints {
            if let Ok(content) = Self::download_url_dataset(endpoint) {
                let parsed_text = Self::extract_clean_text(&content);
                if parsed_text.split_whitespace().count() > 50 {
                    println!(
                        "===> Extracted {} words from HuggingFace server",
                        parsed_text.split_whitespace().count()
                    );
                    return Ok(parsed_text);
                }
            }
        }

        let raw_endpoints = [
            format!(
                "https://huggingface.co/datasets/{}/raw/main/input.txt",
                dataset_repo
            ),
            format!(
                "https://huggingface.co/datasets/{}/raw/main/data.txt",
                dataset_repo
            ),
            format!(
                "https://huggingface.co/datasets/{}/raw/main/train.txt",
                dataset_repo
            ),
            format!(
                "https://huggingface.co/datasets/{}/raw/main/data.jsonl",
                dataset_repo
            ),
            format!(
                "https://huggingface.co/datasets/{}/raw/main/data.json",
                dataset_repo
            ),
        ];

        for endpoint in &raw_endpoints {
            if let Ok(content) = Self::download_url_dataset(endpoint) {
                let parsed_text = Self::extract_clean_text(&content);
                let final_text = if parsed_text.split_whitespace().count() > 20 {
                    parsed_text
                } else {
                    content
                };
                if final_text.split_whitespace().count() > 20 {
                    return Ok(final_text);
                }
            }
        }

        Err(format!(
            "Could not stream raw text from HuggingFace dataset '{}'",
            dataset_repo
        ))
    }

    pub fn extract_clean_text(raw: &str) -> String {
        let trimmed = raw.trim();
        let target_keys = [
            "\"text\":",
            "\"content\":",
            "\"article\":",
            "\"story\":",
            "\"instruction\":",
            "\"output\":",
            "\"sentence\":",
            "\"summary\":",
        ];

        let mut extracted_sentences = Vec::new();

        for &key in &target_keys {
            let mut search_from = 0;
            while let Some(pos) = raw[search_from..].find(key) {
                let key_pos = search_from + pos + key.len();
                if let Some(quote_start) = raw[key_pos..].find('"') {
                    let val_start = key_pos + quote_start + 1;
                    let mut val_end = val_start;
                    let bytes = raw.as_bytes();

                    while val_end < bytes.len() {
                        if bytes[val_end] == b'"' && (val_end == 0 || bytes[val_end - 1] != b'\\') {
                            break;
                        }
                        val_end += 1;
                    }

                    if val_end <= raw.len() && val_end > val_start {
                        let unescaped = raw[val_start..val_end]
                            .replace("\\n", " ")
                            .replace("\\\"", "\"")
                            .replace("\\\\", "\\");

                        let sentence = unescaped.trim();
                        if sentence.split_whitespace().count() >= 3 {
                            extracted_sentences.push(sentence.to_string());
                        }
                    }
                    search_from = (val_end + 1).min(raw.len());
                } else {
                    break;
                }
            }
        }

        if !extracted_sentences.is_empty() {
            extracted_sentences.join("\n")
        } else {
            trimmed.to_string()
        }
    }

    pub fn contradiction_stream() -> String {
        let mut s = String::new();
        s.push_str("the secret access code is 9988 . ");
        s.push_str("system authentication protocol initiated . ");
        s.push_str("network firewall parameters configured . ");
        s.push_str("security credentials updated across distributed nodes . ");
        s.push_str("the secret access code is 1122 . ");
        s.push_str("the secret access code is 1122 . ");
        s.push_str("the secret access code is 1122 . ");
        s
    }

    pub fn mqar_stream(_gap: usize) -> String {
        let pairs = [
            ("manifold", "orthogonal"),
            ("gradient", "convergent"),
            ("tensor", "divergent"),
            ("vector", "decayed"),
            ("entropy", "stable"),
            ("quantum", "active"),
        ];
        let distractors = [
            "the quick brown fox jumps over the lazy dog .",
            "state space models compute sequential representations with linear complexity .",
            "hyperbolic geometry enables hierarchical embedding representations with constant distortion .",
            "continual learning architectures maintain plastic memory without catastrophic interference .",
            "predictive coding minimizes free energy through top down expectation generation .",
            "synaptic consolidation transfers fast episodic weights into slow neocortical matrices .",
        ];
        let mut s = String::new();
        for (i, &(k, v)) in pairs.iter().enumerate() {
            s.push_str(&format!("query {} is {} . ", k, v));
            s.push_str(distractors[i % distractors.len()]);
            s.push(' ');
        }
        s
    }

    pub fn spam_attack_stream(burst_count: usize) -> String {
        let mut s = String::new();
        for _ in 0..6 {
            s.push_str("the speed of light is 300000 . ");
        }
        for _ in 0..burst_count {
            s.push_str("the speed of light is 500 . ");
        }
        s
    }
}