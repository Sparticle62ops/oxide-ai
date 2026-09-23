use oxide_ai_pssa::checkpoint;
use oxide_ai_pssa::cli::{CLIHandler, TrainingOptions};
use oxide_ai_pssa::dataset::TokenizerKind;
use serde_json::Value;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "oxide-training-artifacts-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos()
    ))
}

fn word_options() -> TrainingOptions {
    TrainingOptions {
        epochs: 2,
        latent: 4,
        state: 2,
        key: 2,
        memory: 2,
        chunk: 4,
        lr: 0.005,
        accumulate: 1,
        warmup_steps: 0,
        seed: 19,
        max_tokens: None,
        tokenizer: TokenizerKind::Word,
        vocab_size: 2048,
            resume: None,
    }
}

fn corpus() -> &'static str {
    "alpha beta gamma delta epsilon\nalpha beta gamma delta epsilon\nalpha beta gamma delta epsilon\n"
}

#[test]
fn word_run_has_complete_artifacts_and_matches_pure_training() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let data = temp("word.txt");
    let output = temp("word.pssa");
    let run_dir = temp("word-run");
    let pure = temp("pure.pssa");
    fs::write(&data, corpus()).unwrap();

    let status = Command::new(exe)
        .args([
            "train",
            data.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
            "--tokenizer",
            "word",
            "--epochs",
            "2",
            "--latent",
            "4",
            "--state",
            "2",
            "--key",
            "2",
            "--memory",
            "2",
            "--chunk",
            "4",
            "--lr",
            "0.005",
            "--accumulate",
            "1",
            "--seed",
            "19",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(output.exists());
    assert!(run_dir.join("run.json").exists());
    assert!(run_dir.join("metrics.json").exists());
    for epoch in 0..=2 {
        assert!(run_dir.join(format!("epoch-{epoch:04}.pssa")).exists());
    }

    let run: Value = serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["tokenizer"]["kind"], "word");
    assert_eq!(run["resume"]["supported"], false);
    assert_eq!(
        run["scope"]["tokenizer"],
        "tokenizer fitted on all supplied training text"
    );
    let metrics: Vec<Value> =
        serde_json::from_slice(&fs::read(run_dir.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0]["epoch"], 0);
    assert!(metrics[0]["cross_entropy"].is_null());
    for epoch in 0..=2 {
        assert_eq!(metrics[epoch]["epoch"], epoch);
        let loaded =
            checkpoint::load_checkpoint(run_dir.join(format!("epoch-{epoch:04}.pssa"))).unwrap();
        assert_eq!(
            metrics[epoch]["optimizer_updates"].as_u64().unwrap() as usize,
            loaded.model.step_counter
        );
        assert_eq!(
            metrics[epoch]["checkpoint"],
            format!("epoch-{epoch:04}.pssa")
        );
    }
    assert!(metrics[1]["scored_transitions"].as_u64().unwrap() > 0);
    assert!(
        metrics[2]["optimizer_updates"].as_u64().unwrap()
            > metrics[1]["optimizer_updates"].as_u64().unwrap()
    );

    let (pure_model, _) = CLIHandler::train_corpus(corpus(), &word_options()).unwrap();
    checkpoint::save_model(&pure_model, &pure).unwrap();
    assert_eq!(
        fs::read(run_dir.join("epoch-0002.pssa")).unwrap(),
        fs::read(&pure).unwrap()
    );

    let _ = fs::remove_file(data);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(pure);
    let _ = fs::remove_dir_all(run_dir);
}

#[test]
fn bpe_epoch_checkpoint_generates_without_training_corpus() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let data = temp("bpe.txt");
    let output = temp("bpe.pssa");
    let run_dir = temp("bpe-run");
    fs::write(
        &data,
        "A small byte pair corpus for an embedded tokenizer.\nA second training article follows here.\n",
    )
    .unwrap();
    let status = Command::new(exe)
        .args([
            "train",
            data.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
            "--run-dir",
            run_dir.to_str().unwrap(),
            "--epochs",
            "1",
            "--vocab-size",
            "257",
            "--latent",
            "4",
            "--state",
            "2",
            "--key",
            "2",
            "--memory",
            "2",
            "--chunk",
            "8",
            "--accumulate",
            "1",
            "--seed",
            "23",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let epoch = run_dir.join("epoch-0001.pssa");
    assert!(epoch.exists());
    fs::remove_file(&data).unwrap();
    let generated = Command::new(exe)
        .args([
            "generate",
            "A small",
            "--model",
            epoch.to_str().unwrap(),
            "--temp",
            "0",
            "--max-new-tokens",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );

    let _ = fs::remove_file(output);
    let _ = fs::remove_dir_all(run_dir);
}

#[test]
fn final_save_failure_keeps_completed_epoch_artifacts() {
    let root = temp("final-failure");
    fs::create_dir(&root).unwrap();
    let data = root.join("train.txt");
    let run = root.join("run");
    fs::write(&data, corpus()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args([
            "train",
            data.to_str().unwrap(),
            "--run-dir",
            run.to_str().unwrap(),
            "--out",
            root.to_str().unwrap(),
            "--tokenizer",
            "word",
            "--epochs",
            "1",
            "--latent",
            "4",
            "--state",
            "2",
            "--key",
            "2",
            "--memory",
            "2",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot save checkpoint"));
    let records: Vec<Value> =
        serde_json::from_slice(&fs::read(run.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(records.len(), 2);
    let epoch = checkpoint::load_checkpoint(run.join("epoch-0001.pssa")).unwrap();
    assert_eq!(
        epoch.model.step_counter as u64,
        records[1]["optimizer_updates"].as_u64().unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn final_output_cannot_overwrite_run_metadata() {
    let run = temp("reserved-output");
    let output = Command::new(env!("CARGO_BIN_EXE_oxide_ai_pssa"))
        .args([
            "train",
            "/definitely/missing/dataset",
            "--run-dir",
            run.to_str().unwrap(),
            "--out",
            run.join("metrics.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("reserved run artifact"));
    assert!(fs::read_dir(&run).unwrap().next().is_none());
    fs::remove_dir_all(run).unwrap();
}

#[test]
fn run_directory_must_be_new_and_is_checked_before_dataset_loading() {
    let exe = env!("CARGO_BIN_EXE_oxide_ai_pssa");
    let existing = temp("existing-run");
    fs::create_dir(&existing).unwrap();
    let sentinel = existing.join("sentinel");
    fs::write(&sentinel, b"keep").unwrap();
    let status = Command::new(exe)
        .args([
            "train",
            "/definitely/missing/dataset",
            "--run-dir",
            existing.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    assert!(!existing.join("run.json").exists());
    fs::remove_dir_all(existing).unwrap();

    let empty = temp("empty-run");
    fs::create_dir(&empty).unwrap();
    let status = Command::new(exe)
        .args([
            "train",
            "/definitely/missing/dataset",
            "--run-dir",
            empty.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert!(empty.exists());
    assert!(fs::read_dir(&empty).unwrap().next().is_none());
    fs::remove_dir_all(empty).unwrap();
}
