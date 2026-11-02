"""Offline integration check with a random tiny model, not benchmark scores."""
from pathlib import Path
import ast
import tempfile
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import nbformat
import matplotlib
matplotlib.use("Agg")
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, BertTokenizer

ROOT = Path(__file__).resolve().parent
notebook = nbformat.read(ROOT / "ag_news_classification.ipynb", as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
print("PASS: notebook schema and Python syntax")
torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix="ag_news_check_") as temporary:
    temporary = Path(temporary)
    model_dir = temporary / "tiny_model"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "team", "world", "stock", "new"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(model_dir)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=len(vocab), n_layers=1, n_heads=2, dim=16, hidden_dim=32,
        max_position_embeddings=128, num_labels=4))
    model.save_pretrained(model_dir)
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "RUN_DIR = ROOT" in source:
            source = source.replace('ROOT / "outputs"', 'Path(' + repr(str(temporary)) + ')')
        exec(compile(source, f"cell_{index}", "exec"), scope)
        if "RUN_DIR = ROOT" in cell.source:
            scope.update(MODEL_NAME=str(model_dir), DEVICE=torch.device("cpu"), BATCH_SIZE=4)
            scope["plt"].show = lambda: scope["plt"].close("all")
        if "def read_news" in source:
            assert scope["clean_text"]("hello\\world &amp; test") == "hello world & test"
            assert len(scope["raw_train"]) == 120000
            for name in ("train_df", "val_df", "test_df"):
                scope[name] = scope[name].groupby("label").head(2).reset_index(drop=True)
        if "def train_experiment" in source:
            frozen = scope["build_model"]("frozen_encoder")
            assert all(not p.requires_grad for p in frozen.base_model.parameters())
            before = {n: p.detach().clone() for n, p in frozen.named_parameters()}
            optimizer = torch.optim.AdamW([p for p in frozen.parameters() if p.requires_grad], lr=0.01)
            frozen.train()
            frozen.base_model.eval()
            batch = next(iter(scope["make_loader"](scope["train_data"])))
            frozen(**batch).loss.backward()
            optimizer.step()
            assert all(torch.equal(before[n], p) for n, p in frozen.named_parameters() if not p.requires_grad)
            assert any(not torch.equal(before[n], p) for n, p in frozen.named_parameters() if p.requires_grad)
            full = scope["build_model"]("fine_tuning")
            assert all(p.requires_grad for p in full.parameters())
            print("PASS: frozen backbone unchanged; head updates; fine-tuning enables all parameters")
    assert scope["comparison"].shape[0] == 2
    assert (scope["RUN_DIR"] / "comparison.csv").exists()
    assert len(scope["predict_news"]("A new science discovery")) == 1
    try:
        scope["predict_news"]("")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty input should fail")
print("PASS: training, validation, checkpoints, test reports, plots and inference")
print("Tiny random model only: results are not AG News benchmark metrics.")
