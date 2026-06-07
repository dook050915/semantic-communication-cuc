from pathlib import Path

from data_utils import load_corpus_from_dir, save_sentences


PROJECT_ROOT = Path(__file__).resolve().parents[1]


config = {
    "raw_data_dir": "data/raw/europarl-v7/txt/en",
    "output_path": "data/processed/europarl_en_50k.txt",
    "min_len": 4,
    "max_len": 30,
    "max_sentences": 50000,
}


def resolve_path(path):
    """把配置里的相对路径转成基于项目根目录的绝对路径。"""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    """从 Europarl 原始目录抽取句子，并保存成 processed 文本文件。"""
    raw_data_dir = resolve_path(config["raw_data_dir"])
    output_path = resolve_path(config["output_path"])

    sentences = load_corpus_from_dir(
        str(raw_data_dir),
        min_len=config["min_len"],
        max_len=config["max_len"],
        max_sentences=config["max_sentences"],
    )
    save_sentences(sentences, str(output_path))
    print(f"saved {len(sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
