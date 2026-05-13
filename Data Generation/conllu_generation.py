from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "Subtitles Data"
OUTPUT_FOLDER = BASE_DIR / "Subtitles Data/conllu/"

LANGUAGE = "es"  # Adjust as needed (e.g., "en", "de", "es")
USE_GPU = True

# Adjust depending on VRAM
DOC_BATCH_SIZE = 64

def main():
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU not detected. Running on CPU.")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Find all .ES files recursively
    de_files = sorted(INPUT_FOLDER.rglob("*.ES"))

    total_files = len(de_files)
    print(f"Found {total_files} .ES files.")

    if total_files == 0:
        return

    print(f"\nLoading Stanza pipeline (GPU={USE_GPU})...")

    nlp = stanza.Pipeline(
        lang=LANGUAGE,
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,
        tokenize_batch_size=2000,
        batch_size=DOC_BATCH_SIZE
    )

    processed_count = 0
    failed_count = 0

    for batch_start in range(0, total_files, DOC_BATCH_SIZE):

        batch_end = min(batch_start + DOC_BATCH_SIZE, total_files)

        batch_paths_raw = de_files[batch_start:batch_end]

        batch_paths = []

        # Skip existing outputs
        for src_path in batch_paths_raw:

            relative_path = src_path.relative_to(INPUT_FOLDER)

            out_path = (
                OUTPUT_FOLDER /
                relative_path.with_suffix(".conllu")
            )

            if not out_path.exists():
                batch_paths.append(src_path)
            else:
                processed_count += 1

        print(
            f"\nBatch {batch_start // DOC_BATCH_SIZE + 1} "
            f"({batch_start + 1}–{batch_end} of {total_files})"
        )

        print(
            f"Already exist (skipped): "
            f"{len(batch_paths_raw) - len(batch_paths)}"
        )

        if not batch_paths:
            print("Nothing to process in this batch.")
            continue

        texts = []
        keep_paths = []

        # Read files
        for src_path in batch_paths:

            try:
                text = src_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                if not text.strip():
                    processed_count += 1
                    continue

                texts.append(text)
                keep_paths.append(src_path)

            except Exception as e:
                print(f"Error reading {src_path}: {e}")
                failed_count += 1

        if not texts:
            print("All files in this batch were empty or unreadable.")
            continue

        try:
            # Multi-document processing
            in_docs = [stanza.Document([], text=t) for t in texts]

            out_docs = nlp(in_docs)

            if not isinstance(out_docs, list):
                out_docs = [out_docs]

            for src_path, doc in zip(keep_paths, out_docs):

                try:
                    relative_path = src_path.relative_to(INPUT_FOLDER)

                    out_path = (
                        OUTPUT_FOLDER /
                        relative_path.with_suffix(".conllu")
                    )

                    # Create subdirectories if needed
                    out_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    CoNLL.write_doc2conll(doc, str(out_path))

                    processed_count += 1

                except Exception as e:
                    print(f"Error writing {src_path}: {e}")
                    failed_count += 1

            print(
                f"Processed: {processed_count} / {total_files} "
                f"(failed so far: {failed_count})"
            )

        except Exception as e:
            print(
                f"Error in NLP batch "
                f"{batch_start + 1}–{batch_end}: {e}"
            )

            failed_count += len(keep_paths)

    print("\nAll batches completed.")
    print(f"Total successful: {processed_count}")
    print(f"Total failed: {failed_count}")


if __name__ == "__main__":
    main()