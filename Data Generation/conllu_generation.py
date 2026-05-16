from pathlib import Path
import stanza
from stanza.utils.conll import CoNLL

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FOLDER = BASE_DIR / "Subtitles Data"
OUTPUT_FOLDER = BASE_DIR / "Subtitles Data/conllu"

LANGUAGE = "de"
USE_GPU = True

FILE_EXTENSION = "*.DE"

# =====================================================
# READ LINES
# =====================================================

def read_lines(path):

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines()

    return [
        line.strip()
        for line in lines
        if line.strip()
    ]


# =====================================================
# MAIN
# =====================================================

def main():

    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_FOLDER.rglob(FILE_EXTENSION)
    )

    print(f"Found {len(files)} files.")

    # =================================================
    # LOAD PIPELINE
    # =================================================

    nlp = stanza.Pipeline(
        lang=LANGUAGE,
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=USE_GPU,

        # IMPORTANT
        tokenize_no_ssplit=True
    )

    # =================================================
    # PROCESS FILES
    # =================================================

    for file_id, src_path in enumerate(files, start=1):

        print(f"\n[{file_id}/{len(files)}] {src_path.name}")

        relative = src_path.relative_to(INPUT_FOLDER)

        out_path = (
            OUTPUT_FOLDER /
            relative.with_suffix(".conllu")
        )

        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if out_path.exists():
            print("Skipping existing.")
            continue

        lines = read_lines(src_path)

        print(f"Input aligned lines: {len(lines)}")

        processed_docs = []

        # =============================================
        # PROCESS EACH LINE SEPARATELY
        # =============================================

        for idx, line in enumerate(lines, start=1):

            try:

                # EXACTLY ONE LINE
                doc = nlp(line)

                processed_docs.append(doc)

            except Exception as e:

                print(
                    f"Error at line {idx}: {e}"
                )

        print(
            f"Processed docs: {len(processed_docs)}"
        )
        # =============================================
        # WRITE CONLLU (FIXED)
        # =============================================

        with open(out_path, "w", encoding="utf-8") as f:

            global_sent_id = 1

            for doc in processed_docs:

                # fix sent_id consistently
                for sentence in doc.sentences:
                    sentence.sent_id = str(global_sent_id)
                    global_sent_id += 1

                # write clean CoNLL-U block
                CoNLL.write_doc2conll(doc, f)

                f.write("\n")
# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    main()