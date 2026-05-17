from pathlib import Path
import math

# =========================
# CONFIG
# =========================


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Subtitles Data/conllu/DE_Direkt"

OUTPUT_DIR = BASE_DIR / "Subtitles Data/conllu_chunks/DE_Direkt"


CHUNK_SIZE = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# FUNCTION TO SPLIT CONLLU
# =========================

def read_conllu_sentences(filepath):
    """
    Reads a CoNLL-U file and returns a list of sentences.
    Each sentence is stored as one complete block.
    """
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Sentences separated by blank lines
    sentences = content.split("\n\n")

    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


# =========================
# PROCESS FILES
# =========================

all_files = list(INPUT_DIR.glob("*.conllu"))

print(f"Found {len(all_files)} files")

for file_path in all_files:

    print(f"\nProcessing: {file_path.name}")

    sentences = read_conllu_sentences(file_path)

    total_sentences = len(sentences)

    print(f"Total sentences: {total_sentences}")
    chunks = []

    for i in range(0, total_sentences, CHUNK_SIZE):
        chunks.append(sentences[i:i + CHUNK_SIZE])

    # Merge tiny last chunk
    if len(chunks) > 1 and len(chunks[-1]) < 25:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    num_chunks = math.ceil(total_sentences / CHUNK_SIZE)

    print(f"Creating {num_chunks} chunks")

    for chunk_idx in range(num_chunks):

        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE

        chunk_sentences = sentences[start:end]

        # Skip empty chunks
        if not chunk_sentences:
            continue

        output_filename = (
            f"{file_path.stem}_chunk_{chunk_idx+1:03d}.conllu"
        )

        output_path = OUTPUT_DIR / output_filename

        with open(output_path, "w", encoding="utf-8") as out_f:

            out_f.write("\n\n".join(chunk_sentences))
            out_f.write("\n\n")

        print(
            f"  Wrote {output_filename} "
            f"({len(chunk_sentences)} sentences)"
        )

print("\nDone.")