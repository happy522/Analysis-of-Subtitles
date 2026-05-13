# Analysis-of-Subtitles

## Multi-parallel Cuervos test set
A multi-parallel dataset containing direct/indirect subtitling. Source ES (Mexico), Pivot: EN + EN_CC, Target: DE_ind + DE_dir. Five episodes from Cuervos were selected for the test set. A professional translator, native in German and living in Mexico was commissioned to produce direct translations from Spanish into German. The subtitles were quality proofed by another professional and controversial issues were resolved through agreement. Indirect subtitles as well as source and pivot were obtained from Netflix using the Netflix downloader. 
Preprocessing and Alignment: All subtitle files (.srt) were merged into sentences using punctuation boundaries and written to txt files. These were then manually aligned (Intertext) to create a multiparallel corpus.

### Dataset Size

| Episode | Sentences | Words (ES, EN, DE, Ded) |
|----------|------------|--------------------------|
| S01E02   | 653        |                          |
| S02E06   | 576        |                          |
| S02E08   | 561        |                          |
| S02E10   | 642        |                          |
| S03E07   | 844        | 4567, 4655, 3979, 3928   |
| **Total: 5** | **3276** |                          |
